# Ref: https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/interfaces.py

import sys
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


SAMPLE_RATE = 16000
TOLERABLE_SEQLEN_DIFF = 10


class Featurizer(nn.Module):
    def __init__(
        self,
        upstream,
        feature_selection: str = "hidden_states",
        upstream_device: str = "cuda",
        layer_selection: int = None,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = "Featurizer"

        upstream.eval()
        paired_wavs = [torch.randn(SAMPLE_RATE).to(upstream_device)]
        with torch.no_grad():
            paired_features = upstream(paired_wavs)

        if feature_selection not in paired_features:
            if "hidden_states" in paired_features:
                print(
                    f"[{self.name}] - Warning: {feature_selection} is not a supported args.upstream_feature_selection."
                    f' Using "hidden_states" as the default key.',
                    file=sys.stderr,
                )
                feature_selection = "hidden_states"
            else:
                print(
                    f"[{self.name}] - Error: {feature_selection} is not a supported args.upstream_feature_selection."
                    f' The default key "hidden_states" is also not supported.'
                    f" Please specify -s with the following options: {list(paired_wavs.keys())}",
                    file=sys.stderr,
                )
                raise ValueError
        self.feature_selection = feature_selection
        self.layer_selection = layer_selection
        self.normalize = normalize

        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            print(
                f"[{self.name}] - Take a list of {self.layer_num} features and weighted sum them.",
                file=sys.stderr,
            )
            self.weights = nn.Parameter(torch.zeros(self.layer_num))
            feature = self._weighted_sum([f.cpu() for f in feature])
        else:
            feature = feature.cpu()

        self.output_dim = feature.size(-1)
        if hasattr(upstream, "get_downsample_rates"):
            self.downsample_rate = upstream.get_downsample_rates(feature_selection)
            print(
                f"[{self.name}] - The selected feature {feature_selection}'s downsample rate is {self.downsample_rate}",
                file=sys.stderr,
            )
        else:
            self.downsample_rate = round(
                max(len(wav) for wav in paired_wavs) / feature.size(1)
            )
            print(
                f"[{self.name}] - Warning: The provided upstream does not give statis downsample rate"
                ' by the "get_downsample_rates" interface (see upstream/example/expert.py).'
                " The downsample rate is calculated dynamically basing on the shape of the"
                f" input waveforms v.s. the output features: {self.downsample_rate}",
                file=sys.stderr,
            )

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

        if isinstance(feature, dict):
            feature = list(feature.values())

        if isinstance(feature, (list, tuple)) and len(feature) == 1:
            feature = feature[0]

        if isinstance(feature, (list, tuple)) and isinstance(self.layer_selection, int):
            feature = feature[self.layer_selection]

        return feature

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature), (
            "If you run into this error, there is a great chance"
            " you are finetuning the upstream with wav2vec2's transformer blocks"
            " in weighted-sum mode (default), including wav2vec2, hubert, and decoar2."
            " These models use the layerdrop technique which causes the different number"
            " of layer forwards between different model forwards, resulting in different"
            " number of hidden states for different model forwards. Hence, finetuning"
            " these upstreams is essentially incompatible with weight-sum mode unless"
            " you turn off the layerdrop option in fairseq. See:"
            " https://github.com/pytorch/fairseq/blob/f6abcc2a67328bee8b15c596bb626ce2d720aae6/fairseq/models/wav2vec/wav2vec2.py#L857"
            " However, since finetuning upstreams will backward the gradient through all layers"
            " which serves the same functionality as weighted-sum: all layers can be used for different"
            " downstream tasks. Hence instead of finetuning upstream with weighted-sum, we suggest to"
            " follow the more common setting: finetuning upstream with the last layer. Please use the"
            " following options: --upstream_trainable --upstream_feature_selection last_hidden_state."
            " Or: -f -s last_hidden_state"
        )
        stacked_feature = torch.stack(feature, dim=0)

        if self.normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],)
            )

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def tolist(self, paired_wavs: List[Tensor], paired_feature: Tensor):
        assert paired_feature.dim() == 3, "(batch_size, max_seq_len, feat_dim)"
        feature_len = [round(len(wav) / self.downsample_rate) for wav in paired_wavs]
        length_diff = abs(
            paired_feature.size(1)
            - round(max([len(wav) for wav in paired_wavs]) / self.downsample_rate)
        )
        assert (
            length_diff < TOLERABLE_SEQLEN_DIFF
        ), f"{length_diff} >= {TOLERABLE_SEQLEN_DIFF}"
        feature = [f[:l] for f, l in zip(paired_feature, feature_len)]
        return feature

    def forward(
        self,
        paired_wavs: List[Tensor],
        paired_features: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
    ):
        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            feature = self._weighted_sum(feature)

        return self.tolist(paired_wavs, feature)
