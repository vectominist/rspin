# Ref: https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/asr/expert.py

from pathlib import Path
from typing import Dict, List, Optional, Union

import librosa
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .s3prl_asr.asr_model import RNNs
from .s3prl_asr.dictionary import Dictionary
from .s3prl_asr.featurizer import Featurizer
from .model import RSpinWavlm


def create_dict(
    state_dict: dict, start_with_key: str, remove_start_with_key: bool = False
) -> dict:
    new_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith(start_with_key):
            new_key = key[len(start_with_key) :] if remove_start_with_key else key
            new_state_dict[new_key] = val
    return new_state_dict


def token_to_word(text):
    # Hard coding but it is only used here for now.
    # Assumption that units are characters. Doesn't handle BPE.
    # Inter-character separator is " " and inter-word separator is "|".
    return text.replace(" ", "").replace("|", " ").strip()


class RSpinUpstreamExpert(nn.Module):
    def __init__(self, ckpt):
        super().__init__()

        self.model = RSpinWavlm.load_from_checkpoint(ckpt)

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        feat_list, padding_mask, codes = self.model(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=False,
            get_code=True,
        )

        outputs = {}

        outputs["hidden_states"] = feat_list
        outputs["last_hidden_state"] = outputs["hidden_states"][-1]
        outputs["codes"] = codes

        return outputs


class RSpinASR(nn.Module):
    def __init__(self, config: dict, rspin_path: str, device=torch.device("cpu")):
        super().__init__()

        self.config = config

        self.rspin_model = RSpinUpstreamExpert(rspin_path).to(device)
        self.rspin_dim = self.rspin_model.model.encoder_cfg.encoder_embed_dim
        self.rspin_rate = self.rspin_model.get_downsample_rates()

        self.featurizer = Featurizer(self.rspin_model).to(device)
        self.dictionary = Dictionary.load(
            str(Path(__file__).parent / "s3prl_asr" / "char.dict")
        )
        self.projector = nn.Linear(self.rspin_dim, self.config["project_dim"]).to(
            device
        )
        self.asr_model = RNNs(
            self.config["project_dim"],
            len(self.dictionary.symbols),
            self.rspin_rate,
            **self.config["RNNs"],
        ).to(device)
        self.blank = self.dictionary.bos()

    @property
    def device(self):
        return next(self.parameters()).device

    def load_audio(self, audio_paths: List[str], device=None) -> List[torch.Tensor]:
        if device is None:
            device = self.device

        wavs = []
        for path in audio_paths:
            wav, _ = librosa.load(path, sr=16000)
            if wav.ndim == 2:
                wav = wav.mean(-1)
            wavs.append(torch.FloatTensor(wav).to(device))

        return wavs

    def decode(self, log_probs: torch.Tensor, log_probs_len: torch.LongTensor):
        pred_tokens_batch = []
        pred_words_batch = []

        for log_prob, in_len in zip(log_probs, log_probs_len):
            log_prob = log_prob[:in_len].unsqueeze(0)

            pred_token_ids = log_prob.argmax(dim=-1).unique_consecutive()
            pred_token_ids = pred_token_ids[pred_token_ids != self.blank].tolist()
            pred_tokens = self.dictionary.string(pred_token_ids)

            pred_words = token_to_word(pred_tokens).split()

            pred_tokens_batch.append(pred_tokens)
            pred_words_batch.append(pred_words)

        return pred_tokens_batch, pred_words_batch

    def forward(
        self,
        wavs: Optional[List[torch.Tensor]] = None,
        audio_paths: Optional[List[str]] = None,
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Transcribe audio files.

        Args:
            wavs (Optional[List[torch.Tensor]], optional): 1-D tensors of audio waveforms. Defaults to None.
            audio_paths (Optional[List[str]], optional): paths to audio files. Defaults to None.

        Returns:
            Dict[str, torch.Tensor | List[str]]: output dictionary (key "transcription" contains the list of transcriptions)
        """

        if wavs is None:
            assert isinstance(audio_paths, list) and len(audio_paths) > 0
            wavs = self.load_audio(audio_paths)
        if wavs[0].device != self.device:
            wavs = [wav.to(self.device) for wav in wavs]
        assert wavs[0].dim() == 1

        rspin_outputs = self.rspin_model(wavs)
        features = self.featurizer(wavs, rspin_outputs)
        features_len = torch.LongTensor([len(feat) for feat in features])
        features = pad_sequence(features, batch_first=True).to(device=self.device)
        features = self.projector(features)
        logits, log_probs_len = self.asr_model(features, features_len)
        log_probs = torch.log_softmax(logits, dim=-1)
        pred_tokens_batch, pred_words_batch = self.decode(
            log_probs.float().contiguous().cpu(), log_probs_len
        )
        transcription = [" ".join(text) for text in pred_words_batch]

        return {
            "features": features,
            "features_len": features_len,
            "logits": logits,
            "log_probs": log_probs,
            "log_probs_len": log_probs_len,
            "pred_tokens_batch": pred_tokens_batch,
            "pred_words_batch": pred_words_batch,
            "transcription": transcription,
        }

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        rspin_path: str,
        map_location=None,
        device=torch.device("cpu"),
    ):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        config = ckpt["Config"]["downstream_expert"]["modelrc"]
        model = cls(config, rspin_path, device)
        model.featurizer.load_state_dict(ckpt["Featurizer"])
        model.projector.load_state_dict(
            create_dict(ckpt["Downstream"], "projector.", True)
        )
        model.asr_model.load_state_dict(create_dict(ckpt["Downstream"], "model.", True))
        return model
