import logging
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # set directory of `rspin/`
        sys.path.append("/path/to/rspin")
        from rspin import RSpinWavlm

        self.model = RSpinWavlm.load_from_checkpoint(ckpt)

    def get_downsample_rates(self, key: str) -> int:
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
