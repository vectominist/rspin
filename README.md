# R-Spin Inference Code

## Introduction

<p align="center"><img src="https://github.com/vectominist/rspin/blob/main/figure/rspin.png?raw=true" alt="Spin framework." width="800"/></p>

This repository is the official PyTorch implementation of the **Robust Speaker-invariant Clustering** (**R-Spin**) proposed in the **NAACL 2024** paper [R-Spin: Efficient Speaker and Noise-invariant Representation Learning with Acoustic Pieces](https://arxiv.org/abs/2311.09117) ([Heng-Jui Chang](https://people.csail.mit.edu/hengjui/) and [James Glass](https://www.csail.mit.edu/person/jim-glass); [MIT CSAIL](https://www.csail.mit.edu/)).


## Citation
Please cite our paper if you find this repository and/or the paper useful.
```bib
@inproceedings{chang-glass-2024-r,
    title = "{R}-Spin: Efficient Speaker and Noise-invariant Representation Learning with Acoustic Pieces",
    author = "Chang, Heng-Jui and Glass, James",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.36",
    doi = "10.18653/v1/2024.naacl-long.36",
    pages = "642--662",
}
```

## Getting Started

1. Install the latest [PyTorch](https://pytorch.org/) and [soundfile](https://github.com/bastibe/python-soundfile).
    ```bash
    pip install torch soundfile librosa
    ```
2. Download checkpoint (WavLM + R-Spin):

| Spin Codebook Size | Acoustic Pieces | Checkpoint                                                                          |
| ------------------ | --------------- | ----------------------------------------------------------------------------------- |
| 32 (best for ASR)  | 40k             | [link](https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_32-40k.pt)   |
| 64                 | 40k             | [link](https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_64-40k.pt)   |
| 128                | 40k             | [link](https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_128-40k.pt)  |
| 256                | 40k             | [link](https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_256-40k.pt)  |
| 512                | 40k             | [link](https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_512-40k.pt)  |
| 1024               | 40k             | [link](https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_1024-40k.pt) |
| 2048               | 40k             | [link](https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_2048-40k.pt) |


## Run Inference Code

```python
import soundfile as sf
import torch
from rspin import RSpinWavlm

# Load model
model = RSpinWavlm.load_from_checkpoint("/path/to/checkpoint").cuda()
model.eval()

# Load audio (needs to be 16kHz)
wav, sr = sf.read("/path/to/audio")
assert sr == 16000
if wav.ndim == 2:
    wav = wav.mean(-1)
wav = torch.FloatTensor(wav).cuda()

# Inference
with torch.inference_mode():
    feat_list, padding_mask, codes = model(wav.unsqueeze(0), get_code=True)
    print(codes[0])
    # feat_list: List[torch.Tensor] (shape: (batch, seq_len, encoder_emb_dim))
    # codes: torch.LongTensor (shape: (batch, seq_len))
```

* The `feat_list` consists a list of all hidden representations (including the CNN feature extractor output).
* The `codes` is a `LongTensor` representing the codeword IDs produced by the Spin codebook.
* See [rspin/model.py](https://github.com/vectominist/rspin/blob/main/rspin/model.py) for more information.


## Application: S3PRL
We use the [s3prl](https://github.com/s3prl/s3prl) toolkit for [SUPERB](https://arxiv.org/abs/2105.01051) downstream tasks.
1. Modify [line 24](https://github.com/vectominist/rspin/blob/main/s3prl_py/rspin/expert.py#L24) of [`s3prl_py/rspin/expert.py`](https://github.com/vectominist/rspin/blob/main/s3prl_py/rspin/expert.py) to the absolute path to `rspin/`.
2. Copy the `s3prl_py/rspin` directory to `s3prl` so that the toolkit can load the models.
```bash
cp -R s3prl_py/rspin ../s3prl/s3prl/upstream/rspin
```
3. Add the following line to `../s3prl/s3prl/hub.py`:
```python
from s3prl.upstream.rspin.hubconf import *
```
4. Try loading R-Spin models with the following methods
```python
from s3prl.nn import S3PRLUpstream

# Method 1 (download checkpoints manually with the provided links)
model = S3PRLUpstream("rspin_local", path_or_url="/path/to/checkpoint")

# Method 2 (download checkpoint with s3prl)
model = S3PRLUpstream("rspin_wavlm_32_40k")
```

## Application: End-to-end ASR (S3PRL)
After training an ASR model with the [s3prl](https://github.com/s3prl/s3prl) toolkit, you may use the model to transcribe speech.
```python
import torch
from rspin import RSpinASR

device = torch.device("cuda")
asr_model = RSpinASR.load_from_checkpoint(
    "/path/to/s3prl/checkpoint", "/path/to/rspin/checkpoint", device=device
)
asr_model.eval()

with torch.no_grad():
    audio_paths = ["/audio/file/1", "/audio/file/2"]
    results = asr_model(audio_paths=audio_paths)
    for transcription in results["transcription"]:
        print(transcription)
```


## References

* [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)
* [s3prl](https://github.com/s3prl/s3prl)
* [PyTorch](https://pytorch.org/)
* [spin](https://github.com/vectominist/spin)


## Contact
If you have any questions, please open an issue or email `hengjui [at] mit.edu`.
