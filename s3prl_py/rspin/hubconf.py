import logging

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def rspin_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, **kwargs)


def rspin_local(*args, **kwargs):
    return rspin_custom(*args, **kwargs)


def rspin_url(*args, **kwargs):
    return rspin_custom(*args, **kwargs)


def rspin_wavlm_32_40k(refresh=False, **kwargs):
    kwargs["ckpt"] = (
        "https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_32-40k.pt"
    )
    return rspin_custom(refresh=refresh, **kwargs)


def rspin_wavlm_64_40k(refresh=False, **kwargs):
    kwargs["ckpt"] = (
        "https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_64-40k.pt"
    )
    return rspin_custom(refresh=refresh, **kwargs)


def rspin_wavlm_128_40k(refresh=False, **kwargs):
    kwargs["ckpt"] = (
        "https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_128-40k.pt"
    )
    return rspin_custom(refresh=refresh, **kwargs)


def rspin_wavlm_256_40k(refresh=False, **kwargs):
    kwargs["ckpt"] = (
        "https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_256-40k.pt"
    )
    return rspin_custom(refresh=refresh, **kwargs)


def rspin_wavlm_512_40k(refresh=False, **kwargs):
    kwargs["ckpt"] = (
        "https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_512-40k.pt"
    )
    return rspin_custom(refresh=refresh, **kwargs)


def rspin_wavlm_1024_40k(refresh=False, **kwargs):
    kwargs["ckpt"] = (
        "https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_1024-40k.pt"
    )
    return rspin_custom(refresh=refresh, **kwargs)


def rspin_wavlm_2048_40k(refresh=False, **kwargs):
    kwargs["ckpt"] = (
        "https://data.csail.mit.edu/public-release-sls/rspin/wavlm_rspin_2048-40k.pt"
    )
    return rspin_custom(refresh=refresh, **kwargs)
