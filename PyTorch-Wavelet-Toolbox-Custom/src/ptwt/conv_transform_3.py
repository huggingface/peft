"""Code for three dimensional padded transforms.

The functions here are based on torch.nn.functional.conv3d and it's transpose.
"""

from __future__ import annotations

from typing import Optional, Union

import pywt
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_same_device_dtype,
    _get_len,
    _outer,
    _pad_symmetric,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import BoundaryMode, WaveletCoeffNd, WaveletDetailDict
from .conv_transform import (
    _adjust_padding_at_reconstruction,
    _get_filter_tensors,
    _get_pad,
    _translate_boundary_strings,
)


def _construct_3d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct three-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        Stacked 3d filters of dimension::

        [8, 1, length, height, width].

        The four filters are ordered ll, lh, hl, hh.
    """
    dim_size = lo.shape[-1]
    size = [dim_size] * 3
    lll = _outer(lo, _outer(lo, lo)).reshape(size)
    llh = _outer(lo, _outer(lo, hi)).reshape(size)
    lhl = _outer(lo, _outer(hi, lo)).reshape(size)
    lhh = _outer(lo, _outer(hi, hi)).reshape(size)
    hll = _outer(hi, _outer(lo, lo)).reshape(size)
    hlh = _outer(hi, _outer(lo, hi)).reshape(size)
    hhl = _outer(hi, _outer(hi, lo)).reshape(size)
    hhh = _outer(hi, _outer(hi, hi)).reshape(size)
    filt = torch.stack([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], 0)
    filt = filt.unsqueeze(1)
    return filt


def _fwt_pad3(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode,
    padding: Optional[tuple[int, int, int, int, int, int]] = None,
) -> torch.Tensor:
    """Pad data for the 3d-FWT.

    This function pads the last three axes.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode: The desired padding mode for extending the signal along the edges.
            See :data:`ptwt.constants.BoundaryMode`.
        padding (tuple[int, int, int, int, int, int], optional): A tuple
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            with the number of padded values on the respective side of the
            last three axes of `data`.
            If None, the padding values are computed based
            on the signal shape and the wavelet length. Defaults to None.

    Returns:
        The padded output tensor.
    """
    pytorch_mode = _translate_boundary_strings(mode)

    if padding is None:
        pad_back, pad_front = _get_pad(data.shape[-3], _get_len(wavelet))
        pad_bottom, pad_top = _get_pad(data.shape[-2], _get_len(wavelet))
        pad_right, pad_left = _get_pad(data.shape[-1], _get_len(wavelet))
    else:
        pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = padding
    if pytorch_mode == "symmetric":
        data_pad = _pad_symmetric(
            data, [(pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right)]
        )
    else:
        data_pad = torch.nn.functional.pad(
            data,
            [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back],
            mode=pytorch_mode,
        )
    return data_pad


def wavedec3(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "zero",
    level: Optional[int] = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
) -> WaveletCoeffNd:
    """Compute a three-dimensional wavelet transform.

    Args:
        data (torch.Tensor): The input data. For example of shape
            ``[batch_size, length, height, width]``
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "zero". See :data:`ptwt.constants.BoundaryMode`.
        level (Optional[int]): The maximum decomposition level.
            This argument defaults to None.
        axes (tuple[int, int, int]): Compute the transform over these axes
            instead of the last three. Defaults to (-3, -2, -1).

    Returns:
        A tuple containing the wavelet coefficients,
        see :data:`ptwt.constants.WaveletCoeffNd`.

    Example:
        >>> import ptwt, torch
        >>> data = torch.randn(5, 16, 16, 16)
        >>> transformed = ptwt.wavedec3(data, "haar", level=2, mode="reflect")
    """
    data, ds = _preprocess_tensor(data, ndim=3, axes=axes)

    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_3d_filt(lo=dec_lo, hi=dec_hi)

    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2], data.shape[-3]], wavelet
        )

    result_lst: list[WaveletDetailDict] = []
    res_lll = data
    for _ in range(level):
        if len(res_lll.shape) == 4:
            res_lll = res_lll.unsqueeze(1)
        res_lll = _fwt_pad3(res_lll, wavelet, mode=mode)
        res = torch.nn.functional.conv3d(res_lll, dec_filt, stride=2)
        res_lll, res_llh, res_lhl, res_lhh, res_hll, res_hlh, res_hhl, res_hhh = [
            sr.squeeze(1) for sr in torch.split(res, 1, 1)
        ]
        result_lst.append(
            {
                "aad": res_llh,
                "ada": res_lhl,
                "add": res_lhh,
                "daa": res_hll,
                "dad": res_hlh,
                "dda": res_hhl,
                "ddd": res_hhh,
            }
        )
    result_lst.reverse()
    coeffs: WaveletCoeffNd = res_lll, *result_lst

    return _postprocess_coeffs(coeffs, ndim=3, ds=ds, axes=axes)


def waverec3(
    coeffs: WaveletCoeffNd,
    wavelet: Union[Wavelet, str],
    axes: tuple[int, int, int] = (-3, -2, -1),
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    Args:
        coeffs (WaveletCoeffNd): The wavelet coefficient tuple
            produced by wavedec3, see :data:`ptwt.constants.WaveletCoeffNd`.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        axes (tuple[int, int, int]): Transform these axes instead of the
            last three. Defaults to (-3, -2, -1).

    Returns:
        The reconstructed four-dimensional signal tensor of shape
        ``[batch, depth, height, width]``.

    Raises:
        ValueError: If coeffs is not in a shape as returned from wavedec3 or
            if the dtype is not supported or if the provided axes input has length
            other than three or if the same axes it repeated three.

    Example:
        >>> import ptwt, torch
        >>> data = torch.randn(5, 16, 16, 16)
        >>> transformed = ptwt.wavedec3(data, "haar", level=2, mode="reflect")
        >>> reconstruction = ptwt.waverec3(transformed, "haar")
    """
    coeffs, ds = _preprocess_coeffs(coeffs, ndim=3, axes=axes)
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_3d_filt(lo=rec_lo, hi=rec_hi)

    res_lll = coeffs[0]
    coeff_dicts = coeffs[1:]
    for c_pos, coeff_dict in enumerate(coeff_dicts):
        if not isinstance(coeff_dict, dict) or len(coeff_dict) != 7:
            raise ValueError(
                f"Unexpected detail coefficient type: {type(coeff_dict)}. Detail "
                "coefficients must be a dict containing 7 tensors as returned by "
                "wavedec3."
            )
        for coeff in coeff_dict.values():
            if res_lll.shape != coeff.shape:
                raise ValueError(
                    "All coefficients on each level must have the same shape"
                )
        res_lll = torch.stack(
            [
                res_lll,
                coeff_dict["aad"],
                coeff_dict["ada"],
                coeff_dict["add"],
                coeff_dict["daa"],
                coeff_dict["dad"],
                coeff_dict["dda"],
                coeff_dict["ddd"],
            ],
            1,
        )
        res_lll = torch.nn.functional.conv_transpose3d(res_lll, rec_filt, stride=2)
        res_lll = res_lll.squeeze(1)

        # remove the padding
        padfr = (2 * filt_len - 3) // 2
        padba = (2 * filt_len - 3) // 2
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        if c_pos + 1 < len(coeff_dicts):
            padr, padl = _adjust_padding_at_reconstruction(
                res_lll.shape[-1], coeff_dicts[c_pos + 1]["aad"].shape[-1], padr, padl
            )
            padb, padt = _adjust_padding_at_reconstruction(
                res_lll.shape[-2], coeff_dicts[c_pos + 1]["aad"].shape[-2], padb, padt
            )
            padba, padfr = _adjust_padding_at_reconstruction(
                res_lll.shape[-3], coeff_dicts[c_pos + 1]["aad"].shape[-3], padba, padfr
            )
        if padt > 0:
            res_lll = res_lll[..., padt:, :]
        if padb > 0:
            res_lll = res_lll[..., :-padb, :]
        if padl > 0:
            res_lll = res_lll[..., padl:]
        if padr > 0:
            res_lll = res_lll[..., :-padr]
        if padfr > 0:
            res_lll = res_lll[..., padfr:, :, :]
        if padba > 0:
            res_lll = res_lll[..., :-padba, :, :]

    return _postprocess_tensor(res_lll, ndim=3, ds=ds, axes=axes)
