# conv_transform.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import pywt
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_same_device_dtype,
    _get_len,
    _pad_symmetric,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import BoundaryMode, WaveletCoeff2d


def _create_tensor(
    filter: Sequence[float], flip: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if flip:
        if isinstance(filter, torch.Tensor):
            return filter.flip(-1).unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
    else:
        if isinstance(filter, torch.Tensor):
            return filter.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)


def _get_filter_tensors(
    wavelet: Union[Wavelet, str],
    flip: bool,
    device: Union[torch.device, str],
    # Changed default from torch.float32 to None:
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        flip (bool): Flip filters left-right, if true.
        device (torch.device or str): PyTorch target device.
        dtype (torch.dtype, optional): If None, defaults to torch.float32.
                                       Otherwise, wavelet filters will be
                                       created in this dtype.

    Returns:
        A tuple (dec_lo, dec_hi, rec_lo, rec_hi) containing
        the four filter tensors
    """
    # If user didn’t specify, default to float32.
    if dtype is None:
        dtype = torch.float32

    wavelet = _as_wavelet(wavelet)
    device = torch.device(device)

    if isinstance(wavelet, tuple):
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet
    else:
        dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo, flip, device, dtype)
    dec_hi_tensor = _create_tensor(dec_hi, flip, device, dtype)
    rec_lo_tensor = _create_tensor(rec_lo, flip, device, dtype)
    rec_hi_tensor = _create_tensor(rec_hi, flip, device, dtype)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> tuple[int, int]:
    # pad to ensure we see all filter positions and for pywt compatability.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2
    padr += data_len % 2
    return padr, padl


def _translate_boundary_strings(pywt_mode: BoundaryMode) -> str:
    if pywt_mode == "constant":
        return "replicate"
    elif pywt_mode == "zero":
        return "constant"
    elif pywt_mode == "reflect":
        return pywt_mode
    elif pywt_mode == "periodic":
        return "circular"
    elif pywt_mode == "symmetric":
        # pytorch does not support symmetric mode, we have our own implementation.
        return pywt_mode
    raise ValueError(f"Padding mode not supported: {pywt_mode}")


def _fwt_pad(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: Optional[BoundaryMode] = None,
    padding: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    if mode is None:
        mode = "reflect"
    pytorch_mode = _translate_boundary_strings(mode)

    if padding is None:
        padr, padl = _get_pad(data.shape[-1], _get_len(wavelet))
    else:
        padl, padr = padding

    if pytorch_mode == "symmetric":
        data_pad = _pad_symmetric(data, [(padl, padr)])
    else:
        data_pad = torch.nn.functional.pad(data, [padl, padr], mode=pytorch_mode)
    return data_pad


def _flatten_2d_coeff_lst(
    coeff_lst_2d: WaveletCoeff2d,
    flatten_tensors: bool = True,
) -> list[torch.Tensor]:
    def _process_tensor(coeff: torch.Tensor) -> torch.Tensor:
        return coeff.flatten() if flatten_tensors else coeff

    flat_coeff_lst = [_process_tensor(coeff_lst_2d[0])]
    for coeff_tuple in coeff_lst_2d[1:]:
        flat_coeff_lst.extend(map(_process_tensor, coeff_tuple))
    return flat_coeff_lst


def _adjust_padding_at_reconstruction(
    res_ll_size: int, coeff_size: int, pad_end: int, pad_start: int
) -> tuple[int, int]:
    pred_size = res_ll_size - (pad_start + pad_end)
    next_size = coeff_size
    if next_size == pred_size:
        pass
    elif next_size == pred_size - 1:
        pad_end += 1
    else:
        raise AssertionError(
            "padding error, please check if dec and rec wavelets are identical."
        )
    return pad_end, pad_start


def wavedec(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axis: int = -1,
) -> list[torch.Tensor]:
    """Compute the analysis (forward) 1d fast wavelet transform."""
    data, ds = _preprocess_tensor(data, ndim=1, axes=axis)

    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    filt_len = dec_lo.shape[-1]
    filt = torch.stack([dec_lo, dec_hi], 0)

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], filt_len)

    result_list = []
    res_lo = data
    for _ in range(level):
        res_lo = _fwt_pad(res_lo, wavelet, mode=mode)
        res = torch.nn.functional.conv1d(res_lo, filt, stride=2)
        res_lo, res_hi = torch.split(res, 1, 1)
        result_list.append(res_hi.squeeze(1))
    result_list.append(res_lo.squeeze(1))
    result_list.reverse()

    return _postprocess_coeffs(result_list, ndim=1, ds=ds, axes=axis)


def waverec(
    coeffs, wavelet: Union[Wavelet, str], axis: int = -1
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients."""
    if not isinstance(coeffs, list):
        coeffs = list(coeffs)
    coeffs, ds = _preprocess_coeffs(coeffs, ndim=1, axes=axis)
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    filt = torch.stack([rec_lo, rec_hi], 0)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        res_lo = torch.stack([res_lo, res_hi], 1)
        res_lo = torch.nn.functional.conv_transpose1d(res_lo, filt, stride=2).squeeze(1)

        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        if c_pos < len(coeffs) - 2:
            padr, padl = _adjust_padding_at_reconstruction(
                res_lo.shape[-1], coeffs[c_pos + 2].shape[-1], padr, padl
            )
        if padl > 0:
            res_lo = res_lo[..., padl:]
        if padr > 0:
            res_lo = res_lo[..., :-padr]

    res_lo = _postprocess_tensor(res_lo, ndim=1, ds=ds, axes=axis)
    return res_lo
