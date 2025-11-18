# Copyright 2021 Moritz Wolter
# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the EUPL v1.2
#
# This file contains code derived from PyTorch-Wavelet-Toolbox:
# https://github.com/v0lta/PyTorch-Wavelet-Toolbox
#
# Original work by Moritz Wolter, licensed under EUPL v1.2
# Modifications and integration by HuggingFace Inc. team

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple, Protocol, TypeAlias, Union, cast, overload

import numpy as np
import torch
from typing_extensions import Unpack

from .wavelet import Wavelet as minimal_wavelet


class WaveletDetailTuple2d(NamedTuple):
    horizontal: torch.Tensor
    vertical: torch.Tensor
    diagonal: torch.Tensor


WaveletCoeff2d: TypeAlias = tuple[torch.Tensor, Unpack[tuple[WaveletDetailTuple2d, ...]]]
WaveletDetailDict: TypeAlias = dict[str, torch.Tensor]
WaveletCoeffNd: TypeAlias = tuple[torch.Tensor, Unpack[tuple[WaveletDetailDict, ...]]]


class Wavelet(Protocol):
    name: str
    dec_lo: Sequence[float]
    dec_hi: Sequence[float]
    rec_lo: Sequence[float]
    rec_hi: Sequence[float]
    dec_len: int
    rec_len: int
    filter_bank: tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]

    def __len__(self) -> int:
        return len(self.dec_lo)


class WaveletTensorTuple(NamedTuple):
    dec_lo: torch.Tensor
    dec_hi: torch.Tensor
    rec_lo: torch.Tensor
    rec_hi: torch.Tensor

    @classmethod
    def from_wavelet(cls, wavelet: Wavelet, dtype: torch.dtype) -> "WaveletTensorTuple":
        return cls(
            torch.tensor(wavelet.dec_lo, dtype=dtype),
            torch.tensor(wavelet.dec_hi, dtype=dtype),
            torch.tensor(wavelet.rec_lo, dtype=dtype),
            torch.tensor(wavelet.rec_hi, dtype=dtype),
        )


def _as_wavelet(wavelet: Union[Wavelet, str]) -> Wavelet:
    if isinstance(wavelet, str):
        return minimal_wavelet(wavelet)
    else:
        return wavelet


def _is_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


def _check_if_tensor(array: Any) -> torch.Tensor:
    if not isinstance(array, torch.Tensor):
        raise ValueError("First element of coeffs must be the approximation coefficient tensor.")
    return array


def _check_axes_argument(axes: Sequence[int]) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError("Cant transform the same axis twice.")


def _check_same_device(tensor: torch.Tensor, torch_device: torch.device) -> torch.Tensor:
    if torch_device != tensor.device:
        raise ValueError("coefficients must be on the same device")
    return tensor


def _check_same_dtype(tensor: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    if torch_dtype != tensor.dtype:
        raise ValueError("coefficients must have the same dtype")
    return tensor


@overload
def _coeff_tree_map(
    coeffs: list[torch.Tensor], function: Callable[[torch.Tensor], torch.Tensor]
) -> list[torch.Tensor]: ...
@overload
def _coeff_tree_map(coeffs: WaveletCoeff2d, function: Callable[[torch.Tensor], torch.Tensor]) -> WaveletCoeff2d: ...
@overload
def _coeff_tree_map(coeffs: WaveletCoeffNd, function: Callable[[torch.Tensor], torch.Tensor]) -> WaveletCoeffNd: ...
def _coeff_tree_map(coeffs, function):
    approx = function(coeffs[0])
    result_lst: list[Any] = []
    for element in coeffs[1:]:
        if isinstance(element, tuple):
            result_lst.append(WaveletDetailTuple2d(function(element[0]), function(element[1]), function(element[2])))
        elif isinstance(element, dict):
            new_dict = {key: function(value) for key, value in element.items()}
            result_lst.append(new_dict)
        elif isinstance(element, torch.Tensor):
            result_lst.append(function(element))
        else:
            raise ValueError(f"Unexpected input type {type(element)}")
    if not result_lst:
        return [approx] if isinstance(coeffs, list) else (approx,)
    elif isinstance(result_lst[0], torch.Tensor):
        return [approx] + cast(list[torch.Tensor], result_lst)
    else:
        cast_result_lst = cast(Union[list[WaveletDetailDict], list[WaveletDetailTuple2d]], result_lst)
        return (approx, *cast_result_lst)


def _check_same_device_dtype(
    coeffs: Union[list[torch.Tensor], WaveletCoeff2d, WaveletCoeffNd],
) -> tuple[torch.device, torch.dtype]:
    c = _check_if_tensor(coeffs[0])
    torch_device, torch_dtype = c.device, c.dtype
    _coeff_tree_map(coeffs, partial(_check_same_device, torch_device=torch_device))
    _coeff_tree_map(coeffs, partial(_check_same_dtype, torch_dtype=torch_dtype))
    return torch_device, torch_dtype


def _get_transpose_order(axes: Sequence[int], data_shape: Sequence[int]) -> tuple[list[int], list[int]]:
    axes = [a + len(data_shape) if a < 0 else a for a in axes]
    all_axes = list(range(len(data_shape)))
    remove_transformed = list(filter(lambda a: a not in axes, all_axes))
    return remove_transformed, axes


def _swap_axes(data: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    return torch.permute(data, front + back)


def _undo_swap_axes(data: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    _check_axes_argument(axes)
    front, back = _get_transpose_order(axes, list(data.shape))
    restore_sorted = torch.argsort(torch.tensor(front + back)).tolist()
    return torch.permute(data, restore_sorted)


def _fold_axes(data: torch.Tensor, keep_no: int) -> tuple[torch.Tensor, list[int]]:
    dshape = list(data.shape)
    return (torch.reshape(data, [int(np.prod(dshape[:-keep_no]))] + dshape[-keep_no:]), dshape)


def _unfold_axes(data: torch.Tensor, ds: list[int], keep_no: int) -> torch.Tensor:
    return torch.reshape(data, ds[:-keep_no] + list(data.shape[-keep_no:]))


def _preprocess_coeffs(coeffs, ndim: int, axes, add_channel_dim: bool = False):
    if isinstance(axes, int):
        axes = (axes,)
    torch_dtype = _check_if_tensor(coeffs[0]).dtype
    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")
    if ndim <= 0:
        raise ValueError("Number of dimensions must be positive")
    if tuple(axes) != tuple(range(-ndim, 0)):
        if len(axes) != ndim:
            raise ValueError(f"{ndim}D transforms work with {ndim} axes.")
        else:
            swap_fn = partial(_swap_axes, axes=axes)
            coeffs = _coeff_tree_map(coeffs, swap_fn)
    ds = list(coeffs[0].shape)
    if len(ds) < ndim:
        raise ValueError(f"At least {ndim} input dimensions required.")
    elif len(ds) == ndim:
        coeffs = _coeff_tree_map(coeffs, lambda x: x.unsqueeze(0))
    elif len(ds) > ndim + 1:
        coeffs = _coeff_tree_map(coeffs, lambda t: _fold_axes(t, ndim)[0])
    if add_channel_dim:
        coeffs = _coeff_tree_map(coeffs, lambda x: x.unsqueeze(1))
    return coeffs, ds


def _postprocess_coeffs(coeffs, ndim: int, ds: list[int], axes):
    if isinstance(axes, int):
        axes = (axes,)
    if ndim <= 0:
        raise ValueError("Number of dimensions must be positive")
    if len(ds) < ndim:
        raise ValueError(f"At least {ndim} input dimensions required.")
    elif len(ds) == ndim:
        coeffs = _coeff_tree_map(coeffs, lambda x: x.squeeze(0))
    elif len(ds) > ndim + 1:
        unfold_axes_fn = partial(_unfold_axes, ds=ds, keep_no=ndim)
        coeffs = _coeff_tree_map(coeffs, unfold_axes_fn)
    if tuple(axes) != tuple(range(-ndim, 0)):
        if len(axes) != ndim:
            raise ValueError(f"{ndim}D transforms work with {ndim} axes.")
        else:
            undo_swap_fn = partial(_undo_swap_axes, axes=axes)
            coeffs = _coeff_tree_map(coeffs, undo_swap_fn)
    return coeffs


def _postprocess_tensor(
    data: torch.Tensor, ndim: int, ds: list[int], axes: Union[tuple[int, ...], int]
) -> torch.Tensor:
    return _postprocess_coeffs(coeffs=[data], ndim=ndim, ds=ds, axes=axes)[0]


def _get_filter_tensors(
    wavelet: Union[Wavelet, str], flip: bool, device: torch.device, dtype: torch.dtype
) -> WaveletTensorTuple:
    wavelet = _as_wavelet(wavelet)
    if flip:
        filters = WaveletTensorTuple(
            torch.tensor(wavelet.rec_lo, device=device, dtype=dtype),
            torch.tensor(wavelet.rec_hi, device=device, dtype=dtype),
            torch.tensor(wavelet.dec_lo, device=device, dtype=dtype),
            torch.tensor(wavelet.dec_hi, device=device, dtype=dtype),
        )
    else:
        filters = WaveletTensorTuple.from_wavelet(wavelet, dtype=dtype)
        filters = WaveletTensorTuple(
            filters.dec_lo.to(device),
            filters.dec_hi.to(device),
            filters.rec_lo.to(device),
            filters.rec_hi.to(device),
        )
    return filters


def _adjust_padding_at_reconstruction(tensor_len: int, coeff_len: int, padr: int, padl: int) -> tuple[int, int]:
    if 2 * coeff_len - tensor_len == 1:
        padr += 1
    elif 2 * coeff_len - tensor_len != 0:
        raise ValueError("incorrect padding")
    return padr, padl


def _construct_2d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    filt = filt.unsqueeze(1)
    return filt


def waverec2d(
    coeffs: WaveletCoeff2d,
    wavelet: Union[Wavelet, str],
    axes: tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    coeffs, ds = _preprocess_coeffs(coeffs, ndim=2, axes=axes)
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(wavelet, flip=False, device=torch_device, dtype=torch_dtype)
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_2d_filt(lo=rec_lo, hi=rec_hi)

    res_ll = coeffs[0]
    for c_pos, coeff_tuple in enumerate(coeffs[1:]):
        if not isinstance(coeff_tuple, tuple) or len(coeff_tuple) != 3:
            raise ValueError(f"Unexpected detail coefficient type: {type(coeff_tuple)}. Must be a 3-tuple.")

        curr_shape = res_ll.shape
        for coeff in coeff_tuple:
            if coeff.shape != curr_shape:
                raise ValueError("All coefficients on each level must have the same shape")

        res_lh, res_hl, res_hh = coeff_tuple
        res_ll = torch.stack([res_ll, res_lh, res_hl, res_hh], 1)
        res_ll = torch.nn.functional.conv_transpose2d(res_ll, rec_filt, stride=2).squeeze(1)

        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        if c_pos < len(coeffs) - 2:
            padr, padl = _adjust_padding_at_reconstruction(
                res_ll.shape[-1], coeffs[c_pos + 2][0].shape[-1], padr, padl
            )
            padb, padt = _adjust_padding_at_reconstruction(
                res_ll.shape[-2], coeffs[c_pos + 2][0].shape[-2], padb, padt
            )

        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]

    res_ll = _postprocess_tensor(res_ll, ndim=2, ds=ds, axes=axes)
    return res_ll
