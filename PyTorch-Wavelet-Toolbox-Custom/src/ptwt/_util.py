"""Utility methods to compute wavelet decompositions from a dataset."""

from __future__ import annotations

import functools
import typing
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, NamedTuple, Optional, Protocol, Union, cast, overload

import numpy as np
import pywt
import torch
from typing_extensions import ParamSpec, TypeVar

from .constants import (
    OrthogonalizeMethod,
    WaveletCoeff2d,
    WaveletCoeffNd,
    WaveletDetailDict,
    WaveletDetailTuple2d,
)


class Wavelet(Protocol):
    """Wavelet object interface, based on the pywt wavelet object."""

    name: str
    dec_lo: Sequence[float]
    dec_hi: Sequence[float]
    rec_lo: Sequence[float]
    rec_hi: Sequence[float]
    dec_len: int
    rec_len: int
    filter_bank: tuple[
        Sequence[float], Sequence[float], Sequence[float], Sequence[float]
    ]

    def __len__(self) -> int:
        """Return the number of filter coefficients."""
        return len(self.dec_lo)


class WaveletTensorTuple(NamedTuple):
    """Named tuple containing the wavelet filter bank to use in JIT code."""

    dec_lo: torch.Tensor
    dec_hi: torch.Tensor
    rec_lo: torch.Tensor
    rec_hi: torch.Tensor

    @property
    def dec_len(self) -> int:
        """Length of decomposition filters."""
        return len(self.dec_lo)

    @property
    def rec_len(self) -> int:
        """Length of reconstruction filters."""
        return len(self.rec_lo)

    @property
    def filter_bank(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter bank of the wavelet."""
        return self

    @classmethod
    def from_wavelet(cls, wavelet: Wavelet, dtype: torch.dtype) -> WaveletTensorTuple:
        """Construct Wavelet named tuple from wavelet protocol member."""
        return cls(
            torch.tensor(wavelet.dec_lo, dtype=dtype),
            torch.tensor(wavelet.dec_hi, dtype=dtype),
            torch.tensor(wavelet.rec_lo, dtype=dtype),
            torch.tensor(wavelet.rec_hi, dtype=dtype),
        )


def _as_wavelet(wavelet: Union[Wavelet, str]) -> Wavelet:
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        The input wavelet object or the pywt wavelet object described by the input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _is_orthogonalize_method_supported(
    orthogonalization: Optional[OrthogonalizeMethod],
) -> bool:
    return orthogonalization in typing.get_args(OrthogonalizeMethod)


def _is_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


def _get_len(wavelet: Union[tuple[torch.Tensor, ...], str, Wavelet]) -> int:
    """Get number of filter coefficients for various wavelet data types."""
    if isinstance(wavelet, tuple):
        return wavelet[0].shape[0]
    else:
        return len(_as_wavelet(wavelet))


def _pad_symmetric_1d(signal: torch.Tensor, pad_list: tuple[int, int]) -> torch.Tensor:
    padl, padr = pad_list
    dimlen = signal.shape[0]
    if padl > dimlen or padr > dimlen:
        if padl > dimlen:
            signal = _pad_symmetric_1d(signal, (dimlen, 0))
            padl = padl - dimlen
        if padr > dimlen:
            signal = _pad_symmetric_1d(signal, (0, dimlen))
            padr = padr - dimlen
        return _pad_symmetric_1d(signal, (padl, padr))
    else:
        cat_list = [signal]
        if padl > 0:
            cat_list.insert(0, signal[:padl].flip(0))
        if padr > 0:
            cat_list.append(signal[-padr::].flip(0))
        return torch.cat(cat_list, dim=0)


def _pad_symmetric(
    signal: torch.Tensor, pad_lists: Sequence[tuple[int, int]]
) -> torch.Tensor:
    if len(signal.shape) < len(pad_lists):
        raise ValueError("not enough dimensions to pad.")

    dims = len(signal.shape) - 1
    for pos, pad_list in enumerate(pad_lists[::-1]):
        current_axis = dims - pos
        signal = signal.transpose(0, current_axis)
        signal = _pad_symmetric_1d(signal, pad_list)
        signal = signal.transpose(current_axis, 0)
    return signal


def _fold_axes(data: torch.Tensor, keep_no: int) -> tuple[torch.Tensor, list[int]]:
    """Fold unchanged leading dimensions into a single batch dimension.

    Args:
        data (torch.Tensor): The input data array.
        keep_no (int): The number of dimensions to keep.

    Returns:
        A tuple (result_tensor, input_shape) where result_tensor is the
        folded result array, and input_shape the shape of the original input.
    """
    dshape = list(data.shape)
    return (
        torch.reshape(data, [int(np.prod(dshape[:-keep_no]))] + dshape[-keep_no:]),
        dshape,
    )


def _unfold_axes(data: torch.Tensor, ds: list[int], keep_no: int) -> torch.Tensor:
    """Unfold i.e. [batch*channel,height,widht] to [batch,channel,height,width]."""
    return torch.reshape(data, ds[:-keep_no] + list(data.shape[-keep_no:]))


def _check_if_tensor(array: Any) -> torch.Tensor:
    if not isinstance(array, torch.Tensor):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient tensor."
        )
    return array


def _check_axes_argument(axes: Sequence[int]) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError("Cant transform the same axis twice.")


def _check_same_device(
    tensor: torch.Tensor, torch_device: torch.device
) -> torch.Tensor:
    if torch_device != tensor.device:
        raise ValueError("coefficients must be on the same device")
    return tensor


def _check_same_dtype(tensor: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    if torch_dtype != tensor.dtype:
        raise ValueError("coefficients must have the same dtype")
    return tensor


def _check_same_device_dtype(
    coeffs: Union[list[torch.Tensor], WaveletCoeff2d, WaveletCoeffNd],
) -> tuple[torch.device, torch.dtype]:
    """Check coefficients for dtype and device consistency.

    Check that all coefficient tensors in `coeffs` have the same
    device and dtype.

    Args:
        coeffs (Wavelet coefficients): The resulting coefficients of
            a discrete wavelet transform. Can be either of
            `list[torch.Tensor]` (1d case),
            :data:`ptwt.constants.WaveletCoeff2d` (2d case) or
            :data:`ptwt.constants.WaveletCoeffNd` (Nd case).

    Returns:
        A tuple (device, dtype) with the shared device and dtype of
        all tensors in coeffs.
    """
    c = _check_if_tensor(coeffs[0])
    torch_device, torch_dtype = c.device, c.dtype

    # check for all tensors in `coeffs` that the device matches `torch_device`
    _coeff_tree_map(coeffs, partial(_check_same_device, torch_device=torch_device))
    # check for all tensors in `coeffs` that the dtype matches `torch_dtype`
    _coeff_tree_map(coeffs, partial(_check_same_dtype, torch_dtype=torch_dtype))

    return torch_device, torch_dtype


def _get_transpose_order(
    axes: Sequence[int], data_shape: Sequence[int]
) -> tuple[list[int], list[int]]:
    axes = list(map(lambda a: a + len(data_shape) if a < 0 else a, axes))
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


@overload
def _coeff_tree_map(
    coeffs: list[torch.Tensor],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]: ...


@overload
def _coeff_tree_map(
    coeffs: WaveletCoeff2d,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> WaveletCoeff2d: ...


@overload
def _coeff_tree_map(
    coeffs: WaveletCoeffNd,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> WaveletCoeffNd: ...


def _coeff_tree_map(
    coeffs: Union[list[torch.Tensor], WaveletCoeff2d, WaveletCoeffNd],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> Union[list[torch.Tensor], WaveletCoeff2d, WaveletCoeffNd]:
    """Apply `function` to all tensor elements in `coeffs`.

    Applying a function to all tensors in the (potentially nested)
    coefficient data structure is a common requirement in coefficient
    pre- and postprocessing. This function saves us from having to loop
    over the coefficient data structures in processing.

    Conceptually, this function is inspired by the
    pytree processing philosophy of the JAX framework, see
    https://jax.readthedocs.io/en/latest/working-with-pytrees.html

    Raises:
        ValueError: If the input type is not supported.
    """
    approx = function(coeffs[0])
    result_lst: list[
        Union[
            torch.Tensor,
            WaveletDetailDict,
            WaveletDetailTuple2d,
        ]
    ] = []
    for element in coeffs[1:]:
        if isinstance(element, tuple):
            result_lst.append(
                WaveletDetailTuple2d(
                    function(element[0]),
                    function(element[1]),
                    function(element[2]),
                )
            )
        elif isinstance(element, dict):
            new_dict = {key: function(value) for key, value in element.items()}
            result_lst.append(new_dict)
        elif isinstance(element, torch.Tensor):
            result_lst.append(function(element))
        else:
            raise ValueError(f"Unexpected input type {type(element)}")

    if not result_lst:
        # if only approximation coeff:
        # use list iff data is a list
        return [approx] if isinstance(coeffs, list) else (approx,)
    elif isinstance(result_lst[0], torch.Tensor):
        # if the first detail coeff is tensor
        # -> all are tensors -> return a list
        return [approx] + cast(list[torch.Tensor], result_lst)
    else:
        # cast since we assume that the full list is of the same type
        cast_result_lst = cast(
            Union[list[WaveletDetailDict], list[WaveletDetailTuple2d]], result_lst
        )
        return approx, *cast_result_lst


# 1d case
@overload
def _preprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: Literal[1],
    axes: int,
    add_channel_dim: bool = False,
) -> tuple[list[torch.Tensor], list[int]]: ...


# 2d case
@overload
def _preprocess_coeffs(
    coeffs: WaveletCoeff2d,
    ndim: Literal[2],
    axes: tuple[int, int],
    add_channel_dim: bool = False,
) -> tuple[WaveletCoeff2d, list[int]]: ...


# Nd case
@overload
def _preprocess_coeffs(
    coeffs: WaveletCoeffNd,
    ndim: int,
    axes: tuple[int, ...],
    add_channel_dim: bool = False,
) -> tuple[WaveletCoeffNd, list[int]]: ...


# list of nd tensors
@overload
def _preprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: int,
    axes: Union[tuple[int, ...], int],
    add_channel_dim: bool = False,
) -> tuple[list[torch.Tensor], list[int]]: ...


def _preprocess_coeffs(
    coeffs: Union[
        list[torch.Tensor],
        WaveletCoeff2d,
        WaveletCoeffNd,
    ],
    ndim: int,
    axes: Union[tuple[int, ...], int],
    add_channel_dim: bool = False,
) -> tuple[
    Union[
        list[torch.Tensor],
        WaveletCoeff2d,
        WaveletCoeffNd,
    ],
    list[int],
]:
    """Preprocess coeff tensor dimensions.

    For each coefficient tensor in `coeffs` the transformed axes
    as specified by `axes` are moved to be the last.
    Adds a batch dim if a coefficient tensor has none.
    If it has has multiple batch dimensions, they are folded into a single
    batch dimension.

    Args:
        coeffs (Wavelet coefficients): The resulting coefficients of
            a discrete wavelet transform. Can be either of
            `list[torch.Tensor]` (1d case),
            :data:`ptwt.constants.WaveletCoeff2d` (2d case) or
            :data:`ptwt.constants.WaveletCoeffNd` (Nd case).
        ndim (int): The number of axes :math:`N` on which the transformation
            was applied.
        axes (int or tuple of ints): Axes on which the transform was calculated.
        add_channel_dim (bool): If True, ensures that all returned coefficients
            have at least `:math:`N + 2` axes by potentially adding a new axis at dim 1.
            Defaults to False.

    Returns:
        A tuple ``(coeffs, ds)`` where ``coeffs`` are the transformed
        coefficients and ``ds`` contains the original shape of ``coeffs[0]``.
        If `add_channel_dim` is True, all coefficient tensors have
        :math:`N + 2` axes ([B, 1, c1, ..., cN]).
        otherwise :math:`N + 1` ([B, c1, ..., cN]).

    Raises:
        ValueError: If the input dtype is unsupported or `ndim` does not
            fit to the passed `axes` or `coeffs` dimensions.
    """
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
            # for all tensors in `coeffs`: swap the axes
            swap_fn = partial(_swap_axes, axes=axes)
            coeffs = _coeff_tree_map(coeffs, swap_fn)

    # Fold axes for the wavelets
    ds = list(coeffs[0].shape)
    if len(ds) < ndim:
        raise ValueError(f"At least {ndim} input dimensions required.")
    elif len(ds) == ndim:
        # for all tensors in `coeffs`: unsqueeze(0)
        coeffs = _coeff_tree_map(coeffs, lambda x: x.unsqueeze(0))
    elif len(ds) > ndim + 1:
        # for all tensors in `coeffs`: fold leading dims to batch dim
        coeffs = _coeff_tree_map(coeffs, lambda t: _fold_axes(t, ndim)[0])

    if add_channel_dim:
        # for all tensors in `coeffs`: add channel dim
        coeffs = _coeff_tree_map(coeffs, lambda x: x.unsqueeze(1))

    return coeffs, ds


# 1d case
@overload
def _postprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: Literal[1],
    ds: list[int],
    axes: int,
) -> list[torch.Tensor]: ...


# 2d case
@overload
def _postprocess_coeffs(
    coeffs: WaveletCoeff2d,
    ndim: Literal[2],
    ds: list[int],
    axes: tuple[int, int],
) -> WaveletCoeff2d: ...


# Nd case
@overload
def _postprocess_coeffs(
    coeffs: WaveletCoeffNd,
    ndim: int,
    ds: list[int],
    axes: tuple[int, ...],
) -> WaveletCoeffNd: ...


# list of nd tensors
@overload
def _postprocess_coeffs(
    coeffs: list[torch.Tensor],
    ndim: int,
    ds: list[int],
    axes: Union[tuple[int, ...], int],
) -> list[torch.Tensor]: ...


def _postprocess_coeffs(
    coeffs: Union[
        list[torch.Tensor],
        WaveletCoeff2d,
        WaveletCoeffNd,
    ],
    ndim: int,
    ds: list[int],
    axes: Union[tuple[int, ...], int],
) -> Union[
    list[torch.Tensor],
    WaveletCoeff2d,
    WaveletCoeffNd,
]:
    """Postprocess coeff tensor dimensions.

    This revereses the operations of :func:`_preprocess_coeffs`.

    Unfolds potentially folded batch dimensions and removes any added
    dimensions.
    The transformed axes as specified by `axes` are moved back to their
    original position.

    Args:
        coeffs (Wavelet coefficients): The preprocessed coefficients of
            a discrete wavelet transform. Can be either of
            `list[torch.Tensor]` (1d case),
            :data:`ptwt.constants.WaveletCoeff2d` (2d case) or
            :data:`ptwt.constants.WaveletCoeffNd` (Nd case).
        ndim (int): The number of axes :math:`N` on which the transformation was
            applied.
        ds (list of ints): The shape of the original first coefficient before
            preprocessing, i.e. of ``coeffs[0]``.
        axes (int or tuple of ints): Axes on which the transform was calculated.

    Returns:
        The result of undoing the preprocessing operations on `coeffs`.

    Raises:
        ValueError: If `ndim` does not fit to the passed `axes`
            or `coeffs` dimensions.
    """
    if isinstance(axes, int):
        axes = (axes,)

    if ndim <= 0:
        raise ValueError("Number of dimensions must be positive")

    # Fold axes for the wavelets
    if len(ds) < ndim:
        raise ValueError(f"At least {ndim} input dimensions required.")
    elif len(ds) == ndim:
        # for all tensors in `coeffs`: remove batch dim
        coeffs = _coeff_tree_map(coeffs, lambda x: x.squeeze(0))
    elif len(ds) > ndim + 1:
        # for all tensors in `coeffs`: unfold batch dim
        unfold_axes_fn = partial(_unfold_axes, ds=ds, keep_no=ndim)
        coeffs = _coeff_tree_map(coeffs, unfold_axes_fn)

    if tuple(axes) != tuple(range(-ndim, 0)):
        if len(axes) != ndim:
            raise ValueError(f"{ndim}D transforms work with {ndim} axes.")
        else:
            # for all tensors in `coeffs`: undo axes swapping
            undo_swap_fn = partial(_undo_swap_axes, axes=axes)
            coeffs = _coeff_tree_map(coeffs, undo_swap_fn)

    return coeffs


def _preprocess_tensor(
    data: torch.Tensor,
    ndim: int,
    axes: Union[tuple[int, ...], int],
    add_channel_dim: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Preprocess input tensor dimensions.

    The transformed axes as specified by `axes` are moved to be the last.
    Adds a batch dim if `data` has none.
    If `data` has multiple batch dimensions, they are folded into a single
    batch dimension.

    Args:
        data (torch.Tensor): An input tensor with at least `ndim` axes.
        ndim (int): The number of axes :math:`N` on which the transformation is
            applied.
        axes (int or tuple of ints): Axes on which the transform is calculated.
        add_channel_dim (bool): If True, ensures that the return has at
            least :math:`N + 2` axes by potentially adding a new axis at dim 1.
            Defaults to True.

    Returns:
        A tuple ``(data, ds)`` where ``data`` is the transformed data tensor
        and ``ds`` contains the original shape.
        If `add_channel_dim` is True,
        `data` has :math:`N + 2` axes ([B, 1, d1, ..., dN]).
        otherwise :math:`N + 1` ([B, d1, ..., dN]).
    """
    # interpreting data as the approximation coeffs of a 0-level FWT
    # allows us to reuse the `_preprocess_coeffs` code
    data_lst, ds = _preprocess_coeffs(
        [data], ndim=ndim, axes=axes, add_channel_dim=add_channel_dim
    )
    return data_lst[0], ds


def _postprocess_tensor(
    data: torch.Tensor, ndim: int, ds: list[int], axes: Union[tuple[int, ...], int]
) -> torch.Tensor:
    """Postprocess input tensor dimensions.

    This revereses the operations of :func:`_preprocess_tensor`.

    Unfolds potentially folded batch dimensions and removes any added
    dimensions.
    The transformed axes as specified by `axes` are moved back to their
    original position.

    Args:
        data (torch.Tensor): An preprocessed input tensor.
        ndim (int): The number of axes :math:`N` on which the transformation is
            applied.
        ds (list of ints): The shape of the original input tensor before
            preprocessing.
        axes (int or tuple of ints): Axes on which the transform was calculated.

    Returns:
        The result of undoing the preprocessing operations on `data`.
    """
    # interpreting data as the approximation coeffs of a 0-level FWT
    # allows us to reuse the `_postprocess_coeffs` code
    # return approx, *cast_result_lst
    return _postprocess_coeffs(coeffs=[data], ndim=ndim, ds=ds, axes=axes)[0]


Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def _deprecated_alias(
    **aliases: str,
) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
    """Handle deprecated function and method arguments.

    Use as follows::

        @_deprecated_alias(old_arg='new_arg')
        def myfunc(new_arg):
            ...

    Adapted from https://stackoverflow.com/a/49802489
    """

    def rename_kwargs(
        func_name: str,
        kwargs: Param.kwargs,
        aliases: dict[str, str],
    ) -> None:
        """Rename deprecated kwarg."""
        for alias, new in aliases.items():
            if alias in kwargs:
                if new in kwargs:
                    raise TypeError(
                        f"{func_name} received both {alias} and {new} as arguments!"
                        f" {alias} is deprecated, use {new} instead."
                    )
                warnings.warn(
                    message=(
                        f"`{alias}` is deprecated as an argument to `{func_name}`; use"
                        f" `{new}` instead."
                    ),
                    category=DeprecationWarning,
                    stacklevel=3,
                )
                kwargs[new] = kwargs.pop(alias)

    def deco(f: Callable[Param, RetType]) -> Callable[Param, RetType]:
        @functools.wraps(f)
        def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco
