"""This module implements two-dimensional padded wavelet transforms.

The implementation relies on torch.nn.functional.conv2d and
torch.nn.functional.conv_transpose2d under the hood.
"""

from __future__ import annotations

from typing import Optional, Union

import pywt
import torch

from ._util import (
    Wavelet,
    _check_same_device_dtype,
    _get_len,
    _outer,
    _pad_symmetric,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import BoundaryMode, WaveletCoeff2d, WaveletDetailTuple2d
from .conv_transform import (
    _adjust_padding_at_reconstruction,
    _get_filter_tensors,
    _get_pad,
    _translate_boundary_strings,
)


def _construct_2d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct two-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        Stacked 2d-filters of dimension

        [filt_no, 1, height, width].

        The four filters are ordered ll, lh, hl, hh.

    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    filt = filt.unsqueeze(1)
    return filt


def _fwt_pad2(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: Optional[BoundaryMode] = None,
    padding: Optional[tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    This function pads along the last two axes.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode: The desired padding mode for extending the signal along the edges.
            Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.
        padding (tuple[int, int, int, int], optional): A tuple
            (padl, padr, padt, padb) with the number of padded values
            on the left, right, top and bottom side of the last two
            axes of `data`. If None, the padding values are computed based
            on the signal shape and the wavelet length. Defaults to None.

    Returns:
        The padded output tensor.

    """
    if mode is None:
        mode = "reflect"
    pytorch_mode = _translate_boundary_strings(mode)

    if padding is None:
        padb, padt = _get_pad(data.shape[-2], _get_len(wavelet))
        padr, padl = _get_pad(data.shape[-1], _get_len(wavelet))
    else:
        padl, padr, padt, padb = padding
    if pytorch_mode == "symmetric":
        data_pad = _pad_symmetric(data, [(padt, padb), (padl, padr)])
    else:
        data_pad = torch.nn.functional.pad(
            data, [padl, padr, padt, padb], mode=pytorch_mode
        )
    return data_pad


def wavedec2(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axes: tuple[int, int] = (-2, -1),
) -> WaveletCoeff2d:
    r"""Run a two-dimensional wavelet transformation.

    This function relies on two-dimensional convolutions.
    Outer products allow the construction of 2D-filters from 1D filter arrays
    :ref:`(see fwt-intro) <sec-fwt-2d>`.
    It transforms the last two axes by default.
    This function computes

    .. math::
        \mathbf{x}_s *_2 \mathbf{h}_k = \mathbf{c}_{k, s+1}

    with :math:`k \in [a, h, v, d]` and
    :math:`s \in \mathbb{N}_0` the set of natural numbers,
    where :math:`\mathbf{x}_0` is equal to
    the original input image :math:`\mathbf{X}`.
    :math:`*_2` indicates two dimensional-convolution.
    Computations at subsequent scales work exclusively with
    approximation coefficients :math:`c_{a, s}`
    as inputs. Setting the `level` argument allows choosing the largest scale.

    Args:
        data (torch.Tensor): The input data tensor with any number of dimensions.
            By default 2d inputs are interpreted as ``[height, width]``,
            3d inputs are interpreted as ``[batch_size, height, width]``.
            4d inputs are interpreted as ``[batch_size, channels, height, width]``.
            The ``axes`` argument allows other interpretations.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        mode :
            The desired padding mode for extending the signal along the edges.
            Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.
        level (int): The number of desired scales.
            Defaults to None.
        axes (tuple[int, int]): Compute the transform over these axes instead of the
            last two. Defaults to (-2, -1).

    Returns:
        A tuple containing the wavelet coefficients in pywt order,
        see :data:`ptwt.constants.WaveletCoeff2d`.

    Example:
        >>> import torch
        >>> import ptwt, pywt
        >>> import numpy as np
        >>> from scipy import datasets
        >>> face = np.transpose(datasets.face(),
        >>>                     [2, 0, 1]).astype(np.float64)
        >>> pytorch_face = torch.tensor(face) # try unsqueeze(0)
        >>> coefficients = ptwt.wavedec2(pytorch_face, pywt.Wavelet("haar"),
        >>>                              level=2, mode="zero")

    """
    data, ds = _preprocess_tensor(data, ndim=2, axes=axes)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_2d_filt(lo=dec_lo, hi=dec_hi)

    if level is None:
        level = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst: list[WaveletDetailTuple2d] = []
    res_ll = data
    for _ in range(level):
        res_ll = _fwt_pad2(res_ll, wavelet, mode=mode)
        res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        to_append = WaveletDetailTuple2d(
            res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)
        )
        result_lst.append(to_append)

    result_lst.reverse()
    res_ll = res_ll.squeeze(1)
    result: WaveletCoeff2d = res_ll, *result_lst

    result = _postprocess_coeffs(result, ndim=2, ds=ds, axes=axes)

    return result


def waverec2(
    coeffs: WaveletCoeff2d,
    wavelet: Union[Wavelet, str],
    axes: tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    """Reconstruct a signal from wavelet coefficients.

    This function undoes the effect of the analysis
    or forward transform by running transposed convolutions.

    Args:
        coeffs (WaveletCoeff2d): The wavelet coefficient tuple
            produced by wavedec2. See :data:`ptwt.constants.WaveletCoeff2d`
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
            Refer to the output from ``pywt.wavelist(kind='discrete')``
            for possible choices.
        axes (tuple[int, int]): Compute the transform over these axes instead of the
            last two. Defaults to (-2, -1).

    Returns:
        The reconstructed signal tensor of shape ``[batch, height, width]`` or
        ``[batch, channel, height, width]`` depending on the input to `wavedec2`.

    Raises:
        ValueError: If coeffs is not in a shape as returned from wavedec2 or
            if the dtype is not supported or if the provided axes input has length other
            than two or if the same axes it repeated twice.

    Example:
        >>> import ptwt, pywt, torch
        >>> import numpy as np
        >>> from scipy import datasets
        >>> face = np.transpose(datasets.face(),
        >>>                     [2, 0, 1]).astype(np.float64)
        >>> pytorch_face = torch.tensor(face)
        >>> coefficients = ptwt.wavedec2(pytorch_face, pywt.Wavelet("haar"),
        >>>                              level=2, mode="constant")
        >>> reconstruction = ptwt.waverec2(coefficients, pywt.Wavelet("haar"))

    """
    coeffs, ds = _preprocess_coeffs(coeffs, ndim=2, axes=axes)
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_2d_filt(lo=rec_lo, hi=rec_hi)

    res_ll = coeffs[0]
    for c_pos, coeff_tuple in enumerate(coeffs[1:]):
        if not isinstance(coeff_tuple, tuple) or len(coeff_tuple) != 3:
            raise ValueError(
                f"Unexpected detail coefficient type: {type(coeff_tuple)}. Detail "
                "coefficients must be a 3-tuple of tensors as returned by "
                "wavedec2."
            )

        curr_shape = res_ll.shape
        for coeff in coeff_tuple:
            if coeff.shape != curr_shape:
                raise ValueError(
                    "All coefficients on each level must have the same shape"
                )

        res_lh, res_hl, res_hh = coeff_tuple
        res_ll = torch.stack([res_ll, res_lh, res_hl, res_hh], 1)
        res_ll = torch.nn.functional.conv_transpose2d(
            res_ll, rec_filt, stride=2
        ).squeeze(1)

        # remove the padding
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
