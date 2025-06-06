"""Two-dimensional matrix based fast wavelet transform implementations.

This module uses boundary filters to minimize padding.
"""

from __future__ import annotations

import sys
from typing import Optional, Union, cast

import numpy as np
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_axes_argument,
    _check_same_device_dtype,
    _deprecated_alias,
    _is_orthogonalize_method_supported,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import (
    BoundaryMode,
    OrthogonalizeMethod,
    PaddingMode,
    WaveletCoeff2d,
    WaveletDetailTuple2d,
)
from .conv_transform import _get_filter_tensors
from .conv_transform_2 import _construct_2d_filt, _fwt_pad2
from .matmul_transform import (
    BaseMatrixWaveDec,
    construct_boundary_a,
    construct_boundary_s,
    orthogonalize,
)
from .sparse_math import (
    batch_mm,
    cat_sparse_identity_matrix,
    construct_strided_conv2d_matrix,
)


def _construct_a_2(
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.float64,
    mode: PaddingMode = "sameshift",
) -> torch.Tensor:
    """Construct a raw two-dimensional analysis wavelet transformation matrix.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        height (int): The height of the input image.
        width (int): The width of the input image.
        device (torch.device or str): Where to place the matrix.
        dtype (torch.dtype, optional): Desired matrix data type.
            Defaults to torch.float64.
        mode : The convolution type.
            Options are 'full', 'valid', 'same' and 'sameshift'.
            Defaults to 'sameshift'.

    Returns:
        A sparse fwt analysis matrix.
        The matrices are ordered a, h, v, d or ll, lh, hl, hh.

    Note:
        The constructed matrix is NOT necessarily orthogonal.
        In most cases, construct_boundary_a2d should be used instead.
    """
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=False, device=device, dtype=dtype
    )
    dec_filt = _construct_2d_filt(lo=dec_lo, hi=dec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    analysis_ll = construct_strided_conv2d_matrix(ll, height, width, mode=mode)
    analysis_lh = construct_strided_conv2d_matrix(lh, height, width, mode=mode)
    analysis_hl = construct_strided_conv2d_matrix(hl, height, width, mode=mode)
    analysis_hh = construct_strided_conv2d_matrix(hh, height, width, mode=mode)
    analysis = torch.cat([analysis_ll, analysis_lh, analysis_hl, analysis_hh], 0)
    return analysis


def _construct_s_2(
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.float64,
    mode: PaddingMode = "sameshift",
) -> torch.Tensor:
    """Construct a raw fast wavelet transformation synthesis matrix.

    Note:
        The constructed matrix is NOT necessarily orthogonal.
        In most cases, construct_boundary_s2d should be used instead.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        height (int): The height of the input image, which was originally
            transformed.
        width (int): The width of the input image, which was originally
            transformed.
        device (torch.device): Where to place the synthesis matrix,
            usually CPU or GPU.
        dtype (torch.dtype, optional): The data type the matrix should have.
            Defaults to torch.float64.
        mode : The convolution type.
            Options are 'full', 'valid', 'same' and 'sameshift'.
            Defaults to 'sameshift'.

    Returns:
        The generated fast wavelet synthesis matrix.
    """
    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=True, device=device, dtype=dtype
    )
    dec_filt = _construct_2d_filt(lo=rec_lo, hi=rec_hi)
    ll, lh, hl, hh = dec_filt.squeeze(1)
    synthesis_ll = construct_strided_conv2d_matrix(ll, height, width, mode=mode)
    synthesis_lh = construct_strided_conv2d_matrix(lh, height, width, mode=mode)
    synthesis_hl = construct_strided_conv2d_matrix(hl, height, width, mode=mode)
    synthesis_hh = construct_strided_conv2d_matrix(hh, height, width, mode=mode)
    synthesis = torch.cat(
        [synthesis_ll, synthesis_lh, synthesis_hl, synthesis_hh], 0
    ).coalesce()
    indices = synthesis.indices()
    shape = synthesis.shape
    transpose_indices = torch.stack([indices[1, :], indices[0, :]])
    transpose_synthesis = torch.sparse_coo_tensor(
        transpose_indices, synthesis.values(), size=(shape[1], shape[0]), device=device
    )
    return transpose_synthesis


@_deprecated_alias(boundary="orthogonalization")
def construct_boundary_a2(
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    orthogonalization: OrthogonalizeMethod = "qr",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a boundary fwt matrix for the input wavelet.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        height (int): The height of the input matrix.
            Should be divisible by two.
        width (int): The width of the input matrix.
            Should be divisible by two.
        device (torch.device): Where to place the matrix. Either on
            the CPU or GPU.
        orthogonalization: The method used to orthogonalize
            boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
            Defaults to 'qr'.
        dtype (torch.dtype, optional): The desired data type for the matrix.
            Defaults to torch.float64.

    .. versionchanged:: 1.10
        The argument `boundary` has been renamed to `orthogonalization`.

    Returns:
        A sparse fwt matrix, with orthogonalized boundary wavelets.
    """
    wavelet = _as_wavelet(wavelet)
    a = _construct_a_2(wavelet, height, width, device, dtype=dtype, mode="sameshift")
    orth_a = orthogonalize(
        a, wavelet.dec_len**2, method=orthogonalization
    )  # noqa: BLK100
    return orth_a


@_deprecated_alias(boundary="orthogonalization")
def construct_boundary_s2(
    wavelet: Union[Wavelet, str],
    height: int,
    width: int,
    device: Union[torch.device, str],
    *,
    orthogonalization: OrthogonalizeMethod = "qr",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a 2d-fwt matrix, with boundary wavelets.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        height (int): The original height of the input matrix.
        width (int): The width of the original input matrix.
        device (torch.device): Choose CPU or GPU.
        orthogonalization: The method used to orthogonalize
            boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
            Defaults to 'qr'.
        dtype (torch.dtype, optional): The data type of the
            sparse matrix, choose float32 or 64.
            Defaults to torch.float64.

    .. versionchanged:: 1.10
        The argument `boundary` has been renamed to `orthogonalization`.

    Returns:
        The synthesis matrix, used to compute the inverse fast wavelet transform.
    """
    wavelet = _as_wavelet(wavelet)
    s = _construct_s_2(wavelet, height, width, device, dtype=dtype)
    orth_s = orthogonalize(
        s.transpose(1, 0),
        wavelet.rec_len**2,
        method=orthogonalization,  # noqa: BLK100
    ).transpose(1, 0)
    return orth_s


def _matrix_pad_2(height: int, width: int) -> tuple[int, int, tuple[bool, bool]]:
    pad_tuple = (False, False)
    if height % 2 != 0:
        height += 1
        pad_tuple = (pad_tuple[0], True)
    if width % 2 != 0:
        width += 1
        pad_tuple = (True, pad_tuple[1])
    return height, width, pad_tuple


class MatrixWavedec2(BaseMatrixWaveDec):
    """Experimental sparse matrix 2d wavelet transform.

    For a completely pad-free transform,
    input images are expected to be divisible by two.
    For multiscale transforms all intermediate
    scale dimensions should be divisible
    by two, i.e. ``128, 128 -> 64, 64 -> 32, 32`` would work
    well for a level three transform.
    In this case multiplication with the `sparse_fwt_operator`
    property is equivalent.

    Note:
        Constructing the sparse fwt-matrix is expensive.
        For longer wavelets, high-level transforms, and large
        input images this may take a while.
        The matrix is therefore constructed only once.
        In the non-separable case, it can be accessed via
        the sparse_fwt_operator property.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> from scipy import datasets
        >>> face = datasets.face()[:256, :256, :].astype(np.float32)
        >>> pt_face = torch.tensor(face).permute([2, 0, 1])
        >>> matrixfwt = ptwt.MatrixWavedec2(pywt.Wavelet("haar"), level=2)
        >>> mat_coeff = matrixfwt(pt_face)
    """

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        axes: tuple[int, int] = (-2, -1),
        orthogonalization: OrthogonalizeMethod = "qr",
        separable: bool = True,
        odd_coeff_padding_mode: BoundaryMode = "zero",
    ):
        """Create a new matrix fwt object.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            level (int, optional): The level up to which to compute the fwt. If None,
                the maximum level based on the signal length is chosen. Defaults to
                None.
            axes (int, int): A tuple with the axes to transform.
                Defaults to (-2, -1).
            orthogonalization: The method used to orthogonalize
                boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
                Defaults to 'qr'.
            separable (bool): If this flag is set, a separable transformation
                is used, i.e. a 1d transformation along each axis.
                Matrix construction is significantly faster for separable
                transformations since only a small constant-size part of the
                matrices must be orthogonalized. Defaults to True.
            odd_coeff_padding_mode: The constructed FWT matrices require inputs
                with even lengths. Thus, any odd-length approximation coefficients
                are padded to an even length using this mode,
                see :data:`ptwt.constants.BoundaryMode`.
                Defaults to 'zero'.

        .. versionchanged:: 1.10
            The argument `boundary` has been renamed to `orthogonalization`.

        Raises:
            NotImplementedError: If the selected `orthogonalization` mode
                is not supported.
            ValueError: If the wavelet filters have different lengths.
        """
        self.wavelet = _as_wavelet(wavelet)
        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            _check_axes_argument(axes)
            self.axes = axes
        self.level = level
        self.orthogonalization = orthogonalization
        self.odd_coeff_padding_mode = odd_coeff_padding_mode
        self.separable = separable
        self.input_signal_shape: Optional[tuple[int, int]] = None
        self.fwt_matrix_list: list[
            Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        ] = []
        self.pad_list: list[tuple[bool, bool]] = []
        self.padded = False

        if not _is_orthogonalize_method_supported(self.orthogonalization):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    @property
    def sparse_fwt_operator(self) -> torch.Tensor:
        """Compute the operator matrix for padding-free cases.

        This property exists to make the transformation matrix available.
        To benefit from code handling odd-length levels call the object.

        Returns:
            The sparse 2d-fwt operator matrix.

        Raises:
            NotImplementedError: if a separable transformation was used or if padding
                had to be used in the creation of the transformation matrices.
            ValueError: If no level transformation matrices are stored (most likely
                since the object was not called yet).
        """
        if self.separable:
            raise NotImplementedError

        # in the non-separable case the list entries are tensors
        fwt_matrix_list = cast(list[torch.Tensor], self.fwt_matrix_list)

        if len(fwt_matrix_list) == 1:
            return fwt_matrix_list[0]
        elif len(fwt_matrix_list) > 1:
            if self.padded:
                raise NotImplementedError

            fwt_matrix = fwt_matrix_list[0]
            for scale_mat in fwt_matrix_list[1:]:
                scale_mat = cat_sparse_identity_matrix(scale_mat, fwt_matrix.shape[0])
                fwt_matrix = torch.sparse.mm(scale_mat, fwt_matrix)
            return fwt_matrix
        else:
            raise ValueError(
                "Call this object first to create the transformation matrices for each "
                "level."
            )

    def _construct_analysis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError
        self.fwt_matrix_list = []
        self.size_list = []
        self.pad_list = []
        self.padded = False

        filt_len = self.wavelet.dec_len
        current_height, current_width = self.input_signal_shape
        for curr_level in range(1, self.level + 1):
            if current_height < filt_len or current_width < filt_len:
                # we have reached the max decomposition depth.
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f"height and width ({current_height}, {current_width}) is smaller "
                    f"then the filter length {filt_len}. Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            # the conv matrices require even length inputs.
            current_height, current_width, pad_tuple = _matrix_pad_2(
                current_height, current_width
            )
            if any(pad_tuple):
                self.padded = True
            self.pad_list.append(pad_tuple)
            self.size_list.append((current_height, current_width))
            if self.separable:
                analysis_matrix_rows = construct_boundary_a(
                    wavelet=self.wavelet,
                    length=current_height,
                    orthogonalization=self.orthogonalization,
                    device=device,
                    dtype=dtype,
                )
                analysis_matrix_cols = construct_boundary_a(
                    wavelet=self.wavelet,
                    length=current_width,
                    orthogonalization=self.orthogonalization,
                    device=device,
                    dtype=dtype,
                )
                self.fwt_matrix_list.append(
                    (analysis_matrix_rows, analysis_matrix_cols)
                )
            else:
                analysis_matrix_2d = construct_boundary_a2(
                    wavelet=self.wavelet,
                    height=current_height,
                    width=current_width,
                    orthogonalization=self.orthogonalization,
                    device=device,
                    dtype=dtype,
                )
                self.fwt_matrix_list.append(analysis_matrix_2d)
            current_height = current_height // 2
            current_width = current_width // 2
        self.size_list.append((current_height, current_width))

    def __call__(self, input_signal: torch.Tensor) -> WaveletCoeff2d:
        """Compute the fwt for the given input signal.

        The fwt matrix is set up during the first call
        and stored for future use.

        Args:
            input_signal (torch.Tensor): An input signal of shape
                ``[batch_size, height, width]``.
                2d inputs are interpreted as ``[height, width]``.
                4d inputs as ``[batch_size, channels, height, width]``.
                This transform affects the last two dimensions.

        Returns:
            The resulting coefficients per level are stored in a pywt style tuple,
            see :data:`ptwt.constants.WaveletCoeff2d`.

        Raises:
            ValueError: If the decomposition level is not a positive integer
                or if the input signal has not the expected shape.
        """
        input_signal, ds = _preprocess_tensor(
            input_signal, ndim=2, axes=self.axes, add_channel_dim=False
        )

        batch_size, height, width = input_signal.shape

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != height
            or self.input_signal_shape[1] != width
        ):
            self.input_signal_shape = height, width
            re_build = True

        if self.level is None:
            wlen = len(self.wavelet)
            self.level = int(
                np.min([np.log2(height / (wlen - 1)), np.log2(width / (wlen - 1))])
            )
            re_build = True
        elif self.level <= 0:
            raise ValueError("level must be a positive integer.")

        if not self.fwt_matrix_list or re_build:
            self._construct_analysis_matrices(
                device=input_signal.device, dtype=input_signal.dtype
            )

        def _add_padding(signal: torch.Tensor, pad: tuple[bool, bool]) -> torch.Tensor:
            if pad[0] or pad[1]:
                padding_0 = (0, 1) if pad[0] else (0, 0)
                padding_1 = (0, 1) if pad[1] else (0, 0)

                signal = _fwt_pad2(
                    signal,
                    wavelet=self.wavelet,
                    mode=self.odd_coeff_padding_mode,
                    padding=padding_0 + padding_1,
                )
            return signal

        split_list: list[WaveletDetailTuple2d] = []
        if self.separable:
            ll = input_signal
            for scale, fwt_mats in enumerate(self.fwt_matrix_list):
                fwt_row_matrix, fwt_col_matrix = fwt_mats
                pad = self.pad_list[scale]
                current_height, current_width = self.size_list[scale]
                ll = _add_padding(ll, pad)

                ll = batch_mm(fwt_col_matrix, ll.transpose(-2, -1)).transpose(-2, -1)
                ll = batch_mm(fwt_row_matrix, ll)

                a_coeffs, d_coeffs = torch.split(ll, current_height // 2, dim=-2)
                ll, lh = torch.split(a_coeffs, current_width // 2, dim=-1)
                hl, hh = torch.split(d_coeffs, current_width // 2, dim=-1)

                split_list.append(WaveletDetailTuple2d(lh, hl, hh))
        else:
            ll = input_signal.transpose(-2, -1).reshape([batch_size, -1]).T
            for scale, fwt_matrix in enumerate(self.fwt_matrix_list):
                fwt_matrix = cast(torch.Tensor, fwt_matrix)
                pad = self.pad_list[scale]
                size = self.size_list[scale]
                if pad[0] or pad[1]:
                    ll_reshape = ll.T.reshape(
                        batch_size, size[1] - int(pad[0]), size[0] - int(pad[1])
                    ).transpose(2, 1)
                    ll_reshape = _add_padding(ll_reshape, pad)
                    ll = ll_reshape.transpose(2, 1).reshape([batch_size, -1]).T
                coefficients = torch.sparse.mm(fwt_matrix, ll)
                # get the ll,
                four_split = torch.split(
                    coefficients, int(np.prod((size[0] // 2, size[1] // 2)))
                )
                reshaped = cast(
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    tuple(
                        (
                            el.T.reshape(
                                batch_size, size[1] // 2, size[0] // 2
                            ).transpose(2, 1)
                        )
                        for el in four_split[1:]
                    ),
                )
                split_list.append(WaveletDetailTuple2d(*reshaped))
                ll = four_split[0]
            ll = ll.T.reshape(batch_size, size[1] // 2, size[0] // 2).transpose(2, 1)

        split_list.reverse()
        result: WaveletCoeff2d = ll, *split_list

        result = _postprocess_coeffs(result, ndim=2, ds=ds, axes=self.axes)

        return result


class MatrixWaverec2(object):
    """Synthesis or inverse matrix based-wavelet transformation object.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> from scipy import datasets
        >>> face = datasets.face()[:256, :256, :].astype(np.float32)
        >>> pt_face = torch.tensor(face).permute([2, 0, 1])
        >>> matrixfwt = ptwt.MatrixWavedec2(pywt.Wavelet("haar"), level=2)
        >>> mat_coeff = matrixfwt(pt_face)
        >>> matrixifwt = ptwt.MatrixWaverec2(pywt.Wavelet("haar"))
        >>> reconstruction = matrixifwt(mat_coeff)
    """

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        axes: tuple[int, int] = (-2, -1),
        orthogonalization: OrthogonalizeMethod = "qr",
        separable: bool = True,
    ):
        """Create the inverse matrix-based fast wavelet transformation.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            axes (int, int): The axes transformed by waverec2.
                Defaults to (-2, -1).
            orthogonalization: The method used to orthogonalize
                boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
                Defaults to 'qr'.
            separable (bool): If this flag is set, a separable transformation
                is used, i.e. a 1d transformation along each axis. This is significantly
                faster than a non-separable transformation since only a small constant-
                size part of the matrices must be orthogonalized.
                For invertibility, the analysis and synthesis values must be identical!
                Defaults to True.

        .. versionchanged:: 1.10
            The argument `boundary` has been renamed to `orthogonalization`.

        Raises:
            NotImplementedError: If the selected `orthogonalization` mode
                is not supported.
            ValueError: If the wavelet filters have different lengths.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.orthogonalization = orthogonalization
        self.separable = separable

        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            _check_axes_argument(axes)
            self.axes = axes

        self.ifwt_matrix_list: list[
            Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        ] = []
        self.level: Optional[int] = None
        self.input_signal_shape: Optional[tuple[int, int]] = None

        self.padded = False

        if not _is_orthogonalize_method_supported(self.orthogonalization):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    @property
    def sparse_ifwt_operator(self) -> torch.Tensor:
        """Compute the ifwt operator matrix for pad-free cases.

        Returns:
            The sparse 2d ifwt operator matrix.

        Raises:
            NotImplementedError: if a separable transformation was used or if padding
                had to be used in the creation of the transformation matrices.
            ValueError: If no level transformation matrices are stored (most likely
                since the object was not called yet).
        """
        if self.separable:
            raise NotImplementedError

        # in the non-separable case the list entries are tensors
        ifwt_matrix_list = cast(list[torch.Tensor], self.ifwt_matrix_list)

        if len(ifwt_matrix_list) == 1:
            return ifwt_matrix_list[0]
        elif len(ifwt_matrix_list) > 1:
            if self.padded:
                raise NotImplementedError

            ifwt_matrix = ifwt_matrix_list[-1]
            for scale_mat in ifwt_matrix_list[:-1][::-1]:
                ifwt_matrix = cat_sparse_identity_matrix(
                    ifwt_matrix, scale_mat.shape[0]
                )
                ifwt_matrix = torch.sparse.mm(scale_mat, ifwt_matrix)
            return ifwt_matrix
        else:
            raise ValueError(
                "Call this object first to create the transformation matrices for each "
                "level."
            )

    def _construct_synthesis_matrices(
        self,
        device: Union[torch.device, str],
        dtype: torch.dtype,
    ) -> None:
        self.ifwt_matrix_list = []
        self.padded = False
        if self.level is None or self.input_signal_shape is None:
            raise AssertionError

        current_height, current_width = self.input_signal_shape
        filt_len = self.wavelet.rec_len

        for curr_level in range(1, self.level + 1):
            if current_height < filt_len or current_width < filt_len:
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input shape {self.input_signal_shape}"
                    f". At level {curr_level}, at least one of the current signal "
                    f"height and width ({current_height}, {current_width}) is smaller "
                    f"then the filter length {filt_len}. Therefore, the transformation "
                    f"is only computed up to the decomposition level {curr_level-1}.\n"
                )
                break
            current_height, current_width, pad_tuple = _matrix_pad_2(
                current_height, current_width
            )
            if any(pad_tuple):
                self.padded = True
            if self.separable:
                synthesis_matrix_rows = construct_boundary_s(
                    wavelet=self.wavelet,
                    length=current_height,
                    orthogonalization=self.orthogonalization,
                    device=device,
                    dtype=dtype,
                )
                synthesis_matrix_cols = construct_boundary_s(
                    wavelet=self.wavelet,
                    length=current_width,
                    orthogonalization=self.orthogonalization,
                    device=device,
                    dtype=dtype,
                )
                self.ifwt_matrix_list.append(
                    (synthesis_matrix_rows, synthesis_matrix_cols)
                )
            else:
                synthesis_matrix_2d = construct_boundary_s2(
                    self.wavelet,
                    current_height,
                    current_width,
                    orthogonalization=self.orthogonalization,
                    device=device,
                    dtype=dtype,
                )
                self.ifwt_matrix_list.append(synthesis_matrix_2d)
            current_height = current_height // 2
            current_width = current_width // 2

    def __call__(
        self,
        coefficients: WaveletCoeff2d,
    ) -> torch.Tensor:
        """Compute the inverse matrix 2d fast wavelet transform.

        Args:
            coefficients (WaveletCoeff2d): The coefficient tuple as returned
                by the `MatrixWavedec2` object,
                see :data:`ptwt.constants.WaveletCoeff2d`.

        Returns:
            The original signal reconstruction. For example of shape
            ``[batch_size, height, width]`` or ``[batch_size, channels, height, width]``
            depending on the input to the forward transform and the value
            of the `axis` argument.

        Raises:
            ValueError: If the decomposition level is not a positive integer or if the
                coefficients are not in the shape as it is returned from a
                `MatrixWavedec2` object.
        """
        coefficients, ds = _preprocess_coeffs(coefficients, ndim=2, axes=self.axes)
        torch_device, torch_dtype = _check_same_device_dtype(coefficients)

        level = len(coefficients) - 1
        height, width = tuple(c * 2 for c in coefficients[-1][0].shape[-2:])

        re_build = False
        if (
            self.input_signal_shape is None
            or self.input_signal_shape[0] != height
            or self.input_signal_shape[1] != width
        ):
            self.input_signal_shape = height, width
            re_build = True

        if self.level != level:
            self.level = level
            re_build = True

        if not self.ifwt_matrix_list or re_build:
            self._construct_synthesis_matrices(
                device=torch_device,
                dtype=torch_dtype,
            )

        ll = coefficients[0]
        batch_size = ll.shape[0]
        for c_pos, coeff_tuple in enumerate(coefficients[1:]):
            if not isinstance(coeff_tuple, tuple) or len(coeff_tuple) != 3:
                raise ValueError(
                    f"Unexpected detail coefficient type: {type(coeff_tuple)}. Detail "
                    "coefficients must be a 3-tuple of tensors as returned by "
                    "MatrixWavedec2."
                )

            curr_shape = ll.shape
            for coeff in coeff_tuple:
                if coeff.shape != curr_shape:
                    raise ValueError(
                        "All coefficients on each level must have the same shape"
                    )

            lh, hl, hh = coeff_tuple

            if self.separable:
                synthesis_matrix_rows, synthesis_matrix_cols = self.ifwt_matrix_list[
                    ::-1
                ][c_pos]
                a_coeffs = torch.cat((ll, lh), -1)
                d_coeffs = torch.cat((hl, hh), -1)
                coeff_tensor = torch.cat((a_coeffs, d_coeffs), -2)
                if len(curr_shape) == 2:
                    coeff_tensor = coeff_tensor.unsqueeze(0)
                ll = batch_mm(
                    synthesis_matrix_cols, coeff_tensor.transpose(-2, -1)
                ).transpose(-2, -1)
                ll = batch_mm(synthesis_matrix_rows, ll)
            else:
                ll = torch.cat(
                    [
                        ll.transpose(2, 1).reshape([batch_size, -1]),
                        lh.transpose(2, 1).reshape([batch_size, -1]),
                        hl.transpose(2, 1).reshape([batch_size, -1]),
                        hh.transpose(2, 1).reshape([batch_size, -1]),
                    ],
                    -1,
                )
                ifwt_mat = cast(torch.Tensor, self.ifwt_matrix_list[::-1][c_pos])
                ll = cast(torch.Tensor, torch.sparse.mm(ifwt_mat, ll.T))

            if not self.separable:
                pred_len = [s * 2 for s in curr_shape[-2:]][::-1]
                ll = ll.T.reshape([batch_size] + pred_len).transpose(2, 1)
                pred_len = list(ll.shape[1:])
            else:
                pred_len = [s * 2 for s in curr_shape[-2:]]
            # remove the padding
            if c_pos < len(coefficients) - 2:
                next_len = list(coefficients[c_pos + 2][0].shape[-2:])
                if pred_len != next_len:
                    if pred_len[0] != next_len[0]:
                        ll = ll[:, :-1, :]
                    if pred_len[1] != next_len[1]:
                        ll = ll[:, :, :-1]

        ll = _postprocess_tensor(ll, ndim=2, ds=ds, axes=self.axes)

        return ll
