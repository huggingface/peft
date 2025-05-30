"""Implement matrix-based fwt and ifwt.

This module uses boundary filters instead of padding.

The implementation is based on the description
in Strang Nguyen (p. 32), as well as the description
of boundary filters in "Ripples in Mathematics" section 10.3 .
"""

import sys
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _check_same_device_dtype,
    _deprecated_alias,
    _is_orthogonalize_method_supported,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_tensor,
)
from .constants import BoundaryMode, OrthogonalizeMethod
from .conv_transform import _fwt_pad, _get_filter_tensors
from .sparse_math import (
    _orth_by_gram_schmidt,
    _orth_by_qr,
    cat_sparse_identity_matrix,
    construct_strided_conv_matrix,
)


def _construct_a(
    wavelet: Union[Wavelet, str],
    length: int,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a raw analysis matrix.

    The resulting matrix will only be orthogonal in the Haar case,
    in most cases, you will want to use construct_boundary_a instead.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        length (int): The length of the input signal to transform.
        device (torch.device or str, optional): Where to create the matrix.
            Choose a torch device or device name. Defaults to "cpu".
        dtype (optional): The desired torch datatype. Choose torch.float32
            or torch.float64. Defaults to torch.float64.

    Returns:
        The sparse raw analysis matrix.
    """
    wavelet = _as_wavelet(wavelet)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=False, device=device, dtype=dtype
    )
    analysis_lo = construct_strided_conv_matrix(
        dec_lo.squeeze(), length, 2, mode="sameshift"
    )
    analysis_hi = construct_strided_conv_matrix(
        dec_hi.squeeze(), length, 2, mode="sameshift"
    )
    analysis = torch.cat([analysis_lo, analysis_hi])
    return analysis


def _construct_s(
    wavelet: Union[Wavelet, str],
    length: int,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a raw synthesis matrix.

    The construced matrix is NOT necessary orthogonal.
    In most cases construct_boundary_s should be used instead.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        length (int): The length of the originally transformed signal.
        device (torch.device, optional): Choose cuda or cpu.
            Defaults to torch.device("cpu").
        dtype ([type], optional): The desired data type. Choose torch.float32
            or torch.float64. Defaults to torch.float64.

    Returns:
        The raw sparse synthesis matrix.
    """
    wavelet = _as_wavelet(wavelet)
    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=True, device=device, dtype=dtype
    )
    synthesis_lo = construct_strided_conv_matrix(
        rec_lo.squeeze(), length, 2, mode="sameshift"
    )
    synthesis_hi = construct_strided_conv_matrix(
        rec_hi.squeeze(), length, 2, mode="sameshift"
    )
    synthesis = torch.cat([synthesis_lo, synthesis_hi])
    return synthesis.transpose(0, 1)


def _get_to_orthogonalize(matrix: torch.Tensor, filt_len: int) -> torch.Tensor:
    """Find matrix rows with fewer entries than filt_len.

    The returned rows will need to be orthogonalized.

    Args:
        matrix (torch.Tensor): The wavelet matrix under consideration.
        filt_len (int): The number of entries we would expect per row.

    Returns:
        The row indices with too few entries.
    """
    unique, count = torch.unique_consecutive(
        matrix.coalesce().indices()[0, :], return_counts=True
    )
    return unique[count != filt_len]


def orthogonalize(
    matrix: torch.Tensor, filt_len: int, method: OrthogonalizeMethod = "qr"
) -> torch.Tensor:
    """Orthogonalization for sparse filter matrices.

    Args:
        matrix (torch.Tensor): The sparse filter matrix to orthogonalize.
        filt_len (int): The length of the wavelet filter coefficients.
        method : The orthogonalization method to use. Choose qr
            or gramschmidt. The dense qr code will run much faster
            than sparse gramschidt. Choose gramschmidt if qr fails.
            Defaults to qr.

    Returns:
        Orthogonal sparse transformation matrix.

    Raises:
        ValueError: If an invalid orthogonalization method is given
    """
    to_orthogonalize = _get_to_orthogonalize(matrix, filt_len)
    if len(to_orthogonalize) == 0:
        return matrix
    if method == "qr":
        return _orth_by_qr(matrix, to_orthogonalize)
    elif method == "gramschmidt":
        return _orth_by_gram_schmidt(matrix, to_orthogonalize)
    raise ValueError(f"Invalid orthogonalization method: {method}")


class BaseMatrixWaveDec:
    """A base class for matrix wavedec."""


class MatrixWavedec(BaseMatrixWaveDec):
    """Compute the sparse matrix fast wavelet transform.

    Intermediate scale results must be divisible
    by two. A working third-level transform
    could use an input length of 128.
    This would lead to intermediate resolutions
    of 64 and 32. All are divisible by two.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> matrix_wavedec = ptwt.MatrixWavedec(
        >>>     pywt.Wavelet('haar'), level=2)
        >>> coefficients = matrix_wavedec(data_torch)
    """

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        level: Optional[int] = None,
        axis: Optional[int] = -1,
        orthogonalization: OrthogonalizeMethod = "qr",
        odd_coeff_padding_mode: BoundaryMode = "zero",
    ) -> None:
        """Create a sparse matrix fast wavelet transform object.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            level (int, optional): The level up to which to compute the fwt. If None,
                the maximum level based on the signal length is chosen. Defaults to
                None.
            axis (int, optional): The axis we would like to transform.
                Defaults to -1.
            orthogonalization: The method used to orthogonalize
                boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
                Defaults to 'qr'.
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
            ValueError: If the wavelet filters have different lengths or
                if axis is not an integer.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.level = level
        self.odd_coeff_padding_mode = odd_coeff_padding_mode
        self.orthogonalization = orthogonalization

        if isinstance(axis, int):
            self.axis = axis
        else:
            raise ValueError("MatrixWavedec transforms a single axis only.")

        self.input_length: Optional[int] = None
        self.fwt_matrix_list: list[torch.Tensor] = []
        self.pad_list: list[bool] = []
        self.padded = False
        self.size_list: list[int] = []

        if not _is_orthogonalize_method_supported(self.orthogonalization):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    @property
    def sparse_fwt_operator(self) -> torch.Tensor:
        """The sparse transformation operator.

        If the input signal at all levels is divisible by two,
        the whole operation is padding-free and can be expressed
        as a single matrix multiply.

        The operation ``torch.sparse.mm(sparse_fwt_operator, data.T)``
        computes a batched fwt.

        This property exists to make the operator matrix transparent.
        Calling the object will handle odd-length inputs properly.

        Raises:
            NotImplementedError: if padding had to be used in the creation of the
                transformation matrices.
            ValueError: If no level transformation matrices are stored (most likely
                since the object was not called yet).
        """
        if len(self.fwt_matrix_list) == 1:
            return self.fwt_matrix_list[0]
        elif len(self.fwt_matrix_list) > 1:
            if self.padded:
                raise NotImplementedError

            fwt_matrix = self.fwt_matrix_list[0]
            for scale_mat in self.fwt_matrix_list[1:]:
                scale_mat = cat_sparse_identity_matrix(scale_mat, fwt_matrix.shape[0])
                fwt_matrix = torch.sparse.mm(scale_mat, fwt_matrix)
            return fwt_matrix
        else:
            raise ValueError(
                "Call this object first to create the transformation matrices for each "
                "level."
            )

    def _construct_analysis_matrices(
        self, device: Union[torch.device, str], dtype: torch.dtype
    ) -> None:
        if self.level is None or self.input_length is None:
            raise AssertionError
        self.fwt_matrix_list = []
        self.size_list = []
        self.pad_list = []
        self.padded = False

        filt_len = self.wavelet.dec_len
        curr_length = self.input_length
        for curr_level in range(1, self.level + 1):
            if curr_length < filt_len:
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input size {self.input_length}. At "
                    f"level {curr_level}, the current signal length {curr_length} is "
                    f"smaller than the filter length {filt_len}. Therefore, the "
                    "transformation is only computed up to the decomposition level "
                    f"{curr_level-1}.\n"
                )
                break

            if curr_length % 2 != 0:
                # padding
                curr_length += 1
                self.padded = True
                self.pad_list.append(True)
            else:
                self.pad_list.append(False)

            self.size_list.append(curr_length)

            an = construct_boundary_a(
                self.wavelet,
                curr_length,
                orthogonalization=self.orthogonalization,
                device=device,
                dtype=dtype,
            )
            self.fwt_matrix_list.append(an)
            curr_length = curr_length // 2

        self.size_list.append(curr_length)

    def __call__(self, input_signal: torch.Tensor) -> list[torch.Tensor]:
        """Compute the matrix fwt for the given input signal.

        Matrix FWTs are used to avoid padding.

        Args:
            input_signal (torch.Tensor): Batched input data.
                An example shape could be ``[batch_size, time]``.
                Inputs can have any dimension.
                This transform affects the last axis by default.
                Use the axis argument in the constructor to choose
                another axis.

        Returns:
            A list with the coefficient tensor for each scale.

        Raises:
            ValueError: If the decomposition level is not a positive integer
                or if the input signal has not the expected shape.
        """
        input_signal, ds = _preprocess_tensor(
            input_signal,
            ndim=1,
            axes=self.axis,
            add_channel_dim=False,
        )

        if input_signal.shape[-1] % 2 != 0:
            # odd length input
            input_signal = _fwt_pad(
                input_signal,
                wavelet=self.wavelet,
                mode=self.odd_coeff_padding_mode,
                padding=(0, 1),
            )

        _, length = input_signal.shape

        re_build = False
        if self.input_length != length:
            self.input_length = length
            re_build = True

        if self.level is None:
            wlen = len(self.wavelet)
            self.level = int(np.log2(length / (wlen - 1)))
            re_build = True
        elif self.level <= 0:
            raise ValueError("level must be a positive integer.")

        if not self.fwt_matrix_list or re_build:
            self._construct_analysis_matrices(
                device=input_signal.device, dtype=input_signal.dtype
            )

        lo = input_signal.T
        split_list = []
        for scale, fwt_matrix in enumerate(self.fwt_matrix_list):
            if self.pad_list[scale]:
                # fix odd coefficients lengths for the conv matrix to work.
                lo = lo.T.unsqueeze(1)
                lo = _fwt_pad(
                    lo,
                    wavelet=self.wavelet,
                    mode=self.odd_coeff_padding_mode,
                    padding=(0, 1),
                )
                lo = lo.squeeze(1).T
            coefficients = torch.sparse.mm(fwt_matrix, lo)
            lo, hi = torch.split(coefficients, coefficients.shape[0] // 2, dim=0)
            split_list.append(hi)
        split_list.append(lo)
        # undo the transpose we used to handle the batch dimension.
        result_list = [s.T for s in split_list[::-1]]

        # unfold if necessary
        return _postprocess_coeffs(result_list, ndim=1, ds=ds, axes=self.axis)


@_deprecated_alias(boundary="orthogonalization")
def construct_boundary_a(
    wavelet: Union[Wavelet, str],
    length: int,
    device: Union[torch.device, str] = "cpu",
    orthogonalization: OrthogonalizeMethod = "qr",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a boundary-wavelet filter 1d-analysis matrix.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        length (int): The number of entries in the input signal.
        orthogonalization: The method used to orthogonalize
            boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
            Defaults to 'qr'.
        device: Where to place the matrix. Choose cpu or cuda.
            Defaults to cpu.
        dtype: Choose float32 or float64.

    .. versionchanged:: 1.10
        The argument `boundary` has been renamed to `orthogonalization`.

    Returns:
        The sparse analysis matrix.
    """
    wavelet = _as_wavelet(wavelet)
    a_full = _construct_a(wavelet, length, dtype=dtype, device=device)
    a_orth = orthogonalize(a_full, wavelet.dec_len, method=orthogonalization)
    return a_orth


@_deprecated_alias(boundary="orthogonalization")
def construct_boundary_s(
    wavelet: Union[Wavelet, str],
    length: int,
    device: Union[torch.device, str] = "cpu",
    orthogonalization: OrthogonalizeMethod = "qr",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a boundary-wavelet filter 1d-synthesis matarix.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        length (int): The number of entries in the input signal.
        device (torch.device): Where to place the matrix.
            Choose cpu or cuda. Defaults to cpu.
        orthogonalization: The method used to orthogonalize
            boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
            Defaults to 'qr'.
        dtype: Choose torch.float32 or torch.float64.
            Defaults to torch.float64.

    .. versionchanged:: 1.10
        The argument `boundary` has been renamed to `orthogonalization`.

    Returns:
        The sparse synthesis matrix.
    """
    wavelet = _as_wavelet(wavelet)
    s_full = _construct_s(wavelet, length, dtype=dtype, device=device)
    s_orth = orthogonalize(
        s_full.transpose(1, 0), wavelet.rec_len, method=orthogonalization
    )
    return s_orth.transpose(1, 0)


class MatrixWaverec(object):
    """Matrix-based inverse fast wavelet transform.

    Example:
        >>> import ptwt, torch, pywt
        >>> import numpy as np
        >>> # generate an input of even length.
        >>> data = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> data_torch = torch.from_numpy(data.astype(np.float32))
        >>> matrix_wavedec = ptwt.MatrixWavedec(
        >>>     pywt.Wavelet('haar'), level=2)
        >>> coefficients = matrix_wavedec(data_torch)
        >>> matrix_waverec = ptwt.MatrixWaverec(
        >>>     pywt.Wavelet('haar'))
        >>> reconstruction = matrix_waverec(coefficients)
    """

    @_deprecated_alias(boundary="orthogonalization")
    def __init__(
        self,
        wavelet: Union[Wavelet, str],
        axis: int = -1,
        orthogonalization: OrthogonalizeMethod = "qr",
    ) -> None:
        """Create the inverse matrix-based fast wavelet transformation.

        Args:
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            axis (int): The axis transformed by the original decomposition
                defaults to -1 or the last axis.
            orthogonalization: The method used to orthogonalize
                boundary filters, see :data:`ptwt.constants.OrthogonalizeMethod`.
                Defaults to 'qr'.

        .. versionchanged:: 1.10
            The argument `boundary` has been renamed to `orthogonalization`.

        Raises:
            NotImplementedError: If the selected `orthogonalization` mode
                is not supported.
            ValueError: If the wavelet filters have different lengths or if
                        axis is not an integer.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.orthogonalization = orthogonalization
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise ValueError("MatrixWaverec transforms a single axis only.")

        self.ifwt_matrix_list: list[torch.Tensor] = []
        self.level: Optional[int] = None
        self.input_length: Optional[int] = None
        self.padded = False

        if not _is_orthogonalize_method_supported(self.orthogonalization):
            raise NotImplementedError

        if self.wavelet.dec_len != self.wavelet.rec_len:
            raise ValueError("All filters must have the same length")

    @property
    def sparse_ifwt_operator(self) -> torch.Tensor:
        """The sparse transformation operator.

        If the input signal at all levels is divisible by two,
        the whole operation is padding-free and can be expressed
        as a single matrix multiply.

        Having concatenated the analysis coefficients,
        torch.sparse.mm(sparse_ifwt_operator, coefficients.T)
        to computes a batched iFWT.

        This functionality is mainly here to make the operator-matrix
        transparent. Calling the object handles padding for odd inputs.

        Raises:
            NotImplementedError: if padding had to be used in the creation of the
                transformation matrices.
            ValueError: If no level transformation matrices are stored (most likely
                since the object was not called yet).
        """
        if len(self.ifwt_matrix_list) == 1:
            return self.ifwt_matrix_list[0]
        elif len(self.ifwt_matrix_list) > 1:
            if self.padded:
                raise NotImplementedError

            ifwt_matrix = self.ifwt_matrix_list[-1]
            for scale_matrix in self.ifwt_matrix_list[:-1][::-1]:
                ifwt_matrix = cat_sparse_identity_matrix(
                    ifwt_matrix, scale_matrix.shape[0]
                )
                ifwt_matrix = torch.sparse.mm(scale_matrix, ifwt_matrix)
            return ifwt_matrix
        else:
            raise ValueError(
                "Call this object first to create the transformation matrices for each "
                "level."
            )

    def _construct_synthesis_matrices(
        self, device: Union[torch.device, str], dtype: torch.dtype
    ) -> None:
        self.ifwt_matrix_list = []
        self.size_list = []
        self.padded = False
        if self.level is None or self.input_length is None:
            raise AssertionError

        filt_len = self.wavelet.rec_len
        curr_length = self.input_length

        for curr_level in range(1, self.level + 1):
            if curr_length < filt_len:
                sys.stderr.write(
                    f"Warning: The selected number of decomposition levels {self.level}"
                    f" is too large for the given input size {self.input_length}. At "
                    f"level {curr_level}, the current signal length {curr_length} is "
                    f"smaller than the filter length {filt_len}. Therefore, the "
                    "transformation is only computed up to the decomposition level "
                    f"{curr_level-1}.\n"
                )
                break

            if curr_length % 2 != 0:
                # padding
                curr_length += 1
                self.padded = True

            self.size_list.append(curr_length)

            sn = construct_boundary_s(
                self.wavelet,
                curr_length,
                orthogonalization=self.orthogonalization,
                device=device,
                dtype=dtype,
            )
            self.ifwt_matrix_list.append(sn)
            curr_length = curr_length // 2

    def __call__(self, coefficients: Sequence[torch.Tensor]) -> torch.Tensor:
        """Run the synthesis or inverse matrix fwt.

        Args:
            coefficients (Sequence[torch.Tensor]): The coefficients produced
                by the forward transform.

        Returns:
            The input signal reconstruction.

        Raises:
            ValueError: If the decomposition level is not a positive integer or if the
                coefficients are not in the shape as it is returned from a
                `MatrixWavedec` object.
        """
        if not isinstance(coefficients, list):
            coefficients = list(coefficients)
        coefficients, ds = _preprocess_coeffs(coefficients, ndim=1, axes=self.axis)
        torch_device, torch_dtype = _check_same_device_dtype(coefficients)

        level = len(coefficients) - 1
        input_length = coefficients[-1].shape[-1] * 2

        re_build = False
        if self.level != level or self.input_length != input_length:
            self.level = level
            self.input_length = input_length
            re_build = True

        if not self.ifwt_matrix_list or re_build:
            self._construct_synthesis_matrices(
                device=torch_device,
                dtype=torch_dtype,
            )

        # transpose the coefficients to handle the batch dimension efficiently.
        coefficients = [c.T for c in coefficients]

        lo = coefficients[0]
        for c_pos, hi in enumerate(coefficients[1:]):
            if lo.shape != hi.shape:
                raise ValueError("coefficients must have the same shape")

            lo = torch.cat([lo, hi], 0)
            lo = torch.sparse.mm(self.ifwt_matrix_list[::-1][c_pos], lo)

            # remove padding
            if c_pos < len(coefficients) - 2:
                pred_len = lo.shape[0]
                next_len = coefficients[c_pos + 2].shape[0]
                if next_len != pred_len:
                    lo = lo[:-1, :]
                    pred_len = lo.shape[0]
                    assert (
                        pred_len == next_len
                    ), "padding error, please open an issue on github"

        res_lo = lo.T

        return _postprocess_tensor(res_lo, ndim=1, ds=ds, axes=self.axis)
