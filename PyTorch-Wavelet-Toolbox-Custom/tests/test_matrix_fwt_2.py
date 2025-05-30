"""Test code for the 2d boundary wavelets."""

from typing import Type

import numpy as np
import pytest
import pywt
import scipy.signal
import torch

from ptwt.constants import BoundaryMode
from ptwt.conv_transform import _flatten_2d_coeff_lst
from ptwt.matmul_transform import BaseMatrixWaveDec, MatrixWavedec, MatrixWaverec
from ptwt.matmul_transform_2 import (
    MatrixWavedec2,
    MatrixWaverec2,
    construct_boundary_a2,
    construct_boundary_s2,
)
from tests.test_convolution_fwt import _compare_coeffs

# Created by moritz ( wolter@cs.uni-bonn.de ), 08.09.21


@pytest.mark.parametrize("size", [(16, 16), (16, 8), (8, 16)])
@pytest.mark.parametrize("wavelet_str", ["db1", "db2", "db3", "db4", "db5"])
def test_analysis_synthesis_matrices2(size: tuple[int, int], wavelet_str: str) -> None:
    """Test the 2d analysis and synthesis matrices for various wavelets."""
    wavelet = pywt.Wavelet(wavelet_str)
    a = construct_boundary_a2(
        wavelet,
        size[0],
        size[1],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    s = construct_boundary_s2(
        wavelet,
        size[0],
        size[1],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    test_inv = torch.sparse.mm(s, a)
    assert test_inv.shape[0] == test_inv.shape[1], "the diagonal matrix must be square."
    test_eye = torch.eye(test_inv.shape[0])
    assert np.allclose(test_inv.to_dense().numpy(), test_eye.numpy())


@pytest.mark.slow
@pytest.mark.parametrize("size", [(8, 16), (16, 8), (15, 16), (16, 15), (16, 16)])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_matrix_analysis_fwt_2d_haar(size: tuple[int, int], level: int) -> None:
    """Test the fwt-2d matrix-haar transform, should be equal to the pywt."""
    face = np.mean(
        scipy.datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])], -1
    ).astype(np.float64)
    wavelet = pywt.Wavelet("haar")
    matrixfwt = MatrixWavedec2(wavelet, level=level, separable=False)
    mat_coeff = matrixfwt(torch.from_numpy(face))
    conv_coeff = pywt.wavedec2(face, wavelet, level=level, mode="zero")
    flat_mat_coeff = torch.cat(_flatten_2d_coeff_lst(mat_coeff), -1)
    flat_conv_coeff = np.concatenate(_flatten_2d_coeff_lst(conv_coeff), -1)
    test = np.allclose(flat_mat_coeff.numpy(), flat_conv_coeff)
    test2 = np.allclose(mat_coeff[0].squeeze(0).numpy(), conv_coeff[0])
    test3 = np.allclose(mat_coeff[1][0].squeeze(0).numpy(), conv_coeff[1][0])
    assert test and test2 and test3


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_str", ["haar", "db2", "db3", "db4"])
@pytest.mark.parametrize(
    "size",
    [
        (32, 16),
        (16, 32),
        (25, 26),
        (26, 25),
        (25, 25),
        (16, 16),
        (15, 15),
        (16, 15),
        (15, 16),
        (33, 31),
        (31, 33),
    ],
)
@pytest.mark.parametrize("level", [1, 2, 3, None])
@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize(
    "mode", ["reflect", "zero", "constant", "periodic", "symmetric"]
)
def test_boundary_matrix_fwt_2d(
    wavelet_str: str,
    size: tuple[int, int],
    level: int,
    separable: bool,
    mode: BoundaryMode,
) -> None:
    """Ensure the boundary matrix fwt is invertable."""
    face = np.mean(
        scipy.datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])], -1
    ).astype(np.float64)
    wavelet = pywt.Wavelet(wavelet_str)
    matrixfwt = MatrixWavedec2(
        wavelet, level=level, separable=separable, odd_coeff_padding_mode=mode
    )
    mat_coeff = matrixfwt(torch.from_numpy(face))
    matrixifwt = MatrixWaverec2(wavelet, separable=separable)
    reconstruction = matrixifwt(mat_coeff).squeeze(0)
    # remove the padding
    if size[0] % 2 != 0:
        reconstruction = reconstruction[:-1, :]
    if size[1] % 2 != 0:
        reconstruction = reconstruction[:, :-1]
    assert np.allclose(reconstruction.numpy(), face)
    # test the operator matrices
    if not separable and not matrixfwt.padded and not matrixifwt.padded:
        # padding happens outside of the matrix structure.
        # our matrices therefore only describe pad-free cases.
        test_mat = torch.sparse.mm(
            matrixifwt.sparse_ifwt_operator, matrixfwt.sparse_fwt_operator
        )
        assert np.allclose(test_mat.to_dense().numpy(), np.eye(test_mat.shape[0]))


@pytest.mark.parametrize("wavelet_str", ["db1", "db2"])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("size", [(16, 16), (32, 16), (16, 32)])
@pytest.mark.parametrize("separable", [False, True])
def test_batched_2d_matrix_fwt_ifwt(
    wavelet_str: str, level: int, size: tuple[int, int], separable: bool
) -> None:
    """Ensure the batched matrix fwt works properly."""
    face = scipy.datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])].astype(
        np.float64
    )
    pt_face = torch.from_numpy(face).permute([2, 0, 1])
    wavelet = pywt.Wavelet(wavelet_str)
    matrixfwt = MatrixWavedec2(wavelet, level=level, separable=separable)
    mat_coeff = matrixfwt(pt_face)
    matrixifwt = MatrixWaverec2(wavelet, separable=separable)
    reconstruction = matrixifwt(mat_coeff)
    assert (
        np.allclose(reconstruction[0].numpy(), face[:, :, 0])
        and np.allclose(reconstruction[1].numpy(), face[:, :, 1])
        and np.allclose(reconstruction[2].numpy(), face[:, :, 2])
    )


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_str", ["haar", "db4"])
@pytest.mark.parametrize("separable", [False, True])
def test_matrix_transform_2d_rebuild(wavelet_str: str, separable: bool) -> None:
    """Ensure the boundary matrix fwt is invertable."""
    wavelet = pywt.Wavelet(wavelet_str)
    matrixifwt = MatrixWaverec2(wavelet, separable=separable)
    for level in [4, 1, None]:
        matrixfwt = MatrixWavedec2(wavelet, level=level, separable=separable)
        for size in [[16, 16], [17, 17]]:
            face = np.mean(
                scipy.datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])], -1
            ).astype(np.float64)
            mat_coeff = matrixfwt(torch.from_numpy(face))
            reconstruction = matrixifwt(mat_coeff).squeeze(0)
            # remove the padding
            if size[0] % 2 != 0:
                reconstruction = reconstruction[:-1, :]
            if size[1] % 2 != 0:
                reconstruction = reconstruction[:, :-1]
            assert np.allclose(reconstruction.numpy(), face)
            # test the operator matrices
            if not separable and not matrixfwt.padded and not matrixifwt.padded:
                test_mat = torch.sparse.mm(
                    matrixifwt.sparse_ifwt_operator, matrixfwt.sparse_fwt_operator
                )
                assert np.allclose(
                    test_mat.to_dense().numpy(), np.eye(test_mat.shape[0])
                )


def test_separable_haar_2d() -> None:
    """See if the separable haar coefficients are correct."""
    batch_size = 1
    test_data = torch.rand(batch_size, 32, 32).type(torch.float64)

    pywtl, pywth = pywt.wavedec(test_data.numpy(), "haar", level=1, axis=-1)
    pywtll, pywthl = pywt.wavedec(pywtl, "haar", level=1, axis=-2)
    pywtlh, pywthh = pywt.wavedec(pywth, "haar", level=1, axis=-2)
    pywtres = (pywtll, pywtlh, pywthl, pywthh)

    ptwtres_nested = MatrixWavedec2("haar", 1)(test_data)
    # flatten list
    ptwtres = [tensor for tensor_list in ptwtres_nested for tensor in tensor_list]

    assert all(
        [
            np.allclose(pywt_test, ptwt_test.numpy())
            for pywt_test, ptwt_test in zip(pywtres, ptwtres)
        ]
    )


@pytest.mark.parametrize("size", [[3, 2, 32, 32], [4, 32, 32], [1, 1, 32, 32]])
def test_batch_channel_2d_haar(size: list[int]) -> None:
    """Test matrix fwt-2d leading channel and batch dimension code."""
    signal = torch.randn(*size).type(torch.float64)
    ptwt_coeff = MatrixWavedec2("haar", 2, separable=False)(signal)
    pywt_coeff = pywt.wavedec2(signal, "haar", level=2)

    for ptwtc, pywtc in zip(ptwt_coeff, pywt_coeff):
        if isinstance(ptwtc, torch.Tensor):
            assert np.allclose(ptwtc.numpy(), pywtc)
        else:
            test = [
                np.allclose(ptwtcel, pywtcel) for ptwtcel, pywtcel in zip(ptwtc, pywtc)
            ]
            assert all(test)

    rec = MatrixWaverec2("haar", separable=False)(ptwt_coeff)
    assert np.allclose(rec.numpy(), signal.numpy())


@pytest.mark.parametrize("operator", [MatrixWavedec2, MatrixWavedec])
def test_empty_operators(operator: Type[BaseMatrixWaveDec]) -> None:
    """Check if the error is thrown properly if no matrix was ever built."""
    if operator is MatrixWavedec2:
        matrixfwt = operator(wavelet="haar", separable=False)
    else:
        matrixfwt = operator(wavelet="haar")
    with pytest.raises(ValueError):
        _ = matrixfwt.sparse_fwt_operator


@pytest.mark.parametrize("operator", [MatrixWaverec2, MatrixWaverec])
def test_empty_inverse_operators(operator: Type[BaseMatrixWaveDec]) -> None:
    """Check if the error is thrown properly if no matrix was ever built."""
    if operator is MatrixWaverec2:
        matrixifwt = operator(wavelet="haar", separable=False)
    else:
        matrixifwt = operator(wavelet="haar")
    with pytest.raises(ValueError):
        _ = matrixifwt.sparse_ifwt_operator


@pytest.mark.slow
@pytest.mark.parametrize("axes", ((-2, -1), (-1, -2), (-3, -2), (0, 1), (1, 0)))
def test_axes_2d(axes: tuple[int, int]) -> None:
    """Ensure the axes argument is supported correctly."""
    # TODO: write me.
    data = torch.randn(24, 24, 24, 24, 24).type(torch.float64)
    matrix_wavedec2 = MatrixWavedec2(wavelet="haar", level=3, axes=axes)
    coeff = matrix_wavedec2(data)
    coeff_pywt = pywt.wavedec2(data.numpy(), wavelet="haar", level=3, axes=axes)
    assert len(coeff) == len(coeff_pywt)
    assert _compare_coeffs(coeff, coeff_pywt)

    matrix_waverec2 = MatrixWaverec2("haar", axes=axes)

    rec = matrix_waverec2(coeff)
    assert np.allclose(rec, data)


def test_deprecation() -> None:
    """Ensure the deprecation warning is raised."""
    with pytest.warns(DeprecationWarning):
        MatrixWavedec2("haar", 3, boundary="qr")
