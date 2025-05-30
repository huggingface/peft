"""Test the fwt and ifwt matrices."""

# Written by moritz ( @ wolter.tech ) in 2021

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import pywt
import torch

from ptwt.constants import BoundaryMode, OrthogonalizeMethod
from ptwt.matmul_transform import (
    MatrixWavedec,
    MatrixWaverec,
    _construct_a,
    _construct_s,
    construct_boundary_a,
    construct_boundary_s,
)
from tests._mackey_glass import MackeyGenerator


@pytest.mark.parametrize("size", [8, 16, 24, 32])
def test_analysis_and_synthethis_matrices_db1(size: int) -> None:
    """Ensure the analysis matrix a and the synthesis matrix s invert each other."""
    a_db1 = _construct_a(pywt.Wavelet("db1"), size)
    s_db1 = _construct_s(pywt.Wavelet("db1"), size)
    assert np.allclose(torch.sparse.mm(a_db1, s_db1.to_dense()).numpy(), np.eye(size))


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [16, 32, 64, 128])
def test_fwt_ifwt_haar(level: int, length: int) -> None:
    """Test the Haar case."""
    wavelet = pywt.Wavelet("haar")
    data = np.random.uniform(-1, 1, (length))
    coeffs = pywt.wavedec(data, wavelet, level=level)
    matrix_wavedec = MatrixWavedec(wavelet, level)
    coeffs_matfwt = matrix_wavedec(torch.from_numpy(data))
    test_list = [
        np.allclose(cmfwt.numpy(), cpywt) for cmfwt, cpywt in zip(coeffs_matfwt, coeffs)
    ]
    assert all(test_list)


@pytest.mark.slow
def test_fwt_ifwt_mackey_haar_cuda() -> None:
    """Test the Haar case for a long signal on GPU."""
    wavelet = pywt.Wavelet("haar")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = MackeyGenerator(batch_size=2, tmax=512, delta_t=1, device=device)
    pt_data = torch.squeeze(generator()).type(torch.float64)
    # ensure coefficients are equal.
    coeffs = pywt.wavedec(pt_data.cpu().numpy(), wavelet, level=9)
    matrix_wavedec = MatrixWavedec(wavelet, 9)
    coeffs_matfwt = matrix_wavedec(pt_data)
    test_list = [
        np.allclose(cmfwt.cpu().numpy(), cpywt)
        for cmfwt, cpywt in zip(coeffs_matfwt, coeffs)
    ]
    assert all(test_list)
    # test the inverse fwt.
    matrix_waverec = MatrixWaverec(wavelet)
    reconstructed_data = matrix_waverec(coeffs_matfwt)
    assert np.allclose(pt_data.cpu().numpy(), reconstructed_data.cpu().numpy())


@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 2, 3, 4, None])
@pytest.mark.parametrize("wavelet", ["db2", "db3", "db4", "sym5"])
@pytest.mark.parametrize("size", [[2, 256], [2, 3, 256], [1, 1, 128]])
@pytest.mark.parametrize(
    "mode", ["reflect", "zero", "constant", "periodic", "symmetric"]
)
def test_1d_matrix_fwt_ifwt(
    level: int, wavelet: str, size: list[int], mode: BoundaryMode
) -> None:
    """Test multiple wavelets and levels for a long signal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelet = pywt.Wavelet(wavelet)
    pt_data = torch.randn(*size, device=device).type(torch.float64)
    matrix_wavedec = MatrixWavedec(wavelet, level, odd_coeff_padding_mode=mode)
    coeffs_mat_max = matrix_wavedec(pt_data)
    matrix_waverec = MatrixWaverec(wavelet)
    reconstructed_data = matrix_waverec(coeffs_mat_max)
    assert np.allclose(reconstructed_data.cpu().numpy(), pt_data.cpu().numpy())


@pytest.mark.slow
@pytest.mark.parametrize("size", [24, 64, 128, 256])
@pytest.mark.parametrize(
    "wavelet",
    [
        pywt.Wavelet("db2"),
        pywt.Wavelet("db4"),
        pywt.Wavelet("db6"),
        pywt.Wavelet("db8"),
    ],
)
def test_boundary_filter_analysis_and_synthethis_matrices(
    size: int, wavelet: pywt.Wavelet
) -> None:
    """Check 1d the 1d-fwt matrices for orthogonality and invertability."""
    analysis_matrix = construct_boundary_a(
        wavelet, size, boundary="gramschmidt"
    ).to_dense()
    synthesis_matrix = construct_boundary_s(
        wavelet, size, boundary="gramschmidt"
    ).to_dense()
    # s_db2 = construct_s(pywt.Wavelet("db8"), size)
    # test_eye_inv = torch.sparse.mm(a_db8, s_db2.to_dense()).numpy()
    test_eye_orth = torch.mm(analysis_matrix.transpose(1, 0), analysis_matrix).numpy()
    test_eye_inv = torch.mm(analysis_matrix, synthesis_matrix).numpy()
    err_inv = np.mean(np.abs(test_eye_inv - np.eye(size)))
    err_orth = np.mean(np.abs(test_eye_orth - np.eye(size)))
    print(wavelet.name, "orthogonal error", err_orth, "size", size)
    print(wavelet.name, "inverse error", err_inv, "size", size)
    assert err_orth < 1e-8
    assert err_inv < 1e-8


@pytest.mark.parametrize("wavelet_str", ["db2", "db3", "haar"])
@pytest.mark.parametrize(
    "data",
    [
        np.random.randn(32),
        np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]),
        np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]),
        np.random.randn(18),
        np.random.randn(19),
    ],
)
@pytest.mark.parametrize("level", [2, 1])
@pytest.mark.parametrize("boundary", ["gramschmidt", "qr"])
def test_boundary_transform_1d(
    wavelet_str: str, data: npt.NDArray[Any], level: int, boundary: OrthogonalizeMethod
) -> None:
    """Ensure matrix fwt reconstructions are pywt compatible."""
    data_torch = torch.from_numpy(data.astype(np.float64))
    wavelet = pywt.Wavelet(wavelet_str)
    matrix_wavedec = MatrixWavedec(wavelet, level=level, orthogonalization=boundary)
    coeffs = matrix_wavedec(data_torch)
    matrix_waverec = MatrixWaverec(wavelet, orthogonalization=boundary)
    rec = matrix_waverec(coeffs)
    rec_pywt = pywt.waverec(
        pywt.wavedec(data_torch.numpy(), wavelet, mode="zero"), wavelet
    )
    error = np.sum(np.abs(rec_pywt - rec.numpy()))
    print(
        "wavelet: {},".format(wavelet_str),
        "level: {},".format(level),
        "shape: {},".format(data.shape[-1]),
        "error {:2.2e}".format(error),
    )
    assert np.allclose(rec.numpy(), rec_pywt)
    # test the operator matrices
    if not matrix_wavedec.padded and not matrix_waverec.padded:
        test_mat = torch.sparse.mm(
            matrix_waverec.sparse_ifwt_operator,
            matrix_wavedec.sparse_fwt_operator,
        )
        assert np.allclose(test_mat.to_dense().numpy(), np.eye(test_mat.shape[0]))


@pytest.mark.parametrize("wavelet_str", ["db2", "db3", "haar"])
@pytest.mark.parametrize("boundary", ["qr", "gramschmidt"])
def test_matrix_transform_1d_rebuild(
    wavelet_str: str, boundary: OrthogonalizeMethod
) -> None:
    """Ensure matrix fwt reconstructions are pywt compatible."""
    data_list = [np.random.randn(18), np.random.randn(21)]
    wavelet = pywt.Wavelet(wavelet_str)
    matrix_waverec = MatrixWaverec(wavelet, orthogonalization=boundary)
    for level in [2, 1]:
        matrix_wavedec = MatrixWavedec(wavelet, level=level, orthogonalization=boundary)
        for data in data_list:
            data_torch = torch.from_numpy(data.astype(np.float64))
            coeffs = matrix_wavedec(data_torch)
            rec = matrix_waverec(coeffs)
            rec_pywt = pywt.waverec(
                pywt.wavedec(data_torch.numpy(), wavelet, mode="zero"), wavelet
            )
            assert np.allclose(rec.numpy(), rec_pywt)
            # test the operator matrices
            if not matrix_wavedec.padded and not matrix_waverec.padded:
                test_mat = torch.sparse.mm(
                    matrix_waverec.sparse_ifwt_operator,
                    matrix_wavedec.sparse_fwt_operator,
                )
                assert np.allclose(
                    test_mat.to_dense().numpy(), np.eye(test_mat.shape[0])
                )


def test_4d_invalid_axis_error() -> None:
    """Test the error for 1d axis arguments."""
    with pytest.raises(ValueError):
        MatrixWavedec("haar", axis=(1, 2))


@pytest.mark.parametrize("size", [[2, 3, 32], [5, 32], [32], [1, 1, 64]])
def test_matrix1d_batch_channel(size: list[int]) -> None:
    """Test if batch and channel support works as expected."""
    data = torch.randn(*size).type(torch.float64)
    matrix_wavedec_1d = MatrixWavedec("haar", 3)
    ptwt_coeff = matrix_wavedec_1d(data)
    pywt_coeff = pywt.wavedec(data.numpy(), "haar", level=3)

    test = [
        np.allclose(ptwtcs.numpy(), pywtcs)
        for ptwtcs, pywtcs in zip(ptwt_coeff, pywt_coeff)
    ]
    assert all(test)

    matrix_waverec_2d = MatrixWaverec("haar")
    rec = matrix_waverec_2d(ptwt_coeff)

    assert np.allclose(data.numpy(), rec.numpy())


@pytest.mark.parametrize("axis", (0, 1, 2, 3, 4))
def test_axis_1d(axis: int) -> None:
    """Ensure the axis argument is supported correctly."""
    data = torch.randn(24, 24, 24, 24, 24).type(torch.float64)
    matrix_wavedec = MatrixWavedec(wavelet="haar", level=3, axis=axis)
    coeff = matrix_wavedec(data)
    coeff_pywt = pywt.wavedec(data.numpy(), wavelet="haar", level=3, axis=axis)
    assert len(coeff) == len(coeff_pywt)
    assert all(
        [np.allclose(coeff, coeff_pywt) for coeff, coeff_pywt in zip(coeff, coeff_pywt)]
    )

    matrix_waverec = MatrixWaverec("haar", axis=axis)

    rec = matrix_waverec(coeff)
    assert np.allclose(rec, data)


def test_deprecation() -> None:
    """Ensure the deprecation warning is raised."""
    with pytest.warns(DeprecationWarning):
        MatrixWavedec("haar", 3, boundary="qr")
