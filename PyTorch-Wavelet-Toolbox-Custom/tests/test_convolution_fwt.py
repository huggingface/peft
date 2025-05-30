"""Test the conv-fwt code."""

from collections.abc import Sequence
from typing import Optional, Union

# Written by moritz ( @ wolter.tech ) in 2021
import numpy as np
import pytest
import pywt
import torch
from scipy import datasets

from ptwt._util import _outer
from ptwt.constants import BoundaryMode
from ptwt.conv_transform import (
    _flatten_2d_coeff_lst,
    _translate_boundary_strings,
    wavedec,
    waverec,
)
from ptwt.conv_transform_2 import wavedec2, waverec2
from ptwt.wavelets_learnable import SoftOrthogonalWavelet
from tests._mackey_glass import MackeyGenerator


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_string", ["db1", "db2", "db3", "db4", "db5", "sym5"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("length", [64, 65])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "mode", ["reflect", "zero", "constant", "periodic", "symmetric"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_fwt1d(
    wavelet_string: str,
    level: Optional[int],
    mode: BoundaryMode,
    length: int,
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    """Test multiple convolution fwt, for various levels and padding options."""
    generator = MackeyGenerator(
        batch_size=batch_size, tmax=length, delta_t=1, device="cpu"
    )
    mackey_data_1 = torch.squeeze(generator(), -1).type(dtype)
    wavelet = pywt.Wavelet(wavelet_string)
    ptcoeff = wavedec(mackey_data_1, wavelet, level=level, mode=mode)
    cptcoeff = torch.cat(ptcoeff, -1)
    py_list = []
    for b_el in range(batch_size):
        py_list.append(
            np.concatenate(
                pywt.wavedec(
                    mackey_data_1[b_el, :].numpy(), wavelet, level=level, mode=mode
                ),
                -1,
            )
        )
    py_coeff = np.stack(py_list)
    assert np.allclose(
        cptcoeff.numpy(), py_coeff, atol=float(np.finfo(py_coeff.dtype).resolution)
    )
    res = waverec(ptcoeff, wavelet)
    assert np.allclose(mackey_data_1.numpy(), res.numpy()[:, : mackey_data_1.shape[-1]])


@pytest.mark.parametrize("size", [[5, 10, 64], [1, 1, 32]])
@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_conv_fwt1d_channel(size: list[int], wavelet: str) -> None:
    """Test channel dimension support."""
    data = torch.randn(*size).type(torch.float64)
    ptwt_coeff = wavedec(data, wavelet)
    pywt_coeff = pywt.wavedec(data.numpy(), wavelet, mode="reflect")
    assert all(
        [
            np.allclose(ptwtc.numpy(), pywtc)
            for ptwtc, pywtc in zip(ptwt_coeff, pywt_coeff)
        ]
    )
    rec = waverec(ptwt_coeff, wavelet)
    assert np.allclose(data.numpy(), rec.numpy())


@pytest.mark.parametrize("size", [[32], [64]])
@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_conv_fwt1d_nobatch(size: list[int], wavelet: str) -> None:
    """1d conv for inputs without batch dim."""
    data = torch.randn(*size).type(torch.float64)
    ptwt_coeff = wavedec(data, wavelet)
    pywt_coeff = pywt.wavedec(data.numpy(), wavelet, mode="reflect")
    assert all(
        [
            np.allclose(ptwtc.numpy(), pywtc)
            for ptwtc, pywtc in zip(ptwt_coeff, pywt_coeff)
        ]
    )
    rec = waverec(ptwt_coeff, wavelet)
    assert np.allclose(data.numpy(), rec.numpy())


def test_ripples_haar_lvl3() -> None:
    """Compute example from page 7 of Ripples in Mathematics, Jensen, la Cour-Harbo."""

    class _MyHaarFilterBank:
        @property
        def filter_bank(self) -> tuple[list[float], ...]:
            """Unscaled Haar wavelet filters."""
            return (
                [1 / 2, 1 / 2.0],
                [-1 / 2.0, 1 / 2.0],
                [1 / 2.0, 1 / 2.0],
                [1 / 2.0, -1 / 2.0],
            )

    data = torch.tensor([56.0, 40.0, 8.0, 24.0, 48.0, 48.0, 40.0, 16.0])
    wavelet = pywt.Wavelet("unscaled Haar Wavelet", filter_bank=_MyHaarFilterBank())
    coeffs = wavedec(data, wavelet, level=3)
    assert torch.squeeze(coeffs[0]).numpy() == 35.0
    assert torch.squeeze(coeffs[1]).numpy() == -3.0
    assert (torch.squeeze(coeffs[2]).numpy() == [16.0, 10.0]).all()
    assert (torch.squeeze(coeffs[3]).numpy() == [8.0, -8.0, 0.0, 12.0]).all()


def test_orth_wavelet() -> None:
    """Test an orthogonal wavelet fwt."""
    generator = MackeyGenerator(batch_size=2, tmax=64, delta_t=1, device="cpu")

    mackey_data_1 = torch.squeeze(generator())
    # orthogonal wavelet object test
    wavelet = pywt.Wavelet("db5")
    orthwave = SoftOrthogonalWavelet(
        torch.tensor(wavelet.rec_lo),
        torch.tensor(wavelet.rec_hi),
        torch.tensor(wavelet.dec_lo),
        torch.tensor(wavelet.dec_hi),
    )
    res = waverec(wavedec(mackey_data_1, orthwave), orthwave)
    assert np.allclose(res.detach().numpy(), mackey_data_1.numpy())


@pytest.mark.parametrize("level", [1, 2, 3, None])
@pytest.mark.parametrize("shape", [(64,), (1, 64), (3, 2, 64), (4, 3, 2, 64)])
def test_1d_multibatch(level: Optional[int], shape: Sequence[int]) -> None:
    """Test 1D conv support for multiple inert batch dimensions."""
    data = torch.randn(*shape, dtype=torch.float64)
    ptwt_coeff = wavedec(data, "haar", level=level)
    pywt_coeff = pywt.wavedec(data, "haar", level=level, mode="reflect")

    # test coefficients
    test_list = _compare_coeffs(ptwt_coeff, pywt_coeff)
    assert all(test_list)

    # test reconstruction
    rec = waverec(ptwt_coeff, "haar")
    assert torch.allclose(rec, data)


@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_1d_axis_arg(axis: int) -> None:
    """Ensure the axis argument works as expected."""
    data = torch.randn([16, 16, 16], dtype=torch.float64)

    ptwtcs = wavedec(data, "haar", level=2, axis=axis)
    pywtcs = pywt.wavedec(data, "haar", level=2, axis=axis)

    test_list = _compare_coeffs(ptwtcs, pywtcs)
    assert all(test_list)

    rec = waverec(ptwtcs, "haar", axis=axis)
    assert torch.allclose(rec, data)


def test_2d_haar_lvl1() -> None:
    """Test a 2d-Haar wavelet conv-fwt."""
    # ------------------------- 2d haar wavelet tests -----------------------
    face = np.transpose(
        datasets.face()[128 : (512 + 128), 256 : (512 + 256)], [2, 0, 1]
    ).astype(np.float64)
    wavelet = pywt.Wavelet("haar")
    # single level haar - 2d
    coeff2d_pywt = pywt.dwt2(face, wavelet, mode="zero")
    coeff2d = wavedec2(torch.from_numpy(face), wavelet, level=1, mode="constant")
    flat_list_pywt = np.concatenate(_flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_list_ptwt = torch.cat(_flatten_2d_coeff_lst(coeff2d), -1)
    assert np.allclose(flat_list_pywt, flat_list_ptwt.numpy())
    rec = waverec2(coeff2d, wavelet).numpy().squeeze()
    assert np.allclose(rec, face)


def test_2d_db2_lvl1() -> None:
    """Test a 2d-db2 wavelet conv-fwt."""
    # single level db2 - 2d
    face = np.transpose(
        datasets.face()[256 : (512 + 128), 256 : (512 + 128)], [2, 0, 1]
    ).astype(np.float64)
    wavelet = pywt.Wavelet("db2")
    coeff2d_pywt = pywt.dwt2(face, wavelet, mode="reflect")
    coeff2d = wavedec2(torch.from_numpy(face), wavelet, level=1)
    flat_list_pywt = np.concatenate(_flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_list_ptwt = torch.cat(_flatten_2d_coeff_lst(coeff2d), -1)
    assert np.allclose(flat_list_pywt, flat_list_ptwt.numpy())
    # single level db2 - 2d inverse.
    rec = waverec2(coeff2d, wavelet)
    assert np.allclose(rec.numpy().squeeze(), face)


def test_2d_haar_multi() -> None:
    """Test a 2d-db2 wavelet level 5 conv-fwt."""
    # multi level haar - 2d
    face = np.transpose(
        datasets.face()[256 : (512 + 128), 256 : (512 + 128)], [2, 0, 1]
    ).astype(np.float64)
    wavelet = pywt.Wavelet("haar")
    coeff2d_pywt = pywt.wavedec2(face, wavelet, mode="reflect", level=5)
    coeff2d = wavedec2(torch.from_numpy(face), wavelet, level=5)
    flat_list_pywt = np.concatenate(_flatten_2d_coeff_lst(coeff2d_pywt), -1)
    flat_list_ptwt = torch.cat(_flatten_2d_coeff_lst(coeff2d), -1)
    assert np.allclose(flat_list_pywt, flat_list_ptwt)
    # inverse multi level Harr - 2d
    rec = waverec2(coeff2d, wavelet).numpy().squeeze()
    assert np.allclose(rec, face)


def test_outer() -> None:
    """Test the outer-product implementation."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0])
    res_t = _outer(a, b)
    res_np = np.outer(a.numpy(), b.numpy())
    assert np.allclose(res_t.numpy(), res_np)


@pytest.mark.slow
@pytest.mark.parametrize(
    "wavelet_str", ["haar", "db2", "db3", "db4", "sym4", "rbio2.4", "coif3", "bior2.2"]
)
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [(32, 32), (32, 64), (64, 32), (31, 31)])
@pytest.mark.parametrize(
    "mode", ["reflect", "zero", "constant", "periodic", "symmetric"]
)
def test_2d_wavedec_rec(
    wavelet_str: str, level: Optional[int], size: tuple[int, int], mode: BoundaryMode
) -> None:
    """Ensure pywt.wavedec2 and ptwt.wavedec2 produce the same coefficients.

    wavedec2 and waverec2 must invert each other.
    """
    face = np.transpose(
        datasets.face()[256 : (512 + size[0]), 256 : (512 + size[1])], [2, 0, 1]
    ).astype(np.float64)
    wavelet = pywt.Wavelet(wavelet_str)
    coeff2d = wavedec2(torch.from_numpy(face), wavelet, mode=mode, level=level)
    pywt_coeff2d = pywt.wavedec2(face, wavelet, mode=mode, level=level)
    for pos, coeffs in enumerate(pywt_coeff2d):
        if type(coeffs) is tuple:
            for tuple_pos, tuple_el in enumerate(coeffs):
                assert (
                    tuple_el.shape == coeff2d[pos][tuple_pos].shape
                ), "pywt and ptwt should produce the same shapes."
        else:
            assert (
                coeffs.shape == coeff2d[pos].shape
            ), "pywt and ptwt should produce the same shapes."
    flat_coeff_list_pywt = np.concatenate(_flatten_2d_coeff_lst(pywt_coeff2d), -1)
    flat_coeff_list_ptwt = torch.cat(_flatten_2d_coeff_lst(coeff2d), -1)
    assert np.allclose(flat_coeff_list_pywt, flat_coeff_list_ptwt.numpy())
    rec = waverec2(coeff2d, wavelet)
    rec = rec.numpy().squeeze()
    assert np.allclose(face, rec[:, : face.shape[1], : face.shape[2]])


@pytest.mark.parametrize(
    "size", [(50, 20, 128, 128), (49, 21, 128, 128), (4, 5, 64, 64)]
)
@pytest.mark.parametrize("level", [1, None])
@pytest.mark.parametrize("wavelet", ["haar", "sym3"])
def test_input_4d(
    size: tuple[int, int, int, int], level: Optional[str], wavelet: str
) -> None:
    """Test the error for 4d inputs to wavedec2."""
    data = torch.randn(*size).type(torch.float64)

    pt_res = wavedec2(data, wavelet=wavelet, level=level, mode="reflect")
    pywt_res = pywt.wavedec2(data.numpy(), wavelet=wavelet, level=level, mode="reflect")
    rec = waverec2(pt_res, wavelet)

    # test coefficients
    for ptwtcs, pywtcs in zip(pt_res, pywt_res):
        if isinstance(ptwtcs, tuple):
            assert all(
                (
                    np.allclose(ptwtc.numpy(), pywtc)
                    for ptwtc, pywtc in zip(ptwtcs, pywtcs)
                )
            )
        else:
            assert np.allclose(ptwtcs, pywtcs)

    # test reconstruction.
    assert np.allclose(
        data.numpy(), rec.numpy()[..., : data.shape[-2], : data.shape[-1]]
    )


@pytest.mark.parametrize("padding_str", ["invalid_padding_name"])
def test_incorrect_padding(padding_str: BoundaryMode) -> None:
    """Test expected errors for an invalid padding name."""
    with pytest.raises(ValueError):
        _ = _translate_boundary_strings(padding_str)


def test_input_1d_dimension_error() -> None:
    """Test the error for 1d inputs to wavedec2."""
    with pytest.raises(ValueError):
        data = torch.randn(50)
        wavedec2(data, "haar", level=4)


def _compare_coeffs(
    ptwt_res: Sequence[Union[torch.Tensor, tuple[torch.Tensor, ...]]],
    pywt_res: Sequence[Union[torch.Tensor, tuple[torch.Tensor, ...]]],
) -> list[bool]:
    """Compare coefficient lists.

    Args:
        ptwt_res: Our result list.
        pywt_res: A pyt result list.

    Returns:
        A list with bools from allclose.

    Raises:
        TypeError: In case of a problem with the list structures.
    """
    test_list: list[bool] = []
    for ptwtcs, pywtcs in zip(ptwt_res, pywt_res):
        if isinstance(ptwtcs, tuple) and isinstance(pywtcs, tuple):
            test_list.extend(
                np.allclose(ptwtc.numpy(), pywtc)
                for ptwtc, pywtc in zip(ptwtcs, pywtcs)
            )
        elif isinstance(ptwtcs, torch.Tensor):
            test_list.append(np.allclose(ptwtcs.numpy(), pywtcs))
        else:
            raise TypeError("Invalid coefficient typing.")
    return test_list


@pytest.mark.slow
@pytest.mark.parametrize(
    "size", [(50, 20, 128, 128), (8, 49, 21, 128, 128), (6, 4, 4, 5, 64, 64)]
)
def test_2d_multidim_input(size: tuple[int, ...]) -> None:
    """Test the error for multi-dimensional inputs to wavedec2."""
    data = torch.randn(*size, dtype=torch.float64)
    wavelet = "db2"
    level = 3

    pt_res = wavedec2(data, wavelet=wavelet, level=level, mode="reflect")
    pywt_res = pywt.wavedec2(data.numpy(), wavelet=wavelet, level=level, mode="reflect")
    rec = waverec2(pt_res, wavelet)

    # test coefficients
    test_list = _compare_coeffs(pt_res, pywt_res)
    assert all(test_list)

    # test reconstruction.
    assert np.allclose(
        data.numpy(), rec.numpy()[..., : data.shape[-2], : data.shape[-1]]
    )


@pytest.mark.slow
@pytest.mark.parametrize("axes", [(-2, -1), (-1, -2), (-3, -2), (0, 1), (1, 0)])
def test_2d_axis_argument(axes: tuple[int, int]) -> None:
    """Ensure the axes argument works as expected."""
    data = torch.randn([32, 32, 32, 32], dtype=torch.float64)

    ptwt_coeff = wavedec2(data, "db2", level=3, mode="reflect", axes=axes)
    pywt_coeff = pywt.wavedec2(data, "db2", level=3, mode="reflect", axes=axes)
    rec = waverec2(ptwt_coeff, "db2", axes=axes)

    # test coefficients
    test_list = _compare_coeffs(ptwt_coeff, pywt_coeff)
    assert all(test_list)

    # test reconstruction.
    assert np.allclose(
        data.numpy(), rec.numpy()[..., : data.shape[-2], : data.shape[-1]]
    )


def test_2d_axis_error_axes_count() -> None:
    """Check the error for too many axes."""
    with pytest.raises(ValueError):
        data = torch.randn([32, 32, 32, 32], dtype=torch.float64)
        wavedec2(data, "haar", level=1, axes=(1, 2, 3))


def test_2d_axis_error_axes_repetition() -> None:
    """Check the error for axes repetition."""
    with pytest.raises(ValueError):
        data = torch.randn([32, 32, 32, 32], dtype=torch.float64)
        wavedec2(data, "haar", level=1, axes=(2, 2))
