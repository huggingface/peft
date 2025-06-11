"""Test the stationary wavelet transformation code."""

from typing import Optional

import numpy as np
import pytest
import pywt
import torch

from ptwt.stationary_transform import _circular_pad, iswt, swt


@pytest.mark.parametrize("shape", [(8,), (1, 8), (4, 8), (4, 6, 8), (4, 6, 8, 8)])
def test_circular_pad(shape: tuple[int, ...]) -> None:
    """Test patched circular padding."""
    test_data_np = np.random.rand(*shape).astype(np.float32)
    test_data_pt = torch.from_numpy(test_data_np)

    # torch.nn.functional.pad does always pad the last dimension given
    # a 2D or 3D input, however, numpy pads along all axis given pad
    # width [10, 10]. we need to explicitly set the pad width of the
    # first N-1 dimensions to zero
    expected = np.pad(
        test_data_np,
        [(0, 0) if i != 1 else (10, 10) for i in range(len(shape), 0, -1)],
        mode="wrap",
    )

    if len(shape) < 2:
        # torch.nn.functional.pad only implemented for
        # 2D,3D, 4D, 5Dtensors in circular mode
        with pytest.raises(NotImplementedError):
            _circular_pad(test_data_pt, [10, 10])
    elif len(shape) > 3:
        # torch.nn.functional.pad tries to pad the last 2 dimensions
        # given a 4D input the provided pad width [10, 10] misses values
        # for the second dimension, hence the error
        with pytest.raises(NotImplementedError):  # should be a ValueError though...
            _circular_pad(test_data_pt, [10, 10])
    else:
        actual = _circular_pad(test_data_pt, [10, 10])
        assert expected.shape == tuple(actual.shape)
        assert np.allclose(expected, actual.numpy())


@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [[32], [1, 32], [3, 64], [5, 64]])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "db3", "db4"])
def test_swt_1d(level: Optional[int], size: int, wavelet: str) -> None:
    """Test the 1d swt."""
    signal = torch.from_numpy(np.random.normal(size=size).astype(np.float64))
    ptwt_coeff = swt(signal, wavelet, level=level)
    pywt_coeff = pywt.swt(signal.numpy(), wavelet, level, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])
    assert all(test_list)


@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("size", [[32], [1, 32], [5, 64]])
@pytest.mark.parametrize("wavelet", ["db1", "db2"])
def test_iswt_1d(level: Optional[int], size: int, wavelet: str) -> None:
    """Ensure iswt inverts swt."""
    signal = torch.from_numpy(np.random.normal(size=size).astype(np.float64))
    # signal = np.stack([np.arange(32)] * 3).astype(np.float64)
    ptwt_coeff = swt(signal, wavelet, level=level)
    rec = iswt(ptwt_coeff, wavelet)
    assert torch.allclose(rec, signal)


@pytest.mark.parametrize("size", [[32, 64], [32, 128], [3, 32, 256]])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym3"])
@pytest.mark.parametrize("level", [1, 2, 3, None])
@pytest.mark.parametrize("axis", [1, -1])
def test_swt_1d_slow(level: Optional[int], size: int, wavelet: str, axis: int) -> None:
    """Test the 1d swt."""
    signal = torch.from_numpy(np.random.normal(size=size).astype(np.float64))
    ptwt_coeff = swt(signal, wavelet, level=level, axis=axis)
    pywt_coeff = pywt.swt(
        signal.numpy(), wavelet, level, trim_approx=True, norm=False, axis=axis
    )
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])
    assert all(test_list)
    rec = iswt(ptwt_coeff, wavelet, axis=axis)
    assert torch.allclose(rec, signal)


@pytest.mark.parametrize("size", [[32], [64]])
@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym3"])
@pytest.mark.parametrize("level", [1, 2, 3, None])
def test_swt_1d_nobatch_axis(level: Optional[int], size: int, wavelet: str) -> None:
    """Test the 1d swt."""
    signal = torch.from_numpy(np.random.normal(size=size).astype(np.float64))
    ptwt_coeff = swt(signal, wavelet, level=level)
    pywt_coeff = pywt.swt(signal.numpy(), wavelet, level, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])
    assert all(test_list)
    rec = iswt(ptwt_coeff, wavelet)
    assert torch.allclose(rec, signal)


def test_iswt_1d_int() -> None:
    """Test the 1d swt."""
    signal = torch.from_numpy(np.random.randint(0, 10, size=[16]).astype(np.float64))
    ptwt_coeff = swt(signal, "db2", level=2)
    pywt_coeff = pywt.swt(signal.numpy(), "db2", 2, trim_approx=True, norm=False)
    test_list = []
    for a, b in zip(ptwt_coeff, pywt_coeff):
        test_list.extend([np.allclose(ael.numpy(), bel) for ael, bel in zip(a, b)])
    assert all(test_list)
    rec = iswt(ptwt_coeff, "db2")
    assert torch.allclose(rec, signal)
