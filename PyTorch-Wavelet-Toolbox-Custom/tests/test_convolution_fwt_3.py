"""Test our 3d for loop-convolution based fwt code."""

import typing
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pytest
import pywt
import torch

import ptwt
from ptwt.constants import BoundaryMode


def _expand_dims(
    batch_list: list[Union[npt.NDArray[Any], dict[Any, Any]]]
) -> list[Any]:
    for pos, bel in enumerate(batch_list):
        if isinstance(bel, np.ndarray):
            batch_list[pos] = np.expand_dims(bel, 0)
        elif isinstance(bel, dict):
            for key, item in bel.items():
                batch_list[pos][key] = np.expand_dims(item, 0)
        else:
            raise TypeError(
                "Argument type not supported,\
                             batch_list element should have been a dict."
            )
    return batch_list


def _cat_batch_list(batch_lists: Any) -> Any:
    cat_list = None
    for batch_list in batch_lists:
        batch_list = _expand_dims(batch_list)
        if not cat_list:
            cat_list = batch_list
        else:
            for pos, (cat_el, batch_el) in enumerate(zip(cat_list, batch_list)):
                if isinstance(cat_el, np.ndarray):
                    cat_list[pos] = np.concatenate([cat_el, batch_el])
                elif isinstance(cat_el, dict):
                    for key, tensor in cat_el.items():
                        cat_el[key] = np.concatenate([tensor, batch_el[key]])
                else:
                    raise NotImplementedError()
    return cat_list


@pytest.mark.parametrize(
    "shape",
    [
        (1, 31, 32, 33),
        (1, 64, 64, 64),
        (2, 64, 64, 64),
        (3, 31, 64, 64),
        (3, 64, 31, 64),
        (3, 64, 64, 31),
        (3, 31, 31, 31),
        (3, 32, 32, 32),
        (3, 31, 32, 33),
    ],
)
@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("mode", typing.get_args(BoundaryMode))
def test_waverec3(
    shape: list[int], wavelet: str, level: int, mode: BoundaryMode
) -> None:
    """Ensure the 3d analysis transform is invertible."""
    data = np.random.randn(*shape)
    data_t = torch.from_numpy(data)
    ptwc = ptwt.wavedec3(data_t, wavelet, level=level, mode=mode)
    batch_list = []
    for batch_no in range(data_t.shape[0]):
        pywc = pywt.wavedecn(data_t[batch_no].numpy(), wavelet, level=level, mode=mode)
        batch_list.append(pywc)
    cat_pywc = _cat_batch_list(batch_list)

    # ensure ptwt and pywt coefficients are identical.
    test_list = []
    for a, b in zip(ptwc, cat_pywc):
        if type(a) is torch.Tensor:
            test_list.append(np.allclose(a, b))
        else:
            test_list.extend([np.allclose(a[key], b[key]) for key in a.keys()])

    assert all(test_list)

    # ensure the transforms are invertible.
    rec = ptwt.waverec3(ptwc, wavelet)
    assert np.allclose(
        rec.numpy()[..., : shape[1], : shape[2], : shape[3]], data_t.numpy()
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "size", [[5, 32, 32, 32], [4, 3, 32, 32, 32], [1, 1, 1, 32, 32, 32]]
)
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("wavelet", ["haar", "sym3", "db3"])
@pytest.mark.parametrize("mode", ["zero", "symmetric", "reflect"])
def test_multidim_input(size: list[int], level: int, wavelet: str, mode: str) -> None:
    """Ensure correct folding of multidimensional inputs."""
    data = torch.randn(size, dtype=torch.float64)
    ptwc = ptwt.wavedec3(data, wavelet, level=level, mode=mode)
    # batch_list = []
    # for batch_no in range(data.shape[0]):
    #     pywc = pywt.wavedecn(data[batch_no].numpy(), wavelet, level=level, mode=mode)
    #     batch_list.append(pywc)
    # cat_pywc = _cat_batch_list(batch_list)
    cat_pywc = pywt.wavedecn(data, wavelet, level=level, mode=mode, axes=[-3, -2, -1])

    # ensure ptwt and pywt coefficients are identical.
    test_list = []
    for a, b in zip(ptwc, cat_pywc):
        if type(a) is torch.Tensor:
            test_list.append(np.allclose(a, b))
        else:
            test_list.extend([np.allclose(a[key], b[key]) for key in a.keys()])

    assert all(test_list)

    rec = ptwt.waverec3(ptwc, wavelet)
    assert np.allclose(
        rec.numpy()[..., : size[-3], : size[-2], : size[-1]], data.numpy()
    )


@pytest.mark.slow
@pytest.mark.parametrize("axes", [(-3, -2, -1), (0, 2, 1)])
@pytest.mark.parametrize("level", [1, 2, None])
@pytest.mark.parametrize("mode", ["zero", "symmetric", "reflect"])
def test_axes_arg_3d(
    axes: tuple[int, int, int], level: int, mode: BoundaryMode
) -> None:
    """Test axes argument support."""
    wavelet = "db3"
    data = torch.randn([16, 16, 16, 16, 16], dtype=torch.float64)
    ptwc = ptwt.wavedec3(data, wavelet, level=level, mode=mode, axes=axes)
    cat_pywc = pywt.wavedecn(data, wavelet, level=level, mode=mode, axes=axes)

    # ensure ptwt and pywt coefficients are identical.
    test_list = []
    for a, b in zip(ptwc, cat_pywc):
        if type(a) is torch.Tensor:
            test_list.append(np.allclose(a, b))
        else:
            test_list.extend([np.allclose(a[key], b[key]) for key in a.keys()])

    assert all(test_list)

    rec = ptwt.waverec3(ptwc, wavelet, axes=axes)
    assert np.allclose(data, rec)


def test_2d_dimerror() -> None:
    """Check the error for too many axes."""
    with pytest.raises(ValueError):
        data = torch.randn([32, 32], dtype=torch.float64)
        ptwt.wavedec3(data, "haar")


def test_1d_dimerror() -> None:
    """Check the error for too many axes."""
    with pytest.raises(ValueError):
        data = torch.randn([32], dtype=torch.float64)
        ptwt.wavedec3(data, "haar")
