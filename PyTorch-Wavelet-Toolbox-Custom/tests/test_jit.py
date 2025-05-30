"""Ensure pytorch's torch.jit.trace feature works properly."""

from typing import Optional, Union

import numpy as np
import pytest
import pywt
import torch
from scipy import signal

import ptwt
from ptwt.continuous_transform import _ShannonWavelet
from tests._mackey_glass import MackeyGenerator


def _to_jit_wavedec_fun(
    data: torch.Tensor, wavelet: Union[ptwt.Wavelet, str], level: Optional[int]
) -> list[torch.Tensor]:
    return ptwt.wavedec(data, wavelet, mode="reflect", level=level)


@pytest.mark.slow
@pytest.mark.parametrize("wavelet_string", ["db1", "db3", "db4", "sym5"])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("length", [64, 65])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv_fwt_jit(
    wavelet_string: str, level: int, length: int, batch_size: int, dtype: torch.dtype
) -> None:
    """Test jitting a convolution fwt, for various levels and padding options."""
    generator = MackeyGenerator(
        batch_size=batch_size, tmax=length, delta_t=1, device="cpu"
    )

    mackey_data_1 = torch.squeeze(generator(), -1).type(dtype)
    wavelet = pywt.Wavelet(wavelet_string)
    wavelet = ptwt.WaveletTensorTuple.from_wavelet(wavelet, dtype)

    with pytest.warns(Warning):
        jit_wavedec = torch.jit.trace(  # type: ignore
            _to_jit_wavedec_fun,
            (mackey_data_1, wavelet, torch.tensor(level)),
            strict=False,
        )
        ptcoeff = jit_wavedec(mackey_data_1, wavelet, level=torch.tensor(level))
        jit_waverec = torch.jit.trace(ptwt.waverec, (ptcoeff, wavelet))  # type: ignore
        res = jit_waverec(ptcoeff, wavelet)
    assert np.allclose(mackey_data_1.numpy(), res.numpy()[:, : mackey_data_1.shape[-1]])


def _to_jit_wavedec_2(
    data: torch.Tensor, wavelet: Union[str, ptwt.Wavelet]
) -> list[torch.Tensor]:
    """Ensure uniform datatypes in lists for the tracer.

    Going from list[Union[torch.Tensor, tuple[torch.Tensor]]] to list[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (10, 20, 20), "Changing the chape requires re-tracing."
    coeff = ptwt.wavedec2(data, wavelet, mode="reflect", level=2)
    coeff2 = []
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack(c))
    return coeff2


def _to_jit_waverec_2(
    data: list[torch.Tensor], wavelet: Union[str, ptwt.Wavelet]
) -> torch.Tensor:
    """Undo the stacking from the jit wavedec2 wrapper."""
    d_unstack: list[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = [data[0]]
    for c in data[1:]:
        d_unstack.append(tuple(sc.squeeze(0) for sc in torch.split(c, 1, dim=0)))
    rec = ptwt.waverec2(d_unstack, wavelet)
    return rec


def test_conv_fwt_jit_2d() -> None:
    """Test the jit compilation feature for the wavedec2 function."""
    data = torch.randn(10, 20, 20).type(torch.float64)
    wavelet = pywt.Wavelet("db4")
    coeff = _to_jit_wavedec_2(data, wavelet)
    rec = _to_jit_waverec_2(coeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy())

    wavelet = ptwt.WaveletTensorTuple.from_wavelet(wavelet, dtype=torch.float64)
    with pytest.warns(Warning):
        jit_wavedec2 = torch.jit.trace(  # type: ignore
            _to_jit_wavedec_2,
            (data, wavelet),
            strict=False,
        )
        jit_ptcoeff = jit_wavedec2(data, wavelet)
        # unstack the lists.
        jit_waverec = torch.jit.trace(
            _to_jit_waverec_2, (jit_ptcoeff, wavelet)
        )  # type: ignore
        rec = jit_waverec(jit_ptcoeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy(), atol=1e-7)


def _to_jit_wavedec_3(data: torch.Tensor, wavelet: str) -> list[torch.Tensor]:
    """Ensure uniform datatypes in lists for the tracer.

    Going from list[Union[torch.Tensor, dict[str, torch.Tensor]]] to list[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (10, 20, 20, 20), "Changing the shape requires re-tracing."
    coeff = ptwt.wavedec3(data, wavelet, mode="reflect", level=2)
    coeff2 = []
    keys = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack([c[key] for key in keys]))
    return coeff2


def _to_jit_waverec_3(data: list[torch.Tensor], wavelet: pywt.Wavelet) -> torch.Tensor:
    """Undo the stacking from the jit wavedec3 wrapper."""
    d_unstack: list[Union[torch.Tensor, dict[str, torch.Tensor]]] = [data[0]]
    keys = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
    for c in data[1:]:
        d_unstack.append(
            {key: sc.squeeze(0) for sc, key in zip(torch.split(c, 1, dim=0), keys)}
        )
    rec = ptwt.waverec3(d_unstack, wavelet)
    return rec


def test_conv_fwt_jit_3d() -> None:
    """Test the jit compilation feature for the wavedec3 function."""
    data = torch.randn(10, 20, 20, 20).type(torch.float64)
    wavelet = pywt.Wavelet("db4")
    coeff = _to_jit_wavedec_3(data, wavelet)
    rec = _to_jit_waverec_3(coeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy())

    wavelet = ptwt.WaveletTensorTuple.from_wavelet(wavelet, dtype=torch.float64)
    with pytest.warns(Warning):
        jit_wavedec3 = torch.jit.trace(  # type: ignore
            _to_jit_wavedec_3,
            (data, wavelet),
            strict=False,
        )
        jit_ptcoeff = jit_wavedec3(data, wavelet)
        # unstack the lists.
        jit_waverec = torch.jit.trace(
            _to_jit_waverec_3, (jit_ptcoeff, wavelet)
        )  # type: ignore
        rec = jit_waverec(jit_ptcoeff, wavelet)
    assert np.allclose(rec.squeeze(1).numpy(), data.numpy(), atol=1e-7)


def _to_jit_cwt(sig: torch.Tensor) -> torch.Tensor:
    widths = torch.arange(1, 31)
    wavelet = _ShannonWavelet("shan0.1-0.4")
    sampling_period = (4 / 800) * np.pi
    cwtmatr, _ = ptwt.cwt(sig, widths, wavelet, sampling_period=sampling_period)
    return cwtmatr


def test_cwt_jit() -> None:
    """Test cwt jitting."""
    t = np.linspace(-2, 2, 800, endpoint=False)
    sig = torch.from_numpy(signal.chirp(t, f0=1, f1=12, t1=2, method="linear"))
    with pytest.warns(Warning):
        jit_cwt = torch.jit.trace(_to_jit_cwt, (sig), strict=False)  # type: ignore
    jitcwtmatr = jit_cwt(sig)

    cwtmatr, _ = ptwt.cwt(
        sig,
        torch.arange(1, 31),
        pywt.ContinuousWavelet("shan0.1-0.4"),
        sampling_period=(4 / 800) * np.pi,
    )
    assert np.allclose(jitcwtmatr.numpy(), cwtmatr.numpy())
