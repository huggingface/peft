"""Test the continuous transformation code."""

from typing import Any

import numpy as np
import pytest
import pywt
import torch
from scipy import signal

from ptwt.continuous_transform import _ShannonWavelet, cwt

continuous_wavelets = [
    "cgau1",
    "cgau2",
    "cgau3",
    "cgau4",
    "cgau5",
    "cgau6",
    "cgau7",
    "cgau8",
    "gaus1",
    "gaus2",
    "gaus3",
    "gaus4",
    "gaus5",
    "gaus6",
    "gaus7",
    "gaus8",
    "mexh",
    "morl",
]


@pytest.mark.parametrize("scales", [np.arange(1, 16), 5.0, torch.arange(1, 15)])
@pytest.mark.parametrize("samples", [31, 32])
@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt(wavelet: str, samples: int, scales: Any) -> None:
    """Test the cwt implementation for various wavelets."""
    t = np.linspace(-1, 1, samples, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
    scales_np = scales.numpy() if type(scales) is torch.Tensor else scales
    cwtmatr, freqs = pywt.cwt(data=sig, scales=scales_np, wavelet=wavelet)
    sig = torch.from_numpy(sig)
    cwtmatr_pt, freqs_pt = cwt(sig, scales, wavelet)
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.slow
@pytest.mark.parametrize("cuda", [False, True])
def test_cwt_cuda(cuda: bool, wavelet: str = "cgau6") -> None:
    """Test the cwt implementation with a GPU."""
    t = np.linspace(-1, 1, 400, endpoint=False)
    sig = signal.chirp(t, f0=0, f1=20, t1=1, method="linear")
    cwtmatr, freqs = pywt.cwt(data=sig, scales=np.arange(1, 64), wavelet=wavelet)
    sig = torch.from_numpy(sig)
    if cuda:
        if torch.cuda.is_available():
            sig = sig.cuda()
    cwtmatr_pt, freqs_pt = cwt(sig, np.arange(1, 64), wavelet)
    if cuda:
        cwtmatr_pt = cwtmatr_pt.cpu()
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.parametrize("wavelet", continuous_wavelets)
def test_cwt_batched(wavelet: str) -> None:
    """Test batched transforms."""
    sig = np.random.randn(10, 200)
    widths = np.arange(1, 30)
    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet)
    sig_np = torch.from_numpy(sig)
    cwtmatr_pt, freqs_pt = cwt(sig_np, widths, wavelet)
    assert np.allclose(cwtmatr_pt.numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)


@pytest.mark.parametrize("type", ["shan1-1"])
@pytest.mark.parametrize("grid_size", [8, 9, 10])
def test_nn_schannon_wavefun(type: str, grid_size: int) -> None:
    """Test the wavelet sampling for the differentiable shannon example."""
    pywt_shannon = pywt.ContinuousWavelet(type)
    ptwt_shannon = _ShannonWavelet(type)

    pywt_sample = pywt_shannon.wavefun(grid_size)
    ptwt_sample = ptwt_shannon.wavefun(grid_size)

    assert np.allclose(pywt_sample[0], ptwt_sample[0].detach().numpy())
    assert np.allclose(pywt_sample[1], ptwt_sample[1].detach().numpy())


@pytest.mark.parametrize("scales", [np.arange(1, 16), 5.0, torch.arange(1, 15)])
@pytest.mark.parametrize("samples", [31, 32])
@pytest.mark.parametrize("cuda", [False, True])
def test_nn_cwt(samples: int, scales: Any, cuda: bool) -> None:
    """Test the cwt using a differentiable continuous wavelet."""
    pywt_shannon = pywt.ContinuousWavelet("shan1-1")
    ptwt_shannon = _ShannonWavelet("shan1-1")
    t = np.linspace(-1, 1, samples, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
    scales_np = scales.numpy() if type(scales) is torch.Tensor else scales
    cwtmatr, freqs = pywt.cwt(data=sig, scales=scales_np, wavelet=pywt_shannon)
    sig = torch.from_numpy(sig)
    if cuda:
        if torch.cuda.is_available():
            sig = sig.cuda()
    cwtmatr_pt, freqs_pt = cwt(data=sig, scales=scales, wavelet=ptwt_shannon)
    assert np.allclose(cwtmatr_pt.detach().cpu().numpy(), cwtmatr)
    assert np.allclose(freqs, freqs_pt)
