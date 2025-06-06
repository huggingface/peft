"""PyTorch compatible cwt code.

This module is based on pywt's cwt implementation.
"""

from typing import Any, Union

import numpy as np
import torch
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet
from pywt._functions import scale2frequency
from torch.fft import fft, ifft


def _next_fast_len(n: int) -> int:
    """Round up size to the nearest power of two.

    Given a number of samples `n`, returns the next power of two
    following this number to take advantage of FFT speedup.
    This fallback is less efficient than `scipy.fftpack.next_fast_len`
    """
    return int(2 ** np.ceil(np.log2(n)))


def cwt(
    data: torch.Tensor,
    scales: Union[np.ndarray, torch.Tensor],  # type: ignore
    wavelet: Union[ContinuousWavelet, str],
    sampling_period: float = 1.0,
) -> tuple[torch.Tensor, np.ndarray]:  # type: ignore
    """Compute the single-dimensional continuous wavelet transform.

    This function is a PyTorch port of pywt.cwt as found at:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py

    Args:
        data (torch.Tensor): The input tensor of shape ``[batch_size, time]``.
        scales (torch.Tensor or np.array):
            The wavelet scales to use. One can use
            ``f = pywt.scale2frequency(wavelet, scale)/sampling_period`` to determine
            what physical frequency, ``f``. Here, ``f`` is in hertz when the
            ``sampling_period`` is given in seconds.
        wavelet (ContinuousWavelet or str): The continuous wavelet to work with.
        sampling_period (float): Sampling period for the frequencies output (optional).
            The values computed for ``coefs`` are independent of the choice of
            ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
            period).

    Raises:
        ValueError: If a scale is too small for the input signal.

    Returns:
        A tuple (out_tensor, frequencies). The first tuple-element contains
        the transformation matrix of shape ``[scales, batch, time]``.
        The second element contains an array with frequency information.

    Example:
        >>> import torch, ptwt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> t = np.linspace(-2, 2, 800, endpoint=False)
        >>> sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
        >>> widths = np.arange(1, 31)
        >>> cwtmatr, freqs = ptwt.cwt(
        >>>     torch.from_numpy(sig), widths, "mexh", sampling_period=(4 / 800) * np.pi
        >>> )
    """
    # accept array_like input; make a copy to ensure a contiguous array
    if not isinstance(
        wavelet, (ContinuousWavelet, Wavelet, _DifferentiableContinuousWavelet)
    ):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if type(scales) is torch.Tensor:
        scales = scales.numpy()
    elif np.isscalar(scales):
        scales = np.array([scales])

    if isinstance(wavelet, torch.nn.Module):
        if data.is_cuda:
            wavelet.cuda()

    precision = 10
    int_psi, x = _integrate_wavelet(wavelet, precision=precision)
    if type(wavelet) is ContinuousWavelet:
        int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi
        int_psi = torch.tensor(int_psi, device=data.device)
    elif isinstance(wavelet, torch.nn.Module):
        int_psi = torch.conj(int_psi) if wavelet.complex_cwt else int_psi
    else:
        int_psi = torch.tensor(int_psi, device=data.device)
        x = torch.tensor(x, device=data.device)

    # convert int_psi, x to the same precision as the data
    # x = np.asarray(x, dtype=data.cpu().numpy().real.dtype)

    size_scale0 = -1
    fft_data = None

    out = []
    for scale in scales:
        step = x[1] - x[0]
        j = torch.arange(
            scale * (x[-1] - x[0]) + 1, device=data.device, dtype=data.dtype
        ) / (scale * step)
        j = torch.floor(j).type(torch.long)
        if j[-1] >= len(int_psi):
            # j = np.extract(j < len(int_psi), j)
            j = torch.masked_select(j, j < len(int_psi))
        int_psi_scale = int_psi[j].flip(0)

        # The padding is selected for:
        # - optimal FFT complexity
        # - to be larger than the two signals length to avoid circular
        #   convolution
        size_scale = _next_fast_len(data.shape[-1] + len(int_psi_scale) - 1)
        if size_scale != size_scale0:
            # Must recompute fft_data when the padding size changes.
            fft_data = fft(data, size_scale, dim=-1)
        size_scale0 = size_scale
        fft_wav = fft(int_psi_scale, size_scale, dim=-1)
        conv = ifft(fft_wav * fft_data, dim=-1)
        conv = conv[..., : data.shape[-1] + len(int_psi_scale) - 1]

        coef = -np.sqrt(scale) * torch.diff(conv, dim=-1)

        # transform axis is always -1
        d = (coef.shape[-1] - data.shape[-1]) / 2.0
        if d > 0:
            coef = coef[..., int(np.floor(d)) : -int(np.ceil(d))]
        elif d < 0:
            raise ValueError("Selected scale of {} too small.".format(scale))

        out.append(coef)
    out_tensor = torch.stack(out)
    if type(wavelet) is Wavelet:
        out_tensor = out_tensor.real
    elif isinstance(wavelet, _DifferentiableContinuousWavelet):
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real
        wavelet.cpu()
    else:
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real

    with torch.no_grad():
        frequencies = scale2frequency(wavelet, scales, precision)
        if np.isscalar(frequencies):
            frequencies = np.array([frequencies])
    frequencies /= sampling_period

    if isinstance(wavelet, _DifferentiableContinuousWavelet):
        if data.is_cuda:
            wavelet.cuda()

    return out_tensor, frequencies


def _integrate_wavelet(
    wavelet: Union[ContinuousWavelet, str], precision: int = 8
) -> Any:
    """
    Integrate `psi` wavelet function from -Inf to x using rectangle integration.

    Modified to enable gradient flow through the cwt.

    Ported from:
    https://github.com/PyWavelets/pywt/blob/cef09e7f419aaf4c39b9f778bdc2d54b32fd7337/pywt/_functions.py#L60


    Args:
        wavelet (Wavelet instance or str): Wavelet to integrate.
            If a string, should be the name of a wavelet.
        precision (int): Precision that will be used for wavelet function
            approximation computed with the wavefun(level=precision)
            Wavelet's method. Defaults to 8.

    Returns:
        A tuple (int_psi, x) for orthogonal wavelets;
        for other wavelets, a tuple (int_psi_d, int_psi_r, x) is returned instead.

    Example:
        >>> from pywt import Wavelet, _integrate_wavelet
        >>> wavelet1 = Wavelet('db2')
        >>> [int_psi, x] = _integrate_wavelet(wavelet1, precision=5)
        >>> wavelet2 = Wavelet('bior1.3')
        >>> [int_psi_d, int_psi_r, x] = _integrate_wavelet(wavelet2, precision=5)
    """

    def _integrate(
        arr: Union[np.ndarray, torch.Tensor],  # type: ignore
        step: Union[np.ndarray, torch.Tensor],  # type: ignore
    ) -> Union[np.ndarray, torch.Tensor]:  # type: ignore
        if type(arr) is np.ndarray:
            integral = np.cumsum(arr)
        elif type(arr) is torch.Tensor:
            integral = torch.cumsum(arr, -1)
        else:
            raise TypeError("Only ndarrays or tensors are integratable.")
        integral *= step
        return integral

    if type(wavelet) is str:
        wavelet = DiscreteContinuousWavelet(wavelet)
    elif not isinstance(
        wavelet, (Wavelet, ContinuousWavelet, _DifferentiableContinuousWavelet)
    ):
        wavelet = DiscreteContinuousWavelet(wavelet)

    functions_approximations = wavelet.wavefun(precision)

    if len(functions_approximations) == 2:  # continuous wavelet
        psi, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi, step), x

    elif len(functions_approximations) == 3:  # orthogonal wavelet
        _, psi, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi, step), x

    else:  # biorthogonal wavelet
        _, psi_d, _, psi_r, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi_d, step), _integrate(psi_r, step), x


class _WaveletParameter(torch.nn.Parameter):
    pass


class _DifferentiableContinuousWavelet(
    torch.nn.Module, ContinuousWavelet  # type: ignore
):
    """A base class for learnable Continuous Wavelets."""

    def __init__(self, name: str):
        """Create a trainable shannon wavelet."""
        super().__init__()
        super(ContinuousWavelet, self).__init__()

        self.dtype = torch.float64
        # Use torch nn parameter
        self.bandwidth_par = _WaveletParameter(
            torch.sqrt(torch.tensor(self.bandwidth_frequency, dtype=self.dtype)),
            requires_grad=True,
        )
        self.center_par = _WaveletParameter(
            torch.sqrt(torch.tensor(self.center_frequency, dtype=self.dtype)),
            requires_grad=True,
        )

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid."""
        raise NotImplementedError

    @property
    def bandwidth(self) -> torch.Tensor:
        """Square the bandwith parameter to ensure positive values."""
        return self.bandwidth_par * self.bandwidth_par

    @property
    def center(self) -> torch.Tensor:
        """Square the bandwith parameter to ensure positive values."""
        return self.center_par * self.center_par

    def wavefun(
        self, precision: int, dtype: torch.dtype = torch.float64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Define a grid and evaluate the wavelet on it."""
        length = 2**precision
        # load the bounds from untyped pywt code.
        lower_bound: float = float(self.lower_bound)
        upper_bound: float = float(self.upper_bound)
        grid = torch.linspace(
            lower_bound,
            upper_bound,
            length,
            dtype=dtype,
            device=self.bandwidth_par.device,
        )
        return self(grid), grid


class _ShannonWavelet(_DifferentiableContinuousWavelet):
    """A differentiable Shannon wavelet."""

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid."""
        shannon = (
            torch.sqrt(self.bandwidth)
            * (
                torch.sin(torch.pi * self.bandwidth * grid_values)
                / (torch.pi * self.bandwidth * grid_values)
            )
            * torch.exp(1j * 2 * torch.pi * self.center * grid_values)
        )
        return shannon


class _ComplexMorletWavelet(_DifferentiableContinuousWavelet):
    """A differentiable Shannon wavelet."""

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid."""
        morlet = (
            1.0
            / torch.sqrt(torch.pi * self.bandwidth)
            * torch.exp(-(grid_values**2) / self.bandwidth)
            * torch.exp(1j * 2 * torch.pi * self.center * grid_values)
        )
        return morlet
