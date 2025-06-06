.. _sec-cwt:

Introduction to continuous wavelet transforms
=============================================

This page provides documentation, focusing on the Continuous Wavelet Transform (CWT)
and its application in signal analysis.
See, for example, :cite:`strang1996wavelets,mallat1999wavelet,daubechies1992lectures`
or :cite:`kaiser1994guidetowavelets` for excellent detailed introductions to the topic.

The implementation in :py:meth:`ptwt.continuous_transform.cwt` uses https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py
and efficiently utilizes PyTorch for computation and supports various wavelet functions for flexible signal analysis.

Definition of CWT
------------------
The Continuous Wavelet Transform (CWT) is a mathematical tool
used for analyzing signals in both time and frequency domains simultaneously.
It is based on a wavelet function, which is a small, localized wave-like function,
that is stretched or compressed and shifted across the signal.
The shifted and stretched version of a mother wavelet function
:math:`\psi \in \mathcal{L}_2(\mathbb{R})` is defined by :math:`\psi^{a,b}(t)`,
where :math:`a\in\mathbb{R}\setminus\{0\}` and :math:`b\in\mathbb{R}`
and :math:`\mathcal{L}_2` is a Hilbert space (complete linear space with scalar product), as:

.. math::
    \psi^{a,b}(t) := |a|^{-\frac{1}{2}}\,\psi\left(\frac{t-b}{a}\right)

The CWT of a signal :math:`x\in\mathcal{L}_2(\mathbb{R})` is defined as:

.. math::
    \mathcal{W}_x(a,b) = \langle x,\psi^{a,b} \rangle = |a|^{-\frac{1}{2}}\int_{-\infty}^{\infty} x(t) \, \psi^*\left(\frac{t-b}{a}\right)\,dt

where :math:`\psi^*` denotes the complex conjugate of :math:`\psi`
and :math:`\langle\cdot,\cdot\rangle` the scalar product of two functions.
The CWT essentially provides a time-scale representation of the signal,
revealing information about its frequency content at different time points.

According to :cite:`mertins2020signaltheorie`,
the Fourier transform :math:`\Psi_{a,b}` of the wavelet :math:`\psi_{a,b}` shifted in time and
frequency can serve as an alternative definition of the CWT.
The Fourier transform is directly derived from the properties of
the Fourier Transform regarding time shifting and amplitude modulation:

.. math::
    \Psi_{a,b}(\omega) = |a|^{\frac{1}{2}}\,\text{e}^{-\text{i}\omega b}\,\Psi(\omega)\,

where :math:`\Psi` denotes the Fourier transform of the mother wavelet :math:`\psi`, :math:`\text{i}`
denotes the imaginary unit and :math:`\omega` the frequency of the signal.
Consequently, the CWT :math:\mathcal{W}_x` of a signal :math:`x` can be defined as the inverse Fourier transformation,
as described in :cite:`mertins2020signaltheorie`:

.. math::
    \mathcal{W}_x(a,b)=\frac{1}{2\pi}\langle X,\Psi_{a,b}\rangle=\frac{|a|^\frac{1}{2}}{2\pi}\int_{-\infty}^{\infty}X(\omega)\Psi^*(a\omega)\,\text{e}^{\text{i}\omega b}\,\text{d}\omega

where :math:`X` denotes the Fourier transform of the signal :math:`x`.
This mathematical representation can be used for implementing the CWT.

Time-Scale Analysis with the CWT
--------------------------------
The CWT introduces the concept of scale alongside time,
allowing for a scalable analysis of the signal.
By varying the scale parameter :math:`a`,
the CWT provides information about frequency characteristics around :math:`t=b`.
This property makes the CWT particularly suitable for analyzing high-frequency components of signals concerning time.
Additionally, the resolution in time and frequency is dependent on the scale parameter :math:`a`,
with smaller :math:`|a|` yielding finer time resolution
and larger :math:`|a|` providing better frequency resolution.

In digital signal processing, signals are discrete, necessitating considerations for implementing the CWT.
The CWT is inherently continuous,
but numerical approximations are employed for computation.
Unlike the Discrete Wavelet Transform, which operates on discrete time steps and scales,
CWT offers a near-continuous analysis.
This is achieved using methods like the FFT for efficient computation in the frequency domain.

To use continuous wavelets effectively for audio signal analysis, certain criteria must be met.
Wavelets should be mean-free and satisfy
admissibility conditions ensuring signal energy normalization and invertibility.
Commonly used wavelets such as the Morlet and Mexican-Hat
wavelets are preferred for their properties in time-frequency representation.
The choice of wavelet impacts the trade-off between time and
frequency resolution, with different wavelets offering advantages in specific applications.

**Key Points**:

* The CWT is based on a wavelet function, denoted as :math:`\psi(t)`.
    This function is scaled and shifted to create a family of wavelets :math:`\psi^{a,b}`,
    where :math:`a` represents the scale parameter and :math:`b` represents the translation parameter.

* CWT Formula:
    The CWT of a signal :math:`x(t)` with respect to a wavelet :math:`\psi^{a,b}`
    is given by the inner product of the signal and the translated/scaled wavelet.

* Interpretation:
    The CWT represents the coefficients of a series expansion of the signal in terms of wavelets.
    This allows the signal to be reconstructed from a weighted superposition of the wavelets.

* Properties:
    The wavelet function :math:`\psi` must satisfy certain conditions, such as being admissible,
    which ensures that the CWT can accurately represent the signal without loss of information.

* Time-Scale Analysis:
    One significant aspect of CWT is its ability to analyze signals
    in both time and scale (or frequency) domains simultaneously.
    By varying the scale parameter,
    the CWT can provide information about frequency content at different time points in the signal.

* Implementation Considerations:
    Implementing CWT requires discretizing the scales and performing convolutions.
    Efficient algorithms, such as Fast Fourier Transform (FFT),
    are often utilized for numerical computations.


Bibliography
------------

.. bibliography::
