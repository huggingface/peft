******************************************
Pytorch Wavelet Toolbox (`ptwt`) 
******************************************

.. image:: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml/badge.svg 
    :target: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/actions/workflows/tests.yml
    :alt: GitHub Actions

.. image:: https://readthedocs.org/projects/pytorch-wavelet-toolbox/badge/?version=latest
    :target: https://pytorch-wavelet-toolbox.readthedocs.io/en/latest/ptwt.html
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/pyversions/ptwt
    :target: https://pypi.org/project/ptwt/
    :alt: PyPI Versions

.. image:: https://img.shields.io/pypi/v/ptwt
    :target: https://pypi.org/project/ptwt/
    :alt: PyPI - Project

.. image:: https://img.shields.io/pypi/l/ptwt
    :target: https://github.com/v0lta/PyTorch-Wavelet-Toolbox/blob/main/LICENSE
    :alt: PyPI - License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black code style

.. image:: https://static.pepy.tech/personalized-badge/ptwt?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
 :target: https://pepy.tech/project/ptwt


Welcome to the PyTorch wavelet toolbox. This package implements discrete-(DWT) as well as continuous-(CWT) wavelet transforms:

- the fast wavelet transform (fwt) via ``wavedec`` and its inverse by providing the ``waverec`` function,
- the two-dimensional fwt is called ``wavedec2`` the synthesis counterpart ``waverec2``,
- ``wavedec3`` and ``waverec3`` cover the three-dimensional analysis and synthesis case,
- ``fswavedec2``, ``fswavedec3``, ``fswaverec2`` and ``fswaverec3`` support separable transformations.
- ``MatrixWavedec`` and ``MatrixWaverec`` implement sparse-matrix-based fast wavelet transforms with boundary filters,
- 2d sparse-matrix transforms with separable & non-separable boundary filters are available,
- ``MatrixWavedec3`` and ``MatrixWaverec3`` allow separable 3D-fwt's with boundary filters.
- ``cwt`` computes a one-dimensional continuous forward transform,
- single and two-dimensional wavelet packet forward and backward transforms are available via the ``WaveletPacket`` and ``WaveletPacket2D`` objects,
- finally, this package provides adaptive wavelet support (experimental).

This toolbox extends `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_. In addition to boundary wavelets, we provide GPU and gradient support via a PyTorch backend.
Complete documentation is available at: https://pytorch-wavelet-toolbox.readthedocs.io/en/latest/ptwt.html

This toolbox is independent work. Meta or the PyTorch team have not endorsed it.

**Installation**

Install the toolbox via pip or clone this repository. In order to use ``pip``, type:

.. code-block:: sh

    pip install ptwt
  

You can remove it later by typing ``pip uninstall ptwt``.

Example usage:
""""""""""""""
**Single dimensional transform**

One way to compute fast wavelet transforms is to rely on padding and
convolution. Consider the following example: 

.. code-block:: python

  import torch
  import numpy as np
  import pywt
  import ptwt  # use "from src import ptwt" for a cloned the repo
  
  # generate an input of even length.
  data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
  data_torch = torch.from_numpy(data.astype(np.float32))
  wavelet = pywt.Wavelet('haar')
  
  # compare the forward fwt coefficients
  print(pywt.wavedec(data, wavelet, mode='zero', level=2))
  print(ptwt.wavedec(data_torch, wavelet, mode='zero', level=2))
  
  # invert the fwt.
  print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode='zero'),
                     wavelet))


The functions ``wavedec`` and ``waverec`` compute the 1d-fwt and its inverse.
Internally both rely on ``conv1d``, and its transposed counterpart ``conv_transpose1d``
from the ``torch.nn.functional`` module. This toolbox also supports discrete wavelets
see ``pywt.wavelist(kind='discrete')``. I have tested
Daubechies-Wavelets ``db-x`` and symlets ``sym-x``, are usually a good starting point. 

**Two-dimensional transform**

Analog to the 1d-case ``wavedec2`` and ``waverec2`` rely on 
``conv2d``, and its transposed counterpart ``conv_transpose2d``.
To test an example, run:


.. code-block:: python

  import ptwt, pywt, torch
  import numpy as np
  import scipy.misc

  face = np.transpose(scipy.datasets.face(),
                          [2, 0, 1]).astype(np.float64)
  pytorch_face = torch.tensor(face)
  coefficients = ptwt.wavedec2(pytorch_face, pywt.Wavelet("haar"),
                                  level=2, mode="constant")
  reconstruction = ptwt.waverec2(coefficients, pywt.Wavelet("haar"))
  np.max(np.abs(face - reconstruction.squeeze(1).numpy()))


**Speed tests**

Speed tests comparing our tools to related libraries are `available <https://github.com/v0lta/PyTorch-Wavelet-Toolbox/tree/main/examples/speed_tests/>`_.


**Boundary Wavelets with Sparse-Matrices**

In addition to convolution and padding approaches,
sparse-matrix-based code with boundary wavelet support is available.
In contrast to padding, boundary wavelets do not add extra pixels at 
the edges.
Internally, boundary wavelet support relies on ``torch.sparse.mm``.
Generate 1d sparse matrix forward and backward transforms with the
``MatrixWavedec`` and ``MatrixWaverec`` classes.
Reconsidering the 1d case, try:

.. code-block:: python

  import torch
  import numpy as np
  import pywt
  import ptwt  # use "from src import ptwt" for a cloned the repo
  
  # generate an input of even length.
  data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
  data_torch = torch.from_numpy(data.astype(np.float32))
  # forward
  matrix_wavedec = ptwt.MatrixWavedec(pywt.Wavelet("haar"), level=2)
  coeff = matrix_wavedec(data_torch)
  print(coeff)
  # backward 
  matrix_waverec = ptwt.MatrixWaverec(pywt.Wavelet("haar"))
  rec = matrix_waverec(coeff)
  print(rec)


The process for the 2d transforms ``MatrixWavedec2``, ``MatrixWaverec2`` works similarly.
By default, a separable transformation is used.
To use a non-separable transformation, pass ``separable=False`` to ``MatrixWavedec2`` and ``MatrixWaverec2``.
Separable transformations use a 1D transformation along both axes, which might be faster since fewer matrix entries
have to be orthogonalized.


**Adaptive** **Wavelets**

Experimental code to train an adaptive wavelet layer in PyTorch is available in the ``examples`` folder. In addition to static wavelets
from pywt,

- Adaptive product-filters
- and optimizable orthogonal-wavelets are supported.

See https://github.com/v0lta/PyTorch-Wavelet-Toolbox/tree/main/examples/network_compression/ for a complete implementation.


**Testing**

The ``tests`` folder contains multiple tests to allow independent verification of this toolbox.
The GitHub workflow executes a subset of all tests for efficiency reasons. 
After cloning the repository, moving into the main directory, and installing ``nox`` with ``pip install nox`` run

.. code-block:: sh

  nox --session test



for all existing tests.

Citation
""""""""

If you use this work in a scientific context, please cite the following:

.. code-block::

  @article{JMLR:v25:23-0636,
    author  = {Moritz Wolter and Felix Blanke and Jochen Garcke and Charles Tapley Hoyt},
    title   = {ptwt - The PyTorch Wavelet Toolbox},
    journal = {Journal of Machine Learning Research},
    year    = {2024},
    volume  = {25},
    number  = {80},
    pages   = {1--7},
    url     = {http://jmlr.org/papers/v25/23-0636.html}
  }
