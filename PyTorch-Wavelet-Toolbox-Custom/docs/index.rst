.. PyTorch-Wavelet-Toolbox documentation master file, created by
   sphinx-quickstart on Thu Oct 14 15:19:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _PyTorch install instructions: https://pytorch.org/get-started/locally/

PyTorch Wavelet Toolbox (``ptwt``)
==================================

``ptwt`` brings wavelet transforms to PyTorch. The code is open-source, follow the GitHub link above
to go to the source. This package is listed in the Python Package Index (PyPI). It's best installed via pip.
GPU support depends on PyTorch. To take advantage of GPU-processing follow the `PyTorch install instructions`_.
Install the version that best suits your system's hardware setup. Once PyTorch is set up, type the following
to get started:

.. code-block:: sh

    pip install ptwt

This documentation aims to explain the foundations of wavelet theory, introduce the ``ptwt`` package by example, and
deliver a complete documentation of all functions. Readers who are already familiar with the theory should directly
jump to the examples or the API documentation using the navigation on the left.

``ptwt`` is built to be `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ compatible.
It should be possible to switch back and forth with relative ease.

If you use this work in a scientific context, please cite the following paper:

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

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   intro
   intro_cwt
   examples


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Public API

   ptwt

