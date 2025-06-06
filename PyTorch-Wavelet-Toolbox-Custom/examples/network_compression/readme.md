#### Adaptive Wavelets
```mnist_compression.py``` trains a CNN on MNIST with an adaptive-wavelet
compressed linear layer. The wavelets in the linear layer are learned using gradient descent.

See https://arxiv.org/pdf/2004.09569v3.pdf for a detailed description of the method.

Running this example requires the following steps:
  - clone this repository,
  - install `ptwt`,
  - and execute ```python mnist_compression.py```.
