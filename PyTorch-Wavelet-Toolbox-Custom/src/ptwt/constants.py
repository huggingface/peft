"""Constants and types used throughout the PyTorch Wavelet Toolbox."""

from typing import Literal, NamedTuple, Union

import torch
from typing_extensions import TypeAlias, Unpack

__all__ = [
    "BoundaryMode",
    "ExtendedBoundaryMode",
    "PaddingMode",
    "OrthogonalizeMethod",
    "WaveletDetailTuple2d",
    "WaveletCoeff2d",
    "WaveletCoeff2dSeparable",
    "WaveletCoeffNd",
    "WaveletDetailDict",
]

BoundaryMode = Literal["constant", "zero", "reflect", "periodic", "symmetric"]
"""
This is a type literal for the way of padding used at boundaries.

- Refection padding mirrors samples along the border (``reflect``)
- Zero padding pads zeros (``zero``)
- Constant padding replicates border values (``constant``)
- Periodic padding cyclically repeats samples (``periodic``)
- Symmetric padding mirrors samples along the border (``symmetric``)
"""

ExtendedBoundaryMode = Union[Literal["boundary"], BoundaryMode]
"""
This is a type literal for the way of handling signal boundaries.

This is either a form of padding (see :data:`ptwt.constants.BoundaryMode`
for padding options) or ``boundary`` to use boundary wavelets.
"""

PaddingMode = Literal["full", "valid", "same", "sameshift"]
"""
The padding mode is used when construction convolution matrices.
"""

OrthogonalizeMethod = Literal["qr", "gramschmidt"]
"""
The method for orthogonalizing a matrix.

1. ``qr`` relies on pytorch's dense QR implementation, it is fast but memory hungry.
2. ``gramschmidt`` option is sparse, memory efficient, and slow.

Choose ``gramschmidt`` if ``qr`` runs out of memory.
"""

PacketNodeOrder = Literal["freq", "natural"]
"""
This is a type literal for the order of wavelet packet tree nodes.

- frequency order (``freq``)
- natural order (``natural``)
"""


class WaveletDetailTuple2d(NamedTuple):
    """Detail coefficients of a 2d wavelet transform for a given level.

    This is a type alias for a named tuple ``(H, V, D)`` of detail coefficient tensors
    where ``H`` denotes horizontal, ``V`` vertical and ``D`` diagonal coefficients.
    """

    horizontal: torch.Tensor
    vertical: torch.Tensor
    diagonal: torch.Tensor


WaveletDetailDict: TypeAlias = dict[str, torch.Tensor]
"""Type alias for a dict containing detail coefficient for a given level.

This type alias represents the detail coefficient tensors of a given level for
a wavelet transform in :math:`N` dimensions as the values of a dictionary.
Its keys are a string of length :math:`N` describing the detail coefficient
by the applied filter for each axis. The string consists only of chars 'a' and 'd'
where 'a' denotes the low pass or approximation filter and 'd' the high-pass
or detail filter.
For a 3d transform, the dictionary thus uses the keys::

("aad", "ada", "add", "daa", "dad", "dda", "ddd")

Alias of ``dict[str, torch.Tensor]``
"""


# Note: This data structure was chosen to follow pywt's conventions
WaveletCoeff2d: TypeAlias = tuple[
    torch.Tensor, Unpack[tuple[WaveletDetailTuple2d, ...]]
]
"""Type alias for 2d wavelet transform results.

This type alias represents the result of a 2d wavelet transform
with :math:`n` levels as a tuple ``(A, Tn, ..., T1)`` of length :math:`n + 1`.
``A`` denotes a tensor of approximation coefficients for the `n`-th level
of decomposition. ``Tl`` is a tuple of detail coefficients for level ``l``,
see :data:`ptwt.constants.WaveletDetailTuple2d`.

Note that this type always contains an approximation coefficient tensor but does not
necesseraily contain any detail coefficients.

Alias of ``tuple[torch.Tensor, *tuple[WaveletDetailTuple2d, ...]]``
"""

# Note: This data structure was chosen to follow pywt's conventions
WaveletCoeffNd: TypeAlias = tuple[torch.Tensor, Unpack[tuple[WaveletDetailDict, ...]]]
"""Type alias for wavelet transform results in any dimension.

This type alias represents the result of a Nd wavelet transform
with :math:`n` levels as a tuple ``(A, Dn, ..., D1)`` of length :math:`n + 1`.
``A`` denotes a tensor of approximation coefficients for the `n`-th level
of decomposition. ``Dl`` is a dictionary of detail coefficients for level ``l``,
see :data:`ptwt.constants.WaveletDetailDict`.

Note that this type always contains an approximation coefficient tensor but does not
necesseraily contain any detail coefficients.

Alias of ``tuple[torch.Tensor, *tuple[WaveletDetailDict, ...]]``
"""

WaveletCoeff2dSeparable: TypeAlias = WaveletCoeffNd
"""Type alias for separable 2d wavelet transform results.

This is an alias of :data:`ptwt.constants.WaveletCoeffNd`.
It is used to emphasize the use of :data:`ptwt.constants.WaveletDetailDict`
for detail coefficients in a 2d setting -- in contrast to
:data:`ptwt.constants.WaveletCoeff2d`.

Alias of :data:`ptwt.constants.WaveletCoeffNd`, i.e. of
``tuple[torch.Tensor, *tuple[WaveletDetailDict, ...]]``.
"""
