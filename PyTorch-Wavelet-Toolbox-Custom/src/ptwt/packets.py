"""Compute analysis wavelet packet representations."""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import product
from typing import TYPE_CHECKING, Literal, Optional, Union, overload

import numpy as np
import pywt
import torch

from ._util import (
    Wavelet,
    _as_wavelet,
    _deprecated_alias,
    _is_orthogonalize_method_supported,
    _swap_axes,
    _undo_swap_axes,
)
from .constants import (
    ExtendedBoundaryMode,
    OrthogonalizeMethod,
    PacketNodeOrder,
    WaveletCoeff2d,
    WaveletCoeffNd,
    WaveletDetailTuple2d,
)
from .conv_transform import wavedec, waverec
from .conv_transform_2 import wavedec2, waverec2
from .matmul_transform import MatrixWavedec, MatrixWaverec
from .matmul_transform_2 import MatrixWavedec2, MatrixWaverec2
from .separable_conv_transform import fswavedec2, fswaverec2

if TYPE_CHECKING:
    BaseDict = collections.UserDict[str, torch.Tensor]
else:
    BaseDict = collections.UserDict


def _wpfreq(fs: float, level: int) -> list[float]:
    """Compute the frequencies for a fully decomposed 1d packet tree.

       The packet transform linearly subdivides all frequencies
       from zero up to the Nyquist frequency.

    Args:
        fs (float): The sampling frequency.
        level (int): The decomposition level.

    Returns:
        The frequency bins of the packets in frequency order.
    """
    n = np.array(range(int(np.power(2.0, level))))
    freqs = (fs / 2.0) * (n / (np.power(2.0, level)))
    return list(freqs)


class WaveletPacket(BaseDict):
    """Implements a single-dimensional wavelet packets analysis transform."""

    @_deprecated_alias(boundary_orthogonalization="orthogonalization")
    def __init__(
        self,
        data: Optional[torch.Tensor],
        wavelet: Union[Wavelet, str],
        mode: ExtendedBoundaryMode = "reflect",
        maxlevel: Optional[int] = None,
        axis: int = -1,
        orthogonalization: OrthogonalizeMethod = "qr",
    ) -> None:
        """Create a wavelet packet decomposition object.

        The packet tree is initialized lazily, i.e. a coefficient is only
        calculated as it is retrieved. This allows for partial expansion
        of the wavelet packet tree.

        Args:
            data (torch.Tensor, optional): The input data array of shape ``[time]``,
                ``[batch_size, time]`` or ``[batch_size, channels, time]``.
                If None, the object is initialized without
                performing a decomposition.
                The time axis is transformed by default.
                Use the ``axis`` argument to choose another dimension.
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            mode : The desired padding method. If you select 'boundary',
                the sparse matrix backend will be used. Defaults to 'reflect'.
            maxlevel (int, optional): Value is passed on to `transform`.
                The highest decomposition level to compute. If None, the maximum level
                is determined from the input data shape. Defaults to None.
            axis (int): The axis to transform. Defaults to -1.
            orthogonalization: The orthogonalization method
                to use in the sparse matrix backend,
                see :data:`ptwt.constants.OrthogonalizeMethod`.
                Only used if `mode` equals 'boundary'. Defaults to 'qr'.

        .. versionchanged:: 1.10
            The argument `boundary_orthogonalization` has been renamed to
            `orthogonalization`.

        Raises:
            NotImplementedError: If the selected `orthogonalization` mode
                is not supported.

        Example:
            >>> import torch, pywt, ptwt
            >>> import numpy as np
            >>> import scipy.signal
            >>> import matplotlib.pyplot as plt
            >>> t = np.linspace(0, 10, 1500)
            >>> w = scipy.signal.chirp(t, f0=1, f1=50, t1=10, method="linear")
            >>> wp = ptwt.WaveletPacket(data=torch.from_numpy(w.astype(np.float32)),
            >>>     wavelet=pywt.Wavelet("db3"), mode="reflect")
            >>> np_lst = [wp[node] for node in wp.get_level(5)]
            >>> viz = np.stack(np_lst).squeeze()
            >>> plt.imshow(np.abs(viz))
            >>> plt.show()
        """
        self.wavelet = _as_wavelet(wavelet)
        self.mode = mode
        self.orthogonalization = orthogonalization
        self._matrix_wavedec_dict: dict[int, MatrixWavedec] = {}
        self._matrix_waverec_dict: dict[int, MatrixWaverec] = {}
        self.maxlevel: Optional[int] = None
        self.axis = axis

        self._filter_keys = {"a", "d"}

        if not _is_orthogonalize_method_supported(self.orthogonalization):
            raise NotImplementedError

        if data is not None:
            self.transform(data, maxlevel)
        else:
            self.data = {}

    def transform(
        self,
        data: torch.Tensor,
        maxlevel: Optional[int] = None,
    ) -> WaveletPacket:
        """Lazily calculate the 1d wavelet packet transform for the input data.

        The packet tree is initialized lazily, i.e. a coefficient is only
        calculated as it is retrieved. This allows for partial expansion
        of the wavelet packet tree.

        The transform function allows reusing the same object.

        Args:
            data (torch.Tensor): The input data array of shape ``[time]``
                or ``[batch_size, time]``.
            maxlevel (int, optional): The highest decomposition level to compute.
                If None, the maximum level is determined from the input data shape.
                Defaults to None.

        Returns:
            This wavelet packet object (to allow call chaining).
        """
        self.data = {"": data}
        if maxlevel is None:
            maxlevel = pywt.dwt_max_level(data.shape[self.axis], self.wavelet.dec_len)
        self.maxlevel = maxlevel
        return self

    def initialize(self, keys: Iterable[str]) -> None:
        """Initialize the wavelet packet tree partially.

        Args:
            keys (Iterable[str]): An iterable yielding the keys of the
                tree nodes to initialize.
        """
        it = (self[key] for key in keys)
        # exhaust iterator without storing all values
        collections.deque(it, maxlen=0)

    def reconstruct(self) -> WaveletPacket:
        """Recursively reconstruct the input starting from the leaf nodes.

        Reconstruction replaces the input data originally assigned to this object.

        Note:
           Only changes to leaf node data impact the results,
           since changes in all other nodes will be replaced with
           a reconstruction from the leaves.

        Example:
            >>> import numpy as np
            >>> import ptwt, torch
            >>> signal = np.random.randn(1, 16)
            >>> ptwp = ptwt.WaveletPacket(torch.from_numpy(signal), "haar",
            >>>     mode="boundary", maxlevel=2)
            >>> # initialize other leaf nodes
            >>> ptwp.initialize(["ad", "da", "dd"])
            >>> ptwp["aa"] = torch.zeros_like(ptwp["ad"])
            >>> ptwp.reconstruct()
            >>> print(ptwp[""])

        Raises:
            KeyError: if any leaf node data is not present.
        """
        if self.maxlevel is None:
            self.maxlevel = pywt.dwt_max_level(self[""].shape[-1], self.wavelet.dec_len)

        for level in reversed(range(self.maxlevel)):
            for node in self.get_level(level):
                # check if any children is not available
                # we need to check manually to avoid lazy init
                for child in self._filter_keys:
                    if node + child not in self:
                        raise KeyError(f"Key {node + child} not found")

                data_a = self[node + "a"]
                data_d = self[node + "d"]
                rec = self._get_waverec(data_a.shape[self.axis])([data_a, data_d])
                if level > 0:
                    if rec.shape[self.axis] != self[node].shape[self.axis]:
                        assert (
                            rec.shape[self.axis] == self[node].shape[self.axis] + 1
                        ), "padding error, please open an issue on github"
                        rec = rec.swapaxes(self.axis, -1)[..., :-1].swapaxes(
                            -1, self.axis
                        )
                self[node] = rec
        return self

    def _get_wavedec(
        self,
        length: int,
    ) -> Callable[[torch.Tensor], list[torch.Tensor]]:
        if self.mode == "boundary":
            if length not in self._matrix_wavedec_dict.keys():
                self._matrix_wavedec_dict[length] = MatrixWavedec(
                    self.wavelet,
                    level=1,
                    orthogonalization=self.orthogonalization,
                    axis=self.axis,
                )
            return self._matrix_wavedec_dict[length]
        else:
            return partial(
                wavedec, wavelet=self.wavelet, level=1, mode=self.mode, axis=self.axis
            )

    def _get_waverec(
        self,
        length: int,
    ) -> Callable[[Sequence[torch.Tensor]], torch.Tensor]:
        if self.mode == "boundary":
            if length not in self._matrix_waverec_dict.keys():
                self._matrix_waverec_dict[length] = MatrixWaverec(
                    self.wavelet,
                    orthogonalization=self.orthogonalization,
                    axis=self.axis,
                )
            return self._matrix_waverec_dict[length]
        else:
            return partial(waverec, wavelet=self.wavelet, axis=self.axis)

    @staticmethod
    def get_level(level: int, order: PacketNodeOrder = "freq") -> list[str]:
        """Return the paths to the filter tree nodes.

        Args:
            level (int): The depth of the tree.
            order: The order the paths are in.
                Choose from frequency order (``freq``) and
                natural order (``natural``).
                Defaults to ``freq``.

        Returns:
            A list with the paths to each node.

        Raises:
            ValueError: If `order` is neither ``freq`` nor ``natural``.
        """
        if order == "freq":
            return WaveletPacket._get_graycode_order(level)
        elif order == "natural":
            return ["".join(p) for p in product(["a", "d"], repeat=level)]
        else:
            raise ValueError(
                f"Unsupported order '{order}'. Choose from 'freq' and 'natural'."
            )

    @staticmethod
    def _get_graycode_order(level: int, x: str = "a", y: str = "d") -> list[str]:
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        if level == 0:
            return [""]
        else:
            return graycode_order

    def _expand_node(self, path: str) -> None:
        data = self[path]
        res_lo, res_hi = self._get_wavedec(data.shape[self.axis])(data)
        self.data[path + "a"] = res_lo
        self.data[path + "d"] = res_hi

    def __getitem__(self, key: str) -> torch.Tensor:
        """Access the coefficients in the wavelet packets tree.

        Args:
            key (str): The key of the accessed coefficients. The string may only consist
                of the chars 'a' and 'd' where 'a' denotes the low pass or
                approximation filter and 'd' the high-pass or detail filter.

        Returns:
            The accessed wavelet packet coefficients.

        Raises:
            ValueError: If the wavelet packet tree is not initialized.
            KeyError: If no wavelet coefficients are indexed by the specified key
                and a lazy initialization fails.
        """
        if self.maxlevel is None:
            raise ValueError(
                "The wavelet packet tree must be initialized via 'transform' before "
                "its values can be accessed!"
            )
        if key not in self and len(key) > self.maxlevel:
            raise KeyError(
                f"The requested level {len(key)} with key '{key}' is too large and "
                "cannot be accessed! This wavelet packet tree is initialized with "
                f"maximum level {self.maxlevel}."
            )
        elif key not in self:
            if key == "":
                raise ValueError(
                    "The requested root of the packet tree cannot be accessed! "
                    "The wavelet packet tree is not properly initialized. "
                    "Run `transform` before accessing tree values."
                )
            elif key[-1] not in self._filter_keys:
                raise ValueError(
                    f"Invalid key '{key}'. All chars in the key must be of the "
                    f"set {self._filter_keys}."
                )
            # calculate data from parent
            self._expand_node(key[:-1])
        return super().__getitem__(key)


class WaveletPacket2D(BaseDict):
    """Two-dimensional wavelet packets.

    Example code illustrating the use of this class is available at:
    https://github.com/v0lta/PyTorch-Wavelet-Toolbox/tree/main/examples/deepfake_analysis
    """

    @_deprecated_alias(boundary_orthogonalization="orthogonalization")
    def __init__(
        self,
        data: Optional[torch.Tensor],
        wavelet: Union[Wavelet, str],
        mode: ExtendedBoundaryMode = "reflect",
        maxlevel: Optional[int] = None,
        axes: tuple[int, int] = (-2, -1),
        orthogonalization: OrthogonalizeMethod = "qr",
        separable: bool = False,
    ) -> None:
        """Create a 2D-Wavelet packet tree.

        The packet tree is initialized lazily, i.e. a coefficient is only
        calculated as it is retrieved. This allows for partial expansion
        of the wavelet packet tree.

        Args:
            data (torch.tensor, optional): The input data tensor.
                For example of shape ``[batch_size, height, width]`` or
                ``[batch_size, channels, height, width]``.
                If None, the object is initialized without performing
                a decomposition.
            wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
                Refer to the output from ``pywt.wavelist(kind='discrete')``
                for possible choices.
            mode : A string indicating the desired padding mode.
                If you select 'boundary', the sparse matrix backend is used.
                Defaults to 'reflect'
            maxlevel (int, optional): Value is passed on to `transform`.
                The highest decomposition level to compute. If None, the maximum level
                is determined from the input data shape. Defaults to None.
            axes ([int, int], optional): The tensor axes that should be transformed.
                Defaults to (-2, -1).
            orthogonalization : The orthogonalization method
                to use in the sparse matrix backend,
                see :data:`ptwt.constants.OrthogonalizeMethod`.
                Only used if `mode` equals 'boundary'. Defaults to 'qr'.
            separable (bool): If true, a separable transform is performed,
                i.e. each image axis is transformed separately. Defaults to False.

        .. versionchanged:: 1.10
            The argument `boundary_orthogonalization` has been renamed to
            `orthogonalization`.

        Raises:
            NotImplementedError: If the selected `orthogonalization` mode
                is not supported.
        """
        self.wavelet = _as_wavelet(wavelet)
        self.mode = mode
        self.orthogonalization = orthogonalization
        self.separable = separable
        self.matrix_wavedec2_dict: dict[tuple[int, ...], MatrixWavedec2] = {}
        self.matrix_waverec2_dict: dict[tuple[int, ...], MatrixWaverec2] = {}
        self.axes = axes
        self._filter_keys = {"a", "h", "v", "d"}

        if not _is_orthogonalize_method_supported(self.orthogonalization):
            raise NotImplementedError

        self.maxlevel: Optional[int] = None
        if data is not None:
            self.transform(data, maxlevel)
        else:
            self.data = {}

    def transform(
        self,
        data: torch.Tensor,
        maxlevel: Optional[int] = None,
    ) -> WaveletPacket2D:
        """Lazily calculate the 2d wavelet packet transform for the input data.

        The packet tree is initialized lazily, i.e. a coefficient is only
        calculated as it is retrieved. This allows for partial expansion
        of the wavelet packet tree.

        The transform function allows reusing the same object.

        Args:
            data (torch.tensor): The input data tensor
                of shape ``[batch_size, height, width]``.
            maxlevel (int, optional): The highest decomposition level to compute.
                If None, the maximum level is determined from the input data shape.
                Defaults to None.

        Returns:
            This wavelet packet object (to allow call chaining).
        """
        self.data = {"": data}
        if maxlevel is None:
            min_transform_size = min(_swap_axes(data, self.axes).shape[-2:])
            maxlevel = pywt.dwt_max_level(min_transform_size, self.wavelet.dec_len)
        self.maxlevel = maxlevel

        return self

    def initialize(self, keys: Iterable[str]) -> None:
        """Initialize the wavelet packet tree partially.

        Args:
            keys (Iterable[str]): An iterable yielding the keys of the
                tree nodes to initialize.
        """
        it = (self[key] for key in keys)
        # exhaust iterator without storing all values
        collections.deque(it, maxlen=0)

    def reconstruct(self) -> WaveletPacket2D:
        """Recursively reconstruct the input starting from the leaf nodes.

        Note:
           Only changes to leaf node data impact the results,
           since changes in all other nodes will be replaced with
           a reconstruction from the leaves.

        Raises:
            KeyError: if any leaf node data is not present.
        """
        if self.maxlevel is None:
            min_transform_size = min(_swap_axes(self[""], self.axes).shape[-2:])
            self.maxlevel = pywt.dwt_max_level(min_transform_size, self.wavelet.dec_len)

        for level in reversed(range(self.maxlevel)):
            for node in WaveletPacket2D.get_natural_order(level):
                # check if any children is not available
                # we need to check manually to avoid lazy init
                for child in self._filter_keys:
                    if node + child not in self:
                        raise KeyError(f"Key {node + child} not found")

                data_a = self[node + "a"]
                data_h = self[node + "h"]
                data_v = self[node + "v"]
                data_d = self[node + "d"]
                transform_size = _swap_axes(data_a, self.axes).shape[-2:]
                rec = self._get_waverec(transform_size)(
                    (data_a, WaveletDetailTuple2d(data_h, data_v, data_d))
                )
                if level > 0:
                    rec = _swap_axes(rec, self.axes)
                    swapped_node = _swap_axes(self[node], self.axes)
                    if rec.shape[-1] != swapped_node.shape[-1]:
                        assert (
                            rec.shape[-1] == swapped_node.shape[-1] + 1
                        ), "padding error, please open an issue on GitHub"
                        rec = rec[..., :-1]
                    if rec.shape[-2] != swapped_node.shape[-2]:
                        assert (
                            rec.shape[-2] == swapped_node.shape[-2] + 1
                        ), "padding error, please open an issue on GitHub"
                        rec = rec[..., :-1, :]
                    rec = _undo_swap_axes(rec, self.axes)
                self[node] = rec
        return self

    def _expand_node(self, path: str) -> None:
        data = self[path]
        transform_size = _swap_axes(data, self.axes).shape[-2:]
        result = self._get_wavedec(transform_size)(data)

        # assert for type checking
        assert len(result) == 2
        result_a, (result_h, result_v, result_d) = result
        self.data[path + "a"] = result_a
        self.data[path + "h"] = result_h
        self.data[path + "v"] = result_v
        self.data[path + "d"] = result_d

    def _get_wavedec(self, shape: tuple[int, ...]) -> Callable[
        [torch.Tensor],
        WaveletCoeff2d,
    ]:
        if self.mode == "boundary":
            shape = tuple(shape)
            if shape not in self.matrix_wavedec2_dict.keys():
                self.matrix_wavedec2_dict[shape] = MatrixWavedec2(
                    self.wavelet,
                    level=1,
                    axes=self.axes,
                    orthogonalization=self.orthogonalization,
                    separable=self.separable,
                )
            fun = self.matrix_wavedec2_dict[shape]
            return fun
        elif self.separable:
            return self._transform_fsdict_to_tuple_func(
                partial(
                    fswavedec2,
                    wavelet=self.wavelet,
                    level=1,
                    mode=self.mode,
                    axes=self.axes,
                )
            )
        else:
            return partial(
                wavedec2, wavelet=self.wavelet, level=1, mode=self.mode, axes=self.axes
            )

    def _get_waverec(
        self, shape: tuple[int, ...]
    ) -> Callable[[WaveletCoeff2d], torch.Tensor]:
        if self.mode == "boundary":
            shape = tuple(shape)
            if shape not in self.matrix_waverec2_dict.keys():
                self.matrix_waverec2_dict[shape] = MatrixWaverec2(
                    self.wavelet,
                    axes=self.axes,
                    orthogonalization=self.orthogonalization,
                    separable=self.separable,
                )
            return self.matrix_waverec2_dict[shape]
        elif self.separable:
            return self._transform_tuple_to_fsdict_func(
                partial(fswaverec2, wavelet=self.wavelet, axes=self.axes)
            )
        else:
            return partial(waverec2, wavelet=self.wavelet, axes=self.axes)

    def _transform_fsdict_to_tuple_func(
        self,
        fs_dict_func: Callable[[torch.Tensor], WaveletCoeffNd],
    ) -> Callable[[torch.Tensor], WaveletCoeff2d]:
        def _tuple_func(
            data: torch.Tensor,
        ) -> WaveletCoeff2d:
            fs_dict_data = fs_dict_func(data)
            # assert for type checking
            assert len(fs_dict_data) == 2
            a_coeff, fsdict = fs_dict_data
            return (
                a_coeff,
                WaveletDetailTuple2d(fsdict["ad"], fsdict["da"], fsdict["dd"]),
            )

        return _tuple_func

    def _transform_tuple_to_fsdict_func(
        self,
        fsdict_func: Callable[[WaveletCoeffNd], torch.Tensor],
    ) -> Callable[[WaveletCoeff2d], torch.Tensor]:
        def _fsdict_func(coeffs: WaveletCoeff2d) -> torch.Tensor:
            # assert for type checking
            assert len(coeffs) == 2
            a, (h, v, d) = coeffs
            return fsdict_func((a, {"ad": h, "da": v, "dd": d}))

        return _fsdict_func

    def __getitem__(self, key: str) -> torch.Tensor:
        """Access the coefficients in the wavelet packets tree.

        Args:
            key (str): The key of the accessed coefficients.
                The string may only consist
                of the following chars: 'a', 'h', 'v', 'd'
                The chars correspond to the selected coefficients for a level
                where 'a' denotes the approximation coefficients and
                'h' horizontal, 'v' vertical  and 'd' diagonal details coefficients.

        Returns:
            The accessed wavelet packet coefficients.

        Raises:
            ValueError: If the wavelet packet tree is not initialized.
            KeyError: If no wavelet coefficients are indexed by the specified key
                and a lazy initialization fails.
        """
        if self.maxlevel is None:
            raise ValueError(
                "The wavelet packet tree must be initialized via 'transform' before "
                "its values can be accessed!"
            )
        if key not in self and len(key) > self.maxlevel:
            raise KeyError(
                f"The requested level {len(key)} with key '{key}' is too large and "
                "cannot be accessed! This wavelet packet tree is initialized with "
                f"maximum level {self.maxlevel}."
            )
        elif key not in self:
            if key == "":
                raise ValueError(
                    "The requested root of the packet tree cannot be accessed! "
                    "The wavelet packet tree is not properly initialized. "
                    "Run `transform` before accessing tree values."
                )
            elif key[-1] not in self._filter_keys:
                raise ValueError(
                    f"Invalid key '{key}'. All chars in the key must be of the "
                    f"set {self._filter_keys}."
                )
            # calculate data from parent
            self._expand_node(key[:-1])

        return super().__getitem__(key)

    @overload
    @staticmethod
    def get_level(level: int, order: Literal["freq"]) -> list[list[str]]: ...

    @overload
    @staticmethod
    def get_level(level: int, order: Literal["natural"]) -> list[str]: ...

    @staticmethod
    def get_level(
        level: int, order: PacketNodeOrder = "freq"
    ) -> Union[list[str], list[list[str]]]:
        """Return the paths to the filter tree nodes.

        Args:
            level (int): The depth of the tree.
            order: The order the paths are in.
                Choose from frequency order (``freq``) and
                natural order (``natural``).
                Defaults to ``freq``.

        Returns:
            A list with the paths to each node.

        Raises:
            ValueError: If `order` is neither ``freq`` nor ``natural``.
        """
        if order == "freq":
            return WaveletPacket2D.get_freq_order(level)
        elif order == "natural":
            return WaveletPacket2D.get_natural_order(level)
        else:
            raise ValueError(
                f"Unsupported order '{order}'. Choose from 'freq' and 'natural'."
            )

    @staticmethod
    def get_natural_order(level: int) -> list[str]:
        """Get the natural ordering for a given decomposition level.

        Args:
            level (int): The decomposition level.

        Returns:
            A list with the filter order strings.
        """
        return ["".join(p) for p in product(["a", "h", "v", "d"], repeat=level)]

    @staticmethod
    def get_freq_order(level: int) -> list[list[str]]:
        """Get the frequency order for a given packet decomposition level.

        Use this code to create two-dimensional frequency orderings.

        Args:
            level (int): The number of decomposition scales.

        Returns:
            A list with the tree nodes in frequency order.

        Note:
            Adapted from:
            https://github.com/PyWavelets/pywt/blob/master/pywt/_wavelet_packets.py

            The code elements denote the filter application order. The filters
            are named following the pywt convention as:
            a - LL, low-low coefficients
            h - LH, low-high coefficients
            v - HL, high-low coefficients
            d - HH, high-high coefficients
        """
        wp_natural_path = product(["a", "h", "v", "d"], repeat=level)

        def _get_graycode_order(level: int, x: str = "a", y: str = "d") -> list[str]:
            graycode_order = [x, y]
            for _ in range(level - 1):
                graycode_order = [x + path for path in graycode_order] + [
                    y + path for path in graycode_order[::-1]
                ]
            return graycode_order

        def _expand_2d_path(path: tuple[str, ...]) -> tuple[str, str]:
            expanded_paths = {"d": "hh", "h": "hl", "v": "lh", "a": "ll"}
            return (
                "".join([expanded_paths[p][0] for p in path]),
                "".join([expanded_paths[p][1] for p in path]),
            )

        nodes_dict: dict[str, dict[str, tuple[str, ...]]] = {}
        for (row_path, col_path), node in [
            (_expand_2d_path(node), node) for node in wp_natural_path
        ]:
            nodes_dict.setdefault(row_path, {})[col_path] = node
        graycode_order = _get_graycode_order(level, x="l", y="h")
        nodes = [nodes_dict[path] for path in graycode_order if path in nodes_dict]
        result = []
        for row in nodes:
            result.append(
                ["".join(row[path]) for path in graycode_order if path in row]
            )
        return result
