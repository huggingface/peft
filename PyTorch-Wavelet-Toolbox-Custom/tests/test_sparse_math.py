"""Test the sparse math code from ptwt.sparse_math."""

import numpy as np
import pytest
import scipy.signal
import torch
from scipy import datasets

from ptwt.constants import PaddingMode
from ptwt.sparse_math import (
    batch_mm,
    construct_conv2d_matrix,
    construct_conv_matrix,
    construct_strided_conv2d_matrix,
    construct_strided_conv_matrix,
    sparse_kron,
)

# Written by moritz ( @ wolter.tech ) in 2021


def test_kron() -> None:
    """Test the implementation by evaluation.

    The example is taken from
    https://de.wikipedia.org/wiki/Kronecker-Produkt
    """
    a = torch.tensor([[1, 2], [3, 2], [5, 6]]).to_sparse()
    b = torch.tensor([[7, 8], [9, 0]]).to_sparse()
    sparse_result = sparse_kron(a, b)
    dense_result = torch.kron(a.to_dense(), b.to_dense())
    err = torch.sum(torch.abs(sparse_result.to_dense() - dense_result))
    condition = np.allclose(sparse_result.to_dense().numpy(), dense_result.numpy())
    print("error {:2.2f}".format(err), condition)
    assert condition


@pytest.mark.parametrize(
    "test_filter", [torch.rand([2]), torch.rand([3]), torch.rand([4])]
)
@pytest.mark.parametrize("input_signal", [torch.rand([8]), torch.rand([9])])
@pytest.mark.parametrize("padding", ["full", "same", "valid"])
def test_conv_matrix(
    test_filter: torch.Tensor, input_signal: torch.Tensor, padding: PaddingMode
) -> None:
    """Test the 1d sparse convolution matrix code."""
    conv_matrix = construct_conv_matrix(test_filter, len(input_signal), mode=padding)
    mm_conv_res = torch.sparse.mm(conv_matrix, input_signal.unsqueeze(-1)).squeeze()
    conv_res = scipy.signal.convolve(input_signal.numpy(), test_filter.numpy(), padding)
    error = np.sum(np.abs(conv_res - mm_conv_res.numpy()))
    print("1d conv matrix error", padding, error, len(test_filter), len(input_signal))
    assert np.allclose(conv_res, mm_conv_res.numpy())


@pytest.mark.parametrize(
    "test_filter",
    [
        torch.tensor([1.0, 0]),
        torch.rand([2]),
        torch.rand([3]),
        torch.rand([4]),
    ],
)
@pytest.mark.parametrize(
    "input_signal",
    [
        torch.tensor([0.0, 1, 2, 3, 4, 5, 6, 7]),
        torch.rand([8]),
        torch.rand([9]),
    ],
)
@pytest.mark.parametrize("mode", ["valid", "same"])
def test_strided_conv_matrix(
    test_filter: torch.Tensor, input_signal: torch.Tensor, mode: PaddingMode
) -> None:
    """Test the strided 1d sparse convolution matrix code."""
    strided_conv_matrix = construct_strided_conv_matrix(
        test_filter, len(input_signal), 2, mode=mode
    )
    mm_conv_res = torch.sparse.mm(
        strided_conv_matrix, input_signal.unsqueeze(-1)
    ).squeeze()
    if mode == "same":
        height_offset = len(input_signal) % 2
        padding = (len(test_filter) // 2, len(test_filter) // 2 - 1 + height_offset)
        input_signal = torch.nn.functional.pad(input_signal, padding)

    torch_conv_res = torch.nn.functional.conv1d(
        input_signal.unsqueeze(0).unsqueeze(0),
        test_filter.flip(0).unsqueeze(0).unsqueeze(0),
        stride=2,
    ).squeeze()
    error = torch.sum(torch.abs(mm_conv_res - torch_conv_res))
    print(
        "filter shape {:2}".format(tuple(test_filter.shape)[0]),
        "signal shape {:2}".format(tuple(input_signal.shape)[0]),
        "error {:2.2e}".format(error.item()),
    )
    assert np.allclose(mm_conv_res.numpy(), torch_conv_res.numpy())


@pytest.mark.parametrize(
    "filter_shape",
    [
        (2, 2),
        (3, 3),
        (3, 2),
        (2, 3),
        (5, 3),
        (3, 5),
        (2, 5),
        (5, 2),
        (4, 4),
    ],
)
@pytest.mark.parametrize(
    "size",
    [
        (5, 5),
        (10, 10),
        (16, 16),
        (8, 16),
        (16, 8),
        (16, 7),
        (7, 16),
        (15, 15),
    ],
)
@pytest.mark.parametrize("mode", ["same", "full", "valid"])
@pytest.mark.parametrize("fully_sparse", [True, False])
def test_conv_matrix_2d(
    filter_shape: tuple[int, int],
    size: tuple[int, int],
    mode: PaddingMode,
    fully_sparse: bool,
) -> None:
    """Test the validity of the 2d convolution matrix code.

    It should be equivalent to signal convolve2d.
    """
    test_filter = torch.rand(filter_shape)
    test_filter = test_filter.unsqueeze(0).unsqueeze(0)
    face = datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])]
    face = np.mean(face, -1).astype(np.float32)
    res_scipy = scipy.signal.convolve2d(face, test_filter.squeeze().numpy(), mode=mode)

    face = torch.from_numpy(face)
    face = face.unsqueeze(0).unsqueeze(0)
    conv_matrix2d = construct_conv2d_matrix(
        test_filter.squeeze(), size[0], size[1], mode=mode, fully_sparse=fully_sparse
    )
    res_flat = torch.sparse.mm(
        conv_matrix2d, face.transpose(-2, -1).flatten().unsqueeze(-1)
    )
    res_mm = torch.reshape(res_flat, [res_scipy.shape[1], res_scipy.shape[0]]).T
    assert np.allclose(res_scipy, res_mm)


@pytest.mark.slow
@pytest.mark.parametrize("filter_shape", [(3, 3), (2, 2), (4, 4), (3, 2), (2, 3)])
@pytest.mark.parametrize(
    "size",
    [
        (14, 14),
        (8, 16),
        (16, 8),
        (17, 8),
        (8, 17),
        (7, 7),
        (7, 8),
        (8, 7),
    ],
)
@pytest.mark.parametrize("mode", ["full", "valid"])
def test_strided_conv_matrix_2d(
    filter_shape: tuple[int, int], size: tuple[int, int], mode: PaddingMode
) -> None:
    """Test strided convolution matrices with full and valid padding."""
    test_filter = torch.rand(filter_shape)
    test_filter = test_filter.unsqueeze(0).unsqueeze(0)
    face = datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])]
    face = np.mean(face, -1)
    face = torch.from_numpy(face.astype(np.float32))
    face = face.unsqueeze(0).unsqueeze(0)

    if mode == "full":
        padding = (filter_shape[0] - 1, filter_shape[1] - 1)
    elif mode == "valid":
        padding = (0, 0)
    torch_res = torch.nn.functional.conv2d(
        face, test_filter.flip(2, 3), padding=padding, stride=2
    ).squeeze()

    strided_matrix = construct_strided_conv2d_matrix(
        test_filter.squeeze(), size[0], size[1], stride=2, mode=mode
    )
    res_flat_stride = torch.sparse.mm(
        strided_matrix, face.transpose(-2, -1).flatten().unsqueeze(-1)
    )

    if mode == "full":
        output_shape = [
            int(np.ceil((filter_shape[1] + size[1] - 1) / 2)),
            int(np.ceil((filter_shape[0] + size[0] - 1) / 2)),
        ]
    elif mode == "valid":
        output_shape = [
            (size[1] - (filter_shape[1])) // 2 + 1,
            (size[0] - (filter_shape[0])) // 2 + 1,
        ]
    res_mm_stride = torch.reshape(res_flat_stride, output_shape).T
    assert np.allclose(torch_res.numpy(), res_mm_stride.numpy())


@pytest.mark.parametrize("filter_shape", [(3, 3), (4, 4), (4, 3), (3, 4)])
@pytest.mark.parametrize(
    "size", [(7, 8), (8, 7), (7, 7), (8, 8), (16, 16), (8, 16), (16, 8)]
)
def test_strided_conv_matrix_2d_same(
    filter_shape: tuple[int, int], size: tuple[int, int]
) -> None:
    """Test strided conv matrix with same padding."""
    stride = 2
    test_filter = torch.rand(filter_shape)
    test_filter = test_filter.unsqueeze(0).unsqueeze(0)
    face = datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])]
    face = np.mean(face, -1)
    face = torch.from_numpy(face.astype(np.float32))
    face = face.unsqueeze(0).unsqueeze(0)
    padding = _get_2d_same_padding(filter_shape, size)
    face_pad = torch.nn.functional.pad(face, padding)
    torch_res = torch.nn.functional.conv2d(
        face_pad, test_filter.flip(2, 3), stride=stride
    ).squeeze()
    strided_matrix = construct_strided_conv2d_matrix(
        test_filter.squeeze(),
        face.shape[-2],
        face.shape[-1],
        stride=stride,
        mode="same",
    )
    res_flat_stride = torch.sparse.mm(
        strided_matrix, face.transpose(-2, -1).flatten().unsqueeze(-1)
    )
    output_shape = torch_res.shape
    res_mm_stride = torch.reshape(res_flat_stride, (output_shape[1], output_shape[0])).T
    assert np.allclose(torch_res.numpy(), res_mm_stride.numpy())


def _get_2d_same_padding(
    filter_shape: tuple[int, int], input_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    height_offset = input_size[0] % 2
    width_offset = input_size[1] % 2
    padding = (
        filter_shape[1] // 2,
        filter_shape[1] // 2 - 1 + width_offset,
        filter_shape[0] // 2,
        filter_shape[0] // 2 - 1 + height_offset,
    )
    return padding


@pytest.mark.slow
@pytest.mark.parametrize("size", [(256, 512), (512, 256)])
def test_strided_conv_matrix_2d_sameshift(size: tuple[int, int]) -> None:
    """Test strided conv matrix with sameshift padding."""
    stride = 2
    filter_shape = (3, 3)
    # test_filter = torch.rand(filter_shape)
    test_filter = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [
                0.0,
                1.0,
                0.1,
            ],
            [0.0, 0.1, 0.1],
        ]
    )
    test_filter2 = torch.tensor(
        [
            [1.0, 0.1, 0.0],
            [
                0.1,
                0.1,
                0.0,
            ],
            [0.0, 0.0, 0.0],
        ]
    )
    test_filter = test_filter.unsqueeze(0).unsqueeze(0)
    test_filter2 = test_filter2.unsqueeze(0).unsqueeze(0)
    face = datasets.face()[256 : (256 + size[0]), 256 : (256 + size[1])]
    face = np.mean(face, -1)
    face = torch.from_numpy(face.astype(np.float32))
    face = face.unsqueeze(0).unsqueeze(0)
    padding = _get_2d_same_padding(filter_shape, size)
    face_pad = torch.nn.functional.pad(face, padding)
    torch_res = torch.nn.functional.conv2d(
        face_pad, test_filter2.flip(2, 3), stride=stride
    ).squeeze()
    strided_matrix = construct_strided_conv2d_matrix(
        filter=test_filter.squeeze(),
        input_rows=face.shape[-2],
        input_columns=face.shape[-1],
        stride=stride,
        mode="sameshift",
    )
    coefficients = torch.sparse.mm(
        strided_matrix, face.transpose(-2, -1).flatten().unsqueeze(-1)
    )
    output_shape = torch_res.shape
    res_mm_stride = torch.reshape(coefficients, (output_shape[1], output_shape[0])).T
    assert np.allclose(torch_res.numpy(), res_mm_stride.numpy())


def test_mode_error_2d() -> None:
    """Test the invalid padding-error."""
    test_filter = torch.rand([3, 3])
    with pytest.raises(ValueError):
        _ = construct_conv2d_matrix(test_filter, 32, 32, mode="invalid_mode")
    with pytest.raises(ValueError):
        _ = construct_strided_conv2d_matrix(test_filter, 32, 32, mode="invalid_mode")


def test_mode_error() -> None:
    """Test the invalid padding-error."""
    test_filter = torch.rand([3, 3])
    with pytest.raises(ValueError):
        _ = construct_conv_matrix(test_filter, 32, mode="invalid_mode")
    with pytest.raises(ValueError):
        _ = construct_strided_conv_matrix(test_filter, 32, mode="invalid_mode")


def test_shape_error() -> None:
    """Check the shape error in the batch_mm function."""
    with pytest.raises(ValueError):
        _ = batch_mm(torch.rand(8, 8), torch.rand(1, 6, 8))
