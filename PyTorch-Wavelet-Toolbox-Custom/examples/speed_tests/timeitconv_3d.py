import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

import ptwt


def _to_jit_wavedec_3(data, wavelet):
    """Ensure uniform datatypes in lists for the tracer.

    Going from List[Union[torch.Tensor, Dict[str, torch.Tensor]]] to List[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (32, 100, 100, 100), "Changing the chape requires re-tracing."
    coeff = ptwt.wavedec3(data, wavelet, mode="reflect", level=3)
    coeff2 = []
    keys = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack([c[key] for key in keys]))
    return coeff2


if __name__ == "__main__":
    length = 1e2
    repetitions = 100

    pywt_time_cpu = []
    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_gpu_jit = []

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length), int(length)).astype(
            np.float32
        )
        start = time.perf_counter()
        res = pywt.wavedecn(data, "db5", level=3, axes=range(1, 3), mode="periodic")
        end = time.perf_counter()

        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length), int(length)).astype(
            np.float32
        )
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.wavedec3(data, "db5", level=3, mode="periodic")
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length), int(length)).astype(
            np.float32
        )
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = ptwt.wavedec3(data, "db5", level=3, mode="periodic")
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

    wavelet = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
        _to_jit_wavedec_3,
        (data.cuda(), wavelet),
        strict=False,
    )
    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length), int(length)).astype(
            np.float32
        )
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu_jit.append(end - start)

    print("3d fwt results")
    print(
        f"3d-pywt-cpu    :{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}"
    )
    print(
        f"3d-ptwt-cpu    :{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}"
    )
    print(
        f"3d-ptwt-gpu    :{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}"
    )
    print(
        f"3d-ptwt-gpu-jit:{np.mean(ptwt_time_gpu_jit):5.5f} +- {np.std(ptwt_time_gpu_jit):5.5f}"
    )

    time_stack = np.stack(
        [pywt_time_cpu, ptwt_time_cpu, ptwt_time_gpu, ptwt_time_gpu_jit], -1
    )
    plt.boxplot(time_stack)
    plt.yscale("log")
    plt.xticks([1, 2, 3, 4], ["pywt-cpu", "ptwt-cpu", "ptwt-gpu", "ptwt-gpu-jit"])
    plt.xticks(rotation=20)
    plt.ylabel("runtime [s]")
    plt.title("DWT-3D")
    plt.savefig("./figs/timeitconv3d.png")
