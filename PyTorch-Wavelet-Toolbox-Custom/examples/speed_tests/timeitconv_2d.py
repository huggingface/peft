import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

import ptwt


def _to_jit_wavedec_2(data: torch.Tensor, wavelet) -> list[torch.Tensor]:
    """Ensure uniform datatypes in lists for the tracer.
    Going from list[Union[torch.Tensor, list[torch.Tensor]]] to list[torch.Tensor]
    means we have to stack the lists in the output.
    """
    assert data.shape == (32, 1e3, 1e3), "Changing the chape requires re-tracing."
    coeff = ptwt.wavedec2(data, wavelet, mode="periodic", level=5)
    coeff2 = []
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack(c))
    return coeff2


if __name__ == "__main__":
    repetitions = 100
    length = 1e3

    pywt_time_cpu = []

    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_gpu_jit = []

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        start = time.perf_counter()
        pywt_res = pywt.wavedec2(data, "db5", level=5, mode="periodic")
        end = time.perf_counter()
        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.wavedec2(data, "db5", mode="periodic", level=5)
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()

        start = time.perf_counter()
        res = ptwt.wavedec2(data, "db5", mode="periodic", level=5)
        torch.cuda.synchronize()
        end = time.perf_counter()

        ptwt_time_gpu.append(end - start)

    wavelet = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
        _to_jit_wavedec_2,
        (data.cuda(), wavelet),
        strict=False,
    )

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()

        pc_start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        torch.cuda.synchronize()
        pc_end = time.perf_counter()
        ptwt_time_gpu_jit.append(pc_end - pc_start)

    print("2d fwt results")
    print(
        f"2d-pywt-cpu    :{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}"
    )
    print(
        f"2d-ptwt-cpu    :{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}"
    )
    print(
        f"2d-ptwt-gpu    :{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}"
    )
    print(
        f"2d-ptwt-gpu-jit:{np.mean(ptwt_time_gpu_jit):5.5f} +- {np.std(ptwt_time_gpu_jit):5.5f}"
    )
    # plt.semilogy(pywt_time_cpu, label='pywt-cpu')
    # plt.semilogy(ptwt_time_cpu, label='ptwt-cpu')
    # plt.semilogy(ptwt_time_gpu, label='ptwt-gpu')
    # plt.semilogy(ptwt_time_gpu_jit, label='ptwt-jit')
    # plt.legend()
    # plt.xlabel('repetition')
    # plt.ylabel('runtime [s]')
    # plt.show()
    # plt.clf()

    time_stack = np.stack(
        [pywt_time_cpu, ptwt_time_cpu, ptwt_time_gpu, ptwt_time_gpu_jit], -1
    )
    plt.boxplot(time_stack)
    plt.yscale("log")
    plt.xticks([1, 2, 3, 4], ["pywt-cpu", "ptwt-cpu", "ptwt-gpu", "ptwt-gpu-jit"])
    plt.xticks(rotation=20)
    plt.ylabel("runtime [s]")
    plt.title("DWT-2D")
    plt.savefig("./figs/timeitconv2d.png")
