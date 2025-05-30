import time
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

import ptwt


def _jit_wavedec_fun(data, wavelet):
    return ptwt.wavedec(data, wavelet, mode="periodic", level=10)


if __name__ == "__main__":
    length = 1e6
    repetitions = 100

    pywt_time_cpu = []
    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_cpu_jit = []
    ptwt_time_gpu_jit = []

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        start = time.perf_counter()
        res = pywt.wavedec(data, "db5", level=10, mode="periodic")
        end = time.perf_counter()
        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.wavedec(data, "db5", level=10, mode="periodic")
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    wavelet = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
        _jit_wavedec_fun,
        (data, wavelet),
        strict=False,
    )

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        end = time.perf_counter()
        ptwt_time_cpu_jit.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()

        start = time.perf_counter()
        res = ptwt.wavedec(data, "db5", level=10, mode="periodic")
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

    wavelet = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet("db5"), torch.float32)
    jit_wavedec = torch.jit.trace(
        _jit_wavedec_fun,
        (data.cuda(), wavelet),
        strict=False,
    )

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()

        start = time.perf_counter()
        res = jit_wavedec(data, wavelet)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu_jit.append(end - start)

    print("1d fwt results")
    print(
        f"1d-pywt-cpu    :{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}"
    )
    print(
        f"1d-ptwt-cpu    :{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}"
    )
    print(
        f"1d-ptwt-cpu-jit:{np.mean(ptwt_time_cpu_jit):5.5f} +- {np.std(ptwt_time_cpu_jit):5.5f}"
    )
    print(
        f"1d-ptwt-gpu    :{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}"
    )
    print(
        f"1d-ptwt-gpu-jit:{np.mean(ptwt_time_gpu_jit):5.5f} +- {np.std(ptwt_time_gpu_jit):5.5f}"
    )
    # plt.semilogy(pywt_time_cpu, label='pywt-cpu')
    # plt.semilogy(ptwt_time_cpu, label='ptwt-cpu')
    # plt.semilogy(ptwt_time_cpu_jit, label='ptwt-cpu-jit')
    # plt.semilogy(ptwt_time_gpu, label='ptwt-gpu')
    # plt.semilogy(ptwt_time_gpu_jit, label='ptwt-gpu-jit')
    # plt.legend()
    # plt.xlabel('repetition')
    # plt.ylabel('runtime [s]')
    # plt.show()
    time_stack = np.stack(
        [
            pywt_time_cpu,
            ptwt_time_cpu,
            ptwt_time_cpu_jit,
            ptwt_time_gpu,
            ptwt_time_gpu_jit,
        ],
        -1,
    )
    plt.boxplot(time_stack)
    plt.yscale("log")
    plt.xticks(
        [1, 2, 3, 4, 5],
        ["pywt-cpu", "ptwt-cpu", "ptwt-cpu-jit", "ptwt-gpu", "ptwt-gpu-jit"],
    )
    plt.xticks(rotation=20)
    plt.ylabel("runtime [s]")
    plt.title("DWT-1D")
    plt.savefig("./figs/timeitconv1d.png")
    # plt.show()
