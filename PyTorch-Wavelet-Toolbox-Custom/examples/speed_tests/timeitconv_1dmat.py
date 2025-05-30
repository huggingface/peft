import time

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

import ptwt

if __name__ == "__main__":
    length = 1e6
    repetitions = 100

    pywt_time_cpu = []
    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_gpu_mat = []

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        start = time.perf_counter()
        res = pywt.wavedec(data, "db5", level=10)
        end = time.perf_counter()
        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.wavedec(data, "db5", level=10)
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = ptwt.wavedec(data, "db5", level=10)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

    matrix_wavedec = ptwt.MatrixWavedec("db5", 10)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = matrix_wavedec(data)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu_mat.append(end - start)

    print("1d boundary results")
    print(f"1d-pywt-cpu:{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}")
    print(f"1d-ptwt-cpu:{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}")
    print(f"1d-ptwt-gpu:{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}")
    print(
        f"1d-ptwt-jit:{np.mean(ptwt_time_gpu_mat):5.5f} +- {np.std(ptwt_time_gpu_mat):5.5f}"
    )

    time_stack = np.stack(
        [pywt_time_cpu, ptwt_time_cpu, ptwt_time_gpu, ptwt_time_gpu_mat], -1
    )
    plt.boxplot(time_stack)
    plt.yscale("log")
    plt.xticks([1, 2, 3, 4], ["pywt-cpu", "ptwt-cpu", "ptwt-gpu", "ptwt-gpu-boundary"])
    plt.xticks(rotation=20)
    plt.ylabel("runtime [s]")
    plt.title("DWT-1D-boundary")
    plt.savefig("./figs/timeitconv1dmat.png")
