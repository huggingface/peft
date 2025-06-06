import time

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

import ptwt

if __name__ == "__main__":
    repetitions = int(100)
    length = 1e3

    pywt_time_cpu = []

    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_mat = []

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        start = time.perf_counter()
        res = pywt.wavedec2(data, "db5", level=5)
        end = time.perf_counter()
        pywt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data)
        start = time.perf_counter()
        res = ptwt.wavedec2(data, "db5", level=5)
        end = time.perf_counter()
        ptwt_time_cpu.append(end - start)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = ptwt.wavedec2(data, "db5", level=5)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_gpu.append(end - start)

    matrix_wavedec2 = ptwt.MatrixWavedec2("db5", 5)

    for _ in range(repetitions):
        data = np.random.randn(32, int(length), int(length)).astype(np.float32)
        data = torch.from_numpy(data).cuda()
        start = time.perf_counter()
        res = matrix_wavedec2(data)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ptwt_time_mat.append(end - start)

    print("2d boundary results.")
    print(f"2d-pywt-cpu:{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}")
    print(f"2d-ptwt-cpu:{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}")
    print(f"2d-ptwt-gpu:{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}")
    print(f"2d-ptwt-mat:{np.mean(ptwt_time_mat):5.5f} +- {np.std(ptwt_time_mat):5.5f}")

    time_stack = np.stack(
        [pywt_time_cpu, ptwt_time_cpu, ptwt_time_gpu, ptwt_time_mat], -1
    )
    plt.boxplot(time_stack)
    plt.yscale("log")
    plt.xticks([1, 2, 3, 4], ["pywt-cpu", "ptwt-cpu", "ptwt-gpu", "ptwt-gpu-boundary"])
    plt.xticks(rotation=20)
    plt.ylabel("runtime [s]")
    plt.title("DWT-2D-boundary")
    plt.savefig("./figs/timeitconv2dmat.png")
