import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.signal
import torch

# use from src.ptwt.packets if you cloned the repo instead of using pip.
from ptwt import WaveletPacket

fs = 1000
t = np.linspace(0, 2, int(2 // (1 / fs)))
w = np.sin(256 * np.pi * t**2)

wavelet = pywt.Wavelet("sym8")
wp = WaveletPacket(
    data=torch.from_numpy(w.astype(np.float32)), wavelet=wavelet, mode="boundary"
)
level = 5
nodes = wp.get_level(level)
np_lst = []
for node in nodes:
    np_lst.append(wp[node])
viz = np.stack(np_lst).squeeze()

n = list(range(int(np.power(2, level))))
freqs = (fs / 2) * (n / (np.power(2, level)))

xticks = list(range(viz.shape[-1]))[::6]
xlabels = np.round(np.linspace(min(t), max(t), viz.shape[-1]), 2)[::6]

fig, axs = plt.subplots(2)
axs[0].plot(t, w)
axs[0].set_title("Analyzed signal")
axs[0].set_xlabel("time [s]")
axs[0].set_ylabel("magnitude")

axs[1].set_title("Wavelet packet analysis")
axs[1].imshow(np.abs(viz))
axs[1].set_xlabel("time [s]")
axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xlabels)
axs[1].set_ylabel("frequency [Hz]")
axs[1].set_yticks(n[::4])
axs[1].set_yticklabels(freqs[::4])
axs[1].invert_yaxis()
plt.show()
