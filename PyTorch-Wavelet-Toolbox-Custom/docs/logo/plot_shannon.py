import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    bandwidth = 1
    center = 0
    grid_values = np.linspace(-6, 6, 500)
    shannon = (
        np.sqrt(bandwidth)
        * (np.sin(np.pi * bandwidth * grid_values) / (np.pi * bandwidth * grid_values))
        * np.exp(1j * 2 * np.pi * center * grid_values)
    )
    plt.plot(shannon, linewidth=20.0)
    plt.axis("off")
    plt.savefig("shannon.png")
    plt.show()
