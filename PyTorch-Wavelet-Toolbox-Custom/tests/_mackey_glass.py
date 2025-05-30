"""Generate artificial time-series data for debugging purposes."""

from typing import Optional, Union

import torch


def generate_mackey(
    batch_size: int = 100,
    tmax: float = 200,
    delta_t: float = 1.0,
    rnd: bool = True,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """Generate synthetic training data using the Mackey system of equations.

    dx/dt = beta*(x'/(1+x'))
    See http://www.scholarpedia.org/article/Mackey-Glass_equation
    for reference.

    The system is simulated using a forward euler scheme
    (https://en.wikipedia.org/wiki/Euler_method).

    Args:
        batch_size (int): The number of simulated series to return.
            Defaults to 100.
        tmax (float): Total time to simulate. Defaults to 200.
        delta_t (float): Size of the time step. Defaults to 1.0.
        rnd (bool): If true use a random initial state.
            Defaults to True.
        device (torch.device or str): Choose cpu or cuda. Defaults to "cuda".

    Returns:
        torch.Tensor: A Tensor of shape [batch_size, time, 1],
    """
    steps = int(tmax / delta_t) + 200

    # multi-dimensional data.
    def _mackey(
        x: torch.Tensor, tau: int, gamma: float = 0.1, beta: float = 0.2, n: int = 10
    ) -> torch.Tensor:
        return beta * x[:, -tau] / (1 + torch.pow(x[:, -tau], n)) - gamma * x[:, -1]

    tau = int(17 * (1 / delta_t))
    x0 = torch.ones([tau], device=device)
    x0 = torch.stack(batch_size * [x0], dim=0)
    if rnd:
        # print('Mackey initial state is random.')
        x0 += torch.empty(x0.shape, device=device).uniform_(-0.1, 0.1)
    else:
        x0 += [-0.01, 0.02]

    x = x0
    # forward_euler
    for _ in range(steps):
        res = torch.unsqueeze(x[:, -1] + delta_t * _mackey(x, tau), -1)
        x = torch.cat([x, res], -1)
    discard = 200 + tau
    return x[:, discard:]


class MackeyGenerator:
    """Generates lorenz attractor data in 1 or 3d on the GPU."""

    def __init__(
        self,
        batch_size: int,
        tmax: float,
        delta_t: float,
        block_size: Optional[int] = None,
        restore_and_plot: bool = False,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        """Create a Mackey-Glass simulator.

        Args:
            batch_size (int): Total number of samples to generate.
            tmax (float): Total simulation time.
            delta_t (float): Time step.
            block_size (int, optional): Length of mean blocks. Defaults to None.
            restore_and_plot (bool): Deactivate random init. Defaults to False.
            device (torch.device or str): Choose cpu or cuda. Defaults to "cuda".
        """
        self.batch_size = batch_size
        self.tmax = tmax
        self.delta_t = delta_t
        self.block_size = block_size
        self.restore_and_plot = restore_and_plot
        self.device = device

    def __call__(self) -> torch.Tensor:
        """Simulate a batch and return the result."""
        data_nd = generate_mackey(
            tmax=self.tmax,
            delta_t=self.delta_t,
            batch_size=self.batch_size,
            rnd=not self.restore_and_plot,
            device=self.device,
        )
        data_nd = torch.unsqueeze(data_nd, -1)
        return data_nd
