from types import SimpleNamespace

import numpy as np

from pypulseq.opts import Opts


def make_spiral_grad(
    channel: str,
    waveform: np.ndarray,
    times: np.ndarray,
    system: Opts = Opts(),
    max_grad: float = 0,
    max_slew: float = 0,
    delay: float = 0,
) -> SimpleNamespace:
    """
    Creates a gradient event with arbitrary waveform.

    Parameters
    ----------
    channel : str
        Orientation of gradient event of arbitrary shape. Must be one of `x`, `y` or `z`.
    waveform : numpy.ndarray
        Spiral waveform.
    times : np.ndarray
        Time points of the spiral.
    system : Opts, optional, default=Opts()
        System limits.
    max_grad : float, optional, default=0
        Maximum gradient strength.
    max_slew : float, optional, default=0
        Maximum slew rate.
    delay : float, optional, default=0
        Delay in milliseconds (ms).

    Returns
    -------
    grad : SimpleNamespace
        Gradient event with arbitrary waveform.

    Raises
    ------
    ValueError
        If invalid `channel` is passed. Must be one of x, y or z.
        If slew rate is violated.
        If gradient amplitude is violated.
    """
    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of x, y or z. Passed: {channel}"
        )

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    g = waveform
    slew = np.squeeze(np.subtract(g[1:], g[:-1]) / system.grad_raster_time)

    if max(abs(slew)) >= max_slew:
        raise ValueError(f"Slew rate violation {max(abs(slew)) / max_slew * 100}")
    if max(abs(g)) >= max_grad:
        raise ValueError(f"Gradient amplitude violation {max(abs(g)) / max_grad * 100}")

    grad = SimpleNamespace()
    grad.type = "grad"
    grad.channel = channel
    grad.waveform = g
    grad.delay = delay
    grad.t = times
    # True timing and aux shape data
    grad.tt = (np.arange(1, len(g) + 1) - 0.5) * system.grad_raster_time
    grad.first = (3 * g[0] - g[1]) * 0.5  # Extrapolate by 1/2 gradient rasters
    grad.last = (g[-1] * 3 - g[-2]) * 0.5  # Extrapolate by 1/2 gradient rasters

    return grad


from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from pypulseq.opts import Opts


def traj_to_grad_spiral(
    k: np.ndarray, nb_interleaves=1, raster_time: float = Opts().grad_raster_time
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert k-space trajectory `k` into gradient waveform in compliance with `raster_time` gradient raster time.

    Parameters
    ----------
    k : numpy.ndarray
        K-space trajectory to be converted into gradient waveform.
    raster_time : float, optional, default=Opts().grad_raster_time
        Gradient raster time.
    nb_interleaves : int, optional, default=1,
        number of interleaves in the case of the spiral trajectory

    Returns
    -------
    g : numpy.ndarray
        Gradient waveform.
    sr : numpy.ndarray
        Slew rate.
    """

    size_tot = np.shape(k)[1]
    size_interleaves = size_tot / nb_interleaves

    # Compute finite difference for gradients in Hz/m
    g = np.zeros([2, size_tot])
    for i in range(nb_interleaves):
        g[:, int(i * size_interleaves)] = k[:, int(i * size_interleaves)] / raster_time
        g[:, int(i * size_interleaves + 1) : int((i + 1) * size_interleaves)] = (
            k[:, int(i * size_interleaves + 1) : int((i + 1) * size_interleaves)]
            - k[:, int(i * size_interleaves) : int((i + 1) * size_interleaves - 1)]
        ) / raster_time

    # Compute the slew rate (time derivative of the gradient)
    sr = np.zeros([2, size_tot])
    for i in range(nb_interleaves):
        sr[:, int(i * size_interleaves)] = g[:, int(i * size_interleaves)] / raster_time
        sr[:, int(i * size_interleaves + 1) : int((i + 1) * size_interleaves)] = (
            g[:, int(i * size_interleaves + 1) : int((i + 1) * size_interleaves)]
            - g[:, int(i * size_interleaves) : int((i + 1) * size_interleaves - 1)]
        ) / raster_time

    # Now we think how to post-process the results:
    # gradient is now sampled between the k-points whilst the slew rate is between the gradient points
    # sr = np.zeros([2,size-1])
    # sr[:, 0] = sr0[:, 0]
    # sr[:, 0:-1] = 0.5 * (sr0[:, 0:-1] + sr0[:, 1:])
    # sr[:, -1] = sr0[:, -1]

    return g, sr
