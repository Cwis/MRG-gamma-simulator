from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from pypulseq.opts import Opts


def traj_to_grad_spiral_v2(
    k: np.ndarray, nb_interleaves=1, raster_time: float = Opts().grad_raster_time
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert k-space trajectory `k` into gradient waveform in compliance with `raster_time` gradient raster time.

    Parameters
    ----------
    k : numpy.ndarray
        K-space trajectory to be converted into gradient waveform.
    nb_interleaves : int, optional, default=1,
        number of interleaves in the case of the spiral trajectory
    raster_time : float, optional, default=Opts().grad_raster_time
        Gradient raster time.

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
