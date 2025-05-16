#!/usr/bin/env python
# coding: utf-8

"""
This is an experimental k-space trajectory focused on borders acquisition only and designed for GammaMRI project
"""

import math

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from pypulseq.decompress_shape import decompress_shape

# ======
# SETUP
# ======
# Define system limits
system = pp.Opts(
    max_grad=20,  # based on doi: 10.1002/jmri.26637
    grad_unit="mT/m",
    max_slew=50,  # based on doi: 10.1002/jmri.26637
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    grad_raster_time=20e-6,  # Cameleon can be faster (192kHz so 5e-6s)
    adc_dead_time=10e-6,
    gamma=1.37e6,  # Xe
)

# Create a new sequence object
seq = pp.Sequence(system=system)

# Define user parameters
fov = 0.03
Nx = 7
Ny = 7
alpha = 10  # alpha=°, flip_angle=rad
slice_thickness = 3e-3
nb_slice = 1
delta_k = 1 / fov

# Define timings
TE = 5.5e-3
TR = 10e-3

rf_spoiling_inc = 117  # RF spoiling increment (117 ou 123)

# ======
# CREATE EVENTS
# ======

"RF sinc pulse with slice select and slice select rephasing"
rf, gz, gzr = pp.make_sinc_pulse(
    flip_angle=alpha * math.pi / 180,
    duration=6e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    system=system,
    return_gz=True,
)  # TBW=2 (rapid imaging), =4 (180), =8 (90), =12 (slab and saturation) https://inst.eecs.berkeley.edu/~ee225e/sp13/notes/Lecture13.pdf

" Readout gradient "
gx = pp.make_trapezoid(
    channel="x", flat_area=Nx * delta_k, flat_time=3.2e-3, system=system
)
gy = pp.make_trapezoid(
    channel="y", flat_area=Ny * delta_k, flat_time=3.2e-3, system=system
)
gx_2 = pp.make_trapezoid(
    channel="x", flat_area=-Nx * delta_k, flat_time=3.2e-3, system=system
)
gy_2 = pp.make_trapezoid(
    channel="y", flat_area=-Ny * delta_k, flat_time=3.2e-3, system=system
)
adc = pp.make_adc(
    num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
)
gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=2e-3, system=system)

"Rephasing gradient"
gz_reph = pp.make_trapezoid(
    channel="z", area=-gz.area / 2, duration=2e-3, system=system
)
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k

# Define gradient spoiling
# gx_spoil = pp.make_trapezoid(channel='x', area=2 * Nx * delta_k, system=system)
# gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = (
    np.ceil(
        (
            TE
            - pp.calc_duration(gx_pre)
            - gz.fall_time
            - gz.flat_time / 2
            - pp.calc_duration(gx) / 2
        )
        / seq.grad_raster_time
    )
    * seq.grad_raster_time
)
delay_TR = (
    np.ceil(
        (
            TR
            - pp.calc_duration(gz)
            - pp.calc_duration(gx_pre)
            - pp.calc_duration(gx)
            - delay_TE
        )
        / seq.grad_raster_time
    )
    * seq.grad_raster_time
)

assert np.all(delay_TE >= 0)
# assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0

# ======
# CONSTRUCT SEQUENCE
# ======

for i in range(nb_slice):
    rf.phase_offset = rf_phase / 180 * np.pi
    adc.phase_offset = rf_phase / 180 * np.pi
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
    seq.add_block(rf, gz)
    "Phase encoding gradient"
    gy_pre = pp.make_trapezoid(
        channel="y",
        area=phase_areas[i],
        duration=pp.calc_duration(gx_pre),
        system=system,
    )
    seq.add_block(gx_pre, gy_pre, gz_reph)
    seq.add_block(pp.make_delay(delay_TE))
    seq.add_block(gx, adc)
    seq.add_block(gy, adc)
    seq.add_block(gx_2, adc)
    seq.add_block(gy_2, adc)
    gy_pre.amplitude = -gy_pre.amplitude
    seq.add_block(pp.make_delay(delay_TR), gy_pre)

(
    ok,
    error_report,
) = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]


# ======
# VISUALIZATION
# ======
seq.plot()

# Trajectory calculation and plotting
ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
"Plot the k-space trajectory as a function of time"
plt.figure(1)
plt.plot(time_axis, ktraj.T[0:, 0], label="x axis")
plt.plot(time_axis, ktraj.T[0:, 1], label="y axis")
plt.plot(time_axis, ktraj.T[0:, 2], label="z axis")
plt.plot(
    t_adc, ktraj_adc[0], ".", label="sampling points"
)  # Plot sampling points on the kx-axis
plt.legend(loc="upper left")
plt.suptitle("k-space trajectory as a function of time")
plt.xlabel("Time")
plt.ylabel("k amplitude")


"Plot the 2D k-space trajectory"
plt.figure(2)
plt.plot(ktraj[0], ktraj[1], "b", label="trajectory")  # 2D plot
plt.axis("equal")  # Enforce aspect ratio for the correct trajectory display
plt.plot(
    ktraj_adc[0], ktraj_adc[1], "r.", label="sampling points"
)  # Plot  sampling points
plt.legend(loc="upper left")
plt.suptitle("2D k-space trajectory")
plt.xlabel("kx")
plt.ylabel("ky")

plt.show()

# Prepare the sequence output for the scanner
seq.set_definition("FOV", [fov, fov, slice_thickness])
seq.set_definition("Name", "border")

seq.write("border_pypulseq.seq")

# Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
# slew-rate limits
rep = seq.test_report()
print(rep)
