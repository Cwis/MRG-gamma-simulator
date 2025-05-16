#!/usr/bin/env python
# coding: utf-8

"""
This is an experimental variable density spiral sequence adapted from Benedicte Delattre's spiralsampling_vds_imprvd5.m
 Matlab code implemented during her phD.

It is made to work with pypulseq library:
    Ravi, Keerthi, Sairam Geethanath, and John Vaughan. "PyPulseq: A Python Package for MRI Pulse Sequence Design." Journal
of Open Source Software 4.42 (2019): 1725.

    Ravi, Keerthi Sravan, et al. "Pulseq-Graphical Programming Interface: Open source visual environment for prototyping
pulse sequences and integrated magnetic resonance imaging algorithm development." Magnetic resonance imaging 52 (2018):
9-15.

Rewinding part is from "Spiral Balanced Steady-State Free Precession Cardiac Imaging" MRM 53:1468-1473 (2005)
"""

import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from traj_to_grad_spiral_v2 import traj_to_grad_spiral_v2
from make_spiral_grad import make_spiral_grad

# ======
# SETUP
# ======

# Define system
# ==============
# Most 1.5T to 3.0T superconducting whole body scanners have maximum gradient strengths in the range of 30-45 mT/m,
# while lower field (<0.5T) permanent scanners are in the 15-25 mT/m range.
system = pp.Opts(
    max_grad=20,  # based on doi: 10.1002/jmri.26637
    grad_unit="mT/m",
    max_slew=100,  # 50 based on doi: 10.1002/jmri.26637
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    # rf_raster_time=20e-6,   #todo: verify this
    grad_raster_time=20e-6,  # Cameleon can be faster (192kHz so 5e-6s)
    gamma=-1.24e6,  # 129m Xe
)

# Use around 10% safety margin on max_slew_rate (T/m/s) and max_grad for spiral trajectory (T/m)
max_grad = 0.9 * (system.max_grad / (system.gamma))
max_slew = 0.9 * (system.max_slew / (system.gamma))

# We need the rise time for rewinding gradient calculation
min_rise_time = max_grad / max_slew

# We also need to calculate S being the maximum allowed rotatable slew rate defined as:
S = 1 / np.sqrt(2) * max_grad / min_rise_time
max_slew = S

# Create a new sequence object
seq = pp.Sequence(system=system)

# Define user parameters
# =========================
# b0 = 0.05 #T
fov = 0.03  # m
slice_thickness = 3e-3  # m
n_slices = 1
nb_interleaves = 1
matrix_size = 16
undersamp_factor = 0.8  # undersampling factor: = 1 is a fully sampled k-space whereas
# <1 corresponds to radial undersampling of k-space while angular sampling stays constant
density_parameter = (
    0.5  # oversampling in kspace = alpha_d for Zhao, Archimedeaan spiral if =1
)

# =============================
# SPIRAL TRAJECTORY CALCULATION
# =============================

# Calculate parameters according to user parameters
# ====================================================

# Calculate the number of turns according to Nyquist criteria
if 2 * nb_interleaves / matrix_size < 1:
    nb_turns = np.ceil(
        (
            1
            - (1 - 2 * nb_interleaves / (matrix_size * undersamp_factor)) #todo: 1- ???
            ** (1 / density_parameter)
        )
        ** -1
    )

    print("Number of turns: ", nb_turns)
else:
    raise ValueError("Number of interleaves to high for given matrix size")

# Calculate key parameters
k_max = matrix_size / (2 * fov)  # maximum kspace sampled value = lambda for Zhao, kFOV = 2kmax
omega = 2 * np.pi * nb_turns


# Calculate constant parts for following tau calculation
# ======================================================
# Constant for slew rate limited regime (first part of the spiral around the center)
const_slew = np.sqrt(abs(system.gamma) * max_slew / (k_max * omega**2)) * (
    1 + density_parameter / 2
)

# Constant for amplitude limited regime (second part of the spiral)
const_amp = abs(system.gamma) * max_grad / (k_max * omega) * (density_parameter + 1)

# Design trajectory
# ==================

# To find the good P, set a low value first (probably a lot of iterations),
# then enter that value here so only one iteration will be needed afterward (cf line 376)
P = 820000  # minimum data point index for which slew rate = max_slew_rate / 2
# (minimum value to avoid slew-rate overshoot, depends on the performance of the chosen scanner)
P_low = 0
P_high = P

exit_OK = 0  # =1 when a suitable solution has been found
limit_OK = 0  # =1 when a limit between the two regimes has been found (ie, a suitable P has been found)
calc_OK = 0  # =1 when the end of gradient amplitude limited has been found
cpt_L = 0  # number of iterations to find the right P

while exit_OK == 0:

    if cpt_L == 100:
        raise ValueError("Too many iterations!")
    # todo: quicker way to find the right P?
    if cpt_L > 0:  # starts at the second iteration
        print("boucleA")
        if limit_OK == 0:
            print("boucleB1")
            if calc_OK == 1:
                print("boucleB1a")
                P_high = P
                limit_OK = 1
            else:  # start
                print("boucleB1b")
                P_low = P
                P = 2 * P_high
                P_high = P

        if (
            limit_OK == 1
        ):  # fine-tune between Phigh and Plow to find the best solution (minimal value)
            print("boucleB2")
            interval = P_high - P_low
            interval_2 = interval / 2

            if (calc_OK == 1) & (interval > 1):
                print("boucleB2a")
                P_high = P
                interval = P_high - P_low
                interval_2 = interval / 2
                print("Interval_2: ", interval_2)
                P = np.ceil(interval_2) + P_low
                print("P = ceil(interval_2)+P_low =: ", P)
            elif (calc_OK == 0) & (interval > 1):
                print("boucleB2b")
                P_low = P
                P = np.ceil(interval_2) + P_low
                print("Interval_2: ", interval_2)
            elif (calc_OK == 0) & (interval == 1):
                print("boucleB2c")
                P = P_low
                interval = P_high - P_low
                exit_OK = 1
            elif (calc_OK == 1) & (interval == 1):
                print("boucleB2d")
                P = P_high
                interval = P_high - P_low
                exit_OK = 1
            print("Interval: ", interval)
    print("P: ", P)
    # First part: Slew-rate limited regime, stops when grad_cur_amp = max_grad or tau = 1
    # ------------------------------------------------------------------------------------
    # When the number of interleaves is increased, the trajectory leads to large slew-rate overflow
    # for small k-space values so Zhao et al. proposed the following solution to regularize the slew rate at the origin:
    # Slew rate exponentially increases to its max value so slew(t) = max_slew * (1 - exp(-t/L))**2
    # The parameter L is used to regularize the slew rate at the origin
    # L is chosen by setting slew(t) = max_slew / 2 for Pth data point
    # So we get max_slew / 2 = max_slew * (1 - exp(-t/L))**2 ==> So L = - t / (np.log(1 - 1 / np.sqrt(2)))
    # t is replaced by P * system.grad_raster_time (Pth data point)
    L = -P * system.grad_raster_time / (np.log(1 - 1 / np.sqrt(2)))

    grad_cur_amp = 0
    time = 0
    time_vector, tau, grad_x, grad_y, slew_x, slew_y = (
        np.zeros(
            1, dtype=float
        ),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
    )

    while (grad_cur_amp < max_grad) and (
        tau[-1] <= 1
    ):  # Condition added for tau so the trajectory stops when kmax is reached
        # Zhao's solution by setting slew(t) = max_slew(1-e(-t/L))**2
        tau = np.append(
            tau,
            (const_slew * (time + L * np.exp(-time / L) - L))
            ** (1 / (1 + density_parameter / 2)),
        )

        grad_x = np.append(
            grad_x,
            k_max
            / abs(system.gamma)
            * (1 / system.grad_raster_time)
            * (
                (tau[-1] ** density_parameter) * np.cos(omega * tau[-1])
                - tau[-2] ** density_parameter * np.cos(omega * tau[-2])
            ),
        )
        grad_y = np.append(
            grad_y,
            k_max
            / abs(system.gamma)
            * (1 / system.grad_raster_time)
            * (
                (tau[-1] ** density_parameter) * np.sin(omega * tau[-1])
                - tau[-2] ** density_parameter * np.sin(omega * tau[-2])
            ),
        )

        slew_x = np.append(
            slew_x, (grad_x[-1] - grad_x[-2]) * (1 / system.grad_raster_time)
        )
        slew_y = np.append(
            slew_y, (grad_y[-1] - grad_y[-2]) * (1 / system.grad_raster_time)
        )

        grad_cur_amp = max(grad_x[-1], grad_y[-1])
        time += system.grad_raster_time
        time_vector = np.append(time_vector, time)

    print(
        "First part in slew rate regime done : Current gradient amp. =",
        grad_cur_amp,
        "      Max gradient = ",
        max_grad,
    )

    # Transition time between the two regimes:
    transition_time = time - system.grad_raster_time

    t0b = (
                 (tau[-1] - tau[-2])
                 * (1 / system.grad_raster_time)
                 * (density_parameter + 1)
                 * const_amp ** ((-1) / (density_parameter + 1))
         ) ** (-(density_parameter + 1) / density_parameter)

    delta_taub = tau[-1] - ((const_amp) * t0b) ** (1 / (1 + density_parameter))

    t0b += system.grad_raster_time


    # Second part: Amplitude limited regime, time to reach tau=1 (ie when k=kmax)
    # ---------------------------------------------------------------------------

    while tau[-1] <= 1:
        # General solution for amplitude limited regime,
        # we add delta_tau to keep the continuity
        tau = np.append(tau, (const_amp * t0b) ** (1 / (density_parameter + 1)) + delta_taub)

        grad_x = np.append(
            grad_x,
            k_max
            / abs(system.gamma)
            * (1 / system.grad_raster_time)
            * (
                (tau[-1] ** density_parameter) * np.cos(omega * tau[-1])
                - tau[-2] ** density_parameter * np.cos(omega * tau[-2])
            ),
        )
        grad_y = np.append(
            grad_y,
            k_max
            / abs(system.gamma)
            * (1 / system.grad_raster_time)
            * (
                (tau[-1] ** density_parameter) * np.sin(omega * tau[-1])
                - tau[-2] ** density_parameter * np.sin(omega * tau[-2])
            ),
        )

        slew_x = np.append(
            slew_x, (grad_x[-1] - grad_x[-2]) * (1 / system.grad_raster_time)
        )
        slew_y = np.append(
            slew_y, (grad_y[-1] - grad_y[-2]) * (1 / system.grad_raster_time)
        )

        time += system.grad_raster_time
        time_vector = np.append(time_vector, time)

        t0b += system.grad_raster_time

    plt.figure(1)
    plt.suptitle("Spiral gradients - Initial")
    plt.subplot(221)
    plt.plot(time_vector, grad_x, "b")
    plt.title("grad_x")
    plt.axvline(x=transition_time, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (T/m)")
    plt.subplot(222)
    plt.plot(time_vector, grad_y, "g")
    plt.title("grad_y")
    plt.axvline(x=transition_time, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (T/m)")
    plt.subplot(223)
    plt.plot(time_vector, slew_x, "b")
    plt.title("slew_x")
    plt.axvline(x=transition_time, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (T/m)")
    plt.subplot(224)
    plt.plot(time_vector, slew_y, "g")
    plt.title("slew_y")
    plt.axvline(x=transition_time, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (T/m)")
    plt.gcf().subplots_adjust(
        left=0.125, bottom=0.1, right=0.98, top=0.9, wspace=0.47, hspace=0.55
    )
    plt.show()

    calc_OK = 1
    # The line below needs to be uncommented if the right P has already been found:
    exit_OK = 1

    # Evaluation of the solution
    # ----------------------------

    if max(np.abs(grad_x)) > max_grad  \
            or max(np.abs(grad_y)) > max_grad \
            or max(np.abs(slew_x)) > max_slew \
            or max(np.abs(slew_y)) > max_slew:
        print("Exceeds gradient maximum specifications with margin !")
        if max(np.abs(grad_x)) > (system.max_grad / system.gamma) \
                or max(np.abs(grad_y)) > (system.max_grad / system.gamma) \
                or max(np.abs(slew_x)) > (system.max_slew / system.gamma) \
                or max(np.abs(slew_y)) > (system.max_slew / system.gamma):
            print("Exceeds absolute gradient maximum specifications !")
            calc_OK = 0
            exit_OK = 0
        else:
            calc_OK = 1
            exit_OK = 1

    cpt_L += 1

    readout_time = len(grad_x) * system.grad_raster_time

    print(
        [
            "Readout time = ",
            readout_time,
            "s   (time_vector[-1] =",
            time_vector[-1],
            ")",
        ]
    )
    print(f"Max grad x = {max(np.abs(grad_x))} Max grad y = {max(np.abs(grad_y))}")
    print(f"Max slew x = {max(np.abs(slew_x))} Max slew y = {max(np.abs(slew_y))}")

Grad = grad_x + 1j * grad_y
Slew = slew_x + 1j * slew_y

# K-Space trajectory
# ==================

k_size = len(grad_x)
kx = np.zeros(k_size, dtype=float)
ky = np.zeros(k_size, dtype=float)

for i in range(k_size):
    kx[i] = abs(system.gamma) / (1 / system.grad_raster_time) * (grad_x[i]) + kx[i - 1]
    ky[i] = abs(system.gamma) / (1 / system.grad_raster_time) * (grad_y[i]) + ky[i - 1]

# Interleaves distribution
# --------------------------
kx_inter = np.zeros(len(kx) * nb_interleaves, dtype=float)
ky_inter = np.zeros(len(ky) * nb_interleaves, dtype=float)

# ==> 2pi/Nb method
plt.figure(3)
plt.suptitle("k-space trajectory (2pi/Nb) - Initial")
plt.xlabel("kx")
plt.ylabel("ky")
for i in range(nb_interleaves):
    kx_tmp = kx * np.cos(2 * np.pi / nb_interleaves * i) + ky * np.sin(
        2 * np.pi / nb_interleaves * i
    )
    ky_tmp = -kx * np.sin(2 * np.pi / nb_interleaves * i) + ky * np.cos(
        2 * np.pi / nb_interleaves * i
    )
    kx_inter[len(kx) * i : len(kx) * (i + 1)] = kx_tmp
    ky_inter[len(ky) * i : len(ky) * (i + 1)] = ky_tmp
    plt.plot(kx_tmp, ky_tmp)
plt.show()


# ==> Golden angle method
"""
plt.figure(3)
plt.suptitle("k-space trajectory (golden angle) - Initial")
plt.xlabel("kx")
plt.ylabel("ky")
for i in range(nb_interleaves):
    kx_tmp = kx * np.cos(
        2 * 222.4969 * (i-1)
    ) + ky * np.sin(2 * 222.4969 * (i-1))
    ky_tmp = -kx * np.sin(
        2 * 222.4969 * (i-1)
    ) + ky * np.cos(2 * 222.4969 * (i-1))
    kx_inter[len(kx) * i : len(kx) * (i + 1)] = kx_tmp
    ky_inter[len(ky) * i : len(ky) * (i + 1)] = ky_tmp
    plt.plot(kx_tmp, ky_tmp)
plt.show()
"""


# ===================================
# PYPULSEQ COMPATIBILITY VERIFICATION
# ===================================
# Using a pypulseq function is usefull to get the SimpleNamespaces that will be then directly added to the .seq file
k_pp = np.array([kx_inter, ky_inter])
grad_pp, slew_pp = traj_to_grad_spiral_v2(k_pp, nb_interleaves, system.grad_raster_time)
size_interleaves = grad_pp.shape[1] / nb_interleaves

# Verification of the gradients and slew rate
plt.figure(4)
plt.suptitle("Pypulseq compatibility verification")
for i in range(nb_interleaves):
    plt.subplot(221)
    plt.title("Gradient - PP")
    plt.plot(
        grad_pp[0, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
        grad_pp[1, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
    )
    plt.xlabel("Gx Amplitude (Hz/m)")
    plt.ylabel("Gy Amplitude (Hz/m)")

    plt.subplot(223)
    plt.title("Slew Rate - PP")
    plt.plot(
        slew_pp[0, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
        slew_pp[1, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
    )
    plt.xlabel("SRx Amplitude (Hz/m/s)")
    plt.ylabel("SRy Amplitude (Hz/m/s)")

plt.subplot(222)
plt.title("Gradient - Initial")
plt.plot(grad_x * abs(system.gamma), grad_y * abs(system.gamma))
plt.xlabel("Gx Amplitude (Hz/m)")
plt.ylabel("Gy Amplitude (Hz/m)")

plt.subplot(224)
plt.title("Slew Rate - Initial")
plt.plot(slew_x * abs(system.gamma), slew_y * abs(system.gamma))
plt.xlabel("SRx Amplitude (Hz/m/s)")
plt.ylabel("SRy Amplitude (Hz/m/s)")
plt.gcf().subplots_adjust(
    left=0.15, bottom=0.1, right=0.97, top=0.85, wspace=0.5, hspace=0.6
)


plt.show()


# ================
# SEQUENCE EVENTS
# ================

# Create 90 degree slice selection pulse and associated gradients
# ------------------------------------------------------------------
rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=2.7e-2,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=2,
    return_gz=True,
)

# Calculate ADC
# -----------------

# adc_time = system.grad_raster_time * len(grad_x_filtered)
adc_samples_per_segment = grad_pp.shape[1] / nb_interleaves  # = new size_interleaves
adc_samples = nb_interleaves * adc_samples_per_segment
adc_dwell = readout_time / adc_samples_per_segment
adc_segment_duration = adc_samples_per_segment * adc_dwell

if (
    round(adc_segment_duration % system.grad_raster_time) > np.finfo(float).eps
):  # round is used because mod in Python gives float results on floats
    raise ValueError("ADC segmentation model results in incorrect segment duration")

adc = pp.make_adc(
    num_samples=adc_samples_per_segment,
    # dwell=adc_dwell, #Cannot be used if duration is
    duration=readout_time,
    system=system,
    delay=pp.calc_duration(gz_reph),
)

# Extend spiral_grad_shape by repeating the last sample
# this is needed to accomodate for the ADC tuning delay
# spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1]]


"""
 because of the ADC alignment requirements the sampling window possibly
 extends past the end of the trajectory (these points will have to be
 discarded in the reconstruction, which is no problem). However, the
 ramp-down parts and the Z-spoiler now have to be added to the readout
 block otherwise there will be a gap inbetween
 gz_spoil.delay=mr.calcDuration(gx);
 gx_spoil.delay=gz_spoil.delay;
 gy_spoil.delay=gz_spoil.delay;
 gx_combined=mr.addGradients([gx,gx_spoil], lims);
 gy_combined=mr.addGradients([gy,gy_spoil], lims);
 gz_combined=mr.addGradients([gzReph,gz_spoil], lims);
 """

# Define sequence blocks
# -------------------------

# For the verification plot inside the loop
plt.figure(5)
ax = plt.gca()

for i in range(n_slices):
    for j in range(nb_interleaves):

        # RF
        # -----
        seq.add_block(rf, gz)
        rf.freq_offset = gz.amplitude * slice_thickness * (i - (n_slices - 1) / 2)
        # todo: gx.freq_offset / gy.freq_offset ?

        # Readout gradients
        # ---------------------
        # Extract the waveforms from grad_pp
        waveform_gx = grad_pp[
            0,
            int(j * adc_samples_per_segment) : int((j + 1) * adc_samples_per_segment),
        ]
        waveform_gy = grad_pp[
            1,
            int(j * adc_samples_per_segment) : int((j + 1) * adc_samples_per_segment),
        ]

        # Make spiral gradients
        gx = make_spiral_grad(
            channel="x",
            waveform=waveform_gx,
            system=system,
            delay=pp.calc_duration(
                gz_reph
            ),  # todo: delay may be only for the first interleave?
        )
        gy = make_spiral_grad(
            channel="y",
            waveform=waveform_gy,
            system=system,
            delay=pp.calc_duration(gz_reph),
        )

        # Add spiral gradients
        seq.add_block(gx, gy, adc)

        # Verification plot
        color = next(ax._get_lines.prop_cycler)["color"]
        plt.suptitle("Spiral gradients for each interleave")
        # plt.subplot(221)
        plt.subplot(211)
        plt.plot(time_vector, waveform_gx, color=color)
        plt.title("Gx - PP - Before Rotation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Hz/m)")
        plt.subplot(212)
        plt.plot(time_vector, waveform_gy, color=color)
        plt.title("Gy - PP - Before Rotation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Hz/m)")
        plt.gcf().subplots_adjust(
            left=0.16, bottom=0.1, right=0.98, top=0.85, wspace=0.5, hspace=0.6
        )
    plt.show()

# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed! Error listing follows: ")
    print(error_report)

# Set the definitions
seq.set_definition("FOV", [fov, fov, slice_thickness])
seq.set_definition("Name", "spiral")
# seq.set_definition("MaxAdcSegmentLength", adc_samples_per_segment)

# Output sequence for scanner
seq.write("spiral_v1.seq")

# Plot sequence waveforms
seq.plot()

# Single-function call for trajectory calculation
ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

# For plot purpose, new_time_vector needs to be calculated to fit ktraj length
ktraj_time_vector = np.zeros(1)
for i in range(1, int(ktraj.size / 3 / nb_interleaves)):
    ktraj_time_vector = np.append(ktraj_time_vector, i * system.grad_raster_time)

# K-space trajectory through time
# ---------------------------------
plt.figure(8)
# 3D plot
# ax = plt.axes(projection='3d')
# ax.plot3D(ktraj[0,:], ktraj[1,:], ktraj[2,:])

# 2D plot
plt.subplot(311)
plt.suptitle("K-space trajectory through time")
plt.plot(ktraj[0, :], label="Gx")
# plt.plot(ktraj_adc[0,:], ".")
plt.legend(loc="upper left")
plt.subplot(312)
plt.ylabel("k amplitude")
plt.plot(ktraj[1, :], label="Gy")
# plt.plot(ktraj_adc[1,:], ".")
plt.legend(loc="upper left")
plt.subplot(313)
plt.xlabel("Time (s)")
plt.plot(ktraj[2, :], label="Gz")
# plt.plot(ktraj_adc[2,:], ".")
plt.legend(loc="upper left")
plt.show()

# Plot k-space for each inerleave
# ----------------------------------
for i in range(nb_interleaves):

    # k-space trajectory as a function of time
    plt.figure(9)
    plt.suptitle("k-space trajectory as a function of time")
    plt.title(f"Interleave {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("k amplitude")
    plt.plot(
        ktraj_time_vector,
        ktraj[
            0,
            i
            * int(ktraj.size / 3 / nb_interleaves) : (i + 1)
            * int(ktraj.size / 3 / nb_interleaves),
        ],
        "b",
        label="x axis",
    )
    plt.plot(
        ktraj_time_vector,
        ktraj[
            1,
            i
            * int(ktraj.size / 3 / nb_interleaves) : (i + 1)
            * int(ktraj.size / 3 / nb_interleaves),
        ],
        "g",
        label="y axis",
    )
    plt.legend(loc="upper left")
    plt.show()


# 2D k-space trajectory (a part from the previous loop for plot purpose)
plt.figure(10)
plt.suptitle("2D k-space trajectory")
for i in range(nb_interleaves):

    plt.subplot(211)
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.plot(
        ktraj[
            0,
            int(i * (ktraj.size / 3 / nb_interleaves)) : int(
                (i + 1) * (ktraj.size / 3 / nb_interleaves)
            ),
        ],
        ktraj[
            1,
            int(i * (ktraj.size / 3 / nb_interleaves)) : int(
                (i + 1) * (ktraj.size / 3 / nb_interleaves)
            ),
        ],
        label=f"Interleave {i}",
    )
    plt.legend(loc="upper left")

    plt.subplot(212)
    plt.title("Sampling points")
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.gcf().subplots_adjust(
        left=0.12, bottom=0.1, right=0.98, top=0.85, wspace=0.5, hspace=0.6
    )
    plt.plot(
        ktraj_adc[
            0,
            int(i * (ktraj.size / 3 / nb_interleaves)) : int(
                (i + 1) * (ktraj.size / 3 / nb_interleaves)
            ),
        ],
        ktraj_adc[
            1,
            int(i * (ktraj.size / 3 / nb_interleaves)) : int(
                (i + 1) * (ktraj.size / 3 / nb_interleaves)
            ),
        ],
        ".",
        label=f"Interleave {i}",
    )
    plt.legend(loc="upper left")
plt.show()

test = 1
