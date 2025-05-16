#!/usr/bin/env python
# coding: utf-8

"""
This is an experimental variable density spiral sequence adapted from Benedicte Delattre's spiralsampling_vds_imprvd5.m code
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
import timeit
import sympy as sym
import mpmath as mp
from sympy import Symbol, nsolve, solveset, S

import pypulseq as pp
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from traj_to_grad_spiral_v2 import traj_to_grad_spiral_v2
from make_rewinding_grad import make_rewinding_grad
from make_spiral_grad import make_spiral_grad

# ======
# SETUP
# ======

# Set system limits
# Most 1.5T to 3.0T superconducting whole body scanners have maximum gradient strengths in the range of 30-45 mT/m,
# while lower field (<0.5T) permanent scanners are in the 15-25 mT/m range.
system = pp.Opts(
    max_grad=25.22,
    grad_unit="mT/m",
    max_slew=170,
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    grad_raster_time=5e-6,
)

# Use around 10% safety margin on max_slew_rate (T/m/s) and max_grad for spiral trajectory (T/m)
max_grad = 0.928 * (system.max_grad / (system.gamma))
max_slew = 0.9 * (system.max_slew / (system.gamma))

# Create a new sequence object
seq = pp.Sequence(system=system)

# Define user parameters
b0 = 3  # ?
fov = 0.06
slice_thickness = 3e-3
n_slices = 1
nb_interleaves = 2
matrix_size = 16
undersamp_factor = 0.4  # undersampling factor: = 1 is a fully sampled k-space whereas
# <1 corresponds to radial undersampling of k-space while angular sampling stays constant
density_parameter = (
    4  # oversampling in kspace = alpha_d for Zhao, Archimedeaan spiral if =1
)

# =============================
# SPIRAL TRAJECTORY CALCULATION
# =============================

# Parameters calculation according to user parameters
# ====================================================

# Calculate the number of turns according to Nyquist criteria
if 2 * nb_interleaves / matrix_size < 1:
    nb_turns = np.ceil(
        (
            1
            - (1 - 2 * nb_interleaves / (matrix_size * undersamp_factor))
            ** (1 / density_parameter)
        )
        ** (-1)
    )
    print("Number of turns: ", nb_turns)
else:
    raise ValueError("Number of interleaves to high for given matrix size")

# Calculate key parameters
k_max = matrix_size / (2 * fov)  # maximum kspace sampled value = lambda for Zhao
w = 2 * np.pi * nb_turns


# Tau calculation
# ===============

# Time to reach tau=1 in slew rate limited regime (first part of the spiral around the center)
tau_slew = (
    np.sqrt((system.gamma * max_slew) / (k_max * w ** 2)) * (1 + density_parameter / 2)
) ** -1

# Time to reach tau=1 (ie when k=kmax) in amplitude limited regime (second part of the spiral)
tau_amp = ((system.gamma * max_grad) / (k_max * w) * (1 + density_parameter)) ** -1
tau_amp_2 = (
    (system.gamma * max_grad) / (k_max * w) * (1 + density_parameter)
)  # to facilitate next calculations


# Trajectory designing
# ====================

# To find the good P, set a low value first (probbly a lot of iterations),
# then enter that value here so only one iteration will be needed afterward (cf line 347)
P = 400  # minimum data point index for which slew rate = max_slew_rate / 2
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
    # todo: quicker way to find the right P
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

    # L is Zhao proposition to regularize the slew rate at the origin (slew(t) = max_slew / 2 for Pth data point)
    L = -P * system.grad_raster_time / (np.log(1 - 1 / np.sqrt(2)))
    print("P = ", P)
    ind = 1
    grad_cur_amp = 0
    t1, t2 = system.grad_raster_time, system.grad_raster_time
    time_vector, final_time_vector, tau, grad_x, grad_y, slew_x, slew_y = (
        np.zeros(2, dtype=float),
        np.zeros(2, dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
    )

    # First part: Slew-rate limited regime, stops when grad_cur_amp = max_grad
    while grad_cur_amp < max_grad:
        # Zhao's solution by setting slew(t) = max_slew(1-e(-t/L))**2
        tau = np.append(
            tau,
            ((t1 + L * np.exp(-t1 / L) - L) / tau_slew)
            ** (1 / (density_parameter / 2 + 1)),
        )

        grad_x = np.append(
            grad_x,
            k_max
            / system.gamma
            * (1 / system.grad_raster_time)
            * (
                (tau[ind] ** density_parameter) * np.cos(w * tau[ind])
                - tau[ind - 1] ** density_parameter * np.cos(w * tau[ind - 1])
            ),
        )

        grad_y = np.append(
            grad_y,
            (
                k_max
                / system.gamma
                * (1 / system.grad_raster_time)
                * (
                    (tau[ind] ** density_parameter) * np.sin(w * tau[ind])
                    - tau[ind - 1] ** density_parameter * np.sin(w * tau[ind - 1])
                )
            ),
        )

        slew_x = np.append(
            slew_x, (grad_x[ind] - grad_x[ind - 1]) * (1 / system.grad_raster_time)
        )
        slew_y = np.append(
            slew_y, (grad_y[ind] - grad_y[ind - 1]) * (1 / system.grad_raster_time)
        )

        grad_cur_amp = max(grad_x[ind], grad_y[ind])
        ind += 1
        t1 += system.grad_raster_time
        t2 += system.grad_raster_time
        final_time_vector = np.append(final_time_vector, t1)
        time_vector = np.append(time_vector, t2)

    print("First part in slew rate regime done : ", grad_cur_amp, "> ", max_grad)

    t2 = (
        (tau[-1] - tau[-2])
        * (1 / system.grad_raster_time)
        * (density_parameter + 1)
        * tau_amp_2 ** ((-1) / (density_parameter + 1))
    ) ** (-(density_parameter + 1) / density_parameter)

    delta_tau = tau[-1] - (
        system.gamma * max_grad / (k_max * w) * (1 + density_parameter) * t2
    ) ** (1 / (1 + density_parameter))

    t2 += system.grad_raster_time

    # Second part: Amplitude limited regime
    while tau[ind - 1] <= 1:
        # general solution for amplitude limited regime
        tau = np.append(
            tau,
            (system.gamma * max_grad / (k_max * w) * (1 + density_parameter) * t2)
            ** (1 / (1 + density_parameter))
            + delta_tau,
        )

        grad_x = np.append(
            grad_x,
            (
                k_max
                / system.gamma
                * (1 / system.grad_raster_time)
                * (
                    (tau[ind] ** density_parameter) * np.cos(w * tau[ind])
                    - tau[ind - 1] ** density_parameter * np.cos(w * tau[ind - 1])
                )
            ),
        )
        grad_y = np.append(
            grad_y,
            (
                k_max
                / system.gamma
                * (1 / system.grad_raster_time)
                * (
                    (tau[ind] ** density_parameter) * np.sin(w * tau[ind])
                    - tau[ind - 1] ** density_parameter * np.sin(w * tau[ind - 1])
                )
            ),
        )

        slew_x = np.append(
            slew_x, (grad_x[ind] - grad_x[ind - 1]) * (1 / system.grad_raster_time)
        )
        slew_y = np.append(
            slew_y, (grad_y[ind] - grad_y[ind - 1]) * (1 / system.grad_raster_time)
        )

        ind += 1
        t1 += system.grad_raster_time
        t2 += system.grad_raster_time
        final_time_vector = np.append(final_time_vector, t1)
        time_vector = np.append(time_vector, t2)

    plt.figure(1)
    plt.subplot(221)
    plt.plot(grad_x)
    plt.title("grad_x")
    plt.subplot(222)
    plt.plot(grad_y)
    plt.title("grad_y")
    plt.subplot(223)
    plt.plot(slew_x)
    plt.title("slew_x")
    plt.subplot(224)
    plt.plot(slew_y)
    plt.title("slew_y")
    plt.show()

    calc_OK = 1
    exit_OK = (
        1  # This line needs to be uncommented if the right P has already been found
    )

    if (max(np.abs(grad_x)) > max_grad) or (max(np.abs(grad_y)) > max_grad):
        print("Exceeds gradient maximum specifications ! - before temporal filter")
    if (max(np.abs(slew_x)) > max_slew) or (np.max(np.abs(slew_y)) > max_slew):
        print("Exceeds slew rate specifications ! - before temporal filter")

    # Temporal filter to reduce the arrays length
    cpt_filter = 0
    grad_filtered_size = int(np.ceil((len(grad_x) - 2) / 2))
    grad_x_filtered, grad_y_filtered, slew_x_filtered, slew_y_filtered = (
        np.zeros((grad_filtered_size), dtype=float),
        np.zeros((grad_filtered_size), dtype=float),
        np.zeros((grad_filtered_size), dtype=float),
        np.zeros((grad_filtered_size), dtype=float),
    )

    for i in range(0, (len(grad_x) - 2), 2):
        grad_x_filtered[cpt_filter] = (grad_x[i] + grad_x[i + 1] + grad_x[i + 2]) / 3
        grad_y_filtered[cpt_filter] = (grad_y[i] + grad_y[i + 1] + grad_y[i + 2]) / 3

        slew_x_filtered[cpt_filter] = (
            (grad_x_filtered[cpt_filter] - grad_x_filtered[cpt_filter - 1])
            * (1 / system.grad_raster_time)
            / 2
        )
        slew_y_filtered[cpt_filter] = (
            (grad_y_filtered[cpt_filter] - grad_y_filtered[cpt_filter - 1])
            * (1 / system.grad_raster_time)
            / 2
        )

        cpt_filter += 1

    plt.figure(2)
    plt.subplot(221)
    plt.plot(grad_x_filtered)
    plt.title("grad_x_filtered")
    plt.subplot(222)
    plt.plot(grad_y_filtered)
    plt.title("grad_y_filtered")
    plt.subplot(223)
    plt.plot(slew_x_filtered)
    plt.title("slew_x_filtered")
    plt.subplot(224)
    plt.plot(slew_y_filtered)
    plt.title("slew_y_filtered")
    plt.show()

    # Evaluation of the solution
    if max(np.abs(grad_x_filtered)) > (system.max_grad / system.gamma) or max(
        np.abs(grad_y_filtered)
    ) > (system.max_grad / system.gamma):
        print("Exceeds gradient maximum specifications !")
        calc_OK = 0
        exit_OK = 0

    if max(np.abs(slew_x_filtered)) > (system.max_slew / system.gamma) or max(
        np.abs(slew_y_filtered)
    ) > (system.max_slew / system.gamma):
        print("Exceeds slew rate specifications !")
        calc_OK = 0
        exit_OK = 0

    cpt_L += 1

    # New time vector after filtre
    new_time_vector = np.zeros((grad_filtered_size), dtype=float)
    for i in range(grad_filtered_size):
        new_time_vector[i] = system.grad_raster_time * i

    readout_time = len(grad_x_filtered) * system.grad_raster_time
    print(["Readout time = ", readout_time, "s"])


Grad = grad_x_filtered + 1j * grad_y_filtered
Slew = slew_x_filtered + 1j * slew_y_filtered

# K-Space trajectory
# ==================

k_filtered_size = len(grad_x_filtered)
kx_filtered = np.zeros(k_filtered_size, dtype=float)
ky_filtered = np.zeros(k_filtered_size, dtype=float)

for i in range(k_filtered_size):
    kx_filtered[i] = (
        system.gamma * 2 / (1 / system.grad_raster_time) * (grad_x_filtered[i])
        + kx_filtered[i - 1]
    )
    ky_filtered[i] = (
        system.gamma * 2 / (1 / system.grad_raster_time) * (grad_y_filtered[i])
        + ky_filtered[i - 1]
    )

# Interleaves distribution
kx = np.zeros(len(kx_filtered) * nb_interleaves, dtype=float)
ky = np.zeros(len(ky_filtered) * nb_interleaves, dtype=float)

# ==> 2pi/Nb method
plt.figure(3)
for i in range(nb_interleaves):
    kx_tmp = kx_filtered * np.cos(
        2 * np.pi / nb_interleaves * i
    ) + ky_filtered * np.sin(2 * np.pi / nb_interleaves * i)
    ky_tmp = -kx_filtered * np.sin(
        2 * np.pi / nb_interleaves * i
    ) + ky_filtered * np.cos(2 * np.pi / nb_interleaves * i)
    kx[len(kx_filtered) * i : len(kx_filtered) * (i + 1)] = kx_tmp
    ky[len(ky_filtered) * i : len(ky_filtered) * (i + 1)] = ky_tmp
    plt.plot(kx_tmp, ky_tmp)

plt.title("Final k-space trajectory")
plt.show()

# ==> Golden angle method
"""
plt.figure(4)
for i in range(nb_interleaves):
    kx_tmp = kx_filtered * np.cos(
        2 * 222.4969 * (i-1)
    ) + ky_filtered * np.sin(2 * 222.4969 * (i-1))
    ky_tmp = -kx_filtered * np.sin(
        2 * 222.4969 * (i-1)
    ) + ky_filtered * np.cos(2 * 222.4969 * (i-1))
    kx[len(kx_filtered) * i : len(kx_filtered) * (i + 1)] = kx_tmp
    ky[len(ky_filtered) * i : len(ky_filtered) * (i + 1)] = ky_tmp
    plt.plot(kx_tmp, ky_tmp)

plt.title("Final k-space trajectory - golden angle method")
plt.show()
"""


# ===================================
# PYPULSEQ COMPATIBILITY VERIFICATION
# ===================================
# Using a pypulseq function is usefull to get the SimpleNamespaces that will be then directly added to the .seq file
k_pp = np.array([kx, ky])
grad_pp, slew_pp = traj_to_grad_spiral_v2(k_pp, nb_interleaves)
size_interleaves = grad_pp.shape[1] / nb_interleaves

# Verification of the gradients
grad_filtered = np.array(
    [grad_x_filtered * system.gamma, grad_y_filtered * system.gamma]
)
# Verification of the slew rates
slew_filtered = np.array(
    [slew_x_filtered * system.gamma, slew_y_filtered * system.gamma]
)

plt.figure(5)
for i in range(nb_interleaves):
    plt.subplot(221)
    plt.title("grad_pp")
    plt.plot(
        grad_pp[0, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
        grad_pp[1, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
    )
    plt.subplot(223)
    plt.title("slew_pp")
    plt.plot(
        slew_pp[0, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
        slew_pp[1, int(i * size_interleaves) : int((i + 1) * size_interleaves)],
    )

plt.subplot(222)
plt.title("grad_filtered")
plt.plot(grad_filtered[0], grad_filtered[1])

plt.subplot(224)
plt.title("slew_filtered")
plt.plot(slew_filtered[0], slew_filtered[1])

plt.show()


# ================
# SEQUENCE EVENTS
# ================

# Create fat-sat pulse (rf_fs) and associated spoiler gradient (gz_fs)
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * b0 * system.gamma
rf_fs = pp.make_gauss_pulse(
    flip_angle=110 * np.pi / 180,
    system=system,
    duration=8e-3,
    bandwidth=abs(sat_freq),
    freq_offset=sat_freq,
)
gz_fs = pp.make_trapezoid(
    channel="z", system=system, delay=pp.calc_duration(rf_fs), area=1 / 1e-4
)  # spoil up to 0.1mm

# Create 90 degree slice selection pulse and associated gradients
rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=3e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    return_gz=True,
)

# Calculate ADC

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
    # dwell=adc_dwell,
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

# We need the rise time for rewinding gradient calculation
min_rise_time = system.max_grad / system.max_slew

# We also need to calculate S being the maximum allowed rotatable slew rate defined as:
S = 1 / np.sqrt(2) * max_grad / min_rise_time

for i in range(n_slices):
    for j in range(nb_interleaves):
        # Fat-sat
        # seq.add_block(rf_fs, gz_fs)
        # RF
        seq.add_block(rf, gz)
        rf.freq_offset = gz.amplitude * slice_thickness * (i - (n_slices - 1) / 2)
        # Readout gradients
        waveform_gx = grad_pp[
            0,
            int((j * adc_samples_per_segment) + j) : int(
                ((j + 1) * adc_samples_per_segment) + j
            ),
        ]
        waveform_gy = grad_pp[
            1,
            int((j * adc_samples_per_segment) + j) : int(
                ((j + 1) * adc_samples_per_segment) + j
            ),
        ]

        # Prior to designing the rewinders, the spiral gradients are rotated so that y gradient ends with 0 amplitude
        theta = -(np.arctan(waveform_gy[-1] / waveform_gx[-1]))
        rot_matrice = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        spiral = np.array([waveform_gx, waveform_gy])
        rot_spiral = np.array(np.dot(rot_matrice, spiral))
        rot_grad_x = rot_spiral[0, :]
        rot_grad_y = rot_spiral[1, :]

        gx = make_spiral_grad(
            channel="x",
            waveform=rot_grad_x,
            times=new_time_vector,
            delay=pp.calc_duration(gz_reph),
        )
        gy = make_spiral_grad(
            channel="y",
            waveform=rot_grad_y,
            times=new_time_vector,
            delay=pp.calc_duration(gz_reph),
        )
        seq.add_block(gx, gy, adc)

        #####################################################################################
        # todo: Add a latest step for a smooth rewinding
        # MO and M1 calculation
        M0_x, M1_x, M0_y, M1_y = 0, 0, 0, 0
        for i in range(len(rot_grad_x) - 1):
            M0_x += (
                1
                / 2
                * (new_time_vector[i + 1] - new_time_vector[i])
                * (rot_grad_x[i] + rot_grad_x[i + 1])
            )
            M0_y += (
                1
                / 2
                * (new_time_vector[i + 1] - new_time_vector[i])
                * (rot_grad_y[i] + rot_grad_y[i + 1])
            )
            M1_x += (
                1
                / 2
                * (new_time_vector[i + 1] - new_time_vector[i])
                * (
                    new_time_vector[i] * rot_grad_x[i]
                    + new_time_vector[i + 1] * rot_grad_x[i + 1]
                )
            )
            M1_y += (
                1
                / 2
                * (new_time_vector[i + 1] - new_time_vector[i])
                * (
                    new_time_vector[i] * rot_grad_y[i]
                    + new_time_vector[i + 1] * rot_grad_y[i + 1]
                )
            )

        # Solve the moment-nulling problem for the x axis:
        tau_x1 = (abs(rot_grad_x[-1]) / system.gamma) / S

        # This solution should be better but is not
        # st1 = timeit.default_timer()
        # mp.dps = 15  # decimal precision
        #
        # tau_x2 = Symbol('x2')
        # tau_x3 = Symbol('x3')
        # equ1_x = (tau_x1 ** 2 / 2 - tau_x2 ** 2 + tau_x3 ** 2) - (M0_x/S)
        # equ2_x = (tau_x1 ** 3 / 6 - tau_x2 ** 2 * (tau_x1 + tau_x2) + tau_x3 ** 2 * (tau_x1 + 2 * tau_x2 + tau_x3)) - (M1_x/S)
        #
        # sol_x = nsolve((equ1_x, equ2_x), (tau_x2, tau_x3), (0.00023, 0.00009))
        # print("The solutions for x axis rewinding are:\n", "tau_x1: ", tau_x1, " tau_x2: ", sol_x[0], " tau_x3: ",
        #       sol_x[1])
        #
        # st2 = timeit.default_timer()
        # print("RUN TIME : {0}".format(st2 - st1))

        # This solution is way faster but may be less robust
        st1 = timeit.default_timer()

        # todo: find the good constraints to find the good solutions for x and y
        def fx(variables):
            (tau_x2, tau_x3) = variables
            eq1 = (tau_x1 ** 2 / 2 - tau_x2 ** 2 + tau_x3 ** 2) + (M0_x / S)
            eq2 = (
                tau_x1 ** 3 / 6
                - tau_x2 ** 2 * (tau_x1 + tau_x2)
                + tau_x3 ** 2 * (tau_x1 + 2 * tau_x2 + tau_x3)
            ) + (M1_x / S)

            return [eq1, eq2]

        sol_x = opt.fsolve(fx, (0.0003, 0.0001))
        print(
            "The solutions for fx are:\ntau_x1: ",
            tau_x1,
            " tau_x2: ",
            sol_x[0],
            " tau_x3: ",
            sol_x[1],
        )

        st2 = timeit.default_timer()
        print("RUN TIME : {0}".format(st2 - st1))

        # Solve the moment-nulling problem for the y axis:

        # st1 = timeit.default_timer()
        #
        # tau_y1 = Symbol('y1')
        # tau_y2 = Symbol('y2')
        # M0_y = (tau_y1 ** 2 - tau_y2 ** 2) * -S
        # M1_y = (tau_y1 ** 3 - (2 * tau_y1 * tau_y2 ** 2) - tau_y2 ** 3) * -S
        # sol_y = nsolve((M0_y, M1_y), (tau_y1, tau_y2), (0.0001, 0.0001))
        # print("The solutions for y axis rewinding are:\n", "tau_y1: ", sol_y[0], " tau_y2: ", sol_y[1])
        # st2 = timeit.default_timer()
        # print("RUN TIME : {0}".format(st2 - st1))

        st1 = timeit.default_timer()

        def fy(variables):
            (tau_y1, tau_y2) = variables
            eq1 = (tau_y1 ** 2 - tau_y2 ** 2) + (M0_y / S)
            eq2 = (tau_y1 ** 3 - (2 * tau_y1 * tau_y2 ** 2) - tau_y2 ** 3) + (M1_y / S)

            return [eq1, eq2]

        sol_y = opt.fsolve(fy, (0.0006, 0.0001))
        print("The solutions for fy are:\ntau_y1: ", sol_y[0], " tau_y2: ", sol_y[1])

        st2 = timeit.default_timer()
        print("RUN TIME : {0}".format(st2 - st1))

        rew_x, rew_y = make_rewinding_grad(
            channel_1="x",
            channel_2="y",
            tau_list_1=np.array([tau_x1, sol_x[0], sol_x[1]], dtype=float),
            tau_list_2=np.array([sol_y[0], sol_y[1]], dtype=float),
            last_grad=(rot_grad_x[-1] / system.gamma),
            rot_slew=S,
            max_slew=max_slew,
        )

        # todo: compare the time of each rewinding and add zeros (amp = 0, time = raster_time) at the end of the shortest
        #############################################################################

        # Spoilers (area should be >>> kmax)
        # gz_spoil = pp.make_trapezoid(
        #     channel="z", system=system, area=1 / fov * matrix_size * 4
        # )
        # gx_spoil = pp.make_extended_trapezoid(
        #     channel="x",
        #     times=[0, pp.calc_duration(gz_spoil)],
        #     amplitudes=[
        #         grad_cur_amp,
        #         0,
        #     ],  # last value is zero so we need to use the one before
        # )
        # gy_spoil = pp.make_extended_trapezoid(
        #     channel="y",
        #     times=[0, pp.calc_duration(gz_spoil)],
        #     amplitudes=[grad_cur_amp, 0],
        # )
        # seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        # Rewinding
        # seq.add_block(rew_x, rew_y)


# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()

if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed! Error listing follows: ")
    print(error_report)


seq.set_definition("FOV", [fov, fov, slice_thickness])
seq.set_definition("Name", "spiral")
# seq.set_definition("MaxAdcSegmentLength", adc_samples_per_segment)

# Output sequence for scanner
seq.write("spiral_v1.seq")

# Plot sequence waveforms
seq.plot()

# Single-function call for trajectory calculation
ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

# Plot k-spaces
plt.figure(6)
plt.subplot(311)
plt.title("entire k-space trajectory")
plt.plot(ktraj)
plt.subplot(312)
plt.title("2D k-space trajectory")
plt.plot(ktraj[0, :], ktraj[1, :], "b")
plt.subplot(313)
plt.title("Samples")
plt.plot(ktraj_adc[0, :], ktraj_adc[1, :], "r.")
plt.show()
