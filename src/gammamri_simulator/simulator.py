"""Simulator of MRI for the GammaMRI project"""

import argparse
import os
import shutil
import timeit
from numbers import Number

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import yaml
from scipy.stats import norm

from gammamri_simulator import FFMPEGwriter, animator


def is_number_list(obj):
    """Check if object is non-empty list of numbers.

    Args:
        obj: object to be checked

    Returns:
        true or false

    """
    return isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], Number)


def load_gradient_from_file(gradient_filename):
    """Read gradient waveform from file and return as list. The file is expected to
    contain a yaml list of the gradient in mT/m, or a field 'grad' holding such a list.

    :param gradient_filename: filename of gradient yaml file.
    :return: Gradient waveform as a list
    """
    with open(gradient_filename, "r") as gradient_file:
        try:
            grad = yaml.safe_load(gradient_file)
        except yaml.YAMLError as exc:
            raise Exception(f"Error reading gradient file {gradient_filename}") from exc
        if "grad" in grad:
            grad = grad["grad"]
        if is_number_list(grad):
            return grad
        else:
            raise Exception(
                f"Error reading gradient file {gradient_filename}. "
                + " File must contain a yaml list of numbers."
            )


def calculate_flip_angle(B1, dur, gyro):
    """Calculate flip angle for given B1 waveform and duration.

    :param B1: vector of B1 amplitudes [uT]
    :param dur: duration of pulse [ms]
    :param gyro:
    :return: Pulse flip angle
    """
    dwell = dur / len(B1)
    FA = 0
    for b in B1:
        FA += 360 * (dwell * gyro * np.real(b) * 1e-6)
    return FA


def rf_from_struct(rf_struct):
    """Read RF pulse from struct and return as array.

    :param rf_struct: list of the RF amplitude, or a struct with key 'amp' and
        optionally 'phase', each containing a list of equal length.
        amp is the RF amplitude [uT], and 'phase' is RF phase modulation [degrees].
    :return: RF pulse as a (possibly complex) numpy array
    """
    if is_number_list(rf_struct):
        B1 = np.array(rf_struct)
    elif "amp" in rf_struct and is_number_list(rf_struct["amp"]):
        B1 = np.array(rf_struct["amp"])
        if "phase" in rf_struct:
            if not is_number_list(rf_struct["phase"]):
                raise Exception("'phase' of RF struct must be numerical list")
            elif len(rf_struct["phase"]) != len(B1):
                raise Exception("'amp' and 'phase' of RF struct must have equal length")
            B1 = B1 * np.exp(1j * np.radians(rf_struct["phase"]))
    else:
        raise Exception("Unknown format of RF struct")
    return B1


def load_rf_from_file(rf_filename):
    """Read RF pulse from file and return as array. The file is expected to contain a
    yaml list of the RF amplitude, or a list containing two lists, where the second
    holds the RF phase in degrees.

    :param rf_filename: filename of RF yaml file.
    :return: RF pulse as a numpy array (complex if phase was given)
    """
    with open(rf_filename, "r") as rf_file:
        try:
            rf_struct = yaml.safe_load(rf_file)
        except yaml.YAMLError as exc:
            raise Exception(f"Error reading RF file {rf_filename}") from exc
    return rf_from_struct(rf_struct)


def get_rotation_matrix(angle, axis):
    """Get 3D rotation matrix.

    :param angle: rotation angle in radians.
    :param axis: axis of rotation (0, 1, or 2).
    :return: rotation matrix, numpy array of size [3, 3]
    """
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
    return np.roll(np.roll(rot, axis, axis=0), axis, axis=1)


def spherical2cartesian(spherical):
    """Convert 3D vector from spherical to Cartesian coordinates.

    :param spherical: 3-tuple holding vector length, polar, and azimuthal angle
    :return: Cartesian vector, list of size 3
    """
    length, polar, azim = spherical
    mat = np.dot(
        np.dot(np.array([0, 0, length]), get_rotation_matrix(np.radians(azim), 1)),
        get_rotation_matrix(np.radians(polar), 2),
    )
    return list(mat)


def round_event_time(event_time: float, precision: int = 6) -> float:
    """

    :param event_time:
    :param precision:
    :return:
    """
    return np.round(event_time, decimals=precision)


def merge_event(event, event2merge, event_t):
    """Merge events by adding w1, Gx, Gy, Gz, phase and updating event texts.
    Also update event time t.

    :param event: original event
    :param event2merge: event to be merged
    :param event_t: new event time
    :return: Merged event
    """

    for channel in ["w1", "Gx", "Gy", "Gz", "phase"]:
        if channel in event2merge:
            event[channel] += event2merge[channel]
    for text in ["RFtext", "Gxtext", "Gytext", "Gztext", "spoilText"]:
        if text in event2merge:
            event[text] = event2merge[text]
    if "spoil" in event2merge:
        event["spoil"] = True
    else:
        event["spoil"] = False
    event["t"] = event_t
    return event


def detach_event(event, event2detach, event_t):
    """Detach events by subtracting w1, Gx, Gy, Gz, phase and removing event texts.
    Also update event time t.

    :param event: original event
    :param event2detach: event to be detached
    :param event_t: new event time
    :return: detached event
    """

    for channel in ["w1", "Gx", "Gy", "Gz", "phase"]:
        if channel in event2detach:
            event[channel] -= event2detach[channel]
    for text in ["RFtext", "Gxtext", "Gytext", "Gztext", "spoilText"]:
        if text in event and text in event2detach and event[text] == event2detach[text]:
            del event[text]
    event["t"] = event_t
    return event


def spoil(magnetization_vector):
    """Spoil the transversal magnetization in magnetization vector.

    :param magnetization_vector: magnetization vector, numpy array of size 3.
    :return: spoiled magnetization vector, numpy array of size 3.
    """
    return np.array([0, 0, magnetization_vector[2]])


def empty_event():
    """Creates empty pulse sequence event.

    :return: "empty" pulse sequence event
    """
    return {"w1": 0, "Gx": 0, "Gy": 0, "Gz": 0, "phase": 0, "spoil": False}


def add_events_to_time_vector(time_vector, pulse_sequence):
    """Read event times from pulse_sequence struct and add to input time vector.

    :param time_vector: input time vector [ms]
    :param pulse_sequence: pulse sequence struct of events with event times [ms]
    :return: Array of unique sorted set of times in input time vector and event times.
    """
    time_list = list(time_vector)
    for event in pulse_sequence:
        time_list.append(event["t"])
    return np.unique(round_event_time(np.array(time_list)))


def get_prescribed_time_vector(config, n_tr, animation: bool = False):
    """Get time vector of animations prescribed by 'speed', 'TR', 'fps',
    and 'maxRFspeed' in config.

    :param config: configuration dictionary
    :param n_tr: number of TR:s in time vector
    :param animation: animation mode
    :return: Time vector prescribed by config
    """

    speed_events = config["speed"] + [
        event for event in config["pulseSeq"] if any(["FA" in event, "B1" in event])
    ]
    speed_events = sorted(speed_events, key=lambda event: event["t"])

    kernel_time = np.array([])
    time = 0
    if animation:  # Animation mode
        time_resolution = 1e3  # TODO convert from ms to s
        # time_resolution = float(config["timestep"])
        fps = float(config["fps"])
        speed = float(config["speed"][0]["speed"])

    else:  # Simulation (default) mode
        # time_resolution = 1  # e-3  # 1e3 # TODO read from config
        time_resolution = float(config["timestep"]) * 1e3  # ms
        # time_resolution = 1e-3  # ms
        fps = 1
        speed = 1

    delta_time = time_resolution / fps * speed  # delta time in ms

    for event in speed_events:
        kernel_time = np.concatenate(
            (kernel_time, np.arange(time, event["t"], delta_time)), axis=None
        )
        time = max(time, event["t"])
        if "speed" in event and animation:  # Update delta time only in animation mode
            delta_time = (
                time_resolution / config["fps"] * event["speed"]
            )  # Animation time resolution [msec]
        if "FA" in event or "B1" in event:
            rf_delta_time = delta_time
            if animation:
                rf_delta_time = min(
                    delta_time, time_resolution / config["fps"] * config["maxRFspeed"]
                )  # Time resolution during RF [msec]
            kernel_time = np.concatenate(
                (
                    kernel_time,
                    np.arange(event["t"], event["t"] + event["dur"], rf_delta_time),
                ),
                axis=None,
            )
            time = event["t"] + event["dur"]
    kernel_time = np.concatenate(
        (kernel_time, np.arange(time, config["TR"], delta_time)), axis=None
    )

    time_vector = np.array([])
    for rep in range(n_tr):  # Repeat time vector for each TR
        time_vector = np.concatenate(
            (time_vector, kernel_time + rep * config["TR"]), axis=None
        )
    return np.unique(round_event_time(time_vector))


def arrange_locations(slices, config, key="locations"):
    """Check and setup locations or M0. Set nx, ny, and nz and store in config.

    :param slices: (nested) list of M0 or locations (spatial distribution of Meq).
    :param config: configuration dictionary.
    :param key: pass 'locations' for Meq distribution, and 'M0' for M0 distribution.
    :return:
    """

    if key not in ["M0", "locations"]:
        raise Exception(
            'Argument "key" must be "locations" or "M0", not {}'.format(key)
        )
    if not isinstance(slices, list):
        raise Exception(
            'Expected list in config "{}", not {}'.format(key, type(slices))
        )
    if not isinstance(slices[0], list):
        slices = [slices]
    if not isinstance(slices[0][0], list):
        slices = [slices]
    if key == "M0" and not isinstance(slices[0][0][0], list):
        slices = [slices]
    if "nz" not in config:
        config["nz"] = len(slices)
    elif len(slices) != config["nz"]:
        raise Exception('Config "{}": number of slices do not match'.format(key))
    if "ny" not in config:
        config["ny"] = len(slices[0])
    elif len(slices[0]) != config["ny"]:
        raise Exception('Config "{}": number of rows do not match'.format(key))
    if "nx" not in config:
        config["nx"] = len(slices[0][0])
    elif len(slices[0][0]) != config["nx"]:
        raise Exception('Config "{}": number of elements do not match'.format(key))
    if key == "M0" and len(slices[0][0][0]) != 3:
        raise Exception('Config "{}": inner dimension must be of length 3'.format(key))
    return slices


def derivs(M, t, Meq, w, w1, T1, T2):
    """Bloch equations in rotating frame.

    :param M: magnetization vector.
    :param t: time vector (needed for scipy.integrate.odeint).
    :param Meq: equilibrium magnetization.
    :param w: Larmor frequency :math:`2\\pi\\gamma B_0` [kRad / s].
    :param w1: (complex) B1 rotation frequency :math:`2\\pi\\gamma B_1`  [kRad / s].
    :param T1: longitudinal relaxation time.
    :param T2: transverse relaxation time.
    :return: integrand :math:`\\frac{dM}{dt}`
    """
    dMdt = np.zeros_like(M)
    dMdt[0] = -M[0] / T2 + M[1] * w + M[2] * w1.real
    dMdt[1] = -M[0] * w - M[1] / T2 + M[2] * w1.imag
    dMdt[2] = -M[0] * w1.real - M[1] * w1.imag + (Meq - M[2]) / T1
    return dMdt


def derivs_ivp(t, M, Meq, w, w1, T1, T2):  # Callable fun for solve_ivp
    """Bloch equations in rotating frame.

    :param M: magnetization vector.
    :param t: time vector (needed for scipy.integrate.odeint).
    :param Meq: equilibrium magnetization.
    :param w: Larmor frequency :math:`2\\pi\\gamma B_0` [kRad / s].
    :param w1: (complex) B1 rotation frequency :math:`2\\pi\\gamma B_1`  [kRad / s].
    :param T1: longitudinal relaxation time.
    :param T2: transverse relaxation time.
    :return: integrand :math:`\\frac{dM}{dt}`
    """
    return derivs(M, t, Meq, w, w1, T1, T2)


def get_event_frames(config, event_index):
    """Get first and last frame of event i in config['events'] in terms of config['t']

    :param config: configuration dictionary.
    :param event_index: event index
    :return first_frame: index of first frame in terms of config['t']
    :return last_frame: index of last frame in terms of config['t']
    """
    try:
        first_frame = np.where(config["t"] == config["events"][event_index]["t"])[0][0]
    except IndexError:
        print("Event time not found in time vector")
        raise

    if event_index < len(config["events"]) - 1:
        next_event_time = config["events"][event_index + 1]["t"]
    else:
        next_event_time = config["TR"]
    try:
        last_frame = np.where(config["t"] == next_event_time)[0][0]
    except IndexError:
        print("Event time not found in time vector")
        raise
    return first_frame, last_frame


def apply_pulse_sequence(config, Meq, M0, w, T1, T2, pos0, v, D):
    """Simulate magnetization vector during nTR (+nDummies) applications of pulse seq.

    :param config: configuration dictionary.
    :param Meq: equilibrium magnetization along z axis.
    :param M0: initial state of magnetization vector, numpy array of size 3.
    :param w: Larmor frequency :math:`2\\pi\\gamma B_0` [kRad/s].
    :param T1: longitudinal relaxation time.
    :param T2: transverse relaxation time.
    :param pos0: position (x,y,z) of magnetization vector at t=0 [m].
    :param v: velocity (x,y,z) of spins [mm/s]
    :param D: diffusivity (x,y,z) of spins [:math:`mm^2/s`]
    :return: magnetization vector over time, numpy array of size [7, nFrames].
             1:3 are magnetization, 4:6 are position, 7 is larmor freq with Gs.
    """
    M = np.zeros([len(config["t"]), 3])
    M[0] = M0  # Initial state

    W = np.zeros([len(config["t"]), 1])  # larmor freq with gradients
    W[0] = w  # Initial state

    gyro = float(config["gyro"])

    pos = np.tile(pos0, [len(config["t"]), 1])  # initial position
    if np.linalg.norm(D) > 0:  # diffusion contribution
        for frame in range(1, len(config["t"])):
            delta_time = config["t"][frame] - config["t"][frame - 1]
            for dim in range(3):
                pos[frame][dim] = pos[frame - 1][dim] + norm.rvs(
                    scale=np.sqrt(D[dim] * delta_time * 1e-9)
                )  # standard deviation in meters
            if config["t"][frame] == 0:  # reset position for t=0
                pos[: frame + 1] += np.tile(pos0 - pos[frame], [frame + 1, 1])
    if np.linalg.norm(v) > 0:  # velocity contribution
        pos += np.outer(config["t"], v) * 1e-6

    for rep in range(
        -config["nDummies"], config["nTR"]
    ):  # dummy TRs get negative frame numbers
        start_frame = rep * config["nFramesPerTR"]

        for event_index, event in enumerate(config["events"]):
            first_frame, last_frame = get_event_frames(config, event_index)
            first_frame += start_frame
            last_frame += start_frame

            M0 = M[first_frame]

            if "spoil" in event and event["spoil"]:  # Spoiler event
                M0 = spoil(M0)

            # frequency due to w plus any gradients
            # (use position at firstFrame, i.e. approximate no motion during frame)
            wg = w
            wg += (
                2 * np.pi * gyro * event["Gx"] * pos[first_frame, 0] / 1000
            )  # [kRad/s]
            wg += (
                2 * np.pi * gyro * event["Gy"] * pos[first_frame, 1] / 1000
            )  # [kRad/s]
            wg += (
                2 * np.pi * gyro * event["Gz"] * pos[first_frame, 2] / 1000
            )  # [kRad/s]

            w1 = event["w1"] * np.exp(1j * np.radians(event["phase"]))

            t = config["t"][first_frame : last_frame + 1]
            if len(t) == 0:
                raise Exception("Corrupt config['events']")

            use_ODEint = False
            if use_ODEint:
                # Solve Bloch equation with ODEint
                # TODO check numerical integrator parameters
                # https://www.southampton.ac.uk/~fangohr/teaching/python/book/html/16-scipy.html
                M[first_frame : last_frame + 1] = integrate.odeint(
                    derivs, M0, t, args=(Meq, wg, w1, T1, T2)
                )
            else:
                # Solve Bloch equation with solve_ivp
                t_span = (t[0], t[-1])
                ivp_method = "RK45"
                ivp_solution = integrate.solve_ivp(
                    derivs_ivp,
                    t_span,
                    M0,
                    method=ivp_method,
                    t_eval=t,
                    args=(Meq, wg, w1, T1, T2),
                )
                if ivp_solution.success:
                    M[first_frame : last_frame + 1] = ivp_solution.y.transpose()

            W[first_frame : last_frame + 1] = wg

    return np.concatenate((W, M, pos), 1).transpose()


def simulate_component(config, component, Meq, M0=None, pos=None):
    """Simulate nIsochromats magnetization vectors per component with uniform
    distribution of Larmor frequencies.

    Args:
        config: configuration dictionary.
        component:  component specification from config.
        Meq:    equilibrium magnetization along z axis.
        M0:     initial state of magnetization vector, numpy array of size 3.
        pos:   position (x,y,z) of magnetization vector [m].

    Returns:
        component magnetization vectors over time, numpy array of size
        [nIsochromats, 7, nFrames].  1:3 are magnetization, 4:6 are position, 7 is freq with gradient.

    """
    if not M0:
        M0 = [0, 0, Meq]  # Default initial state is equilibrium magnetization
    if not pos:
        pos = [0, 0, 0]
    v = [component["vx"], component["vy"], component["vz"]]
    D = [component["Dx"], component["Dy"], component["Dz"]]
    # Shifts in ppm for dephasing vectors:
    isochromats = [
        (2 * i + 1 - config["nIsochromats"]) / 2 * config["isochromatStep"]
        + component["CS"]
        for i in range(0, config["nIsochromats"])
    ]
    comp = np.empty((config["nIsochromats"], 7, len(config["t"])))

    for m, isochromat in enumerate(isochromats):
        w = config["w0"] * isochromat * 1e-6  # Demodulated frequency [kRad / s]
        comp[m, :, :] = apply_pulse_sequence(
            config, Meq, M0, w, component["T1"], component["T2"], pos, v, D
        )
    return comp


def check_and_set_locations(sequence_config):
    """

    :param sequence_config:
    :return:
    """
    print("check_and_set_locations")

    if not "collapseLocations" in sequence_config:
        sequence_config["collapseLocations"] = False
    if not "locations" in sequence_config:
        sequence_config["locations"] = arrange_locations([[[1]]], sequence_config)
    else:
        if isinstance(sequence_config["locations"], dict):
            for comp in iter(sequence_config["locations"]):
                sequence_config["locations"][comp] = arrange_locations(
                    sequence_config["locations"][comp], sequence_config
                )
        elif isinstance(sequence_config["locations"], list):
            locs = sequence_config["locations"]
            sequence_config["locations"] = {}
            for comp in [n["name"] for n in sequence_config["components"]]:
                sequence_config["locations"][comp] = arrange_locations(
                    locs, sequence_config
                )
        else:
            raise Exception(
                'Sequence config "locations" should be list or components dict'
            )
    for (FOV, n) in [("FOVx", "nx"), ("FOVy", "ny"), ("FOVz", "nz")]:
        if FOV not in sequence_config:
            sequence_config[FOV] = (
                sequence_config[n] * sequence_config["locSpacing"]
            )  # FOV in m
    if "M0" in sequence_config:
        if isinstance(sequence_config["M0"], dict):
            for comp in iter(sequence_config["M0"]):
                sequence_config["M0"][comp] = arrange_locations(
                    sequence_config["M0"][comp], sequence_config, "M0"
                )
        elif isinstance(sequence_config["M0"], list):
            M0 = sequence_config["M0"]
            sequence_config["M0"] = {}
            for comp in [n["name"] for n in sequence_config["components"]]:
                sequence_config["M0"][comp] = arrange_locations(
                    M0, sequence_config, "M0"
                )
        else:
            raise Exception('Sequence config "M0" should be list or components dict')


def check_and_set_animation_outputs(config):
    """

    :param config:
    :return:
    """
    print("check_and_set_outputs")

    # Check animation output
    for output in config["output"]:
        if "tRange" in output:
            if not len(output["tRange"]) == 2:
                raise Exception('Output "tRange" expected to be 2-tuple')
        elif output["type"] == "psd":
            output["tRange"] = [0, config["TR"]]
        else:
            output["tRange"] = [0, config["nTR"] * config["TR"]]
        if "dpi" not in output:
            output["dpi"] = 100
        if "freeze" not in output:
            output["freeze"] = []
        elif not isinstance(output["freeze"], list):
            output["freeze"] = [output["freeze"]]
        if output["type"] == "3D":
            if "drawAxes" not in output:
                output["drawAxes"] = config["nx"] * config["ny"] * config["nz"] == 1
    if "background" not in config:
        config["background"] = {}
    if "color" not in config["background"]:
        config["background"]["color"] = "black"
    if config["background"]["color"] not in ["black", "white"]:
        raise Exception('Only "black" and "white" supported as background colors')


def set_pulse_sequence(config):
    """

    :param config:
    :return:
    """
    print("set_pulse_sequence")

    # Create non-overlapping events, each with constant w1, Gx, Gy, Gz,
    # including empty "relaxation" events
    config["events"] = []
    ongoing_events = []
    new_event = empty_event()  # Start with empty "relaxation event"
    new_event["t"] = 0
    for i, event in enumerate(config["separatedPulseSeq"]):
        event_time = round_event_time(event["t"])

        # Merge any events starting simultaneously:
        if event_time == new_event["t"]:
            new_event = merge_event(new_event, event, event_time)
        else:
            config["events"].append(dict(new_event))
            new_event = merge_event(new_event, event, event_time)

        if "dur" in event:  # event is ongoing unless no 'dur', i.e. spoiler event
            ongoing_events.append(event)

            # sort ongoing events according to event end time:
            sorted(
                ongoing_events,
                key=lambda event: event["t"] + event["dur"],
                reverse=False,
            )

        if event is config["separatedPulseSeq"][-1]:
            next_event_time = round_event_time(config["TR"])
        else:
            next_event_time = round_event_time(config["separatedPulseSeq"][i + 1]["t"])
        for stopping_event in [
            event
            for event in ongoing_events[::-1]
            if round_event_time(event["t"] + event["dur"]) <= next_event_time
        ]:
            config["events"].append(dict(new_event))
            new_event = detach_event(
                new_event,
                stopping_event,
                round_event_time(stopping_event["t"] + stopping_event["dur"]),
            )
            ongoing_events.pop()
    config["events"].append(dict(new_event))

    # Set clock vector
    config["kernelClock"] = get_prescribed_time_vector(config, 1)
    config["kernelClock"] = add_events_to_time_vector(
        config["kernelClock"], config["events"]
    )
    if config["kernelClock"][-1] == config["TR"]:
        config["kernelClock"] = config["kernelClock"][:-1]
    config["nFramesPerTR"] = len(config["kernelClock"])
    config["t"] = np.array([])
    for rep in range(
        -config["nDummies"], config["nTR"]
    ):  # Repeat time vector for each TR (dummy TR:s get negative time)
        config["t"] = np.concatenate(
            (config["t"], round_event_time(config["kernelClock"] + rep * config["TR"])),
            axis=None,
        )
    config["t"] = np.concatenate(
        (config["t"], round_event_time(config["nTR"] * config["TR"])), axis=None
    )  # Add end time to time vector
    config["kernelClock"] = np.concatenate(
        (config["kernelClock"], config["TR"]), axis=None
    )  # Add end time to kernel clock


def check_pulse_sequence(config):
    """

    :param config:
    :return:
    """
    print("check_pulse_sequence")
    gyro = float(config["gyro"])
    if "pulseSeq" not in config:
        config["pulseSeq"] = []
    allowed_keys = ["t", "spoil", "dur", "FA", "B1", "phase", "Gx", "Gy", "Gz"]
    for event in config["pulseSeq"]:
        for item in event.keys():  # allowed keys
            if item not in allowed_keys:
                raise Exception('PulseSeq key "{}" not supported'.format(item))
        if not "t" in event:
            raise Exception('All pulseSeq events must have an event time "t"')
        if not any([key in event for key in ["FA", "B1", "Gx", "Gy", "Gz", "spoil"]]):
            raise Exception("Empty events not allowed")
        if event["t"] > config["TR"]:
            raise Exception("pulseSeq event t exceeds TR")
        if "spoil" in event:  # Spoiler event
            if not event["spoil"]:
                raise Exception("Spoiler event must have spoil: true")
            if any([key not in ["t", "spoil"] for key in event]):
                raise Exception(
                    "Spoiler event should only have event time t and spoil: true"
                )
            event["spoiltext"] = "spoiler"
        else:
            if "dur" not in event:
                raise Exception(
                    "All pulseSeq events except spoiler events must have a duration"
                    + "dur [msec]"
                )
            if round_event_time(event["dur"]) == 0:
                raise Exception("Event duration is too short")
            if (event["t"] + event["dur"]) > config["TR"]:
                raise Exception("pulseSeq event t+dur exceeds TR")
        if "phase" in event:
            if not isinstance(event["phase"], Number):
                raise Exception("Event phase [degrees] must be numeric")
            if not ("FA" in event or "B1" in event):
                raise Exception("Only RF events can have a phase")

        if "FA" in event or "B1" in event:  # RF-pulse event (possibly with gradient)

            # combinations not allowed:
            if "B1" in event and not "dur" in event:
                raise Exception('RF-pulse must provide "dur" along with "B1"')

            if "B1" in event:
                if isinstance(event["B1"], Number):
                    event["B1"] = np.array([event["B1"]])
                elif isinstance(event["B1"], str):
                    event["B1"] = load_rf_from_file(event["B1"])
                else:
                    event["B1"] = rf_from_struct(event["B1"])

            # calculate FA prescribed by B1
            if "B1" in event:
                flip_angle = calculate_flip_angle(event["B1"], event["dur"], gyro)

            # calculate B1 or scale it to get prescribed FA
            if "FA" in event:
                if "B1" not in event:
                    event["B1"] = np.array(
                        [event["FA"] / (float(event["dur"]) * 360 * float(gyro) * 1e-6)]
                    )
                else:
                    event["B1"] = (
                        event["B1"] * event["FA"] / flip_angle
                    )  # scale B1 to get prescribed FA
            else:
                event["FA"] = flip_angle

            event["w1"] = [
                2 * np.pi * gyro * B1 * 1e-6 for B1 in event["B1"]
            ]  # kRad / s
            event["RFtext"] = str(int(abs(event["FA"]))) + "\N{DEGREE SIGN}" + "-pulse"
        if any([key in event for key in ["Gx", "Gy", "Gz"]]):  # Gradient (no RF)
            if not ("dur" in event and event["dur"] > 0):
                raise Exception("Gradient must have a specified duration>0 (dur [ms])")
            for g in ["Gx", "Gy", "Gz"]:
                if g in event:
                    if (
                        isinstance(event[g], dict)
                        and "file" in event[g]
                        and "amp" in event[g]
                    ):
                        grad = load_gradient_from_file(event[g]["file"])
                        # CHANGE TO ALLOW FOR NEGATIVE GRADIENTS
                        # event[g] = list(np.array(grad) / np.max(grad) * event[g]["amp"])
                        event[g] = list(np.array(grad))
                    elif not isinstance(event[g], Number) and not (
                        isinstance(event[g], list) and len(event[g]) > 0
                    ):
                        raise Exception("Unknown type {} for B1".format(type(event[g])))

    # Sort pulseSeq according to event time
    config["pulseSeq"] = sorted(config["pulseSeq"], key=lambda event: event["t"])

    # split any pulseSeq events with array values into separate events
    config["separatedPulseSeq"] = []
    for event in config["pulseSeq"]:
        array_lengths = [
            len(event[key])
            for key in ["w1", "Gx", "Gy", "Gz"]
            if key in event and isinstance(event[key], list)
        ]
        if len(array_lengths) > 0:  # arrays in event
            array_length = np.max(array_lengths)
            if len(set(array_lengths)) == 2 and 1 in set(array_lengths):
                # extend any singleton arrays to full length
                for key in ["w1", "Gx", "Gy", "Gz"]:
                    if (
                        key in event
                        and isinstance(event[key], list)
                        and len(event[key]) == 1
                    ):
                        event[key] *= array_length
            elif len(set(array_lengths)) > 1:
                raise Exception(
                    "If w1, Gx, Gy, Gz of an event are provided as lists, equal "
                    + "length is required"
                )
            for i, t in enumerate(
                np.linspace(
                    event["t"], event["t"] + event["dur"], array_length, endpoint=False
                )
            ):
                sub_duration = event["dur"] / array_length
                sub_event = {"t": t, "dur": sub_duration}
                if i == 0 and spoil in event:
                    sub_event["spoil"] = event["spoil"]
                for key in ["w1", "Gx", "Gy", "Gz", "phase", "RFtext"]:
                    if key in event:
                        if type(event[key]) is list:
                            if i < len(event[key]):
                                sub_event[key] = event[key][i]
                            else:
                                raise Exception(
                                    f"Length of {key} does not match other event"
                                    + " properties"
                                )
                        else:
                            sub_event[key] = event[key]
                        if key in ["Gx", "Gy", "Gz"]:
                            sub_event["{}text".format(key)] = "{}: {:2.0f} mT/m".format(
                                key, sub_event[key]
                            )
                config["separatedPulseSeq"].append(sub_event)
        else:
            for key in ["Gx", "Gy", "Gz"]:
                if key in event:
                    event["{}text".format(key)] = "{}: {:2.0f} mT/m".format(
                        key, event[key]
                    )
            config["separatedPulseSeq"].append(event)

    # Sort separatedPulseSeq according to event time
    config["separatedPulseSeq"] = sorted(
        config["separatedPulseSeq"], key=lambda event: event["t"]
    )


def check_and_set_sequence(
    sequence_config,
    physics_config,
    phantom_config,
    default_sequence_config,
):
    """Check and set the acquisition sequence from its configuration YAML file
    with physical setup and phantom inputs

    :param sequence_config: sequence configuration dictionary read from YAML file
    :param physics_config: physics configuration dictionary read from YAML file
    :param phantom_config: phantom configuration dictionary read from YAML file
    :return:
    """
    print("check_and_set_sequence")

    # Check physics parameters and add them to the sequence
    if any([key not in physics_config for key in ["b_zero", "gyro"]]):
        raise Exception("Physics config must contain 'b_zero' and 'gyro'")
    sequence_config["b_zero"] = physics_config["b_zero"]
    sequence_config["gyro"] = physics_config["gyro"]
    if sequence_config["timestep"] < 0:
        sequence_config["timestep"] = physics_config["timestep"]

    if any([key not in sequence_config for key in ["TR"]]):
        raise Exception('Sequence config must contain "TR"')

    if "title" not in sequence_config:
        sequence_config["title"] = ""

    sequence_config["TR"] = round_event_time(sequence_config["TR"])
    if "nTR" not in sequence_config:
        sequence_config["nTR"] = 1

    if "nDummies" not in sequence_config:
        sequence_config["nDummies"] = 0

    sequence_config["w0"] = (
        2 * np.pi * float(sequence_config["gyro"]) * float(sequence_config["b_zero"])
    )  # Larmor frequency [kRad/s]

    if "nIsochromats" not in sequence_config:
        sequence_config["nIsochromats"] = 1
    if "isochromatStep" not in sequence_config:
        if sequence_config["nIsochromats"] > 1:
            raise Exception('Please specify "isochromatStep" [ppm] in sequence config')
        else:
            sequence_config["isochromatStep"] = 0

    if "components" not in phantom_config:
        sequence_config["components"] = [{}]
    else:
        sequence_config["components"] = phantom_config["components"]
    for comp in sequence_config["components"]:
        for (key, default) in [
            ("name", ""),
            ("CS", 0),
            ("T1", np.inf),
            ("T2", np.inf),
            ("vx", 0),
            ("vy", 0),
            ("vz", 0),
            ("Dx", 0),
            ("Dy", 0),
            ("Dz", 0),
        ]:
            if key not in comp:
                comp[key] = default
    sequence_config["nComps"] = len(sequence_config["components"])

    if "locSpacing" not in physics_config:
        sequence_config["locSpacing"] = default_sequence_config[
            "loc_spacing"
        ]  # distance between locations [m]
    else:
        sequence_config["locSpacing"] = physics_config["loc_spacing"]

    if "locations" in phantom_config:
        sequence_config["locations"] = phantom_config["locations"]

    if "M0" in phantom_config:
        sequence_config["M0"] = phantom_config["M0"]


def check_and_set_animation(
    sequence_config, animation_config, default_animation_config
):
    """

    :param sequence_config:
    :param animation_config:
    :param default_animation_config:
    :return:
    """
    print("check_and_set_animation")

    if not "output" in animation_config:
        raise Exception(
            "At least one output must be specified in the animation configuration when using animation."
        )
    sequence_config["output"] = animation_config["output"]

    # Frames per second in animation
    if "fps" not in sequence_config:
        sequence_config["fps"] = default_animation_config["fps"]
    else:
        sequence_config["fps"] = animation_config["fps"]

    # Leap factor
    if "leap_factor" not in sequence_config:
        sequence_config["leap_factor"] = default_animation_config["leap_factor"]
    else:
        sequence_config["leap_factor"] = animation_config["leap_factor"]

    # Gif writer
    gif_writer = default_animation_config["gif_writer"].lower()
    if gif_writer == "ffmpeg":
        if not shutil.which("ffmpeg"):
            raise Exception("FFMPEG not found")
    else:
        raise Exception(
            "Argument gif_writer must be 'ffmpeg' in the general configuration."
        )
    sequence_config["gif_writer"] = "ffmpeg"

    # check speed prescription
    if "speed" in animation_config:
        sequence_config["speed"] = animation_config["speed"]
    else:
        sequence_config["speed"] = default_animation_config["speed"]
    if isinstance(sequence_config["speed"], Number):
        sequence_config["speed"] = [{"t": 0, "speed": sequence_config["speed"]}]
    elif isinstance(sequence_config["speed"], list):
        for event in sequence_config["speed"]:
            if not ("t" in event and "speed" in event):
                raise Exception(
                    "Each item in 'speed' list must have field 't' [msec] and 'speed'"
                )
            if event["t"] >= sequence_config["TR"]:
                raise Exception("Specified speed change must be within TR.")
        if not 0 in [event["t"] for event in sequence_config["speed"]]:
            raise Exception("Speed at time 0 must be specified.")
    else:
        raise Exception("Animation config 'speed' must be a number or a list")
    sequence_config["speed"] = sorted(
        sequence_config["speed"], key=lambda event: event["t"]
    )
    if "maxRFspeed" not in animation_config:
        sequence_config["maxRFspeed"] = default_animation_config["max_RF_speed"]
    elif not isinstance(sequence_config["maxRFspeed"], Number):
        raise Exception("Animation config 'maxRFspeed' must be numeric")
    else:
        sequence_config["maxRFspeed"] = animation_config["maxRFspeed"]


def check_physics(physics_config):
    """Check the physics configuration

    :param physics_config: physics configuration read from the YAML file
    """
    if any([key not in physics_config for key in ["b_zero", "gyro"]]):
        raise Exception("Physics config must contain 'b_zero' and 'gyro'")


def check_config(config):
    """Check the general configuration

    :param config: general configuration read from the YAML file
    """
    if "title" not in config:
        config["title"] = "sequence"
    if "outdir" not in config:
        config["outdir"] = "out"
    if "verbose" not in config:
        config["verbose"] = True
    if "use_animation" not in config:
        config["use_animation"] = False
    if "parallelized" not in config:
        config["parallelized"] = False
    if any([key not in config for key in ["sequence_default", "animation_default"]]):
        raise Exception(
            "General config must contain 'sequence_default' and 'animation_default'"
        )
    sequence_default = {
        k: v for element in config["sequence_default"] for k, v in element.items()
    }
    if any([key not in sequence_default for key in ["loc_spacing"]]):
        raise Exception("General config 'sequence_default' must contain 'loc_spacing'")
    animation_default = {
        k: v for element in config["animation_default"] for k, v in element.items()
    }
    if any(
        [
            key not in animation_default
            for key in ["fps", "max_RF_speed", "dpi", "leap_factor"]
        ]
    ):
        raise Exception(
            "General config 'animation_default' must contain 'fps', 'max_RF_speed',"
            + " 'dpi' and 'leap_factor'"
        )


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def simulate(config, vectors):
    """

    :param config:
    :param vectors:
    :return:
    """

    n_iter = config["nz"] * config["ny"] * config["nx"] * len(config["components"])
    i_iter = 0
    printProgressBar(0, n_iter, prefix="Progress:", suffix="Complete", length=100)

    # TODO parallelize
    for z in range(config["nz"]):
        for y in range(config["ny"]):
            for x in range(config["nx"]):
                for c, component in enumerate(config["components"]):
                    if component["name"] in config["locations"]:
                        try:
                            Meq = config["locations"][component["name"]][z][y][x]
                        except:
                            raise Exception(
                                "Is the 'location' matrix shape equal for all"
                                + " components?"
                            )
                    elif isinstance(config["locations"], list):
                        Meq = config["locations"][z][y][x]
                    else:
                        Meq = 0.0
                    if "M0" in config and component["name"] in config["M0"]:
                        try:
                            M0 = spherical2cartesian(
                                config["M0"][component["name"]][z][y][x]
                            )
                        except:
                            raise Exception(
                                "Is the 'M0' matrix shape equal for all components?"
                            )
                    elif "M0" in config and isinstance(config["M0"], list):
                        M0 = spherical2cartesian(config["M0"][z][y][x])
                    else:
                        M0 = None
                    pos = [
                        (x + 0.5 - config["nx"] / 2) * config["locSpacing"],
                        (y + 0.5 - config["ny"] / 2) * config["locSpacing"],
                        (z + 0.5 - config["nz"] / 2) * config["locSpacing"],
                    ]
                    vectors[x, y, z, c, :, :, :] = simulate_component(
                        config, component, Meq, M0, pos
                    )

                    i_iter += 1
                    printProgressBar(
                        i_iter,
                        n_iter,
                        prefix="Progress:",
                        suffix="Complete",
                        length=100,
                    )


# PARALLEL
import itertools
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count


def simulate_parallel(config, vectors):
    """

    :param config:
    :param vectors:
    :return:
    """

    z = range(config["nz"])
    y = range(config["ny"])
    x = range(config["nx"])
    c = range(len(config["components"]))

    paramlist = list(itertools.product(z, y, x, c))

    def func(params):
        z = params[0]
        y = params[1]
        x = params[2]
        c = params[3]
        # return numpy.zeros((1,3,3))

        #
        component = config["components"][c]
        if component["name"] in config["locations"]:
            try:
                Meq = config["locations"][component["name"]][z][y][x]
            except:
                raise Exception(
                    "Is the 'location' matrix shape equal for all" + " components?"
                )
        elif isinstance(config["locations"], list):
            Meq = config["locations"][z][y][x]
        else:
            Meq = 0.0
        if "M0" in config and component["name"] in config["M0"]:
            try:
                M0 = spherical2cartesian(config["M0"][component["name"]][z][y][x])
            except:
                raise Exception("Is the 'M0' matrix shape equal for all components?")
        elif "M0" in config and isinstance(config["M0"], list):
            M0 = spherical2cartesian(config["M0"][z][y][x])
        else:
            M0 = None
        pos = [
            (x + 0.5 - config["nx"] / 2) * config["locSpacing"],
            (y + 0.5 - config["ny"] / 2) * config["locSpacing"],
            (z + 0.5 - config["nz"] / 2) * config["locSpacing"],
        ]
        return simulate_component(config, component, Meq, M0, pos)

    # Generate processes equal to the number of cores
    # njobs = multiprocessing.cpu_count()
    pool = Pool()  # nodes=8)

    njobs = cpu_count()
    print(f"Simulate parallel on the {njobs} cores...")

    # Distribute the parameter sets evenly across the cores
    res = pool.map(func, paramlist)

    n_iter = config["nz"] * config["ny"] * config["nx"] * len(config["components"])
    i_iter = 0
    printProgressBar(0, n_iter, prefix="Progress:", suffix="Complete", length=100)

    for params, sim_res in zip(paramlist, res):
        z = params[0]
        y = params[1]
        x = params[2]
        c = params[3]

        vectors[x, y, z, c, :, :, :] = sim_res

        i_iter += 1
        printProgressBar(
            i_iter,
            n_iter,
            prefix="Progress:",
            suffix="Complete",
            length=100,
        )


def animate(config, vectors):
    """Animate the computed simulation

    :param config:
    :param vectors:
    :return:
    """
    animator.get_text(config)  # prepare text flashes for 3D plot

    vectors = animator.resample_time_frames(vectors, config, True)  # Animation mode
    animator.fade_text_flashes(config)
    delay = int(
        100 / config["fps"] * config["leap_factor"]
    )  # Delay between frames in ticks of 1/100 sec

    output_directory = config["output_directory"]
    for output in config["output"]:
        if output["file"]:
            if output["type"] in ["xy", "z"]:
                signal = np.sum(
                    vectors[:, :, :, :, :, :3, :], (0, 1, 2, 4)
                )  # sum over space and isochromats
                if "normalize" in output and output["normalize"]:
                    for c, comp in enumerate([n["name"] for n in config["components"]]):
                        signal[c, :] /= np.sum(config["locations"][comp])
                signal /= np.max(np.abs(signal))  # scale signal relative to maximum
                if "scale" in output:
                    signal *= output["scale"]
            ffmpeg_writer = FFMPEGwriter.FFMPEGwriter(config["fps"])
            os.makedirs(output_directory, exist_ok=True)
            outfile = os.path.join(output_directory, output["file"])

            output["freezeFrames"] = []
            for t in output["freeze"]:
                output["freezeFrames"].append(np.argmin(np.abs(config["tFrames"] - t)))
            for frame in range(0, len(config["tFrames"]), config["leap_factor"]):
                # Use only every leapFactor frame in animation
                if output["type"] == "3D":
                    fig = animator.plot_frame_3D(config, vectors, frame, output)
                elif output["type"] == "kspace":
                    fig = animator.plot_frame_kspace(config, frame, output)
                elif output["type"] == "psd":
                    fig = animator.plot_frame_psd(config, frame, output)
                elif output["type"] in ["xy", "z"]:
                    fig = animator.plot_frame_mt(config, signal, frame, output)
                plt.draw()

                files_to_save = []
                if frame in output["freezeFrames"]:
                    files_to_save.append(
                        "{}_{}.png".format(
                            ".".join(outfile.split(".")[:-1]), str(frame).zfill(4)
                        )
                    )

                ffmpeg_writer.add_frame(fig)

                for file in files_to_save:
                    print(
                        'Saving frame {}/{} as "{}"'.format(
                            frame + 1, len(config["tFrames"]), file
                        )
                    )
                    plt.savefig(file, facecolor=plt.gcf().get_facecolor())

                plt.close()

            ffmpeg_writer.write(outfile)


def run(
    output_directory: str,
    general_configuration_filename: str,
    physics_configuration_filename: str,
    sequence_configuration_filename: str,
    phantom_configuration_filename: str,
    animation_configuration_filename: str = "",
    timestep: int = -1,
):
    """Run the MRI simulator for the given configurations.

    :param output_directory:
    :param general_configuration_filename:
    :param physics_configuration_filename:
    :param sequence_configuration_filename:
    :param phantom_configuration_filename:
    :param animation_configuration_filename:
    :param timestep:
    :return: magnetizations matrix [x, y , z, t, [sx, sy, sz], wg]
    """
    print("Run GammaMRI-Simulator\n")

    # Read general configuration file
    print("Read general configuration...")
    with open(general_configuration_filename, "r") as general_configuration_file:
        try:
            general_config = yaml.safe_load(general_configuration_file)
        except yaml.YAMLError as exc:
            raise Exception(
                "Error reading the general configuration file: "
                + f"{general_configuration_filename}"
            ) from exc
    check_config(general_config)

    # Check if verbose mode
    verbose = general_config["verbose"]
    if verbose:
        print("\tverbose mode")

    # Check if animation is used
    # use_animation = general_config["use_animation"]
    use_animation = animation_configuration_filename != ""
    if verbose and use_animation:
        print("\tuse animation")

    # Read default sequence values
    default_sequence = {
        key: value
        for element in general_config["sequence_default"]
        for key, value in element.items()
    }

    # Read default animation values
    default_animation = {
        key: value
        for element in general_config["animation_default"]
        for key, value in element.items()
    }
    print("done.\n")

    # Read physics configuration file
    print("Read physics configuration...")
    with open(physics_configuration_filename, "r") as physics_configuration_file:
        try:
            physics_config = yaml.safe_load(physics_configuration_file)
        except yaml.YAMLError as exc:
            raise Exception(
                "Error reading the physics configuration file: "
                + f"{physics_configuration_filename}"
            ) from exc
    check_physics(physics_config)
    if verbose:
        print(
            f"\tB0 = {physics_config['b_zero']} [T];\n\tGyromagnetic ratio = {physics_config['gyro']} [kHz/T]"
        )
    print("done.\n")

    # Read sequence configuration file
    print("Read sequence configuration...")
    with open(sequence_configuration_filename, "r") as sequence_configuration_file:
        try:
            sequence_config = yaml.safe_load(sequence_configuration_file)
        except yaml.YAMLError as exc:
            raise Exception(
                "Error reading the sequence configuration file: "
                + f"{sequence_configuration_filename}"
            ) from exc

    print("done.\n")

    # Read phantom configuration file
    print("Read phantom configuration...")
    with open(phantom_configuration_filename, "r") as phantom_configuration_file:
        try:
            phantom_config = yaml.safe_load(phantom_configuration_file)
        except yaml.YAMLError as exc:
            raise Exception(
                "Error reading the phantom configuration file: "
                + f"{phantom_configuration_filename}"
            ) from exc

    # Set user timestep if defined
    sequence_config["timestep"] = timestep

    # Check and set the sequence configuration for simulation
    check_and_set_sequence(
        sequence_config, physics_config, phantom_config, default_sequence
    )
    sequence_config["output_directory"] = output_directory
    print("done.\n")

    # With animation
    if use_animation:
        # Read sequence configuration file
        print("Read animation configuration...")
        with open(
            animation_configuration_filename, "r"
        ) as animation_configuration_file:
            try:
                animation_config = yaml.safe_load(animation_configuration_file)
            except yaml.YAMLError as exc:
                raise Exception(
                    "Error reading the animation configuration file: "
                    + f"{animation_configuration_filename}"
                ) from exc

        # Check and set the sequence configuration with animation for simulation
        check_and_set_animation(sequence_config, animation_config, default_animation)

        print("done.\n")
    else:
        # TODO remove the need to specify speed if no animation required
        sequence_config["speed"] = default_animation["speed"]

    # Pulse sequence
    check_pulse_sequence(sequence_config)
    set_pulse_sequence(sequence_config)

    # Locations
    check_and_set_locations(sequence_config)

    # Animation output
    if use_animation:
        check_and_set_animation_outputs(sequence_config)

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    sequence_config["output_directory"] = output_directory

    # Output vectors
    vectors = np.empty(
        (
            sequence_config["nx"],
            sequence_config["ny"],
            sequence_config["nz"],
            sequence_config["nComps"],
            sequence_config["nIsochromats"],
            7,
            len(sequence_config["t"]),
        ),
        dtype="float32",
    )
    print(vectors.shape)

    magnetizations = np.empty(
        (
            sequence_config["nx"],
            sequence_config["ny"],
            sequence_config["nz"],
            len(sequence_config["t"]),
            4,
        ),
        dtype="float32",
    )
    print(magnetizations.shape)

    print(f"Time length = {len(sequence_config['t'])}")

    # Simulate
    print("\nStart simulation...")
    simulation_start_time = timeit.default_timer()
    if general_config["parallelized"]:
        simulate_parallel(sequence_config, vectors)
    else:
        simulate(sequence_config, vectors)
    simulation_time = timeit.default_timer() - simulation_start_time
    print(f"\tsimulation took {np.round(simulation_time, decimals=6)} [s] to compute.")
    print("done.")

    # Compile results in magnetization vector

    # Sum iso
    temp_vector = np.sum(vectors, axis=4)

    # Sum comp
    temp_vector = np.sum(temp_vector, axis=3)

    # Change t position in vector and keep only Spin+wg and not Localisation information
    for t in range(vectors.shape[-1]):
        magnetizations[:, :, :, t, :] = temp_vector[:, :, :, 0:4, t]

    # Save simulation results as magnetizazion vector file
    output_vector_npy_filename: str = os.path.join(
        output_directory, "magnetizations.npy"
    )
    np.save(output_vector_npy_filename, magnetizations)  # .npy

    # Save whole vectors
    # output_mvector_npy_filename: str = os.path.join(output_directory, "vectors.npy")
    # np.save(output_mvector_npy_filename, vectors)  # .npy

    # Save magnetizations as raw file + header
    output_mvector_header_filename: str = os.path.join(
        output_directory, "magnetizations.hdr"
    )
    with open(output_mvector_header_filename, "w") as header_file:
        header_file.write(f"dtype={magnetizations.dtype}\n")
        header_file.write(f"shape={magnetizations.shape}\n")

    output_mvector_raw_filename: str = os.path.join(
        output_directory, "magnetizations.raw"
    )
    magnetizations.astype("float32").tofile(output_mvector_raw_filename)

    # Animate
    if use_animation:

        vectors_animation = np.empty(
            (
                sequence_config["nx"],
                sequence_config["ny"],
                sequence_config["nz"],
                sequence_config["nComps"],
                sequence_config["nIsochromats"],
                6,
                len(sequence_config["t"]),
            )
        )
        # print(f"Vector shape = {vectors_animation.shape}")

        # Change t position in vector and keep only Spin+wg and not Localisation information
        # vectors_animation = np.copy(vectors[:, :, :, :, : 1:7, :])
        for t in range(vectors.shape[-1]):
            vectors_animation[:, :, :, :, :, :, t] = vectors[:, :, :, :, :, 1:7, t]

        # print(f"Vector shape = {vectors.shape}")
        # print(f"Vector shape = {vectors_animation.shape}")
        output_avector_npy_filename: str = os.path.join(
            output_directory, "animation_vectors.npy"
        )
        np.save(output_avector_npy_filename, vectors_animation)  # .npy

        # Use agg for matplotlib
        matplotlib.use("agg")

        print("\nStart animation...")
        animation_start_time = timeit.default_timer()
        animate(sequence_config, vectors_animation)
        animation_time = timeit.default_timer() - animation_start_time
        print(
            f"\tanimation took {np.round(animation_time, decimals=6)} [s] to compute."
        )
        print("done.")

    return magnetizations


def parse_and_run():
    """Command line parser. Parse command line and run main program."""

    # Initiate command line parser
    parser = argparse.ArgumentParser(
        description="Simulate magnetization vectors using Bloch equations"
    )
    parser.add_argument(
        "--sequence",
        "-s",
        help="Name of the sequence configuration yaml file",
        type=str,
        default="",
        required=True,
    )
    parser.add_argument(
        "--animation",
        "-a",
        help="Name of the animation configuration yaml file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--physics",
        "-p",
        help="Name of the physics configuration yaml file",
        type=str,
        default="",
        required=True,
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Name of the misc configuration yaml file",
        type=str,
        default="",
        required=True,
    )

    # Parse command line
    args = parser.parse_args()

    # Change dir to main GammaMRI-Simulator
    os.chdir("..")
    print(os.getcwd())

    # Run main program
    run(args.sequence, args.physics, args.config, args.animation)


if __name__ == "__main__":
    parse_and_run()
