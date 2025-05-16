"""Class to generate a sequence from the supported types

sequence_type supported: gre

date: 2021-07-21
author: cc
"""

import os

import numpy as np
import pypulseq as pp
import yaml
from matplotlib import pyplot as plt

from gammamri_seq.pulse_seq import PulseSeq
from gammamri_seq.sequence import Sequence

# from gammamri_seq.spiral_functions import traj_to_grad_spiral, make_spiral_grad
from gammamri_seq.waveforms import (
    RfWaveform,
    TrapGradientWaveform,
    Waveform,
    SpiralGradientWaveform,
)
from gammamri_simulator.physical_setup import PhysicalSetup

from gammamri_seq.traj_to_grad_spiral_v2 import traj_to_grad_spiral_v2
from gammamri_seq.make_spiral_grad import make_spiral_grad


class SequenceGenerator:

    _sequence_types = ["gre", "spiral", "border", "90"]
    PRECISION = 5
    MS_CONVERT = 1e3

    @classmethod
    def generate_sequence(
        self,
        sequence_type: str,
        matrix_shape: tuple,
        # REMOVED FOV AS NOT USED fov,
        title: str,
        outdir: str,
        physics_configuration_filename: str,
        timestep: int = -1,
        duration: float = -1,
        sequence_parameters = None,
    ):
        if not sequence_type in self._sequence_types:
            raise Exception(
                f"Unknown sequence type '{sequence_type}',"
                + "sequence types allowed are: {self.SEQUENCE_TYPE}"
            )
        if sequence_type == "gre":
            return self.generate_gre_sequence(
                matrix_shape,
                title,
                outdir,
                physics_configuration_filename,
                timestep,
                duration,
            )
        elif sequence_type == "spiral":
            return self.generate_spiral_sequence(
                matrix_shape, title, outdir, physics_configuration_filename, timestep,
                duration,
            )
        elif sequence_type == "border":
            return self.generate_border_sequence(
                matrix_shape, title, outdir, physics_configuration_filename, timestep, duration,
            )
        elif sequence_type == "90":
            return self.generate_90_sequence(
                matrix_shape, title, outdir, physics_configuration_filename, timestep, duration, sequence_parameters,
            )

    @classmethod
    def generate_gre_sequence(
        self,
        matrix_shape: tuple,
        title: str,
        outdir: str,
        physics_configuration_filename: str,
        timestep: int = -1,
        duration: float = -1,
    ):
        print("Generate gre sequence")

        fov = 0.03  # 3cm field of view

        sequence_name = f"{title}_gre"
        sequence_pypulseq_filename = f"{sequence_name}.seq"
        sequence_config_filename = f"{sequence_name}.yml"

        # Create waveforms directory if not exists
        waveform_dir = os.path.join(outdir, "waveforms")
        os.makedirs(waveform_dir, exist_ok=True)

        # Physics system
        ps = PhysicalSetup(physics_configuration_filename)

        # Set raster time to timestep to ensure everything is with the same clock
        # TODO handle different raster times for rf and grad
        # raster_time: float = min(ps.grad_raster_time, ps.rf_raster_time)
        raster_time = timestep
        if timestep < 0:
            raster_time = ps.timestep

        # Init the system with the physics configuration
        system = pp.Opts(
            adc_dead_time=ps.adc_dead_time,
            gamma=ps.gamma,
            grad_raster_time=raster_time,
            grad_unit=ps.grad_unit,
            max_grad=ps.max_grad,
            max_slew=ps.max_slew,
            rf_dead_time=ps.rf_dead_time,
            rf_raster_time=raster_time,
            rf_ringdown_time=ps.rf_ringdown_time,
            rise_time=ps.rise_time,
            slew_unit=ps.slew_unit,
        )
        # system = pp.Opts(
        #     max_grad=28,
        #     grad_unit="mT/m",
        #     max_slew=150,
        #     slew_unit="T/m/s",
        #     rf_ringdown_time=20e-6,
        #     rf_dead_time=100e-6,
        #     adc_dead_time=10e-6,
        # )

        # Matrix size
        nz, ny, nx = matrix_shape

        # Matrix size
        nz, ny, nx = matrix_shape

        # Define FOV and resolution
        # fov = 200e-3
        alpha = 10  # alpha=°, flip_angle=rad
        slice_thickness = 3e-3

        # Timing
        # TE = 8e-3  # np.array([4.3e-3])
        # TR = 15e-3

        TE = 4.3e-3  # Echo time
        TR = 10e-3  # Repetition time

        TE = 6e-3  # Echo time
        TR = 15e-3  # Repetition time

        # RF spoiling increment (117 or 123)
        rf_spoiling_inc = 117
        ro_duration = 3.2e-3

        # Create sequence in Pypulseq and GammaMRI format
        seq_pypulseq = pp.Sequence(system=system)
        seq_yaml = Sequence(sequence_name)

        # TODO check TR and nTR exact meaning in simulator format
        seq_yaml.repetition_time = TR * ny
        seq_yaml.number_repetitions = 1

        # RF sinc pulse with slice select and slice select rephasing
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=alpha * np.pi / 180,
            duration=3e-3,  # TODO check ms
            slice_thickness=slice_thickness,
            apodization=0.42,  # 0.5
            time_bw_product=4,
            system=system,
            return_gz=True,
        )  # TBW=2 (rapid imaging), =4 (180), =8 (90), =12 (slab and saturation)
        # https://inst.eecs.berkeley.edu/~ee225e/sp13/notes/Lecture13.pdf

        # TODO pypulseq rf use 1e-6 hard coded for vector length
        rf_hard_code_clock = 1e-6
        rf_correction_ratio = int(raster_time / rf_hard_code_clock)
        rf_corrected_signal = rf.signal[0:-1:rf_correction_ratio].copy()
        rf_corrected_time = rf.t[: len(rf_corrected_signal)].copy()

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        rf.signal = rf_corrected_signal
        rf.t = rf_corrected_time

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        # Convert RF and associated gradient to waveforms of the same size in the format
        # of the GammaMRI-Simulator
        rf_waveform, rf_gz_waveform = Waveform.get_pulse_and_gradient_waveforms(
            rf,
            gz,
            os.path.join(waveform_dir, "rf.yml"),
            raster_time,
            self.PRECISION,
            ps.gamma,
        )

        # Define other gradients and ADC events
        delta_k = 1 / fov

        # Readout gradient
        gx = pp.make_trapezoid(
            channel="x", flat_area=nx * delta_k, flat_time=ro_duration, system=system
        )

        # Convert gradient to waveform in the format of the GammaMRI-Simulator
        gx_waveform = TrapGradientWaveform(
            gx,
            os.path.join(waveform_dir, "gx.yml"),
            raster_time,
            self.PRECISION,
            ps.gamma,
        )

        adc = pp.make_adc(
            num_samples=nx, duration=gx.flat_time, delay=gx.rise_time, system=system
        )  # TODO useful to include in simulation ?

        # pre gradient in x
        gx_pre = pp.make_trapezoid(
            channel="x", area=-gx.area / 2, duration=2e-3, system=system
        )

        # Convert gradient to waveform in the format of the GammaMRI-Simulator
        gx_pre_waveform = TrapGradientWaveform(
            gx_pre,
            os.path.join(waveform_dir, "gx_pre.yml"),
            raster_time,
            self.PRECISION,
            ps.gamma,
        )

        # Rephasing gradient
        gz_reph = pp.make_trapezoid(
            channel="z", area=-gz.area / 2, duration=1e-3, system=system
        )

        # Convert gradient to waveform in the format of the GammaMRI-Simulator
        gz_reph_waveform = TrapGradientWaveform(
            gz_reph,
            os.path.join(waveform_dir, "gz_reph.yml"),
            raster_time,
            self.PRECISION,
            ps.gamma,
        )
        phase_areas = (np.arange(ny) - ny / 2) * delta_k

        # Define gradient spoiling
        gx_spoil = pp.make_trapezoid(channel="x", area=2 * nx * delta_k, system=system)

        # Convert gradient to waveform in the format of the GammaMRI-Simulator
        gx_spoil_waveform = TrapGradientWaveform(
            gx_spoil,
            os.path.join(waveform_dir, "gx_spoil.yml"),
            raster_time,
            self.PRECISION,
            ps.gamma,
        )

        gz_spoil = pp.make_trapezoid(
            channel="z", area=4 / slice_thickness, system=system
        )

        # Convert gradient to waveform in the format of the GammaMRI-Simulator
        gz_spoil_waveform = TrapGradientWaveform(
            gz_spoil,
            os.path.join(waveform_dir, "gz_spoil.yml"),
            raster_time,
            self.PRECISION,
            ps.gamma,
        )

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
                / seq_pypulseq.grad_raster_time
            )
            * seq_pypulseq.grad_raster_time
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
                / seq_pypulseq.grad_raster_time
            )
            * seq_pypulseq.grad_raster_time
        )

        assert np.all(delay_TE >= 0)
        duration_spoil = pp.calc_duration(gx_spoil, gz_spoil)
        assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

        rf_phase = 0
        rf_inc = 0

        # CONSTRUCT SEQUENCE

        # Loop over phase encodes and define sequence blocks
        j = 0
        current_time: float = 0
        for i in range(ny):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            # Pypulseq block
            seq_pypulseq.add_block(rf, gz)

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, rf_waveform.duration())
            pulse_seq.set_flip_angle(alpha)
            pulse_seq.set_b1(rf_waveform.filename)
            pulse_seq.set_gradient_z(rf_gz_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += rf_waveform.duration()

            # Phase encoding gradient
            gy_pre = pp.make_trapezoid(
                channel="y",
                area=phase_areas[i],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )

            # Convert gradient to waveform in the format of the GammaMRI-Simulator
            gy_pre_waveform = TrapGradientWaveform(
                gy_pre,
                os.path.join(waveform_dir, f"gy_pre{i}.yml"),
                raster_time,
                self.PRECISION,
                ps.gamma,
            )

            # Pypulseq block
            seq_pypulseq.add_block(gx_pre, gy_pre, gz_reph)

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, gx_pre_waveform.duration())
            pulse_seq.set_gradient_x(gx_pre_waveform)
            pulse_seq.set_gradient_y(gy_pre_waveform)
            pulse_seq.set_gradient_z(gz_reph_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gx_pre_waveform.duration()

            # Add delay TE
            seq_pypulseq.add_block(pp.make_delay(delay_TE))  # delay_TE[j]))
            current_time += float(delay_TE)

            # Pypulseq block
            seq_pypulseq.add_block(gx, adc)  # ADC is ignored for simulation

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, gx_waveform.duration())
            pulse_seq.set_gradient_x(gx_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gx_waveform.duration()

            gy_pre.amplitude = -gy_pre.amplitude

            # Convert gradient to waveform in the format of the GammaMRI-Simulator
            gy_spoil_waveform = TrapGradientWaveform(
                gy_pre,
                os.path.join(waveform_dir, f"gy_reph{i}.yml"),
                raster_time,
                self.PRECISION,
                ps.gamma,
            )

            # Pypulseq block
            seq_pypulseq.add_block(
                pp.make_delay(delay_TR), gx_spoil, gy_pre, gz_spoil
            )  # delay_TR[j]

            # Yaml equivalent : 3 events because 3 different durations
            pulse_seq = PulseSeq(current_time, gx_spoil_waveform.duration())
            pulse_seq.set_gradient_x(gx_spoil_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            pulse_seq = PulseSeq(current_time, gy_spoil_waveform.duration())
            pulse_seq.set_gradient_y(gy_spoil_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            pulse_seq = PulseSeq(current_time, gz_spoil_waveform.duration())
            pulse_seq.set_gradient_z(gz_spoil_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Add delay to match TR
            current_time += float(delay_TR)

        (
            ok,
            error_report,
        ) = (
            seq_pypulseq.check_timing()
        )  # Check whether the timing of the sequence is correct
        if ok:
            print("Timing check passed successfully")
        else:
            print("Timing check failed. Error listing follows:")
            [print(e) for e in error_report]

        # ======
        # VISUALIZATION
        # ======
        seq_pypulseq.plot()

        # Trajectory calculation and plotting
        (
            ktraj_adc,
            ktraj,
            t_excitation,
            t_refocusing,
            t_adc,
        ) = seq_pypulseq.calculate_kspacePP()
        time_axis = np.arange(1, ktraj.shape[1] + 1) * system.grad_raster_time
        plt.figure()
        plt.plot(time_axis, ktraj.T)  # Plot the entire k-space trajectory
        plt.plot(t_adc, ktraj_adc[0], ".")  # Plot sampling points on the kx-axis
        plt.figure()
        plt.plot(ktraj[0], ktraj[1], "b")  # 2D plot
        plt.axis("equal")  # Enforce aspect ratio for the correct trajectory display
        plt.plot(ktraj_adc[0], ktraj_adc[1], "r.")  # Plot  sampling points
        plt.show()

        # Prepare the sequence output for the scanner
        seq_pypulseq.set_definition("FOV", [fov, fov, slice_thickness])
        seq_pypulseq.set_definition("Name", "gre")

        # Save .seq
        seq_pypulseq.write(os.path.join(outdir, sequence_pypulseq_filename))

        # Save yaml
        with open(os.path.join(outdir, sequence_config_filename), "w") as yaml_file:
            yaml.dump(seq_yaml.dict(), yaml_file, sort_keys=False)

        # Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
        # slew-rate limits
        rep = seq_pypulseq.test_report()
        print(rep)

        return sequence_config_filename

    @classmethod
    def generate_border_sequence(
        cls,
        matrix_shape: tuple,
        title: str,
        outdir: str,
        physics_configuration_filename: str,
        timestep: int = -1,
        duration: float = -1,
    ):
        print("Generate border sequence")

        sequence_name = f"{title}_border"
        sequence_pypulseq_filename = f"{sequence_name}.seq"
        sequence_config_filename = f"{sequence_name}.yml"

        # Create waveforms directory if not exists
        waveform_dir = os.path.join(outdir, "waveforms")
        os.makedirs(waveform_dir, exist_ok=True)

        # Physics system
        ps = PhysicalSetup(physics_configuration_filename)

        # Set raster time to timestep to ensure everything is with the same clock
        # TODO handle different raster times for rf and grad
        # raster_time: float = min(ps.grad_raster_time, ps.rf_raster_time)
        raster_time = timestep
        if timestep < 0:
            raster_time = ps.timestep

        # =============================
        # SYSTEM SETUP
        # =============================

        # Init the system with the physics configuration
        system = pp.Opts(
            adc_dead_time=ps.adc_dead_time,
            gamma=ps.gamma,
            grad_raster_time=ps.grad_raster_time,
            grad_unit=ps.grad_unit,
            max_grad=ps.max_grad,
            max_slew=ps.max_slew,
            rf_dead_time=ps.rf_dead_time,
            rf_raster_time=ps.rf_raster_time,
            rf_ringdown_time=ps.rf_ringdown_time,
            rise_time=ps.rise_time,
            slew_unit=ps.slew_unit,
        )

        fov = 0.03  # 3cm field of view

        # Matrix size
        nz, ny, nx = matrix_shape
        alpha = 90  # 10  # alpha=°, flip_angle=rad
        slice_thickness = 3e-3
        nb_slice = nz
        delta_k = 1 / fov

        # Define timings
        TE = 5.8e-2  # Has been increased so delay_TE is >0
        TR = 9.5e-2  # Has been increased so delay_TR is >0

        rf_spoiling_inc = 117  # RF spoiling increment (117 ou 123)

        # Create sequence in Pypulseq and GammaMRI format
        seq_pypulseq = pp.Sequence(system=system)
        seq_yaml = Sequence(sequence_name)

        # Define events

        # RF sinc pulse with slice select and slice select rephasing
        # Values that do not violate max_grad amplitude:
        #   - duration=2.7e-1 and time_bw_product=4
        #   - duration=2.7e-2 and time_bw_product=2
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=alpha * np.pi / 180,
            duration=2.7e-2,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=2,
            system=system,
            return_gz=True,
        )  # TBW=2 (rapid imaging), =4 (180), =8 (90), =12 (slab and saturation)

        # TODO pypulseq rf use 1e-6 hard coded for vector length
        rf_hard_code_clock = 1e-6
        rf_correction_ratio = int(raster_time / rf_hard_code_clock)
        rf_corrected_signal = rf.signal[0:-1:rf_correction_ratio].copy()
        rf_corrected_time = rf.t[: len(rf_corrected_signal)].copy()

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        rf.signal = rf_corrected_signal
        rf.t = rf_corrected_time

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        # Waveforms for converter
        rf_waveform, rf_gz_waveform = Waveform.get_pulse_and_gradient_waveforms(
            rf,
            gz,
            os.path.join(waveform_dir, "rf.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )

        # Readout gradient
        gx = pp.make_trapezoid(
            channel="x", flat_area=nx * delta_k, flat_time=4.1e-2, system=system
        )
        gx_waveform = TrapGradientWaveform(
            gx,
            os.path.join(waveform_dir, "gx.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )

        gy = pp.make_trapezoid(
            channel="y", flat_area=ny * delta_k, flat_time=4.1e-2, system=system
        )
        gy_waveform = TrapGradientWaveform(
            gy,
            os.path.join(waveform_dir, "gy.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )

        gx_2 = pp.make_trapezoid(
            channel="x", flat_area=-nx * delta_k, flat_time=4.1e-2, system=system
        )
        gx2_waveform = TrapGradientWaveform(
            gx_2,
            os.path.join(waveform_dir, "gx2.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )

        gy_2 = pp.make_trapezoid(
            channel="y", flat_area=-ny * delta_k, flat_time=4.1e-2, system=system
        )
        gy2_waveform = TrapGradientWaveform(
            gy_2,
            os.path.join(waveform_dir, "gy2.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )

        adc = pp.make_adc(
            num_samples=nx, duration=gx.flat_time, delay=gx.rise_time, system=system
        )
        gx_pre = pp.make_trapezoid(
            channel="x", area=-gx.area / 2, duration=2.1e-2, system=system
        )
        gx_pre_waveform = TrapGradientWaveform(
            gx_pre,
            os.path.join(waveform_dir, "gx_pre.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )

        # Rephasing gradient
        gz_reph = pp.make_trapezoid(
            channel="z", area=-gz.area / 2, duration=2.1e-2, system=system
        )
        gz_reph_waveform = TrapGradientWaveform(
            gz_reph,
            os.path.join(waveform_dir, "gz_reph.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )
        phase_areas = (np.arange(ny) - ny / 2) * delta_k

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
                / seq_pypulseq.grad_raster_time
            )
            * seq_pypulseq.grad_raster_time
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
                / seq_pypulseq.grad_raster_time
            )
            * seq_pypulseq.grad_raster_time
        )

        assert np.all(delay_TE >= 0)
        # assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

        rf_phase = 0
        rf_inc = 0

        # ======
        # CONSTRUCT SEQUENCE
        # ======
        current_time: float = 0
        for i in range(nz):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq_pypulseq.add_block(rf, gz)

            # Conversion
            pulse_seq = PulseSeq(current_time, rf_waveform.duration())
            pulse_seq.set_flip_angle(alpha)
            pulse_seq.set_b1(rf_waveform.filename)
            pulse_seq.set_gradient_z(rf_gz_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += rf_waveform.duration()

            "Phase encoding gradient"
            gy_pre = pp.make_trapezoid(
                channel="y",
                area=phase_areas[i],
                duration=pp.calc_duration(gx_pre),
                system=system,
            )

            # Convert gradient to waveform in the format of the GammaMRI-Simulator
            gy_pre_waveform = TrapGradientWaveform(
                gy_pre,
                os.path.join(waveform_dir, f"gy_pre{i}.yml"),
                raster_time,
                cls.PRECISION,
                ps.gamma,
            )

            seq_pypulseq.add_block(gx_pre, gy_pre, gz_reph)

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, gx_pre_waveform.duration())
            pulse_seq.set_gradient_x(gx_pre_waveform)
            pulse_seq.set_gradient_y(gy_pre_waveform)
            pulse_seq.set_gradient_z(gz_reph_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gx_pre_waveform.duration()

            seq_pypulseq.add_block(pp.make_delay(delay_TE))
            current_time += float(delay_TE)

            seq_pypulseq.add_block(gx, adc)

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, gx_waveform.duration())
            pulse_seq.set_gradient_x(gx_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gx_waveform.duration()

            seq_pypulseq.add_block(gy, adc)

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, gy_waveform.duration())
            pulse_seq.set_gradient_y(gy_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gy_waveform.duration()

            seq_pypulseq.add_block(gx_2, adc)

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, gx2_waveform.duration())
            pulse_seq.set_gradient_x(gx2_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gx2_waveform.duration()

            seq_pypulseq.add_block(gy_2, adc)

            # Yaml equivalent
            pulse_seq = PulseSeq(current_time, gy2_waveform.duration())
            pulse_seq.set_gradient_y(gy2_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gy2_waveform.duration()

            gy_pre.amplitude = -gy_pre.amplitude

            # Convert gradient to waveform in the format of the GammaMRI-Simulator
            gy_spoil_waveform = TrapGradientWaveform(
                gy_pre,
                os.path.join(waveform_dir, f"gy_reph{i}.yml"),
                raster_time,
                cls.PRECISION,
                ps.gamma,
            )

            seq_pypulseq.add_block(pp.make_delay(delay_TR), gy_pre)

            pulse_seq = PulseSeq(current_time, gy_spoil_waveform.duration())
            pulse_seq.set_gradient_y(gy_spoil_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += gy_spoil_waveform.duration()

            # Add delay to match TR
            current_time += float(delay_TR)

        # Set timing
        if duration < 0: duration = current_time+raster_time
        seq_yaml.repetition_time = duration #current_time + raster_time
        seq_yaml.number_repetitions = 1

        (
            ok,
            error_report,
        ) = (
            seq_pypulseq.check_timing()
        )  # Check whether the timing of the sequence is correct
        if ok:
            print("Timing check passed successfully")
        else:
            print("Timing check failed. Error listing follows:")
            [print(e) for e in error_report]

        # ======
        # VISUALIZATION
        # ======
        seq_pypulseq.plot()

        # Trajectory calculation and plotting
        (
            ktraj_adc,
            ktraj,
            t_excitation,
            t_refocusing,
            t_adc,
        ) = seq_pypulseq.calculate_kspace()
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

        plt.savefig("myplot.png")

        # Prepare the sequence output for the scanner
        seq_pypulseq.set_definition("FOV", [fov, fov, slice_thickness])
        seq_pypulseq.set_definition("Name", "border")

        # Save .seq
        seq_pypulseq.write(os.path.join(outdir, sequence_pypulseq_filename))

        # Save yaml
        with open(os.path.join(outdir, sequence_config_filename), "w") as yaml_file:
            yaml.dump(seq_yaml.dict(), yaml_file, sort_keys=False)

        # Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
        # slew-rate limits
        rep = seq_pypulseq.test_report()
        print(rep)

        return sequence_config_filename

    @classmethod
    def generate_spiral_sequence(
        cls,
        matrix_shape: tuple,
        title: str,
        outdir: str,
        physics_configuration_filename: str,
        timestep: int = -1,
        duration: float = -1,
    ):
        print("Generate spiral sequence")

        sequence_name = f"{title}_spiral"
        sequence_pypulseq_filename = f"{sequence_name}.seq"
        sequence_config_filename = f"{sequence_name}.yml"

        # Create waveforms directory if not exists
        waveform_dir = os.path.join(outdir, "waveforms")
        os.makedirs(waveform_dir, exist_ok=True)

        # Physics system
        ps = PhysicalSetup(physics_configuration_filename)

        # Set raster time to timestep to ensure everything is with the same clock
        # TODO handle different raster times for rf and grad
        # raster_time: float = min(ps.grad_raster_time, ps.rf_raster_time)
        raster_time = timestep
        if timestep < 0:
            raster_time = ps.timestep

        # =============================
        # SYSTEM SETUP
        # =============================

        # Init the system with the physics configuration
        system = pp.Opts(
            adc_dead_time=ps.adc_dead_time,
            gamma=ps.gamma,
            grad_raster_time=raster_time,
            grad_unit=ps.grad_unit,
            max_grad=ps.max_grad,
            max_slew=ps.max_slew,
            rf_dead_time=ps.rf_dead_time,
            rf_raster_time=raster_time,
            rf_ringdown_time=ps.rf_ringdown_time,
            rise_time=ps.rise_time,
            slew_unit=ps.slew_unit,
        )

        # Matrix size
        nz, ny, nx = matrix_shape

        ## NEW SPIRAL

        # Use around 10% safety margin on max_slew_rate (T/m/s) and max_grad for spiral trajectory (T/m)
        max_grad = 0.9 * (system.max_grad / (system.gamma))
        max_slew = 0.9 * (system.max_slew / (system.gamma))

        # We need the rise time for rewinding gradient calculation
        min_rise_time = max_grad / max_slew

        # We also need to calculate S being the maximum allowed rotatable slew rate defined as:
        S = 1 / np.sqrt(2) * max_grad / min_rise_time
        max_slew = S

        # Create sequence in Pypulseq and GammaMRI format
        seq = pp.Sequence(system=system)
        seq_yaml = Sequence(sequence_name)

        # Define user parameters
        # =========================
        # b0 = 0.05 #T
        fov = 0.03  # m
        slice_thickness = 3e-3  # m
        n_slices = 1
        nb_interleaves = 1
        matrix_size = nx
        undersamp_factor = (
            0.8  # undersampling factor: = 1 is a fully sampled k-space whereas
        )
        # <1 corresponds to radial undersampling of k-space while angular sampling stays constant
        density_parameter = 0.5  # oversampling in kspace = alpha_d for Zhao, Archimedeaan spiral if =1 # TODO

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
                    - (
                        1 - 2 * nb_interleaves / (matrix_size * undersamp_factor)
                    )  # todo: 1- ???
                    ** (1 / density_parameter)
                )
                ** -1
            )

            print("Number of turns: ", nb_turns)
        else:
            raise ValueError("Number of interleaves to high for given matrix size")

        # Calculate key parameters
        k_max = matrix_size / (
            2 * fov
        )  # maximum kspace sampled value = lambda for Zhao, kFOV = 2kmax
        omega = 2 * np.pi * nb_turns

        # Calculate constant parts for following tau calculation
        # ======================================================
        # Constant for slew rate limited regime (first part of the spiral around the center)
        const_slew = np.sqrt(abs(system.gamma) * max_slew / (k_max * omega**2)) * (
            1 + density_parameter / 2
        )

        # Constant for amplitude limited regime (second part of the spiral)
        const_amp = (
            abs(system.gamma) * max_grad / (k_max * omega) * (density_parameter + 1)
        )

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
                np.zeros(1, dtype=float),
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
                tau = np.append(
                    tau, (const_amp * t0b) ** (1 / (density_parameter + 1)) + delta_taub
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
            plt.savefig("plot_spiral.png")

            calc_OK = 1
            # The line below needs to be uncommented if the right P has already been found:
            exit_OK = 1

            # Evaluation of the solution
            # ----------------------------

            if (
                max(np.abs(grad_x)) > max_grad
                or max(np.abs(grad_y)) > max_grad
                or max(np.abs(slew_x)) > max_slew
                or max(np.abs(slew_y)) > max_slew
            ):
                print("Exceeds gradient maximum specifications with margin !")
                if (
                    max(np.abs(grad_x)) > (system.max_grad / system.gamma)
                    or max(np.abs(grad_y)) > (system.max_grad / system.gamma)
                    or max(np.abs(slew_x)) > (system.max_slew / system.gamma)
                    or max(np.abs(slew_y)) > (system.max_slew / system.gamma)
                ):
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
            print(
                f"Max grad x = {max(np.abs(grad_x))} Max grad y = {max(np.abs(grad_y))}"
            )
            print(
                f"Max slew x = {max(np.abs(slew_x))} Max slew y = {max(np.abs(slew_y))}"
            )

        Grad = grad_x + 1j * grad_y
        Slew = slew_x + 1j * slew_y

        # K-Space trajectory
        # ==================

        k_size = len(grad_x)
        kx = np.zeros(k_size, dtype=float)
        ky = np.zeros(k_size, dtype=float)

        for i in range(k_size):
            kx[i] = (
                abs(system.gamma) / (1 / system.grad_raster_time) * (grad_x[i])
                + kx[i - 1]
            )
            ky[i] = (
                abs(system.gamma) / (1 / system.grad_raster_time) * (grad_y[i])
                + ky[i - 1]
            )

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
        grad_pp, slew_pp = traj_to_grad_spiral_v2(
            k_pp, nb_interleaves, system.grad_raster_time
        )
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
        alpha = 90
        rf, gz, gz_reph = pp.make_sinc_pulse(
            flip_angle=np.pi / 2,
            system=system,
            duration=2.7e-2,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=2,
            return_gz=True,
        )

        # Converter

        # TODO pypulseq rf use 1e-6 hard coded for vector length
        rf_hard_code_clock = 1e-6
        rf_correction_ratio = int(raster_time / rf_hard_code_clock)
        rf_corrected_signal = rf.signal[0:-1:rf_correction_ratio].copy()
        rf_corrected_time = rf.t[: len(rf_corrected_signal)].copy()

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        rf.signal = rf_corrected_signal
        rf.t = rf_corrected_time

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        # Waveforms for converter
        rf_waveform, rf_gz_waveform = Waveform.get_pulse_and_gradient_waveforms(
            rf,
            gz,
            os.path.join(waveform_dir, "rf.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )
        # end converter

        # Calculate ADC
        # -----------------

        # adc_time = system.grad_raster_time * len(grad_x_filtered)
        adc_samples_per_segment = (
            grad_pp.shape[1] / nb_interleaves
        )  # = new size_interleaves
        adc_samples = nb_interleaves * adc_samples_per_segment
        adc_dwell = readout_time / adc_samples_per_segment
        adc_segment_duration = adc_samples_per_segment * adc_dwell

        if (
            round(adc_segment_duration % system.grad_raster_time) > np.finfo(float).eps
        ):  # round is used because mod in Python gives float results on floats
            raise ValueError(
                "ADC segmentation model results in incorrect segment duration"
            )

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

        current_time = 0

        for i in range(n_slices):
            for j in range(nb_interleaves):

                # RF
                # -----
                seq.add_block(rf, gz)
                rf.freq_offset = (
                    gz.amplitude * slice_thickness * (i - (n_slices - 1) / 2)
                )
                # todo: gx.freq_offset / gy.freq_offset ?

                # Conversion
                pulse_seq = PulseSeq(current_time, rf_waveform.duration())
                pulse_seq.set_flip_angle(alpha)
                pulse_seq.set_b1(rf_waveform.filename)
                pulse_seq.set_gradient_z(rf_gz_waveform)
                seq_yaml.add_pulse_seq(pulse_seq)

                # Update time
                current_time += rf_waveform.duration()

                # Readout gradients
                # ---------------------
                # Extract the waveforms from grad_pp
                waveform_gx = grad_pp[
                    0,
                    int(j * adc_samples_per_segment) : int(
                        (j + 1) * adc_samples_per_segment
                    ),
                ]
                waveform_gy = grad_pp[
                    1,
                    int(j * adc_samples_per_segment) : int(
                        (j + 1) * adc_samples_per_segment
                    ),
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

                # Conversion
                gx_waveform = SpiralGradientWaveform(
                    gx,
                    os.path.join(waveform_dir, "gx.yml"),
                    raster_time,
                    cls.PRECISION,
                    ps.gamma,
                )
                gy_waveform = SpiralGradientWaveform(
                    gy,
                    os.path.join(waveform_dir, "gy.yml"),
                    raster_time,
                    cls.PRECISION,
                    ps.gamma,
                )

                pulse_seq = PulseSeq(current_time, gx_waveform.duration())
                pulse_seq.set_gradient_x(gx_waveform)
                pulse_seq.set_gradient_y(gy_waveform)
                seq_yaml.add_pulse_seq(pulse_seq)

                # Update time
                current_time += gx_waveform.duration()

                # HALLO

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

        # Set timing
        seq_yaml.repetition_time = current_time + raster_time
        seq_yaml.number_repetitions = 1

        # Set the definitions
        seq.set_definition("FOV", [fov, fov, slice_thickness])
        seq.set_definition("Name", "spiral")
        # seq.set_definition("MaxAdcSegmentLength", adc_samples_per_segment)

        # Save .seq
        seq.write(os.path.join(outdir, sequence_pypulseq_filename))

        # Save yaml
        with open(os.path.join(outdir, sequence_config_filename), "w") as yaml_file:
            yaml.dump(seq_yaml.dict(), yaml_file, sort_keys=False)

        return sequence_config_filename

    @classmethod
    def generate_90_sequence(
        cls,
        matrix_shape: tuple,
        title: str,
        outdir: str,
        physics_configuration_filename: str,
        timestep: int = -1,
        duration: float = -1,
        sequence_parameters = None,
    ):
        print("Generate 90RF pulse only sequence")

        sequence_name = f"{title}_90"
        sequence_config_filename = f"{sequence_name}.yml"

        # Create waveforms directory if not exists
        waveform_dir = os.path.join(outdir, "waveforms")
        os.makedirs(waveform_dir, exist_ok=True)

        # Physics system
        ps = PhysicalSetup(physics_configuration_filename)

        # Set raster time to timestep to ensure everything is with the same clock
        # TODO handle different raster times for rf and grad
        # raster_time: float = min(ps.grad_raster_time, ps.rf_raster_time)
        raster_time = timestep
        if timestep < 0:
            raster_time = ps.timestep

        # =============================
        # SYSTEM SETUP
        # =============================

        # Init the system with the physics configuration
        system = pp.Opts(
            adc_dead_time=ps.adc_dead_time,
            gamma=ps.gamma,
            grad_raster_time=ps.grad_raster_time,
            grad_unit=ps.grad_unit,
            max_grad=ps.max_grad,
            max_slew=ps.max_slew,
            rf_dead_time=ps.rf_dead_time,
            rf_raster_time=ps.rf_raster_time,
            rf_ringdown_time=ps.rf_ringdown_time,
            rise_time=ps.rise_time,
            slew_unit=ps.slew_unit,
        )

        fov = 0.03  # 3cm field of view

        alpha = 90
        rf_duration = 2.7e-2
        slice_thickness = 3e-2
        if sequence_parameters is not None:
            if "duration" in sequence_parameters["RF"]: rf_duration = float(sequence_parameters["RF"]["duration"])
            if "angle" in sequence_parameters["RF"]: alpha = float(sequence_parameters["RF"]["angle"])
            if "thickness" in sequence_parameters["RF"]: slice_thickness = float(sequence_parameters["RF"]["thickness"])

        # Matrix size
        nz, ny, nx = matrix_shape
        nz = 1 # force only one slice
        #alpha = 90  # 10  # alpha=°, flip_angle=rad
        #slice_thickness = 3e-3
        delta_k = 1 / fov

        rf_spoiling_inc = 117  # RF spoiling increment (117 ou 123)

        # Create sequence in Pypulseq and GammaMRI format
        seq_pypulseq = pp.Sequence(system=system)
        seq_yaml = Sequence(sequence_name)

        # Define events

        # RF sinc pulse with slice select and slice select rephasing
        # Values that do not violate max_grad amplitude:
        #   - duration=2.7e-1 and time_bw_product=4
        #   - duration=2.7e-2 and time_bw_product=2
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=alpha * np.pi / 180,
            duration=rf_duration,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=2,
            system=system,
            return_gz=True,
        )  # TBW=2 (rapid imaging), =4 (180), =8 (90), =12 (slab and saturation)

        # TODO pypulseq rf use 1e-6 hard coded for vector length
        rf_hard_code_clock = 1e-6
        rf_correction_ratio = int(raster_time / rf_hard_code_clock)
        rf_corrected_signal = rf.signal[0:-1:rf_correction_ratio].copy()
        rf_corrected_time = rf.t[: len(rf_corrected_signal)].copy()

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        rf.signal = rf_corrected_signal
        rf.t = rf_corrected_time

        dur_rf = pp.calc_duration(rf)
        dur_gz = pp.calc_duration(gz)

        # Waveforms for converter
        rf_waveform, rf_gz_waveform = Waveform.get_pulse_and_gradient_waveforms(
            rf,
            gz,
            os.path.join(waveform_dir, "rf.yml"),
            raster_time,
            cls.PRECISION,
            ps.gamma,
        )

        phase_areas = (np.arange(ny) - ny / 2) * delta_k


        rf_phase = 0
        rf_inc = 0

        # ======
        # CONSTRUCT SEQUENCE
        # ======
        current_time: float = 0
        for i in range(nz):
            rf.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq_pypulseq.add_block(rf, gz)

            # Conversion
            pulse_seq = PulseSeq(current_time, rf_waveform.duration())
            pulse_seq.set_flip_angle(alpha)
            pulse_seq.set_b1(rf_waveform.filename)
            pulse_seq.set_gradient_z(rf_gz_waveform)
            seq_yaml.add_pulse_seq(pulse_seq)

            # Update time
            current_time += rf_waveform.duration()

        # Set timing
        if duration < 0: duration = current_time+raster_time
        seq_yaml.repetition_time = duration #current_time + raster_time
        seq_yaml.number_repetitions = 1

        (
            ok,
            error_report,
        ) = (
            seq_pypulseq.check_timing()
        )  # Check whether the timing of the sequence is correct
        if ok:
            print("Timing check passed successfully")
        else:
            print("Timing check failed. Error listing follows:")
            [print(e) for e in error_report]

        # Save yaml
        with open(os.path.join(outdir, sequence_config_filename), "w") as yaml_file:
            yaml.dump(seq_yaml.dict(), yaml_file, sort_keys=False)

        return sequence_config_filename
