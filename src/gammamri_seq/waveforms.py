import numpy as np
import pypulseq as pp


class Waveform:
    def __init__(
        self,
        event,
        filename: str,
        raster_time: float = 1e-6,
        precision: int = 5,
        gamma: float = 42.576e6,
        save: bool = True,
    ):
        self.event = event
        self.waveform = []
        self.filename = filename
        self.raster_time = raster_time
        self.precision = precision
        self.gamma = gamma

        self.convert(save)

    def duration(self) -> float:
        return self.raster_time * self.waveform.size

    def convert(self, save: bool = True):
        pass

    def write(self) -> bool:
        if self.waveform.size > 0:
            np.savetxt(
                self.filename,
                [self.waveform],
                fmt=f"%.{self.precision}f",
                delimiter=",",
            )
            with open(self.filename, "r+") as waveform_file:
                waveform_data = waveform_file.readlines()
                waveform_file.seek(0)
                new_waveform_data = "[" + waveform_data[0][:-1] + "]\n"
                waveform_file.writelines(new_waveform_data)
        else:
            print("SIZE NOT OK")

    @staticmethod
    def get_pulse_and_gradient_waveforms(
        event_rf,
        event_gradient,
        filename_rf: str,
        raster_time: float = 1e-6,
        precision: int = 5,
        gamma: float = 42.576e6,
    ):
        # Gradient filename
        filename_rf_gradient = f"{filename_rf[:-4]}_gradient.yml"

        # Generate both waveforms without saving them
        rf_waveform = RfWaveform(
            event_rf, filename_rf, raster_time, precision, gamma, False
        )
        rf_gradient_waveform = TrapGradientWaveform(
            event_gradient, filename_rf_gradient, raster_time, precision, gamma, False
        )

        # Match size
        if len(rf_waveform.waveform) > len(rf_gradient_waveform.waveform):
            diff = len(rf_waveform.waveform) - len(rf_gradient_waveform.waveform)
            fill = np.zeros((1, diff), dtype=np.float32)
            rf_gradient_waveform.waveform = np.concatenate(
                rf_gradient_waveform.waveform, fill
            )
        elif len(rf_waveform.waveform) < len(rf_gradient_waveform.waveform):
            diff = len(rf_gradient_waveform.waveform) - len(rf_waveform.waveform)
            fill = np.zeros(diff, dtype=np.float32)
            rf_waveform.waveform = np.append(rf_waveform.waveform, fill)

        # Save and return
        rf_waveform.write()
        rf_gradient_waveform.write()

        return rf_waveform, rf_gradient_waveform


class RfWaveform(Waveform):
    def convert(self, save: bool = True):
        self.waveform = self.event.signal
        if save:
            self.write()


class TrapGradientWaveform(Waveform):
    amplitude: float = 1

    def convert(self, save: bool = True):
        amplitude = np.array(
            [
                0,
                self.event.first,
                self.event.amplitude,
                self.event.amplitude,
                self.event.last,
            ]
        )
        time = np.cumsum(
            [
                0,
                self.event.delay,
                self.event.rise_time,
                self.event.flat_time,
                self.event.fall_time,
            ]
        )
        waveform_hertz = pp.points_to_waveform(amplitude, self.raster_time, time)

        # convert kHz/m to mT/m
        self.waveform = waveform_hertz * 1e3 / self.gamma

        # amplitude
        self.amplitude = float(self.event.amplitude) * 1e3 / self.gamma

        if save:
            self.write()
            
class SpiralGradientWaveform(Waveform):
    amplitude: float = 1

    def convert(self, save: bool = True):
        
        # convert kHz/m to mT/m
        self.waveform = self.event.waveform * 1e3 / self.gamma

        if save:
            self.write()
