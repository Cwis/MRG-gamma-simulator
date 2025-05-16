class PulseSeq:
    def __init__(self, time: float, duration: float):
        self.time: float = time  # ms
        self.duration: float = duration  # ms

        self.flip_angle: float = 0  # °
        self.b1: str = ""  # uT
        self.phase: float = 0  # °
        self.gradient_x = {"file": "", "amp": 1}
        self.gradient_y = {"file": "", "amp": 1}
        self.gradient_z = {"file": "", "amp": 1}

        self.keys = ["t", "dur", "FA", "B1", "phase", "Gx", "Gy", "Gz"]
        self.used_keys = ["t", "dur"]

    def set_flip_angle(self, flip_angle: float = 0, remove: bool = False):
        if remove and "FA" in self.used_keys:
            self.used_keys.remove("FA")
        else:
            self.flip_angle = flip_angle
            self.used_keys.append("FA")

    def set_b1(self, b1: str = "", remove: bool = False):
        if remove and "B1" in self.used_keys:
            self.used_keys.remove("B1")
        else:
            self.b1 = b1
            self.used_keys.append("B1")

    def set_phase(self, phase: float = 0, remove: bool = False):
        if remove and "phase" in self.used_keys:
            self.used_keys.remove("phase")
        else:
            self.phase = phase
            self.used_keys.append("phase")

    def set_gradient_x(self, gradient_x_waveform, remove: bool = False):
        if remove and "Gx" in self.used_keys:
            self.used_keys.remove("Gx")
            self.gradient_x["file"] = ""
            self.gradient_x["amp"] = 1
        else:
            self.gradient_x["file"] = gradient_x_waveform.filename
            self.gradient_x["amp"] = gradient_x_waveform.amplitude
            self.used_keys.append("Gx")

    def set_gradient_y(self, gradient_y_waveform, remove: bool = False):
        if remove and "Gy" in self.used_keys:
            self.used_keys.remove("Gy")
            self.gradient_y["file"] = ""
            self.gradient_y["amp"] = 1
        else:
            self.gradient_y["file"] = gradient_y_waveform.filename
            self.gradient_y["amp"] = gradient_y_waveform.amplitude
            self.used_keys.append("Gy")

    def set_gradient_z(self, gradient_z_waveform, remove: bool = False):
        if remove and "Gz" in self.used_keys:
            self.used_keys.remove("Gz")
            self.gradient_z["file"] = ""
            self.gradient_z["amp"] = 1
        else:
            self.gradient_z["file"] = gradient_z_waveform.filename
            self.gradient_z["amp"] = gradient_z_waveform.amplitude
            self.used_keys.append("Gz")

    def dict(self, precision: int = 5, time_unit_multiplier: float = 1e3):

        self.gradient_x["amp"] = round(self.gradient_x["amp"], precision)
        self.gradient_y["amp"] = round(self.gradient_y["amp"], precision)
        self.gradient_z["amp"] = round(self.gradient_z["amp"], precision)

        # Sort used_keys according to keys order
        # self.used_keys = [ele for ele in self.keys if ele in self.used_keys]
        all_keys_dict = dict(
            zip(
                self.keys,
                [
                    round(self.time * time_unit_multiplier, precision),
                    round(self.duration * time_unit_multiplier, precision),
                    self.flip_angle,
                    self.b1,
                    self.phase,
                    self.gradient_x,
                    self.gradient_y,
                    self.gradient_z,
                ],
            )
        )
        used_dict = {
            key: value for key, value in all_keys_dict.items() if key in self.used_keys
        }
        return used_dict
