"""GammaMRI simulator physical setup"""

from gammamri_simulator.configuration import Configuration


class PhysicalSetup(Configuration):
    """Class for the physical setup of the GammaMRI simulator"""

    def check_configuration(self) -> bool:
        if any([key not in self.configuration for key in self.configuration_keys]):
            # raise Exception(f"Physics config must contain '{key}'")
            print(f"Configuration check fail: physics config must contain '{key}'")
            return False
        return True

    def __init__(self, configuration_filename: str = ""):
        self.adc_dead_time: float = 0
        self.b0: float = 1.5
        self.gamma: float = 42.576e6
        self.grad_raster_time: float = 10e-6
        self.grad_unit: str = "Hz/m"
        self.max_grad: float = 0
        self.max_slew: float = 0
        self.rf_dead_time: float = 0
        self.rf_raster_time: float = 1e-6
        self.rf_ringdown_time: float = 0
        self.rise_time: float = 0
        self.slew_unit: str = "Hz/m/s"
        self.timestep: float = 1e-6
        self.configuration_filename: str = configuration_filename

        self.configuration_keys = [
            "gyro",
            "b_zero",
            "timestep",
            "adc_dead_time",
            "grad_raster_time",
            "grad_unit",
            "max_grad",
            "max_slew",
            "rf_dead_time",
            "rf_raster_time",
            "rf_ringdown_time",
            "rise_time",
            "slew_unit",
            "timestep",
        ]

        # Init from configuration file
        if self.configuration_filename != "":
            self.read_configuration()
            self.adc_dead_time = float(self.configuration["adc_dead_time"])
            self.b_zero = float(self.configuration["b_zero"])
            self.gamma = float(self.configuration["gyro"])
            self.grad_raster_time = float(self.configuration["grad_raster_time"])
            self.grad_unit = self.configuration["grad_unit"]
            self.max_grad = float(self.configuration["max_grad"])
            self.max_slew = float(self.configuration["max_slew"])
            self.rf_dead_time = float(self.configuration["rf_dead_time"])
            self.rf_raster_time = float(self.configuration["rf_raster_time"])
            self.rf_ringdown_time = float(self.configuration["rf_ringdown_time"])
            self.rise_time = float(self.configuration["rise_time"])
            self.slew_unit = self.configuration["slew_unit"]
            self.timestep = float(self.configuration["timestep"])
