import numpy as np


class Component:
    def __init__(
        self,
        name: str,
        T1: float,
        T2: float,
        vx: float = 0,
        vy: float = 0,
        vz: float = 0,
        CS: float = 0,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        locations=[],
        M0=[],
    ):
        """

        :param name: Component name
        :param T1: T1 relaxation time [ms]
        :param T2: T2 relaxation time [ms]
        :param CS: chemical shift [ppm]
        :param vx: velocity in x [mm/s]
        :param vy: velocity in y [mm/s]
        :param vz: velocity in z [mm/s]
        :param dx: diffusion in x [mm^2/s]
        :param dy: diffusion in y [mm^2/s]
        :param dz: diffusion in z [mm^2/s]
        :param locations: matrix representing the equilibrium magnetization.
        :param M0: matrix to specify initial state other than equilibrium.
        Match locations shape with inner additional dimmension of 3: vector length,
        polar angle [°] and azimutal angle [°].
        """
        self.name = name
        self.T1 = T1
        self.T2 = T2
        self.CS = CS
        self.velocity = [vx, vy, vz]
        self.diffusion = [dx, dy, dz]
        self.locations = locations
        self.M0 = M0

        self._keys = ["name", "T1", "T2"]
        self._locations_keys = ["name", "", "T2"]

    def set_locations(self, locations):
        if len(locations.shape) == 2:
            locations = locations.reshape((1, locations.shape[0], locations.shape[1]))
        self.locations = locations

    def get_string_locations(self):
        if len(self.locations.shape) < 3:
            locations_3d = self.locations.reshape(
                (1, self.locations.shape[0], self.locations.shape[1])
            )
        else:
            locations_3d = self.locations

        # Set the print option to have the full array printed
        np_print_options = np.get_printoptions()
        np.set_printoptions(threshold=np.inf, precision=3, linewidth=np.inf)
        locations_str = np.array2string(locations_3d, separator=",").replace("\n", "")
        locations_str = " ".join(locations_str.split())
        np.set_printoptions(**np_print_options)
        return locations_str

    # TODO add all component properties to output if set
    def dict(self):
        return dict(
            zip(
                self._keys,
                [self.name, self.T1, self.T2],
            )
        )
