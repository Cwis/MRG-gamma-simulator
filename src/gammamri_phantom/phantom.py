"""Phantom class for GammaMRI
Create a Shepp-Logan or modified Shepp-Logan phantom

:param matrix_size: size of imaging matrix in pixels (default 256)

:param phantom_type: The type of phantom to produce.
    Either "Modified Shepp-Logan" or "Shepp-Logan". This is overridden
    if ``ellipses`` is also specified.

:param ellipses: Custom set of ellipses to use.  These should be in
    the form::

        [[I, a, b, x0, y0, phi],
        [I, a, b, x0, y0, phi],
                        ...]

    where each row defines an ellipse.

    :I: Additive intensity of the ellipse.
    :a: Length of the major axis.
    :b: Length of the minor axis.
    :x0: Horizontal offset of the centre of the ellipse.
    :y0: Vertical offset of the centre of the ellipse.
    :phi: Counterclockwise rotation of the ellipse in degrees,
        measured as the angle between the horizontal axis and
        the ellipse major axis.

The image bounding box in the algorithm is ``[-1, -1], [1, 1]``,
so the values of ``a``, ``b``, ``x0``, ``y0`` should all be specified with
respect to this box.

:returns: Phantom image

References:

Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
Feb. 1974, p. 232.

Toft, P.; "The Radon Transform - Theory and Implementation",
Ph.D. thesis, Department of Mathematical Modelling, Technical
University of Denmark, June 1996.

"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import yaml

from gammamri_phantom.component import Component


class Phantom:
    _phantom_types = [
        "shepp-logan",
        "modified shepp-logan",
        "3cs",
        "2cd",
        "2c",
        "1c",
        "1p",
        "full1",
        "full2",
    ]

    def __init__(
        self,
        name: str = "phantom",
        matrix_shape: tuple = None,  # TODO add third dimension
        phantom_type: str = "custom",
        ellipses=None,
    ):
        self.name: str = name
        if matrix_shape:
            self._matrix_shape: tuple = self.get_shape_3d(matrix_shape)
        else:
            self._matrix_shape: tuple = None
        self._phantom_type = phantom_type
        if ellipses is not None:
            if np.size(ellipses, 1) != 6:
                raise AssertionError("Wrong number of columns in user phantom")
            else:
                self._ellipses = ellipses
        elif phantom_type != "custom":
            self.set_phantom_type(phantom_type)
        self.number_of_components = 0

        self._components = []
        self._background = None

        self._keys = ["title", "components", "locations"]
        self._used_keys = ["title"]
        self._used_keys.append("components")
        if self._matrix_shape:
            self._used_keys.append("locations")

        self._generated: bool = False
        self._phantom = None

    def _select_phantom_type(self):
        if self._phantom_type.lower() == "shepp-logan":
            self._ellipses = self._shepp_logan()
        elif self._phantom_type.lower() == "modified shepp-logan":
            self._ellipses = self._modified_shepp_logan()
        elif self._phantom_type.lower() == "3cs":
            self._ellipses = self._three_cylinders()
        elif self._phantom_type.lower() == "2cd":
            self._ellipses = self._two_cylinders_dual()
        elif self._phantom_type.lower() == "2c":
            self._ellipses = self._two_cylinders()
        elif self._phantom_type.lower() == "1c":
            self._ellipses = self._one_cylinder()
        elif self._phantom_type.lower() == "1p":
            self._ellipses = self._one_pixel()
        elif self._phantom_type.lower() == "full1":
            self._ellipses = self._full_one()
        elif self._phantom_type.lower() == "full2":
            self._ellipses = self._full_two()
        else:
            raise ValueError("Unknown phantom type: %s" % self._phantom_type)

    @staticmethod
    def get_components_from_ellipses(ellipses):
        return Counter([ellipse[0] for ellipse in ellipses])

    @staticmethod
    def _shepp_logan():
        #  Standard head phantom, taken from Shepp & Logan
        return [
            [2, 0.69, 0.92, 0, 0, 0],
            [-0.98, 0.6624, 0.8740, 0, -0.0184, 0],
            [-0.02, 0.1100, 0.3100, 0.22, 0, -18],
            [-0.02, 0.1600, 0.4100, -0.22, 0, 18],
            [0.01, 0.2100, 0.2500, 0, 0.35, 0],
            [0.01, 0.0460, 0.0460, 0, 0.1, 0],
            [0.02, 0.0460, 0.0460, 0, -0.1, 0],
            [0.01, 0.0460, 0.0230, -0.08, -0.605, 0],
            [0.01, 0.0230, 0.0230, 0, -0.606, 0],
            [0.01, 0.0230, 0.0460, 0.06, -0.605, 0],
        ]

    @staticmethod
    def _modified_shepp_logan():
        #  Modified version of Shepp & Logan's head phantom,
        #  adjusted to improve contrast.  Taken from Toft.
        return [
            [1, 0.69, 0.92, 0, 0, 0],
            [-0.80, 0.6624, 0.8740, 0, -0.0184, 0],
            [-0.20, 0.1100, 0.3100, 0.22, 0, -18],
            [-0.20, 0.1600, 0.4100, -0.22, 0, 18],
            [0.10, 0.2100, 0.2500, 0, 0.35, 0],
            [0.10, 0.0460, 0.0460, 0, 0.1, 0],
            [0.10, 0.0460, 0.0460, 0, -0.1, 0],
            [0.10, 0.0460, 0.0230, -0.08, -0.605, 0],
            [0.10, 0.0230, 0.0230, 0, -0.606, 0],
            [0.10, 0.0230, 0.0460, 0.06, -0.605, 0],
        ]

    @staticmethod
    def _three_cylinders():
        # 3 compartments phantom
        return [
            [1, 0.6900, 0.6900, 0, 0, 0],
            [-0.50, 0.2100, 0.2100, 0.22, -0.1, 0],
            [-0.20, 0.2100, 0.2100, -0.22, -0.1, 0],
            [0.10, 0.2100, 0.2100, 0, 0.28, 0],
        ]

    @staticmethod
    def _two_cylinders():
        # 2 compartments phantom
        return [
            [1, 0.9, 0.9, 0, 0, 0],
            [2, 0.33, 0.33, 0.3, -0.15, 0],
        ]

    @staticmethod
    def _two_cylinders_dual():
        # 2 compartments phantom
        return [
            [1, 0.9, 0.9, 0, 0, 0],
            [2, 0.33, 0.33, 0.3, -0.15, 0],
            [2, 0.33, 0.33, -0.3, 0.15, 0],
        ]

    @staticmethod
    def _one_cylinder():
        # 1 cylinder
        return [
            [1, 0.9, 0.9, 0, 0, 0],
        ]

    @staticmethod
    def _one_pixel():
        # 1 pixel
        return [
            [1, 0.1, 0.1, 0, 0, 0],
        ]

    @staticmethod
    def _full_one():
        # 1 component everywhere
        return [
            [1, 2, 2, 0, 0, 0],
        ]

    @staticmethod
    def _full_two():
        # 1 component everywhere + 1 component in center
        return [
            [1, 2, 2, 0, 0, 0],
            [2, 0.5, 0.5, 0.3, -0.15, 0],
        ]

    @staticmethod
    def _three_quarters():
        # TODO three quarters
        return []

    @staticmethod
    def get_shape_3d(matrix_shape: tuple):
        if not 4 > len(matrix_shape) > 1:
            raise ValueError(f"Phantom matrix shape must be 2D or 3D: {matrix_shape}")
        shape_3d = (1,) + matrix_shape if len(matrix_shape) == 2 else matrix_shape
        if any(s < 1 for s in shape_3d):
            raise ValueError(
                f"Phantom matrix shape must contain values >= 1: {matrix_shape}"
            )
        return shape_3d

    def set_matrix_size(self, matrix_shape: tuple, generate_phantom: bool = False):
        self._matrix_shape = self.get_shape_3d(matrix_shape)
        self._generated = False
        if generate_phantom:
            self.generate()

    def set_phantom_type(self, phantom_type_name: str, generate_phantom: bool = False):
        if phantom_type_name.lower() in self._phantom_types:
            self._phantom_type = phantom_type_name
            self._select_phantom_type()
            self._generated = False
        else:
            raise ValueError(
                f"Unknown phantom type: {phantom_type_name};"
                + f" must be one of {self._phantom_types} "
            )

        if generate_phantom:
            self.generate()

    def set_ellipses(self, ellipses, generate_phantom: bool = False):
        if np.size(ellipses, 1) != 6:
            raise AssertionError("Wrong number of columns in user phantom")
        self._phantom_type = "custom"
        self._ellipses = ellipses
        self._generated = False

        if generate_phantom:
            self.generate()

    def generate(self, matrix_shape: tuple = None):
        """Generate the phantom based on previously set parameters and optionally
        given matrix size.

        :param matrix_shape: (z, y, x)
        """
        ellipses_comps = self.get_components_from_ellipses(self._ellipses)

        if matrix_shape:
            self._matrix_shape = self.get_shape_3d(matrix_shape)
        if self.number_of_components != len(ellipses_comps):
            raise ValueError(
                f"Phantom must have the same number of components"
                + f" ({self.number_of_components}) and of different ellipse intensities"
                + f" ({len(ellipses_comps)})."
            )
        if self._ellipses is None:
            raise ValueError(f"Phantom ellipses or type is not defined.")

        # Init phantom
        phantom2d = np.zeros(
            self._matrix_shape[1:3], dtype=np.float32
        )  # TODO check dtype

        # Create the pixel grid
        ygrid, xgrid = np.mgrid[
            -1 : 1 : (1j * self._matrix_shape[1]), -1 : 1 : (1j * self._matrix_shape[2])
        ]

        # Map components and ellipse intensities
        keys = ellipses_comps.keys()
        values = [c for c, component in enumerate(self._components)]
        map_comps = dict(zip(keys, values))

        # Re-init locations
        background_locations2d = np.ones(self._matrix_shape[1:3])
        components_locations2d = np.zeros(
            (len(ellipses_comps), self._matrix_shape[1], self._matrix_shape[2])
        )
        components_locations3d = np.zeros(
            (
                len(ellipses_comps),
                self._matrix_shape[0],
                self._matrix_shape[1],
                self._matrix_shape[2],
            )
        )

        # Compute locations from ellipses
        for ellipse in self._ellipses:
            intensity = ellipse[0]
            a2 = ellipse[1] ** 2
            b2 = ellipse[2] ** 2
            x0 = ellipse[3]
            y0 = ellipse[4]
            phi = ellipse[5] * np.pi / 180  # Rotation angle in radians

            # Get corresponding component from intensity value
            id_comp = map_comps.get(intensity)

            # Create the offset x and y values for the grid
            x = xgrid - x0
            y = ygrid - y0

            cos_p = np.cos(phi)
            sin_p = np.sin(phi)

            # Find the pixels within the ellipse
            locs = (
                ((x * cos_p + y * sin_p) ** 2) / a2
                + ((y * cos_p - x * sin_p) ** 2) / b2
            ) <= 1

            # Add the ellipse intensity to those pixels
            phantom2d[locs] += intensity

            # Component and background locations
            components_locations2d[id_comp, locs] += 1.0
            background_locations2d[locs] -= 1.0

        # Remove spins from previous components location (not stacking spins)
        nc = components_locations2d.shape[0]
        for id_comp in range(0, nc - 1):
            for id_next_comp in range(id_comp + 1, nc):
                components_locations2d[
                    id_comp, components_locations2d[id_next_comp] == 1
                ] = 0

        # Compute 3D stacks
        for id_comp in map_comps.values():
            comp_loc3d = np.stack(
                [components_locations2d[id_comp] for i in range(self._matrix_shape[0])]
            )
            self._components[id_comp].set_locations(comp_loc3d)

        # Background
        if self.has_background():
            self._components.append(self._background)
            self._components[-1].set_locations(background_locations2d)

        self._phantom = phantom2d
        self._generated = True

    def is_valid(self):
        return self._generated

    def get_phantom(self):
        if self.is_valid():
            return self._phantom
        else:
            print("Phantom not generated!")
            return None

    def get_phantom_type(self):
        return self._phantom_type

    def get_phantom_types(self):
        return self._phantom_types

    def add_component(self, component: Component):
        self._components.append(component)
        self.number_of_components += 1

    def add_background(self, component: Component):
        self._background = component

    def has_background(self) -> bool:
        return self._background is not None

    def dict(self):
        locations_dict = dict(
            zip(
                [component.name for component in self._components],
                [component.get_string_locations() for component in self._components],
            )
        )
        all_keys_dict = dict(
            zip(
                self._keys,
                [
                    self.name,
                    [component.dict() for component in self._components],
                    locations_dict,
                ],
            )
        )
        used_dict = {
            key: value for key, value in all_keys_dict.items() if key in self._used_keys
        }
        return used_dict

    def write(self, phantom_filename: str = "phantom.yml"):
        temp_filename = phantom_filename[:-4] + "temp.yml"
        with open(temp_filename, "w") as yaml_file:
            yaml.dump(self.dict(), yaml_file, sort_keys=False)

        # TODO more elegant way to remove str quote
        with open(temp_filename, "r") as infile, open(phantom_filename, "w") as outfile:
            data = infile.read()
            data = data.replace("'", "")
            outfile.write(data)

        # remove temp
        os.remove(temp_filename)

    def plot(self, filename: str = ""):

        if not self.is_valid():
            print("Cannot plot phantom which has not been generated.")
            return

        titles = [
            f"{self.name} {self._matrix_shape} "
            + f"{self.number_of_components} "
            + f"component{'s' if self.number_of_components > 1 else ''}"
        ]

        colorbar = False
        cmap = "gray"
        image_matrix = self._phantom.copy()

        assert self._phantom.ndim in [2, 3], "image_matrix must have 2 or 3 dimensions"

        if image_matrix.ndim == 2:
            image_matrix = image_matrix.reshape(
                (1, image_matrix.shape[0], image_matrix.shape[1])
            )

        scale = (np.min(image_matrix), np.max(image_matrix))
        vmin, vmax = scale

        tile_shape = (1, image_matrix.shape[0])
        assert (
            np.prod(tile_shape) >= image_matrix.shape[0]
        ), "image tile rows x columns must equal the 3rd dim extent of image_matrix"

        # add empty titles as necessary
        if len(titles) < image_matrix.shape[0]:
            titles.extend(["" for x in range(image_matrix.shape[0] - len(titles))])

        if len(titles) > 0:
            assert (
                len(titles) >= image_matrix.shape[0]
            ), "number of titles must equal 3rd dim extent of image_matrix"

        cols, rows = tile_shape
        fig = plt.figure()
        plt.set_cmap(cmap)
        for z in range(image_matrix.shape[0]):
            ax = fig.add_subplot(cols, rows, z + 1)
            ax.set_title(titles[z])
            ax.set_axis_off()
            imgplot = ax.imshow(
                image_matrix[z, :, :], vmin=vmin, vmax=vmax, picker=True
            )
            if colorbar is True:
                plt.colorbar(imgplot)

        # Save to png file
        if filename != "":
            plt.savefig(filename)

        # Show plot
        plt.show()
