"""Handling of YAML configuration files for the GammaMRI simulator"""


def check_physics(physics_config):
    """Check the physics configuration

    :param physics_config: physics configuration read from the YAML file
    """
    if any([key not in physics_config for key in ["b_zero", "gyro"]]):
        raise Exception("Physics config must contain 'b_zero' and 'gyro'")
