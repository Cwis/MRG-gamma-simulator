"""Configuration class"""
from pathlib import Path

import yaml


class Configuration:

    configuration = None
    configuration_keys = []
    configuration_filename = ""

    def __init__(self, configuration_filename: str = ""):
        self.configuration_filename = configuration_filename

    def read_configuration(self) -> bool:
        """Read YAML configuration file

        :return:
        """
        configuration_path = Path(self.configuration_filename)
        if configuration_path.is_file():
            with open(configuration_path, "r") as configuration_file:
                try:
                    self.configuration = yaml.safe_load(configuration_file)
                except yaml.YAMLError as exc:
                    raise Exception(
                        "Error reading the configuration file: "
                        + f"{configuration_filename}"
                    ) from exc
                return self.check_configuration()
        else:
            raise Exception("Error reading the configuration file: invalid path")

    def check_configuration(self) -> bool:
        """Check the configuration

        :return:
        """
        pass
