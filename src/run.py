"""Run a simulation from a full given configuration

date: 2022-03-21

author: C. Chenes
version: 0.1
"""
import os

import yaml
import argparse

from gammamri_phantom.component import Component
from gammamri_phantom.phantom import Phantom
from gammamri_seq.sequence_generator import SequenceGenerator
from gammamri_simulator import simulator

if __name__ == "__main__":

    print("Gamma-MRI magnetic resonance manipulation of nuclear spins simulation")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configuration",
        help="configuration file (yaml) containing all the required configurations"
        + " files (phantom, sequence, physics) for the simulation run.",
    )
    args = parser.parse_args()
    configuration_filename: str = args.configuration

    print(os.getcwd())

    print(f"Read configuration of the complete setup from:  {configuration_filename}")
    with open(configuration_filename, "r") as configuration_file:
        try:
            setup_configuration = yaml.safe_load(configuration_file)
        except yaml.YAMLError as exc:
            raise Exception(
                "Error reading the configuration file: " + f"{configuration_filename}"
            ) from exc

    # Create output directory
    title = setup_configuration["title"]
    output_directory = os.path.join(setup_configuration["output_directory"], title)
    os.makedirs(output_directory, exist_ok=True)

    # Get general configuration
    general_config_filename = setup_configuration["general_configuration"]

    # Get physics configuration
    physics_config_filename = setup_configuration["physics_configuration"]

    # Get the phantom configuration
    phantom_config_filename = setup_configuration["phantom_configuration"]

    # Get the sequence configuration
    sequence_config_filename = setup_configuration["sequence_configuration"]

    # Get the optional animation configuration
    animation_config_filename = ""
    if "animation_configuration" in setup_configuration:
        animation_config_filename = setup_configuration["animation_configuration"]

    # Simulate with the generated configurations
    print("Simulation...")

    magnetizations = simulator.run(
        output_directory,
        general_config_filename,
        physics_config_filename,
        sequence_config_filename,
        phantom_config_filename,
        animation_config_filename,
    )

    print("done.")
