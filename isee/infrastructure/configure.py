"""
configure.py
Takes user input file and returns settings namespace object
"""

import argparse
import pytraj       # to support pytraj calls in input file
import mdtraj       # to support mdtraj calls in the input file
import numpy        # to support numpy  calls in input file
import numpy as np  # to support numpy  calls even if called as np
import sys
import os
import shutil
import pickle
from jinja2 import Environment, FileSystemLoader
import typing
import pydantic

def configure(input_file, user_working_directory=''):
    """
    Configure the settings namespace based on the config file.

    Parameters
    ----------
    input_file : str
        Name of the configuration file to read
    user_working_directory : str
        User override for working directory (overrides value in input_file), ignored if set to ''

    Returns
    -------
    settings : argparse.Namespace
        Settings namespace object

    """

    class Settings(pydantic.BaseModel):
        # This class initializes the settings object with type hints. After being built, it gets exported as an
        # argparse.Namelist object, just for convenience.

        # Core settings required for all jobs
        job_type: str = 'isEE'
        batch_system: str
        restart: bool
        md_engine: str = 'amber'
        task_manager: str = 'simple'
        init_topology: str
        working_directory: str
        overwrite: bool
        algorithm: str = 'script'

        # Batch template settings
        nodes: int = 1
        ppn: int = 1
        mem: str = '4000mb'
        walltime: str = '02:00:00'
        solver: str = 'sander'
        extra: str = ''

        # File path settings (required for all jobs, but do have sensible defaults)
        path_to_input_files: str = os.path.dirname(os.path.realpath(__file__)) + '/data/input_files'
        path_to_templates: str = os.path.dirname(os.path.realpath(__file__)) + '/data/templates'

        # Settings for isEE algorithm
        degeneracy: int = 0
        initial_coordinates: list = ['']

        # Script for algorithm = 'script'
        mutation_script: list = []

        # Linear Interaction Energy parameters
        ts_mask: str = ''
        lie_alpha: float = 0.18
        lie_beta: float = 0.33

        # Not expected to be set by user
        DEBUG: bool = False     # True causes some functions to return dummy values for testing purposes
        dont_dump: bool = False     # when True, prevents dumping settings to settings.pkl

    # Import config file line-by-line using exec()
    try:
        lines = open(input_file, 'r').readlines()
    except FileNotFoundError:
        try:
            lines = open('atesa/' + input_file, 'r').readlines()     # for testing
        except:
            lines = open(input_file, 'r').readlines()   # to reproduce original error
    line_index = 0
    for line in lines:      # each line in the input file is just python code setting a variable;
        line_index += 1
        try:
            exec(line)      # this means that comments are supported using '#' and whitespace is ignored.
        except Exception as e:
            raise ValueError('error raised while reading line ' + str(int(line_index)) + ' of configuration file '
                             + input_file + ': ' + str(e))

    # Define settings namespace to store all these variables
    config_dict = {}
    config_dict.update(locals())
    settings = argparse.Namespace()
    settings.__dict__.update(Settings(**config_dict))

    # Override working directory if provided with user_working_directory
    if user_working_directory:
        settings.working_directory = user_working_directory

    # Format directories properly (no trailing '/')
    if settings.working_directory[-1] == '/':
        settings.working_directory = settings.working_directory[:-1]
    if settings.path_to_input_files[-1] == '/':
        settings.path_to_input_files = settings.path_to_input_files[:-1]
    if settings.path_to_templates[-1] == '/':
        settings.path_to_templates = settings.path_to_templates[:-1]

    # Set Jinja2 environment
    if os.path.exists(settings.path_to_templates):
        settings.env = Environment(loader=FileSystemLoader(settings.path_to_templates))
    else:
        raise FileNotFoundError('could not locate templates folder: ' + settings.path_to_templates)

    return settings

if __name__ == "__main__":
    configure('','')
