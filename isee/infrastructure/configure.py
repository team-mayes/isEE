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
        job_type: str = 'isee'
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

        # Settings for isEE jobtype
        degeneracy: int = 0
        initial_coordinates: typing.List[str] = ['']    # todo: in isEE as currently written, each thread has to have the same initial coordinates, so either change that or change this
        ts_bonds: typing.Tuple[typing.List[str], typing.List[str], typing.List[float], typing.List[float]] = [[''],[''],[-1],[-1]]
        hmr: bool = False
        min_steps: int = 5000
        storage_directory: str = '/'    # this '/' is removed later in configure
        dry_distance: float = 8.0
        rosetta_mutate: bool = False

        # Initialize charges settings
        initialize_charges: bool = True
        ic_qm_mask: str = ''
        ic_qm_theory: str = 'DFTB3'
        ic_qm_cut: float = 8.0
        ic_qm_charge: int = 0

        # For algorithm = 'script'
        mutation_script: typing.List[typing.List[str]] = []
        # todo: add option for script algorithm to skip WT?

        # For algorithms 'monte_carlo', 'covariance_saturation', and 'subnetwork_hotspots'
        covariance_reference_resid: int = -1
        immutable: typing.List[int] = []

        # For algorithm = 'monte_carlo'
        max_plurality: int = 3
        plural_penalty: float = 1
        skip_wt: bool = False
        shared_history_file: str = ''

        # Linear Interaction Energy parameters
        ts_mask: str = ''
        lie_mask: str = ''
        lie_alpha: float = 0.18
        lie_beta: float = 0.33
        lie_dry: bool = True
        lie_decomposed: bool = False

        # Stability model parameters
        stabilitymodel: str = 'ddgunmean'
        destabilization_cutoff: float = -3.0

        # Other
        restart_terminated_threads: bool = True
        pH: float = 7

        # Custom Amber force fields, if required
        paths_to_forcefields: typing.List[str] = ['']

        # Not expected to be set by user
        SPOOF: bool = False     # True causes simulations to be skipped and scores to be spoofed
        spoof_latent = False    # internal for spoofing
        seq = []                # internal for spoofing
        rmsd_covar = []         # internal for spoofing
        TEST: bool = False      # True causes some functions to use much less rigorous methods, for testing purposes
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
    if settings.storage_directory[-1] == '/':
        settings.storage_directory = settings.storage_directory[:-1]

    # Set Jinja2 environment
    if os.path.exists(settings.path_to_templates):
        settings.env = Environment(loader=FileSystemLoader(settings.path_to_templates))
    else:
        raise FileNotFoundError('could not locate templates folder: ' + settings.path_to_templates)

    return settings

if __name__ == "__main__":
    configure('','')
