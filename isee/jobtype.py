"""
Interface for JobType objects. New JobTypes can be implemented by constructing a new class that inherits from JobType
and implements its abstract methods.
"""

import abc
import os
import sys
import subprocess
import random
import pickle
import argparse
import numpy
import shutil
import time
import pytraj
import mdtraj
import warnings
import copy
import re
import psutil
from isee import utilities
from isee import main
from isee.infrastructure import factory

class JobType(abc.ABC):
    """
    Abstract base class for job types.

    Implements methods for all of the job type-specific tasks that ATESA might need.

    """

    @abc.abstractmethod
    def get_initial_coordinates(self, settings):
        """
        Obtain list of initial coordinate files and copy them to the working directory.

        Parameters
        ----------
        self : Thread
            Methods in the JobType abstract base class are intended to be invoked by Thread objects
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        initial_coordinates : list
            List of strings naming the applicable initial coordinate files that were copied to the working directory

        """

        pass

    @abc.abstractmethod
    def get_next_step(self, settings):
        """
        Determine and return name for next step in the thread given by "self"

        Parameters
        ----------
        self : Thread()
            The Thread object on which to act
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        name : str
            Name for the next step

        """

        pass

    @abc.abstractmethod
    def get_input_file(self, settings):
        """
        Obtain appropriate input file for next job.

        At its most simple, implementations of this method can simply return settings.path_to_input_files + '/' +
        settings.job_type + '_' + settings.md_engine + '.in'

        Parameters
        ----------
        self : Thread
            Methods in the JobType abstract base class are intended to be invoked by Thread objects
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        input_file : str
            Name of the applicable input file

        """

        pass

    @abc.abstractmethod
    def get_batch_template(self, settings):
        """
        Return name of batch template file for the type of job indicated.

        Parameters
        ----------
        self : Thread
            Methods in the JobType abstract base class are intended to be invoked by Thread objects
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        name : str
            Name of the batch file template requested

        """

        pass

    @abc.abstractmethod
    def get_inpcrd(self):
        """
        Return the name of the appropriate inpcrd file for the next step in the thread given by self.

        Parameters
        ----------
        self : Thread
            Methods in the JobType abstract base class are intended to be invoked by Thread objects

        Returns
        -------
        inpcrd : str
            String containing desired file name

        """

        pass

    @abc.abstractmethod
    def update_history(self, settings, **kwargs):
        """
        Update or initialize the history namespace for this job type.

        This namespace is used to store the full history of a threads coordinate and trajectory files, as well as their
        results if necessary.

        If update_history is called with a kwargs containing {'initialize': True}, it simply prepares a blank
        history namespace and returns it. Otherwise, it adds the values of the desired keywords (which are desired
        depends on the implementation) to the corresponding history attributes in the index given by thread.suffix.

        Parameters
        ----------
        self : Thread
            Methods in the JobType abstract base class are intended to be invoked by Thread objects
        settings : argparse.Namespace
            Settings namespace object
        kwargs : dict
            Dictionary of arguments that might be used to update the history object

        Returns
        -------
        None

        """

        pass

    @abc.abstractmethod
    def analyze(self, settings):
        """
        Perform necessary analysis of a completed simulation step and store results as appropriate into self.history

        Parameters
        ----------
        self : Thread
            Methods in the JobType abstract base class are intended to be invoked by Thread objects
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        None

        """

        pass

    @abc.abstractmethod
    def algorithm(self, settings):
        """
        Implement the algorithm that determines the next step and sets up the appropriate self.history attributes

        Parameters
        ----------
        self : Thread
            Methods in the JobType abstract base class are intended to be invoked by Thread objects
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        terminate : Bool
            If True, terminate the entire isEE job

        """

        pass


class isEE(JobType):
    """
    Adapter class for primary isEE jobtype. My current conception is that this will be the only jobtype, but I've held
    off on simplifying the JobType class to only this implementation for now in case that changes

    """

    def get_initial_coordinates(self, settings):
        list_to_return = []
        for item in settings.initial_coordinates:
            if settings.degeneracy > 1:     # implements degeneracy option
                og_item = item
                if '/' in item:
                    item = item[item.rindex('/') + 1:]
                list_to_return += [item + '_' + str(this_index) for this_index in range(settings.degeneracy)]
                for file_to_make in list_to_return:
                    shutil.copy(og_item, settings.working_directory + '/' + file_to_make)
            else:
                og_item = item
                if '/' in item:
                    item = item[item.rindex('/') + 1:]
                list_to_return += [item]
                try:
                    shutil.copy(og_item, settings.working_directory + '/' + item)
                except shutil.SameFileError:
                    pass
        return list_to_return

    def get_next_step(self, settings):
        return 'prod'   # todo: consider cutting this method (only keeping it at the moment to contrast with minimization steps, but I might end up not keeping those output files around)

    def get_input_file(self, settings):
        return settings.path_to_input_files + '/' + settings.job_type + '_' + settings.md_engine + '.in'

    def get_batch_template(self, settings):
        templ = settings.md_engine + '_' + settings.batch_system + '.tpl'
        if os.path.exists(settings.path_to_templates + '/' + templ):
            return templ
        else:
            raise FileNotFoundError('cannot find required template file: ' + templ)

    def get_struct(self):
        return self.history.inpcrd[-1], self.history.tops[-1]

    def update_history(self, settings, **kwargs):
        if 'initialize' in kwargs.keys():
            if kwargs['initialize']:
                self.history = argparse.Namespace()
                self.history.inpcrd = []            # list of strings; initialized by main.init_threads(), updated by algorithm
                self.history.trajs = []             # list of strings; updated by update_history() called by process.py
                self.history.tops = []              # list of strings; initialized by main.init_threads(), updated by algorithm
                self.history.muts = []              # list of strings describing mutations tested; updated by algorithm
                self.history.score = []             # list of scores; updated by analyze()
                self.history.timestamps = []        # list of ints representing seconds since the epoch for the end of each step, updated by ?
            if 'inpcrd' in kwargs.keys():
                self.history.inpcrd.append(kwargs['inpcrd'])
        else:   # self.history should already exist
            if 'nc' in kwargs.keys():
                if len(self.history.trajs) < self.suffix + 1:
                    self.history.trajs.append([])
                    if len(self.history.trajs) < self.suffix + 1:
                        raise IndexError('history.prod_trajs is the wrong length for thread: ' +
                                         self.history.inpcrd[0] + '\nexpected length: ' + str(self.suffix))
                self.history.trajs[self.suffix].append(kwargs['nc'])

    def analyze(self, settings):
        self.history.score.append(utilities.lie(self.history.trajs[-1], self.history.tops[-1], settings))

    def algorithm(self, settings):
        # Get next mutation to apply from the desired algorithm, or else terminate
        this_algorithm = factory.algorithm_factory(settings.algorithm)
        next_step = this_algorithm.get_next_step(None, self, settings)
        if next_step == 'TER':  # algorithm says global termination     # todo: this should be thread termination, right? Global termination should be determined by some higher level process called in the main loop in main.py, I think.
            return True
        self.history.muts.append(next_step)

        # Get index of previous step to use as base for next step
        base_index = 0  # for now we'll just use the original input structure

        # Perform desired mutation
        new_inpcrd, new_top = utilities.mutate(self.history.inpcrd[base_index], self.history.tops[base_index], next_step, settings)

        # Update history and return
        self.history.inpcrd.append(new_inpcrd)
        self.history.tops.append(new_top)
        self.suffix += 1

        return False
