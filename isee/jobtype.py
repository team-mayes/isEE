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
        Obtain list of the appropriate initial coordinate files and copy them to the working directory.

        Parameters
        ----------
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        initial_coordinates : list
            List of strings naming the applicable initial coordinate files that were copied to the working directory

        """

        pass

    @abc.abstractmethod
    def get_next_step(self, thread, settings):
        """
        Determine and return name for next step in the thread given by "self"

        Parameters
        ----------
        thread : Thread()
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
    def get_input_file(self, thread, settings):
        """
        Obtain appropriate input file for next job.

        At its most simple, implementations of this method can simply return settings.path_to_input_files + '/' +
        settings.job_type + '_' + settings.md_engine + '.in'

        Parameters
        ----------
        thread : Thread
            Methods in the JobType abstract base class are intended to operate on Thread objects
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
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        name : str
            Name of the batch file template requested

        """

        pass

    @abc.abstractmethod
    def get_struct(self, thread):
        """
        Return the name of the appropriate inpcrd and topology files for the next step in the thread given by self.

        Parameters
        ----------
        thread : Thread
            Methods in the JobType abstract base class are intended to operate on Thread objects

        Returns
        -------
        inpcrd : str
            String containing desired coordinate file name
        top : str
            String containing desired topology file name

        """

        pass

    @abc.abstractmethod
    def update_history(self, thread, settings, **kwargs):
        """
        Update or initialize the history namespace for this job type.

        This namespace is used to store the full history of a threads coordinate and trajectory files, as well as their
        results if necessary.

        If update_history is called with a kwargs containing {'initialize': True}, it simply prepares a blank
        history namespace and returns it. Otherwise, it adds the values of the desired keywords (which are desired
        depends on the implementation) to the corresponding history attributes in the index given by thread.suffix.

        Implementations of update_history should always save a copy of a freshly initialized thread.history object in a
        pickle file named 'algorithm_history.pkl' if one does not already exist. This is to support parallelization of
        algorithms across threads while still keeping the "shape" of the history object defined in one place (namely,
        in the implementation of update_history.)

        Parameters
        ----------
        thread : Thread
            Methods in the JobType abstract base class are intended to operate on Thread objects
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
    def analyze(self, thread, settings):
        """
        Perform necessary analysis of a completed simulation step and store results as appropriate into thread.history

        Parameters
        ----------
        thread : Thread
            Methods in the JobType abstract base class are intended to operate on Thread objects
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        None

        """

        pass

    @abc.abstractmethod
    def algorithm(self, thread, allthreads, settings):
        """
        Implement the algorithm that determines the next step and sets up the appropriate thread.history attributes

        Parameters
        ----------
        thread : Thread
            Methods in the JobType abstract base class are intended to operate on Thread objects
        allthreads : list
            A list of all of the thread objects to consider during the algorithm.
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        terminate : Bool
            If True, terminate the entire isEE job

        """

        pass

    @abc.abstractmethod
    def gatekeeper(self, thread, settings):
        """
        Return boolean indicating whether job is ready for next interpretation step.

        Parameters
        ----------
        thread : Thread
            Methods in the JobType abstract base class are intended to operate on Thread objects
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        status : bool
            If True, ready for next interpretation step; otherwise, False

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

    def get_next_step(self, thread, settings):
        if thread.history.muts:
            return '_'.join(thread.history.muts[-1])
        else:
            return 'unmutated'

    def get_input_file(self, thread, settings):
        return settings.path_to_input_files + '/' + settings.job_type + '_' + settings.md_engine + '.in'

    def get_batch_template(self, settings):
        templ = settings.md_engine + '_' + settings.batch_system + '.tpl'
        if os.path.exists(settings.path_to_templates + '/' + templ):
            return templ
        else:
            raise FileNotFoundError('cannot find required template file: ' + templ)

    def get_struct(self, thread):
        return thread.history.inpcrd[-1], thread.history.tops[-1]

    def update_history(self, thread, settings, **kwargs):
        if 'initialize' in kwargs.keys():
            if kwargs['initialize']:
                thread.history = argparse.Namespace()
                thread.history.inpcrd = []            # list of strings; initialized by main.init_threads(), updated by algorithm
                thread.history.trajs = []             # list of strings; updated by update_history() called by process.py
                thread.history.tops = []              # list of strings; initialized by main.init_threads(), updated by algorithm
                thread.history.muts = []              # list of lists of strings describing mutations tested; updated by algorithm
                thread.history.score = []             # list of scores; updated by analyze()
                thread.history.timestamps = []        # list of ints representing seconds since the epoch for the end of each step, updated by ?
            if not os.path.exists(settings.working_directory + '/algorithm_history.pkl'):     # initialize algorithm_history file if necessary
                pickle.dump(thread.history, open(settings.working_directory + '/algorithm_history.pkl', 'wb'))  # an empty thread.history template
            if 'inpcrd' in kwargs.keys():
                thread.history.inpcrd.append(kwargs['inpcrd'])
        else:   # thread.history should already exist
            if 'nc' in kwargs.keys():
                if len(thread.history.trajs) < thread.suffix + 1:
                    thread.history.trajs.append([])
                    if len(thread.history.trajs) < thread.suffix + 1:
                        raise IndexError('history.prod_trajs is the wrong length for thread: ' +
                                         thread.history.inpcrd[0] + '\nexpected length: ' + str(thread.suffix + 1) +
                                         ', but it is currently: ' + str(thread.history.trajs))
                thread.history.trajs[thread.suffix].append(kwargs['nc'])

    def analyze(self, thread, settings):
        thread.history.score.append(utilities.lie(thread.history.trajs[-1], thread.history.tops[-1], settings))

    def algorithm(self, thread, allthreads, settings):
        # Get next mutation to apply from the desired algorithm, or else terminate
        this_algorithm = factory.algorithm_factory(settings.algorithm)

        if not thread.history.trajs:  # if this is the first step in this thread
            next_step = this_algorithm.get_first_step(thread, allthreads, settings)
        else:
            next_step = this_algorithm.get_next_step(thread, allthreads, settings)

        if next_step == 'WT':   # do nothing, correct structure is already set in history.tops and history.inpcrd
            return False        # False: do not globally terminate

        if next_step == 'IDLE':
            thread.idle = True
            return False        # False: do not globally terminate

        thread.idle = False       # explicitly reset idle to False whenever we get to this step

        if next_step == 'TER':  # algorithm says thread termination
            thread.terminated = True
            return False        # False: do not globally terminate

        # Else, we have a new mutant to direct the thread to build and simulate
        thread.history.muts.append(next_step)

        # Get index of previous step to use as base for next step
        base_index = 0  # for now we'll just use the original input structure

        # Perform desired mutation
        new_inpcrd, new_top = utilities.mutate(thread.history.inpcrd[base_index], thread.history.tops[base_index], next_step, thread.history.inpcrd[base_index] + '_' + '_'.join(next_step), settings)

        # Update history and return
        thread.history.inpcrd.append(new_inpcrd)
        thread.history.tops.append(new_top)
        thread.suffix += 1

        if not thread.history.trajs:    # if this is the first step in this thread
            # reformat thread data to reflect that 'WT' is being skipped
            thread.history.inpcrd = [thread.history.inpcrd[-1]]
            thread.history.tops = [thread.history.tops[-1]]
            thread.history.muts = [thread.history.muts[-1]]
            thread.suffix = 0

        return False    # False: do not terminate

    def gatekeeper(self, thread, settings):
        # first, check whether this thread is intentionally idling and if so, let the algorithm guide us
        if thread.idle:
            this_algorithm = factory.algorithm_factory(settings.algorithm)
            return this_algorithm.reevaluate_idle(self)

        # if job for this thread has status 'C'ompleted/'C'anceled...
        if thread.get_status(0, settings) == 'C':     # index 0 because there is only ever one element in thread.jobids
            return True
        else:
            return False
