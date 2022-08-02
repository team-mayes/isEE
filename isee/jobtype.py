"""
Interface for JobType objects. New JobTypes can be implemented by constructing a new class that inherits from JobType
and implements its abstract methods.
"""

import abc
import os
import sys
import time
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
            The Thread object on which to operate
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
            The Thread object on which to operate
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
            The Thread object on which to operate

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
            The Thread object on which to operate
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
        Perform necessary analysis of a completed simulation step and store results as appropriate into thread.history.
        If the simulation did not apparently succeed for whatever reason, analyze returns False and terminates.

        Parameters
        ----------
        thread : Thread
            The Thread object on which to operate
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        okay : bool
            True if the simulation went okay, false if it needs to be redone or skipped

        """

        pass

    @abc.abstractmethod
    def algorithm(self, thread, allthreads, settings):
        """
        Implement the algorithm that determines the next step and sets up the appropriate thread.history attributes

        Parameters
        ----------
        thread : Thread
            The Thread object on which to operate
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
    def gatekeeper(self, thread, allthreads, settings):
        """
        Return boolean indicating whether job is ready for next interpretation step.

        Parameters
        ----------
        thread : Thread
            The Thread object on which to operate
        allthreads : list
            List of all thread objects to consider
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
        if thread.history.muts and thread.history.muts[-1]:
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
                thread.history.trajs = []             # list of lists of strings; updated by update_history() called by process.py
                thread.history.tops = []              # list of strings; initialized by main.init_threads(), updated by algorithm
                thread.history.muts = []              # list of lists of strings describing mutations tested; updated by algorithm
                thread.history.score = []             # list of scores; updated by analyze()
                thread.history.timestamps = []        # list of ints representing seconds since the epoch for the end of each step; initialized by main.init_threads(), updated by algorithm
            if not os.path.exists(settings.working_directory + '/algorithm_history.pkl'):     # initialize algorithm_history file if necessary # todo: deprecate?
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
        if settings.degeneracy > 1: # todo: fix (I don't know what the problem is, if there even still is one)
            print('skipping analyze step because it is currently incompatible with degeneracy > 1. This also means that'
                  ' the storage directory (if specified) will not be created. Do your analysis manually.')
            return True
        elif settings.skip_analyze:
            return True
        if not settings.SPOOF:  # default behavior
            if not all([os.path.exists(traj) for traj in thread.history.trajs[-1]]) or any([pytraj.iterload(traj, thread.history.tops[-1]).n_frames == 0 for traj in thread.history.trajs[-1]]):     # if the simulation didn't produce a trajectory
                return False
            if settings.storage_directory:  # move a 'dry' copy to storage, if we have a storage directory
                try:
                    dry_trajs = []
                    for traj in thread.history.trajs[-1]:
                        dry_traj, dry_top = utilities.strip_and_store(traj, thread.history.tops[-1], settings)   # I honestly have no idea why thread.history.trajs[-1] is a list here when it was a string just before?
                        dry_trajs.append(dry_traj)
                except:
                    raise RuntimeError('strip_and_store failed with files: ' + thread.history.trajs[-1][0] + ' and ' +
                                       thread.history.tops[-1])
                thread.history.score.append(utilities.lie(dry_traj, dry_top, settings))

                # Write results in a human-readable format
                if not os.path.exists(settings.storage_directory + '/results.out'):
                    open(settings.storage_directory + '/results.out', 'w').close()

                with open(settings.storage_directory + '/results.out', 'a') as f:
                    f.write(str(thread.history.muts[-1]) + ': ' + str(thread.history.score[-1]) + '\n')

            else:
                thread.history.score.append(utilities.lie(thread.history.trajs[-1][0], thread.history.tops[-1], settings))
        else:   # spoof behavior
            thread.history.score.append(utilities.score_spoof(settings.seq, settings.rmsd_covar, settings))

        return True

    def algorithm(self, thread, allthreads, settings):
        # Get next mutation to apply from the desired algorithm, or else terminate
        this_algorithm = factory.algorithm_factory(settings.algorithm)

        if not thread.history.trajs:  # if this is the first step in this thread
            next_step = this_algorithm.get_first_step(thread, allthreads, settings)
        else:
            next_step = this_algorithm.get_next_step(thread, allthreads, settings)

        if next_step == 'WT':   # do nothing, correct structure is already set in history.tops and history.inpcrd
            # thread.history.muts.append([])  # empty muts entry, for consistency in indexing
            next_step = ['WT']

        elif next_step == 'IDLE':
            thread.idle = True
            return False        # False: do not globally terminate

        thread.idle = False       # explicitly reset idle to False whenever we get to this step

        if next_step == 'TER':  # algorithm says thread termination
            thread.terminated = True
            return False        # False: do not globally terminate

        # Perform desired mutation
        # todo: implement possibility of mutating using something other than initial coordinates/topology as a base?
        if '/' in settings.init_topology:   # todo: this shouldn't be necessary because it should already be done in main.init_threads
            settings.init_topology = settings.init_topology[settings.init_topology.rindex('/') + 1:]
        initial_coordinates_to_mutate = settings.initial_coordinates[0]
        if '/' in initial_coordinates_to_mutate:
            initial_coordinates_to_mutate = initial_coordinates_to_mutate[initial_coordinates_to_mutate.rindex('/') + 1:]
        if next_step == ['WT']:
            mutations = ['']    # need to call mutate for WT to apply ts_bonds to the topology file
        else:
            mutations = next_step
        new_inpcrd, new_top = utilities.mutate(initial_coordinates_to_mutate, settings.init_topology, mutations, initial_coordinates_to_mutate + '_' + '_'.join(next_step), settings)

        # Update history and return
        thread.history.inpcrd.append(new_inpcrd)
        thread.history.tops.append(new_top)
        thread.history.muts.append(next_step)
        thread.history.timestamps.append(time.time())
        thread.suffix += 1

        if not thread.history.trajs:    # if this is the first step in this thread
            thread.history.inpcrd = [thread.history.inpcrd[-1]]
            thread.history.tops = [thread.history.tops[-1]]
            thread.history.muts = [thread.history.muts[-1]]
            thread.history.timestamps = [thread.history.timestamps[-1]]
            thread.suffix = 0

        # Update history file so other threads can see what this one is up to
        this_algorithm.build_algorithm_history(allthreads, settings)

        # Check thread-level termination criteria to see if thread.terminated should be True
        # Do this last so that thread can be restarted without repeating previous step
        if settings.max_steps_per_thread:
            if thread.moves_this_time >= settings.max_steps_per_thread:
                thread.terminated = True
                return False    # False: do not globally terminate

        return False    # False: do not terminate

    def gatekeeper(self, thread, allthreads, settings):
        # First, check whether this thread is intentionally idling and if so, let the algorithm guide us
        if thread.idle:
            # We should be sure at this point that something hasn't gone wrong and idled ALL the threads...
            if all([this_thread.idle for this_thread in allthreads]):
                # Reevaluate idle for all threads
                this_algorithm = factory.algorithm_factory(settings.algorithm)
                for thread in allthreads:
                    thread.idle = this_algorithm.reevaluate_idle(thread, allthreads)

                # If problem persists...
                if all([this_thread.idle for this_thread in allthreads]):
                    raise RuntimeError('all threads are in an idle state, which means something must have gone wrong. '
                                   'Inspect the restart.pkl file for errors.')
            this_algorithm = factory.algorithm_factory(settings.algorithm)
            return this_algorithm.reevaluate_idle(thread, allthreads)

        # If job for this thread has status 'C'ompleted/'C'anceled...
        if all([thread.get_status(ii, settings) == 'C' for ii in range(len(thread.jobids))]):
            # todo: implement restarting if simulation crashed before a certain number of steps completed?
            return True
        else:
            return False
