"""
Interface for Algorithm objects. New Algorithms can be implemented by constructing a new class that inherits from
Algorithm and implements its abstract methods.
"""

import os
import re
import abc
import time
import shutil
import pickle
import mdtraj
import argparse
import subprocess
from isee.infrastructure import factory
from isee import utilities

def buffer_history(algorithm_history):
    """
    Add properly typed placeholder entries to the end of each attribute of algorithm_history in order to keep each
    attribute of the same length.

    All attributes of a properly formatted algorithm_history object are lists, where a given index across any such list
    contains corresponding information. This function simply appends blank placeholder entries to lists that are too
    short, to facilitate cleanly maintaining this feature even when not every list in the object is ready to be updated.

    If the type is not recognized, None is used instead. None is also used where the list contains ints, floats, or
    bools, as using zero or False in this case is more likely to be accidentally misinterpreted as data instead of as
    the absence thereof, as intended.

    Parameters
    ----------
    algorithm_history : argparse.Namespace
        The object on which to act

    Returns
    -------
    None

    """

    # Identify the length of the longest list
    max_len = max([len(item) for item in [algorithm_history.__getattribute__(att) for att in algorithm_history.__dict__.keys()]])

    # Now append the appropriate placeholders where needed
    for attribute in algorithm_history.__dict__.keys():
        try:
            this_type = type(algorithm_history.__getattribute__(attribute)[0])   # type of 0th object in attribute
        except IndexError:  # empty list, no types available
            this_type = type(None)

        to_append = None
        # iterate through objects of various types that evaluate with bool() to False
        # we exclude int, float, and bool examples because these should use None regardless
        for potential_type in ['', None, [], (), {}]:
            if this_type == type(potential_type):
                to_append = potential_type
                break

        while len(algorithm_history.__getattribute__(attribute)) < max_len:
            algorithm_history.__getattribute__(attribute).append(to_append)


class Algorithm(abc.ABC):
    """
    Abstract base class for isEE algorithms.

    Implements methods for all of the task manager-specific tasks that isEE might need.

    """

    def __init__(self):
        ### Load algorithm_history from algorithm_history.pkl ###
        if os.path.exists('algorithm_history.pkl'):
            self.algorithm_history = pickle.load(open('algorithm_history.pkl', 'rb'))
        else:
            raise FileNotFoundError('algorithm_history.pkl not found; it should have been created when the first '
                                    'thread was initialized. Did you do something unusual?')

    @staticmethod
    def dump_and_return(to_return, algorithm_history):
        # Helper function to dump pickle before returning
        pickle.dump(algorithm_history, open('algorithm_history.pkl.bak', 'wb'))
        if not os.path.getsize('algorithm_history.pkl.bak') == 0:
            shutil.copy('algorithm_history.pkl.bak', 'algorithm_history.pkl')  # copying after is safer

        return to_return

    @abc.abstractmethod
    def get_first_step(self, thread, allthreads, settings):
        """
        Determine the first step for a thread. This method should be called before the first 'process' step after a new
        thread is created (as defined by having an empty thread.history.trajs attribute) to allow the algorithm to
        decide what its first step should be.

        Specifically, implementations of this method should either pass on all threads after the first one directly to
        get_next_step (for algorithms where the next step can be determined without information from the first one), or
        else idle each thread after the first one.

        Parameters
        ----------
        thread : Thread()
            Thread object to consider
        allthreads : list
            List of all thread object to consider
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        first_step : str
            Next mutation to apply to the system, formatted as <resid><three-letter-code>; alternatively, 'TER' if
            terminating or 'IDLE' if doing nothing. Finally, can also return 'WT' to indicate that the simulation should
            proceed with the structure passed directly to isEE, without mutation.

        """

        pass

    @abc.abstractmethod
    def get_next_step(self, thread, settings):
        """
        Determine the next step for the thread, or return 'TER' if there is none or 'IDLE' if information from other
        threads is required before the next step can be determined.

        Implementations of this method should always begin by loading a global history object from
        algorithm_history.pkl, which is a namespace object with the same attributes as a thread.history object, which
        should have been created by JobType.update_history the first time it was called. They should then end by dumping
        their copy of the algorithm history to that file. The purpose of this file is to keep track of the global
        history of the algorithm so far in order to facilitate parallelization among multiple threads.

        In general, whenever a new item is added to any of the lists in the algorithm history object, something should
        also be added to the corresponding index of every other list, even if only a placeholder. If the new item is
        being appended to the end of a list, then this requirement can be satisfied by calling buffer_history() on the
        algorithm history object after appending the desired information.

        In the case where necessary information for get_next_step is still being awaited from an assigned thread, this
        function should return the string 'IDLE', to indicate that the thread should do nothing. The functions of 'IDLE'
        and 'TER' are defined by process.process().

        Parameters
        ----------
        thread : Thread()
            Thread object to consider
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        next_step : str
            Next mutation to apply to the system, formatted as <resid><three-letter-code>; alternatively, 'TER' if
            terminating or 'IDLE' if doing nothing.

        """

        pass

    @abc.abstractmethod
    def reevaluate_idle(self, thread):
        """
        Determine if the given idle thread is ready to be passed along to the next step or should remain idle. The
        result from this method can be used as a stand-in for the results of a jobtype.gatekeeper implementation when
        called on an idle thread.

        Parameters
        ----------
        thread : Thread()
            Thread object to consider

        Returns
        -------
        gatekeeper : bool
            If True, then the thread is ready for the next "interpret" step. If False, then it should remain idle.

        """

        pass


class Script(Algorithm):
    """
    Adapter class for the "algorithm" that is simply following the "script" of mutations prescribed by the user in the
    settings.mutation_script list.

    This particular implementation of Algorithm has no conditions that return 'IDLE'.

    """

    def get_first_step(self, thread, allthreads, settings):
        # if this is the first thread in allthreads and there are no history.trajs objects in any threads yet
        if thread == allthreads[0] and not any([bool(item.history.trajs) for item in allthreads]):
            return 'WT'
        else:
            return Script.get_next_step(self, thread, settings) # does calling this here make this method the super?

    def get_next_step(self, thread, settings):
        untried = [item for item in settings.mutation_script if not item in self.algorithm_history.muts]
        try:
            self.algorithm_history.muts.append(untried[0])
            buffer_history(self.algorithm_history)
            return self.dump_and_return(untried[0], self.algorithm_history)   # first untried mutation
        except IndexError:  # no untried mutation remains
            return self.dump_and_return('TER', self.algorithm_history)

    def reevaluate_idle(self, thread):
        return True    # this algorithm never returns 'IDLE', so it's always ready for the next step


class CovarianceSaturation(Algorithm):
    """
    Adapter class for algorithm that finds the residue with the minimum RMSD of covariance profile within the enzyme and
    then performs saturation mutagenesis before repeating.

    RMSD reference is determined by settings.covariance_reference_resid

    """

    def get_first_step(self, thread, allthreads, settings):
        # if this is the first thread in allthreads and there are no history.trajs objects in any threads yet
        if thread == allthreads[0] and not any([bool(item.history.trajs) for item in allthreads]):
            return 'WT'
        else:
            return 'IDLE'   # need to wait for first simulation to finish before proceeding

    def get_next_step(self, thread, settings):
        all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

        if self.algorithm_history.muts == []:   # first ever mutation
            saturation = True   # saturation = True means: pick a new residue to mutate

        # Otherwise, saturation is defined by having tried all the residues in all_resnames (excluding the wild type)
        # elif (the last (len(all_resnames) - 1) mutation resnames are all unique) and\
        #   (all of the last (len(all_resnames) - 1) mutation resnames are in all_resnames) and\
        #   (all of the last (len(all_resnames) - 1) mutation resids are equal to the last resid):
        elif len(set([item[-3:] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]])) == (len(all_resnames) - 1) and\
                set([item[-3:] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]]).issubset(set(all_resnames))\
                and all([item[:-3] == self.algorithm_history.muts[-1][:-3] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]]):
            saturation = True

        else:
            saturation = False

        if saturation:
            # Perform RMSD profile analysis
            rmsd_covar = utilities.covariance_profile(thread, -1, settings)     # -1: operate on most recent trajectory

            # Pick minimum RMSD residue that hasn't already been done
            already_done = set([int(item[:-3]) for item in self.algorithm_history.muts] + [settings.covariance_reference_resid])
            paired = []
            i = 0
            for value in rmsd_covar:
                i += 1
                paired.append([i, value])       # paired resid, rmsd-of-covariance
            paired = sorted(paired, key=lambda x: x[1])         # sorted by ascending rmsd
            resid = settings.covariance_reference_resid
            this_index = 0
            while resid in already_done:    # iterate through paired in order of ascending rmsd until resid is unused
                resid = paired[this_index][0]
                this_index += 1
                if this_index >= len(paired):    # if there are no more residues to mutate
                    return self.dump_and_return('TER', self.algorithm_history)

            next_mut = str(int(resid)) + all_resnames[0]

            algorithm_history.muts.append(next_mut)
            buffer_history(self.algorithm_history)
            return self.dump_and_return(next_mut, self.algorithm_history)
        else:   # unsaturated, so pick an unused mutation on the same residue as the previous mutation
            # First, we need to generate a list of all the mutations that haven't been tried yet on this residue
            done = [item[-3:] for item in self.algorithm_history.muts if item[:-3] == self.algorithm_history.muts[-1][:-3]]
            todo = [item for item in all_resnames if not item in done]

            # Then, remove the wild type residue name from the list
            mtop = mdtraj.load_prmtop(thread.history.tops[0])   # 0th topology corresponds to the "wild type" here
            wt = mtop.residue(int(self.algorithm_history.muts[-1][:-3]) - 1)  # mdtraj topology is zero-indexed, so -1
            if str(wt)[:3] in todo:
                todo.remove(str(wt)[:3])   # wt is formatted as <three letter code><zero-indexed resid>

            next_mut = self.algorithm_history.muts[-1][:-3] + todo[0]

            algorithm_history.muts.append(next_mut)
            buffer_history(self.algorithm_history)
            return self.dump_and_return(next_mut, self.algorithm_history)

    def reevaluate_idle(self, thread):
        # The condition to meet for this algorithm to allow an idle thread to resume is simply that the simulation for
        # the first system (non-mutated) is finished and has had get_next_step called on it
        if os.path.exists('algorithm_history.pkl'):
            algorithm_history = pickle.load(open('algorithm_history.pkl', 'rb'))
            if algorithm_history.muts:
                return True
            else:
                return False
        else:
            return False


class SubnetworkHotspots(Algorithm):
    """
    Adapter class for algorithm that finds subnetworks within the protein structure (as defined by ___) and then uses
    the same method as the CovarianceSaturation algorithm to identify hotspots within subnetworks and perform saturation
    mutagenesis on them. After saturation within each

    """

    def get_first_step(self, thread, allthreads, settings):
        pass

    def get_next_step(self, thread, settings):
        pass

    def reevaluate_idle(self, thread):
        pass


if __name__ == '__main__':
    jobtype = factory.jobtype_factory('isee')
    thread = argparse.Namespace()
    settings = argparse.Namespace()
    settings.mutation_script = ['64ASN']
    settings.algorithm = 'script'
    jobtype.update_history(thread, settings, **{'initialize': True})
    thread.history.trajs = ['extant']
    test = Script()
    next = test.get_first_step(thread, [thread], settings)
    print(next)
