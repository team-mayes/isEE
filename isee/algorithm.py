"""
Interface for Algorithm objects. New Algorithms can be implemented by constructing a new class that inherits from
Algorithm and implements its abstract methods.
"""

import os
import re
import sys
import abc
import copy
import time
import numpy
import shutil
import pickle
import mdtraj
import argparse
import itertools
import subprocess
import tensorflow as tf
from isee.infrastructure import factory
from isee import utilities
from math import factorial
from filelock import Timeout, FileLock

## Deprecated
# def buffer_history(algorithm_history):
#     """
#     Add properly typed placeholder entries to the end of each attribute of algorithm_history in order to keep each
#     attribute of the same length.
#
#     All attributes of a properly formatted algorithm_history object are lists, where a given index across any such list
#     contains corresponding information. This function simply appends blank placeholder entries to lists that are too
#     short, to facilitate cleanly maintaining this feature even when not every list in the object is ready to be updated.
#
#     If the type is not recognized, None is used instead. None is also used where the list contains ints, floats, or
#     bools, as using zero or False in this case is more likely to be accidentally misinterpreted as data instead of as
#     the absence thereof, as intended.
#
#     Parameters
#     ----------
#     algorithm_history : argparse.Namespace
#         The object on which to act
#
#     Returns
#     -------
#     None
#
#     """
#
#     # Identify the length of the longest list
#     max_len = max([len(item) for item in [algorithm_history.__getattribute__(att) for att in algorithm_history.__dict__.keys()]])
#
#     # Now append the appropriate placeholders where needed
#     for attribute in algorithm_history.__dict__.keys():
#         try:
#             this_type = type(algorithm_history.__getattribute__(attribute)[0])   # type of 0th object in attribute
#         except IndexError:  # empty list, no types available
#             this_type = type(None)
#
#         to_append = None
#         # iterate through objects of various types that evaluate with bool() to False
#         # we exclude int, float, and bool examples because these should use None regardless
#         for potential_type in ['', None, [], (), {}]:
#             if this_type == type(potential_type):
#                 to_append = potential_type
#                 break
#
#         while len(algorithm_history.__getattribute__(attribute)) < max_len:
#             algorithm_history.__getattribute__(attribute).append(to_append)


class Algorithm(abc.ABC):
    """
    Abstract base class for isEE algorithms.

    Implements methods for all of the algorithm-specific tasks that isEE might need.

    """

    def __init__(self):
        # define backlog list for algorithms that determine more than one step at a time
        if os.path.exists('backlog.pkl'):
            self.backlog = pickle.load(open('backlog.pkl', 'rb'))
        else:
            self.backlog = []

        ## Deprecated
        # ### Load algorithm_history from algorithm_history.pkl ###
        # if os.path.exists('algorithm_history.pkl'):
        #     self.algorithm_history = pickle.load(open('algorithm_history.pkl', 'rb'))
        # else:
        #     raise FileNotFoundError('algorithm_history.pkl not found; it should have been created when the first '
        #                             'thread was initialized. Did you do something unusual?')

    @staticmethod
    def build_algorithm_history(allthreads, settings):
        # First, establish lock so only one instance of build_algorithm_history will run at a time
        # If we have a shared history file, this needs to go in the same directory as that
        if settings.shared_history_file and '/' in settings.shared_history_file:
            lockfile_directory = settings.shared_history_file[:settings.shared_history_file.rindex('/') + 1]
        else:
            lockfile_directory = ''
        lock = FileLock(lockfile_directory + 'algorithm_history.lock')
        with lock:
            open(lockfile_directory + 'arbitrary_lockfile.lock', 'w').close()
        lock.acquire()
        open(lockfile_directory + 'arbitrary_lockfile.lock', 'w').close()    # this step will block until the lock is released

        # Build algorithm_history afresh from all thread history attributes
        if settings.shared_history_file:
            history_file = settings.shared_history_file
        else:
            history_file = 'algorithm_history.pkl'

        if not os.path.exists(history_file):
            algorithm_history = argparse.Namespace()
            for key in allthreads[0].history.__dict__.keys():
                algorithm_history.__dict__[key] = []    # initialize dictionary
        else:
            algorithm_history = pickle.load(open(history_file, 'rb'))

        # # Why did I write this? I specifically don't want this to be done.
        # def append_lists(to_append_to, list_of_lists):
        #     # Helper function that combines arbitrarily nested lists: [-1,0], [[1,2,3],[4,5,6]] -> [-1,0,1,2,3,4,5,6]
        #     temp = to_append_to
        #     for item in list_of_lists:
        #         for subitem in item:
        #             temp.append(subitem)
        #     return temp

        # for key in list(allthreads[0].history.__dict__.keys()):
        #     print(key)
        #     print(thread.history.__dict__[key])
        #     try:
        #         current = algorithm_history.__dict__[key]
        #     except KeyError:
        #         current = []
        #         algorithm_history.__dict__[key] = current
        #
        #     # algorithm_history.__dict__[key] = append_lists(current, [thread.history.__dict__[key] for thread in allthreads])
        #     algorithm_history.__dict__[key] = current + [thread.history.__dict__[key] for thread in allthreads]

        def merge_namespace(current, new):
            # Helper function to merge contents of 'new' namespace into 'current' namespace, assuming both have the same
            # keys, that each key corresponds to a list, and that there are the same number of entries in each list.
            # If a particular "column" of entries in 'new' is already present in 'current' then it is not merged.]))
            try:    # assert same keys present in current and new
                assert set(current.__dict__.keys()) == set(new.__dict__.keys())
            except AssertionError:
                raise RuntimeError('merge_namespace called with namespaces with non-matching keys.\n' +
                                   ' current keys(): ' + str(current.__dict__.keys()) + '\n' +
                                   ' new keys():     ' + str(new.__dict__.keys()))

            try:    # assert each key corresponds to a list
                assert all([isinstance(item, list) for item in [current.__dict__[key] for key in current.__dict__.keys()]])
                assert all([isinstance(item, list) for item in [new.__dict__[key] for key in new.__dict__.keys()]])
            except AssertionError:
                raise RuntimeError('one or both lists in merge_namespace has at least one attribute that is not a list')

            try:    # assert length of all entries in current dictionary is constant or zero
                assert all([len(current.__dict__[key]) in [0, max([len(current.__dict__[kk]) for kk in current.__dict__.keys()])] for key in list(current.__dict__.keys())])
            except AssertionError:
                raise RuntimeError('merge_namespace called with \'current\' namespace with inconsistent row lengths: \n' +
                                   str(current))
            for key in list(current.__dict__.keys()):
                for col in range(max([len(new.__dict__[kk]) for kk in new.__dict__.keys()])):
                    try:
                        current.__dict__[key].append(new.__dict__[key][col])
                    except IndexError:  # no entry at index 'col' for this key
                        current.__dict__[key].append([])     # append blank list to keep columns equal in length

            try:    # assert length of all entries in current dictionary is constant
                assert all([len(current.__dict__[key]) == max([len(current.__dict__[kk]) for kk in current.__dict__.keys()]) for key in list(current.__dict__.keys())])
            except AssertionError:
                raise RuntimeError('merge_namespace PRODUCED \'current\' namespace with inconsistent row lengths '
                                   'despite having been called with a \'current\' and \'new\' with consistent row '
                                   'lengths: \n' + str(current))

            # Remove duplicate and subset (identical but for missing entries) columns from current.__dict__
            allkeys = [key for key in current.__dict__.keys()]  # keys in (any) specific order, for rebuilding
            columns = [tuple([current.__dict__[key][col] for key in allkeys]) for col in range(len(current.__dict__[list(current.__dict__.keys())[0]]))]
            columns = list(k for k,_ in itertools.groupby(columns))     # remove duplicate columns
            while True:     # remove subset columns # todo: nested loops can be very slow, is this okay?
                removed = False
                for ii in range(len(columns)):
                    col = columns[ii]
                    for jj in range(len(columns)):
                        if not jj == ii:
                            cmp = columns[jj]
                            if all([cmp[kk] in [[], col[kk]] for kk in range(len(cmp))]):
                                columns.remove(cmp)
                                removed = True
                                break   # break out of jj loop
                    if removed:
                        break   # break out of ii loop
                if not removed:
                    break   # break while loop; only reachable if no columns were removed this iteration
            current = argparse.Namespace()
            for key in allkeys:
                current.__dict__[key] = []              # re-initialize dictionary
            for col in range(len(columns)):             # rebuild namespace
                kk = 0      # kk keeps track of index within allkey
                for key in allkeys:
                    current.__dict__[key].append(list(columns)[col][kk])
                    kk += 1

            return current

        # Perform merge
        for thread in allthreads:
            algorithm_history = merge_namespace(algorithm_history, thread.history)

        # Sort every column chronologically by timestamp attribute
        for key in list(algorithm_history.__dict__.keys()):
            algorithm_history.__dict__[key] = [x for _,x in sorted(zip(algorithm_history.__dict__['timestamps'], algorithm_history.__dict__[key]))]

        # Dump pickle file to 'algorithm_history.pkl' regardless of settings.shared_history_file, because every working
        # directory should have its own algorithm history object in addition to the shared one.
        pickle.dump(algorithm_history, open('algorithm_history.pkl.bak', 'wb'))
        if not os.path.getsize('algorithm_history.pkl.bak') == 0:
            shutil.copy('algorithm_history.pkl.bak', 'algorithm_history.pkl')  # copying after is safer
        else:
            raise RuntimeError('failed to dump algorithm history pickle file')

        # Copy to shared file if necessary
        if settings.shared_history_file:
            shutil.copy('algorithm_history.pkl', settings.shared_history_file)

        lock.release()
        return algorithm_history

    # @staticmethod
    # def dump_and_return(to_return, algorithm_history):
    #     # Helper function to dump pickle before returning
    #     pickle.dump(algorithm_history, open('algorithm_history.pkl.bak', 'wb'))
    #     if not os.path.getsize('algorithm_history.pkl.bak') == 0:
    #         shutil.copy('algorithm_history.pkl.bak', 'algorithm_history.pkl')  # copying after is safer
    #
    #     return to_return

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
        first_step : list or str
            List of first mutation(s) to apply to the system, formatted as a list of strings of format
            <resid><three-letter-code>; alternatively, 'TER' if terminating or 'IDLE' if doing nothing. Finally, can
            also return 'WT' to indicate that the simulation should roceed with the structure passed directly to isEE,
            without mutation.

        """

        pass

    @abc.abstractmethod
    def get_next_step(self, thread, allthreads, settings):
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
        allthreads : list
            Full list of Thread() objects for this job (including thread)
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        next_step : list or str
            Next mutation(s) to apply to the system, formatted as a list of strings of format
            <resid><three-letter-code>; alternatively, 'TER' if terminating or 'IDLE' if doing nothing.

        """

        pass

    @abc.abstractmethod
    def reevaluate_idle(self, thread, allthreads):
        """
        Determine if the given idle thread is ready to be passed along to the next step or should remain idle. The
        result from this method can be used as a stand-in for the results of a jobtype.gatekeeper implementation when
        called on an idle thread.

        Parameters
        ----------
        thread : Thread()
            Thread object to consider
        allthreads : list
            List of all thread objects to consider

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
        if thread == allthreads[0] and not any([bool(item.history.trajs) for item in allthreads]) and not settings.skip_wt:
            return 'WT'
        else:
            return Script.get_next_step(self, thread, allthreads, settings)

    def get_next_step(self, thread, allthreads, settings):
        algorithm_history = self.build_algorithm_history(allthreads, settings)

        untried = [item for item in settings.mutation_script if not item in algorithm_history.muts]
        try:
            return untried[0]     # first untried mutation
        except IndexError:  # no untried mutation remains
            return 'TER'

    def reevaluate_idle(self, thread, allthreads):
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
        elif len(allthreads[0].history.trajs) >= 2:     # if this is True, then the first trajectory is completed!
            return CovarianceSaturation.get_next_step(self, thread, allthreads, settings)
        else:
            return 'IDLE'   # need to wait for first simulation to finish before proceeding

    def get_next_step(self, thread, allthreads, settings):
        algorithm_history = self.build_algorithm_history(allthreads, settings)

        all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

        if algorithm_history.muts == [['WT']]:   # first ever mutation
            saturation = True   # saturation = True means: pick a new residue to mutate

        # Otherwise, saturation is defined by having tried all the residues in all_resnames (excluding the wild type)
        # elif (the last (len(all_resnames) - 1) mutation resnames are all unique) and\
        #   (all of the last (len(all_resnames) - 1) mutation resnames are in all_resnames) and\
        #   (all of the last (len(all_resnames) - 1) mutation resids are equal to the last resid):
        elif len(set([item[0][-3:] for item in algorithm_history.muts[-1 * (len(all_resnames) - 1):]])) == (len(all_resnames) - 1) and\
                set([item[0][-3:] for item in algorithm_history.muts[-1 * (len(all_resnames) - 1):]]).issubset(set(all_resnames))\
                and all([item[0][:-3] == algorithm_history.muts[-1][0][:-3] for item in algorithm_history.muts[-1 * (len(all_resnames) - 1):]]):
            saturation = True

        else:
            saturation = False

        if saturation:
            # Perform RMSD profile analysis # todo: change to operate on best-scoring mutant yet found
            if not settings.SPOOF:
                rmsd_covar = utilities.covariance_profile(allthreads[0], 0, settings)  # operate on wild type trajectory
            else:
                rmsd_covar = settings.rmsd_covar

            # Pick minimum RMSD residue that hasn't already been done
            already_done = set([int(item[0][:-3]) for item in algorithm_history.muts if not 'WT' in item] + [settings.covariance_reference_resid])
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
                    return 'TER'

            next_mut = [str(int(resid)) + all_resnames[0]]

            return next_mut
        else:   # unsaturated, so pick an unused mutation on the same residue as the previous mutation
            # First, we need to generate a list of all the mutations that haven't been tried yet on this residue
            done = [item[0][-3:] for item in algorithm_history.muts if item[0][:-3] == algorithm_history.muts[-1][0][:-3]]
            todo = [item for item in all_resnames if not item in done]

            # Then, remove the wild type residue name from the list
            if settings.SPOOF:
                mtop = mdtraj.load_prmtop(settings.init_topology)
            else:
                mtop = mdtraj.load_prmtop(thread.history.tops[0])   # 0th topology corresponds to the "wild type" here
            wt = mtop.residue(int(algorithm_history.muts[-1][0][:-3]) - 1)  # mdtraj topology is zero-indexed, so -1
            if str(wt)[:3] in todo:
                todo.remove(str(wt)[:3])   # wt is formatted as <three letter code><zero-indexed resid>

            next_mut = [algorithm_history.muts[-1][0][:-3] + todo[0]]

            return next_mut

    def reevaluate_idle(self, thread, allthreads):
        # The condition to meet for this algorithm to allow an idle thread to resume is simply that the simulation for
        # the first system (non-mutated) is finished and has had get_next_step called on it
        algorithm_history = self.build_algorithm_history(allthreads, settings)

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
    mutagenesis on them. After saturation within each subnetwork is completed individually, combinations of the top
    scoring mutants in each subnetwork are added together in search of complimentary effects

    """

    # todo: this is currently just CovarianceSaturation, update it
    def get_first_step(self, thread, allthreads, settings):
        # if this is the first thread in allthreads and there are no history.trajs objects in any threads yet
        if thread == allthreads[0] and not any([bool(item.history.trajs) for item in allthreads]):
            return 'WT'
        elif len(allthreads[0].history.trajs) >= 2:  # if this is True, then the first trajectory is completed!
            return SubnetworkHotspots.get_next_step(self, thread, allthreads, settings)
        else:
            return 'IDLE'   # need to wait for first simulation to finish before proceeding

    def get_next_step(self, thread, allthreads, settings):
        algorithm_history = self.build_algorithm_history(allthreads, settings)

        if not settings.TEST:
            all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']
        else:   # much-truncated list to skip much of the busywork, for testing purposes
            all_resnames = ['ALA', 'GLY']

        if algorithm_history.muts == []:   # first ever mutation
            saturation = True   # saturation = True means: pick a new residue to mutate

        # Otherwise, saturation is defined by having tried all the residues in all_resnames (excluding the wild type)
        # elif (the last (len(all_resnames) - 1) mutation resnames are all unique) and\
        #   (all of the last (len(all_resnames) - 1) mutation resnames are in all_resnames) and\
        #   (all of the last (len(all_resnames) - 1) mutation resids are equal to the last resid):
        elif len(set([item[0][-3:] for item in algorithm_history.muts[-1 * (len(all_resnames) - 1):]])) == (len(all_resnames) - 1) and\
                set([item[0][-3:] for item in algorithm_history.muts[-1 * (len(all_resnames) - 1):]]).issubset(set(all_resnames)) and\
                all([item[0][:-3] == algorithm_history.muts[-1][0][:-3] for item in algorithm_history.muts[-1 * (len(all_resnames) - 1):]]):
            saturation = True

        # Alternatively, if there are any double or higher mutants in the algorithm history, consider it saturated
        elif any([len(item) > 1 for item in algorithm_history.muts]):
            saturation = True

        else:
            saturation = False

        if saturation:
            # Perform RMSD profile analysis
            rmsd_covar = utilities.covariance_profile(allthreads[0], 0, settings)     # operate on wild type trajectory

            # Calculate subnetworks
            subnetworks = self.get_subnetworks()

            # Pick minimum RMSD residue for a subnetwork that hasn't already been done
            already_done = set([int(item[0][:-3]) for item in algorithm_history.muts])

            # Choose next unmutated subnetwork
            if self.no_unmut_subnets(allthreads): # no more subnetworks unmutated, so now do combinations
                # This needs to not proceed until all the scores for single mutants have been recorded, so begin by
                # querying reevaluate_idle to see if we need to idle...
                if not self.reevaluate_idle(thread, allthreads):
                    return 'IDLE'

                # Need to build a list of lists containing combinations of the most promising single mutants
                # This will require pairing history.muts and history.score data (can I keep track of score data in algorithm_history?)
                score_and_muts = []   # full paired set of scores and single mutants across all threads to date
                for this_thread in allthreads:
                    for step_index in range(len(this_thread.history.score)):
                        if len(this_thread.history.muts[step_index]) == 1:   # skip anything not a single mutant
                            score_and_muts.append([this_thread.history.score[step_index], this_thread.history.muts[step_index][0]])
                # Now, prune all mutants who are not the best-scoring at that index
                score_and_muts = sorted(score_and_muts, key=lambda x: int(x[1][:-3]))  # sort by mutation index
                score_and_muts.append([0,'None'])   # kludge to make the coming for loop work properly
                this_mut = score_and_muts[0][1]
                best_index = 0
                best_scorers = []   # list of best single mutations at each attempted index
                for item_index in range(len(score_and_muts)):
                    if not score_and_muts[item_index][1][:-3] == this_mut[:-3]:
                        this_mut = score_and_muts[item_index][1][0]
                        best_scorers.append(score_and_muts[best_index][1])
                        best_index = item_index
                    if score_and_muts[item_index][0] < score_and_muts[best_index][0]:
                        best_index = item_index

                # Finally, construct list of combinations to attempt, and pick one:
                combinations = [list(item) for item in list(itertools.combinations(best_scorers, 2)) if not list(item) in self.algorithm_history.muts]
                if combinations:    # if undone combinations remain
                    return combinations[0]
                else:
                    return 'TER'

            # Pick a new mutation in the chosen unmutated subnetwork; reached only if unmutated subnetworks remain
            next_subnetwork = min([subnet_index for subnet_index in range(len(subnetworks)) if not any([int(mut[0][:-3]) in subnetworks[subnet_index] for mut in self.algorithm_history.muts])])
            subnetwork = subnetworks[next_subnetwork]
            paired = []
            for i in range(len(subnetwork)):
                paired.append([subnetwork[i], rmsd_covar[i]])   # paired resid, rmsd-of-covariance
            paired = sorted(paired, key=lambda x: x[1])         # sorted by ascending rmsd
            resid = settings.covariance_reference_resid
            this_index = 0
            while resid in [already_done] + [settings.covariance_reference_resid]:    # iterate through paired in order of ascending rmsd until resid is unused
                resid = paired[this_index][0]
                this_index += 1
                if this_index >= len(paired):    # if there are no more residues to mutate; should be inaccessible
                    return 'TER'

            next_mut = [str(int(resid)) + all_resnames[0]]

            return next_mut
        else:   # unsaturated, so pick an unused mutation on the same residue as the previous mutation
            # First, we need to generate a list of all the mutations that haven't been tried yet on this residue
            done = [item[0][-3:] for item in algorithm_history.muts if item[0][:-3] == algorithm_history.muts[-1][0][:-3]]
            todo = [item for item in all_resnames if not item in done]

            # Then, remove the wild type residue name from the list
            mtop = mdtraj.load_prmtop(thread.history.tops[0])   # 0th topology corresponds to the "wild type" here
            wt = mtop.residue(int(algorithm_history.muts[-1][0][:-3]) - 1)  # mdtraj topology is zero-indexed, so -1
            if str(wt)[:3] in todo:
                todo.remove(str(wt)[:3])   # wt is formatted as <three letter code><zero-indexed resid>
            else:
                print('WARNING: mutating residue whose wild type residue name is not on the list of possible residue '
                      'names: ' + wt + '\nThis may be fine or it may reflect a critical error, so consider carefully.')

            next_mut = [algorithm_history.muts[-1][0][:-3] + todo[0]]

            return next_mut

    def reevaluate_idle(self, thread, allthreads):
        # The first condition to meet for this algorithm to allow an idle thread to resume is simply that the simulation
        # for the first system (non-mutated) is finished and has had get_next_step called on it
        algorithm_history = self.build_algorithm_history(allthreads, settings)

        if os.path.exists('algorithm_history.pkl'):
            algorithm_history = pickle.load(open('algorithm_history.pkl', 'rb'))
            if algorithm_history.muts:
                # The second condition is that, if and only if there are no unmutated subnetworks, there must be a
                # recorded score for each one
                if self.no_unmut_subnets(allthreads):
                    if len(list(itertools.chain.from_iterable([this_thread.history.score for this_thread in allthreads]))) == len(list(itertools.chain.from_iterable([this_thread.history.muts for this_thread in allthreads]))):
                        return True
                    else:
                        return False
                return True
            else:
                return False
        else:
            return False

    def no_unmut_subnets(self, allthreads):
        # Determine from algorithm_history whether there are any subnetworks not containing at least one single mutant
        algorithm_history = self.build_algorithm_history(allthreads, settings)

        subnetworks = self.get_subnetworks()
        if len(set([item[0][:-3] for item in algorithm_history.muts if not item == []])) >= len(subnetworks):    # todo: should/can I do better than this?
            return True
        else:
            return False

    def get_subnetworks(self):
        ### KLUDGE KLUDGE KLUDGE ###
        # todo: remove kludge, implement general solution
        # A list of lists of 1-indexed resids constituting each subnetwork
        # This one is intentionally ordered to put the subnetwork containing the catalytic residue first.
        return [[399, 188, 47, 45, 361, 394, 266, 53, 181, 49, 185, 71, 379, 52, 50, 67, 70, 54, 180, 360, 177, 182, 64, 178,
          363, 69, 51, 362, 260, 56, 261, 418, 48, 365, 367, 357, 72, 44, 41, 176, 368, 391, 392, 393, 184, 395, 396,
          397, 265, 414, 311, 55, 61, 263, 262, 398, 42, 46, 369, 304, 358, 282, 66, 364, 359, 415, 416, 417, 68, 419,
          420, 421, 413, 179, 43, 307, 57, 366, 422, 63, 58, 412, 62, 65, 390, 183, 73, 39, 40, 74, 308, 400],
         [440, 402, 371, 167, 351, 171, 310, 376, 352, 316, 349, 276, 173, 378, 377, 313, 122, 277, 441, 125, 22, 175,
          23, 165, 124, 370, 121, 348, 315, 401, 306, 172, 372, 309, 123, 314],
         [29, 60, 344, 32, 410, 320, 409, 87, 408, 343, 405, 404, 323, 88, 341, 327, 342, 339, 27, 25, 317, 89, 31, 322,
          319, 33, 90, 438, 406, 338, 345, 336, 403, 28, 407, 30, 346, 278, 335, 380, 347, 34, 340, 305, 321, 284, 439,
          26, 286, 318, 285],
         [134, 152, 153, 120, 92, 111, 325, 210, 96, 97, 98, 99, 100, 101, 102, 21, 104, 105, 106, 107, 108, 109, 110,
          209, 112, 113, 114, 115, 116, 117, 118, 20, 91, 324, 93, 119, 24, 350, 94, 146, 128, 129, 130, 131, 132, 133,
          326, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 127, 147, 148, 149, 150, 151, 95, 103, 154, 155,
          156, 157, 158, 159, 160, 161, 162, 163, 164, 126],
         [234, 264, 3, 275, 218, 170, 2, 10, 11, 9, 1, 12, 13, 14, 15, 169, 168, 16, 374, 19, 186, 187, 353, 206, 190,
          191, 192, 193, 194, 195, 196, 197, 198, 199, 8, 201, 224, 203, 204, 375, 202, 207, 208, 205, 18, 17, 211, 213,
          214, 215, 216, 217, 312, 356, 5, 221, 222, 223, 189, 258, 226, 227, 228, 229, 230, 231, 232, 233, 7, 235, 236,
          237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 225,
          166, 4, 212, 174, 219, 220, 259, 354, 267, 268, 269, 270, 271, 272, 373, 274, 273, 200, 6, 355],
         [435, 434, 384, 38, 430, 427, 431, 426, 287, 288, 383, 300, 436, 293, 78, 280, 291, 386, 299, 298, 297, 333,
          303, 81, 84, 82, 387, 388, 334, 85, 295, 79, 425, 76, 294, 83, 80, 292, 385, 301, 289, 296, 75, 411, 283, 279,
          37, 302, 437, 328, 329, 86, 331, 59, 35, 428, 433, 332, 337, 389, 424, 432, 290, 382, 77, 281, 330, 423, 429,
          381, 36]]
        ### KLUDGE KLUDGE KLUDGE ###


class MonteCarlo(Algorithm):
    """
    Adapter class that chooses new mutants to explore using a Monte Carlo selection algorithm.

    The basic idea is that mutants are chosen randomly with likelihood proportional to some known distribution, in this
    case, the RMSD-of-covariance to a reference residue. Multiple mutants are also included by averaging the weights of
    each individual position contributing to them and applying a penality for each additional position above 1. This
    distribution is assumed to be proportional to the underlying "importance" of each mutant (a vague concept).

    The above paragraph describes the method of choosing residue positions, but not the residues that they're mutated
    to. For this, each possible mutant is considered separately (so there are 19 possible mutations in the pool per
    position). In this way positions that have already been sampled for some mutants will naturally be weighted less.
    """

    def get_first_step(self, thread, allthreads, settings):
        # if this is the first thread in allthreads and there are no history.trajs objects in any threads yet
        if thread == allthreads[0] and not any([bool(item.history.trajs) for item in allthreads]) and not settings.skip_wt:
            return 'WT'
        else:
            return MonteCarlo.get_next_step(self, thread, allthreads, settings)

    def get_next_step(self, thread, allthreads, settings):
        # todo: implement any sort of termination criterion?

        algorithm_history = self.build_algorithm_history(allthreads, settings)

        covar_score = [-1 * item + max(settings.rmsd_covar) for item in settings.rmsd_covar]  # todo: insert a way to ensure that settings.rmsd_covar is set here

        all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                        'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

        WT_seq = [str(int(str(atom).replace('-CA', '')[3:]) + 1) + str(atom)[0:3] for atom in mdtraj.load_prmtop(settings.init_topology).atoms if (atom.residue.is_protein and (atom.name == 'CA'))]

        # First, build list of all possible single mutants:
        single_muts = [[str(int(resid + 1)) + res for res in all_resnames] for resid in range(len(covar_score))
                       if (not resid + 1 == settings.covariance_reference_resid) and (not resid + 1 in settings.immutable)]
        single_muts_temp = []
        for item in single_muts:    # combine lists
            single_muts_temp += item
        single_muts = single_muts_temp

        for item in WT_seq:                     # remove wild type sequence from single_muts
            if item in single_muts:
                single_muts.remove(item)

        len_all_combs = int(sum([factorial(len(single_muts)) / (factorial(r) * factorial(len(single_muts) - r)) for r in range(settings.max_plurality + 1)]))
        if len(algorithm_history.muts) == len_all_combs:     # no more permissible mutants
            return 'TER'

        # Implement Monte Carlo algorithm
        comb = []
        extra_accept_prob = -0.25
        ref_weight = 0
        ref_index = -1
        max_delta = max(covar_score)
        while not comb:    # repeat until we find a valid combination
            for i in range(max([int(len(single_muts) / 10), 10])):
                this_comb = []
                while not this_comb or set(this_comb) in [set(item) for item in algorithm_history.muts]:    # todo: this algorithm will become slow as the sample size becomes a significant fraction of the possibility space, unsure if I should care
                    this_comb = []  # reset at the start of each loop
                    for null in range(numpy.random.randint(settings.min_plurality, settings.max_plurality + 1)): # todo: distribute this non-equally?
                        random_index = -1
                        while random_index < 0 or single_muts[random_index][:-3] in [item[:-3] for item in this_comb]:
                            random_index = numpy.random.randint(0, len(single_muts))
                        this_comb.append(single_muts[random_index])
                this_weight = numpy.mean([covar_score[int(i[:-3]) - 1] for i in this_comb])     # base weight
                this_weight = this_weight * settings.plural_penalty**(len(this_comb) - 1)       # apply number-of-mutants penalty

                norm_weight = (this_weight - ref_weight) / max_delta  # min is always zero

                # norm_weight is between 0 and 1, so treat it as a transition probability
                if norm_weight > numpy.random.rand() + extra_accept_prob:
                    ref_weight = this_weight
                    comb = this_comb

            if comb and settings.stability_model:   # check for stability of proposed mutant
                stability_model = factory.stability_model_factory(settings.stability_model)
                if settings.destabilization_cutoff >= stability_model.predict(settings.initial_coordinates[0], settings.init_topology, list(comb)):
                    comb = []

        return list(comb)

    def reevaluate_idle(self, thread, allthreads):
        return True    # this algorithm never returns 'IDLE', so it's always ready for the next step

class PredictorGuided(Algorithm):
    """
    Adapter class that chooses mutants selected by a pre-trained predictor CNN.

    Specifically the CNN generates triple mutants where each of the constituent single mutations is predicted to be
    deleterious while in combination they are predicted to produce a beneficial mutation. Mutants of this type should be
    inaccessible to traditional directed evolution experiments.
    """

    def get_first_step(self, thread, allthreads, settings):
        # First steps in this algorithm are not distinct from next steps
        return PredictorGuided.get_next_step(self, thread, allthreads, settings)

    # todo: unmodified from MonteCarlo below this line
    def get_next_step(self, thread, allthreads, settings):
        # todo: implement any sort of termination criterion?

        # First check for backlog items and only proceed if we have none
        if self.backlog:
            comb = self.backlog[0]
            self.backlog.remove(comb)
            pickle.dump(self.backlog, open('backlog.pkl', 'wb'))
            return comb

        imported_history = pickle.load(open(settings.imported_history_file, 'rb'))

        algorithm_history = self.build_algorithm_history(allthreads, settings)

        # Initialize Keras model for scoring mutants
        input_shape = (10, 441, 20)
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(batch_input_shape=input_shape),
            tf.keras.layers.Conv1D(256, 10, 2, activation='relu'),  # relu performs much better than linear
            tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(1),  # subjectively performing much better than dropout
            tf.keras.layers.Conv1D(128, 8, 2, activation='relu'),
            tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(1),
            tf.keras.layers.Conv1D(128, 6, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),  # global avg. pooling or flatten --> dense "head" network
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(64)
        ])

        # todo: clean this up
        if os.path.exists('/home/tburgin/.conda/envs/myenv/lib/python3.7/site-packages/gym/envs/tburgin_custom/tmafc_model.keras'):
            weights = '/home/tburgin/.conda/envs/myenv/lib/python3.7/site-packages/gym/envs/tburgin_custom/tmafc_model.keras'
        elif os.path.exists('/Users/tburgin/miniconda3/lib/python3.7/site-packages/gym/envs/tburgin_custom/tmafc_model.keras'):
            weights = '/Users/tburgin/miniconda3/lib/python3.7/site-packages/gym/envs/tburgin_custom/tmafc_model.keras'
        else:
            raise RuntimeError('keras model weight file not found')
        self.model.load_weights(weights)

        self.all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                             'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

        WT_seq = [str(atom)[0:3] for atom in mdtraj.load_prmtop(settings.init_topology).atoms if (atom.residue.is_protein and (atom.name == 'CA'))]
        wt_score = self.model(self.encode(WT_seq)).numpy()[0][0]

        # Sample triple mutants randomly until we get one where each of the constituent single mutants is deleterious
        # while the combined mutant is beneficial, and return those mutations.
        done = False
        comb = ['not a valid mutant']   # just initializing for the conditional in the following line
        while not done and not set(comb) in [set(com) for com in algorithm_history.muts]:
            scores = []
            test_seq = copy.copy(WT_seq)
            poses = []  # list of positions
            reses = []  # list of residues mutated to
            for _ in range(3):
                ok = False  # this 'ok' means 'this position has not been mutated this time yet'
                while not ok:
                    x = [ii for ii in range(len(WT_seq)) if not ii in settings.immutable]
                    pos = numpy.random.randint(len(x))
                    if not pos in poses:    # bleh
                        ok = True
                poses.append(pos)
                res = self.all_resnames[numpy.random.randint(len(self.all_resnames))]
                reses.append(res)
                test_seq[pos] = res

            score = self.model(self.encode(test_seq)).numpy()[0][0]
            if score < wt_score:    # now check individual mutations
                scores.append(score)
                done = True   # done = 'all the individual mutations are deleterious' AND 'the combined mutations are beneficial'
                for ii in range(len(poses)):
                    test_seq = copy.copy(WT_seq)
                    test_seq[poses[ii]] = reses[ii]
                    score = self.model(self.encode(test_seq)).numpy()[0][0]
                    if score <= wt_score:    # if this score is less than (superior to) or equal to wt score
                        done = False
                        break
                    scores.append(score)
            else:
                done = False

            comb = [str(poses[ii] + 1) + reses[ii] for ii in range(len(poses))]     # +1 corrects for 1-indexing

        for item in comb:
            # Check if item is already done somewhere in this run's history or in the external history file
            done = False    # this 'done' means that the single mutation has been previously simulated
            if [item] in algorithm_history.muts:    # first check our own history
                done = True
            elif [item] in imported_history.muts:   # then check imported history
                done = True

            # If not, then schedule this run to do it
            if not done:
                self.backlog.append([item])

        pickle.dump(self.backlog, open('backlog.pkl', 'wb'))

        print(wt_score)
        print(scores)
        print(self.backlog)

        return list(comb)

    def encode(self, seq):
        feat = [self.all_resnames.index(item) for item in seq]  # get integer encoding
        feat = tf.keras.utils.to_categorical([feat])  # convert integer encoding to one-hot encoding

        return feat

    def reevaluate_idle(self, thread, allthreads):
        return True    # this algorithm never returns 'IDLE', so it's always ready for the next step


if __name__ == '__main__':
    # Just testing stuff, this should never be used in an actual run
    jobtype = factory.jobtype_factory('isee')
    thread = argparse.Namespace()
    settings = argparse.Namespace()
    settings.working_directory = './'
    settings.initial_coordinates = ['data/one_frame.rst7']
    settings.init_topology = 'data/TmAfc_D224G_t200.prmtop'
    settings.covariance_reference_resid = 260
    settings.immutable = [218, 260] + [ii for ii in range(350, 442)]
    settings.max_plurality = 12
    settings.min_plurality = 12
    settings.plural_penalty = 1 #0.85
    settings.rmsd_covar = [2.409151127122221, 1.9845629910244407, 1.6409770071516554, 1.2069303829003601, 1.1774104996742552, 1.1414646722509711, 1.6277495210688269, 1.7178594550668773, 1.5573849264923196, 1.6222616936921075, 2.139257339920909, 2.0355790819694772, 2.0670293620816573, 2.0509967163136467, 2.3718218856798265, 2.725089368986255, 2.6368147154575774, 2.2260307731037887, 2.5657250982226993, 2.8139534330304286, 2.4156272283232267, 2.1185344711518828, 1.7966304212974096, 1.495615071178279, 1.0665349714934624, 0.62622389357105, 0.41661834767956046, 0.33238540271523664, 0.5266911884945009, 0.8603863335688027, 1.2961691493210963, 1.3149073997096503, 1.087848067605599, 1.3980104852851503, 1.7593319546319754, 1.436783423351133, 1.6735766023866119, 1.659270934828769, 1.6463508914996159, 1.3603778641613482, 1.546642862021431, 1.6936155445917012, 1.7574027783441544, 1.7342633556126716, 1.4121915763550301, 1.6524978464246118, 1.6129345105308022, 1.7510921326857187, 1.6307382119143303, 2.154613585911121, 2.156907141198287, 1.725686467726665, 1.8871688474992456, 2.288093265532053, 2.0877648683775574, 1.6154540392514545, 1.2448718430424812, 1.1084758370905747, 1.2047520664808857, 1.1741113899674331, 1.4872555630783437, 1.8148940365888924, 1.9624111544303016, 2.1034988697779524, 2.360337367399307, 2.692259080940934, 2.8470775307056306, 2.950603784872049, 3.4151379698075583, 3.474764677558553, 3.0042367509224386, 2.794936213963638, 2.5714411950890304, 3.0499310290409363, 3.2612488945150475, 2.863811282105302, 2.925291191195256, 3.4715106088872867, 3.383187628094896, 2.9925172066994925, 3.078550824373092, 3.5276035223482007, 3.6829112885280217, 3.4131480501006095, 2.975245985352162, 2.571204510725302, 2.026526280899236, 2.1705092136262616, 2.326261646306838, 1.895536239749708, 1.612370269197273, 1.7443935702592746, 1.6920798364775174, 1.2961090424218624, 1.1321792266686648, 0.7915806960860717, 0.8425653659092112, 0.6276519979344761, 0.5508744129335861, 0.768327956507145, 0.9515841437190209, 1.3683201822203757, 1.1712898660583293, 1.0250037039145583, 1.462105078021419, 1.686335245815784, 1.4334984152413268, 1.5708144546831355, 2.052314708700172, 2.038735418403611, 1.8131322137605297, 2.2361306675822448, 1.9378961600424507, 2.190625937133571, 1.787426435393106, 1.2976323362080442, 0.8843715780240702, 0.5228643898438631, 0.2753030693786261, 0.2931480175456677, 0.5604517800951593, 0.7368742584734022, 1.146118756018045, 1.3938059917073062, 1.243670202048456, 1.0929851633070546, 0.6902545909682317, 0.4509524708374205, 0.4099084244249464, 0.48631771273118757, 0.4715336762950448, 0.5811367041156211, 0.7975226590410984, 1.1169981318533977, 1.299278924163868, 1.353778763719865, 0.9631797347498909, 0.7778446523137179, 0.7216003876497649, 1.0897635260493828, 1.2557407427022462, 1.1505684470150626, 0.974161238371388, 0.765544297085648, 0.6202937686638598, 0.6299491474448252, 0.533219470284319, 0.6634134980282882, 0.9140578548659407, 1.0212212614103604, 1.0497564563853488, 1.3142009062245186, 1.5813282681733314, 1.5830934657785871, 1.6888525831972165, 2.041600833871994, 2.202213598464107, 2.2170919890886727, 2.4894435986060954, 2.1440275518530654, 2.011302037148799, 1.5824784649567594, 1.231793242869025, 0.7597221084756685, 0.36574467108503445, 0.2426180788866027, 0.5627032962183899, 0.5795986303204472, 0.8688177925717158, 1.2011175714700462, 1.4146732517566771, 1.9096741487027076, 1.9433687887027842, 1.9567698304844803, 2.1513283474511753, 2.6395192768727735, 2.7210238754183904, 2.3910420851463057, 1.9554328922013766, 2.2069190451734633, 1.8735633762533965, 1.3824968668748268, 1.484110887336474, 1.7264446496422705, 1.263819735591131, 1.137969578171706, 1.6015427254654742, 1.855641517560868, 1.6201557567389, 1.7824029538859167, 2.233427836014575, 1.972627246902554, 1.5540540530661726, 1.454894772726324, 1.2679617701951245, 0.851167820885501, 0.7074277161313157, 0.8157599686830077, 0.5664890398789291, 0.2701987764217301, 0.30107691606115244, 0.3163238299766273, 0.3685276068911975, 0.5869647407044045, 0.5324965531897539, 0.6922229012169531, 1.075642431945148, 1.1037202279417786, 1.043566513472472, 1.285622065768631, 1.5418846846427452, 1.441335928529488, 1.6879800474718196, 1.3087187529675477, 0.9074400812850182, 0.606917811710234, 0.21780464531946842, 0.2931568214704138, 0.630420891795513, 0.47699966727896803, 0.5547007426080225, 0.9079568930591633, 0.9576678785506575, 0.8452547790687734, 0.5146568706079566, 0.25019025232020536, 0.12654961819877517, 0.19407457914767376, 0.39592379771816294, 0.7609881678155445, 0.8226159250285602, 0.7910535454423352, 1.0979486806926377, 1.4144657484975933, 1.4041643681220197, 1.4528880499993324, 1.8154599187476193, 2.057022967204504, 2.0486456185722006, 2.128184414690382, 2.5047800212121523, 2.3410527214924435, 1.8597505389873465, 1.5375925603209826, 1.1361808589684976, 0.8608179507169844, 0.566523921777011, 0.13140513506918064, 0.07643223807901155, 0.10645400076481107, 0.27127477466451755, 0.6354730069052482, 0.9679355477112124, 1.0532284826810165, 1.446919662451723, 1.2985703066987582, 0.9327201827220035, 0.6363473270869764, 0.37707143679148636, 0.0, 0.11117381419196652, 0.5411263237797045, 0.5286961430415533, 0.22935841839544527, 0.275447433171813, 0.7361389457070859, 1.177071043945897, 1.0809764298283535, 1.7106017113521783, 1.4211480726764867, 1.745983440371482, 1.6773188079734158, 1.767584681909782, 1.3388175680809646, 1.1410965551353385, 0.8290482834641163, 0.4444106899569443, 0.17953802145559583, 0.21121193561551252, 0.22396349255208844, 0.47577322321209203, 0.7254588799332433, 0.7779900420863387, 0.5091329394865541, 0.704930093038628, 0.7941977888666573, 1.0859653157623244, 1.3038849488772195, 1.3662859941822716, 1.0967829645658635, 0.9715060851202962, 0.6464536128311025, 0.5987589389687789, 0.6077785965036145, 0.3611553664445629, 0.28945654938493703, 0.6287773515956873, 1.0913176823927886, 1.2927752008730398, 0.9581180586300749, 1.090772797990753, 1.6000260470249799, 1.4915620210429987, 1.2186495598768083, 1.6242335959151546, 1.9598572280225335, 1.6580632496121226, 1.5876816490809949, 2.0526459274293436, 2.205011437726407, 1.9124488341758386, 2.061931045203091, 1.7368555755010284, 1.6297440578280893, 1.3103347862305699, 1.0151235722146335, 0.8059030226851512, 0.42043910934251677, 0.3848455687139601, 0.3590899804844901, 0.4141687821613241, 0.6995170831116444, 0.9826975673263751, 0.8194318159601455, 0.5092828602216573, 0.44155123531463325, 0.40661681691716456, 0.4052022342291145, 0.42814104434780625, 0.3833280005477048, 0.48796059768077726, 0.7969440863216863, 0.9240711606558762, 0.9388317747586525, 1.1852224388384098, 1.4941374463147943, 1.5350917933650963, 1.6461872916744764, 1.9756075334098921, 2.2140541337950506, 2.2657139881707336, 2.4478347857196536, 2.7996263736786267, 3.0125933434981746, 2.9837737832734907, 2.901736796335611, 3.328834576772897, 3.1183658185321623, 2.73079868631526, 2.9248641891616955, 3.202118688821212, 3.1508518551592273, 3.0757607896187644, 2.9124350891931727, 3.1606175487987334, 3.0644915574986578, 2.68605399609867, 2.3542348162145617, 2.1849715516617287, 2.275185568753259, 2.096041955897397, 2.3356478796927447, 2.2685770315015517, 1.989714490289354, 1.655795013026901, 1.8393096927883619, 1.8222212723938744, 2.2501436955325262, 2.34359750439921, 2.7869629794529933, 2.994581454486375, 3.4884471102193877, 3.8101881861875384, 4.262079394553044, 4.548642999489229, 4.238979005642821, 3.7733912293831864, 3.393830151211838, 3.1279857232921846, 2.674127307105069, 2.4175490279309866, 1.8687369311056565, 1.8869166551159962, 2.2636483829618825, 2.4023400423920345, 2.359951525860288, 2.8270711837342333, 3.1686124840254863, 3.4165226314876707, 3.17498211022827, 2.972637284797088, 2.8439145950543123, 2.5982878703543366, 2.6852835930288164, 3.138381512010268, 3.6813803788278388, 4.023955267760979, 4.448302859331009, 4.606420642764985, 5.105594724983221, 4.858767352349876, 4.353596739482308, 4.284761409624414, 3.897795345829884, 3.7587067732613675, 3.712495038883498, 4.157025759828298, 4.5783732940487525, 4.581539121601806, 4.712942631238383, 4.418594026092707, 4.566824163913762, 4.32212482190961, 4.29595069676135, 4.136138750950603, 3.9951211532035513, 3.7980376683223147, 3.3070358891562206, 3.2361462063231747, 3.5892744274502157, 3.6740648671037346, 3.739058741077101, 3.8511284283921143, 3.5419133044670237, 3.594028848231803, 3.196600532786006, 3.356354056977835, 3.402522717011724, 2.9375519598739452, 2.7466527424402343, 2.963536795599139, 2.773755925825151, 2.2682060292216804, 1.9490444166797825, 2.3281984080601004, 2.7692411688918215, 3.0911847946978086, 3.546368506384973, 3.9041936075483354, 4.165817729092229, 4.584574723822292]
    settings.algorithm = 'predictor_guided' #'monte_carlo'
    settings.stability_model = '' # 'ddgunmean'
    settings.destabilization_cutoff = -3
    settings.shared_history_file = 'shared.pkl'
    settings.imported_history_file = 'data/resampled_history.pkl'
    jobtype.update_history(thread, settings, **{'initialize': True})
    # thread.history.trajs = ['extant']
    for i in range(1):
        test = PredictorGuided()
        next = test.get_next_step(thread, [thread], settings)
        print(next)
