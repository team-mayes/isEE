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
import itertools
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
        if thread == allthreads[0] and not any([bool(item.history.trajs) for item in allthreads]):
            return 'WT'
        else:
            return Script.get_next_step(self, thread, allthreads, settings)

    def get_next_step(self, thread, allthreads, settings):
        untried = [item for item in settings.mutation_script if not item in self.algorithm_history.muts]
        try:
            self.algorithm_history.muts.append([untried[0]])
            buffer_history(self.algorithm_history)
            return self.dump_and_return([untried[0]], self.algorithm_history)   # first untried mutation
        except IndexError:  # no untried mutation remains
            return self.dump_and_return('TER', self.algorithm_history)

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
        all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

        if self.algorithm_history.muts == []:   # first ever mutation
            saturation = True   # saturation = True means: pick a new residue to mutate

        # Otherwise, saturation is defined by having tried all the residues in all_resnames (excluding the wild type)
        # elif (the last (len(all_resnames) - 1) mutation resnames are all unique) and\
        #   (all of the last (len(all_resnames) - 1) mutation resnames are in all_resnames) and\
        #   (all of the last (len(all_resnames) - 1) mutation resids are equal to the last resid):
        elif len(set([item[0][-3:] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]])) == (len(all_resnames) - 1) and\
                set([item[0][-3:] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]]).issubset(set(all_resnames))\
                and all([item[0][:-3] == self.algorithm_history.muts[-1][0][:-3] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]]):
            saturation = True

        else:
            saturation = False

        if saturation:
            # Perform RMSD profile analysis # todo: change to operate on best-scoring mutant yet found
            rmsd_covar = utilities.covariance_profile(allthreads[0], 0, settings)  # operate on wild type trajectory

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

            next_mut = [str(int(resid)) + all_resnames[0]]

            self.algorithm_history.muts.append(next_mut)
            buffer_history(self.algorithm_history)
            return self.dump_and_return(next_mut, self.algorithm_history)
        else:   # unsaturated, so pick an unused mutation on the same residue as the previous mutation
            # First, we need to generate a list of all the mutations that haven't been tried yet on this residue
            done = [item[0][-3:] for item in self.algorithm_history.muts if item[:-3] == self.algorithm_history.muts[-1][0][:-3]]
            todo = [item for item in all_resnames if not item in done]

            # Then, remove the wild type residue name from the list
            mtop = mdtraj.load_prmtop(thread.history.tops[0])   # 0th topology corresponds to the "wild type" here
            wt = mtop.residue(int(self.algorithm_history.muts[-1][0][:-3]) - 1)  # mdtraj topology is zero-indexed, so -1
            if str(wt)[:3] in todo:
                todo.remove(str(wt)[:3])   # wt is formatted as <three letter code><zero-indexed resid>

            next_mut = [self.algorithm_history.muts[-1][0][:-3] + todo[0]]

            self.algorithm_history.muts.append(next_mut)
            buffer_history(self.algorithm_history)
            return self.dump_and_return(next_mut, self.algorithm_history)

    def reevaluate_idle(self, thread, allthreads):
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
        if not settings.TEST:
            all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']
        else:   # much-truncated list to skip much of the busywork, for testing purposes
            all_resnames = ['ALA', 'GLY']

        if self.algorithm_history.muts == []:   # first ever mutation
            saturation = True   # saturation = True means: pick a new residue to mutate

        # Otherwise, saturation is defined by having tried all the residues in all_resnames (excluding the wild type)
        # elif (the last (len(all_resnames) - 1) mutation resnames are all unique) and\
        #   (all of the last (len(all_resnames) - 1) mutation resnames are in all_resnames) and\
        #   (all of the last (len(all_resnames) - 1) mutation resids are equal to the last resid):
        elif len(set([item[0][-3:] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]])) == (len(all_resnames) - 1) and\
                set([item[0][-3:] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]]).issubset(set(all_resnames)) and\
                all([item[0][:-3] == self.algorithm_history.muts[-1][0][:-3] for item in self.algorithm_history.muts[-1 * (len(all_resnames) - 1):]]):
            saturation = True

        # Alternatively, if there are any double or higher mutants in the algorithm history, consider it saturated
        elif any([len(item) > 1 for item in self.algorithm_history.muts]):
            saturation = True

        else:
            saturation = False

        if saturation:
            # Perform RMSD profile analysis
            rmsd_covar = utilities.covariance_profile(allthreads[0], 0, settings)     # operate on wild type trajectory

            # Calculate subnetworks
            subnetworks = self.get_subnetworks()

            # Pick minimum RMSD residue for a subnetwork that hasn't already been done
            already_done = set([int(item[0][:-3]) for item in self.algorithm_history.muts])

            # Choose next unmutated subnetwork
            if self.no_unmut_subnets(): # no more subnetworks unmutated, so now do combinations
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
                score_and_muts.append([0,['None']])   # kludge to make the coming for loop work properly
                this_mut = score_and_muts[0][1][0]
                best_index = 0
                best_scorers = []   # list of best single mutations at each attempted index
                for item_index in range(len(score_and_muts)):
                    if not score_and_muts[item_index][1][0][:-3] == this_mut[:-3]:
                        this_mut = score_and_muts[item_index][1][0]
                        best_scorers.append(score_and_muts[best_index][1][0])
                        best_index = item_index
                    if score_and_muts[item_index][0] < score_and_muts[best_index][0]:
                        best_index = item_index

                # Finally, construct list of combinations to attempt, and pick one:
                combinations = [list(item) for item in list(itertools.combinations(best_scorers, 2)) if not list(item) in self.algorithm_history.muts]
                if combinations:    # if undone combinations remain
                    self.algorithm_history.muts.append(combinations[0])
                    buffer_history(self.algorithm_history)
                    return self.dump_and_return(combinations[0], self.algorithm_history)
                else:
                    return self.dump_and_return('TER', self.algorithm_history)

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
                    return self.dump_and_return('TER', self.algorithm_history)

            next_mut = [str(int(resid)) + all_resnames[0]]

            self.algorithm_history.muts.append(next_mut)
            buffer_history(self.algorithm_history)
            return self.dump_and_return(next_mut, self.algorithm_history)
        else:   # unsaturated, so pick an unused mutation on the same residue as the previous mutation
            # First, we need to generate a list of all the mutations that haven't been tried yet on this residue
            done = [item[0][-3:] for item in self.algorithm_history.muts if item[0][:-3] == self.algorithm_history.muts[-1][0][:-3]]
            todo = [item for item in all_resnames if not item in done]

            # Then, remove the wild type residue name from the list
            mtop = mdtraj.load_prmtop(thread.history.tops[0])   # 0th topology corresponds to the "wild type" here
            wt = mtop.residue(int(self.algorithm_history.muts[-1][0][:-3]) - 1)  # mdtraj topology is zero-indexed, so -1
            if str(wt)[:3] in todo:
                todo.remove(str(wt)[:3])   # wt is formatted as <three letter code><zero-indexed resid>

            next_mut = [self.algorithm_history.muts[-1][0][:-3] + todo[0]]

            self.algorithm_history.muts.append(next_mut)
            buffer_history(self.algorithm_history)
            return self.dump_and_return(next_mut, self.algorithm_history)

    def reevaluate_idle(self, thread, allthreads):
        # The first condition to meet for this algorithm to allow an idle thread to resume is simply that the simulation
        # for the first system (non-mutated) is finished and has had get_next_step called on it
        if os.path.exists('algorithm_history.pkl'):
            algorithm_history = pickle.load(open('algorithm_history.pkl', 'rb'))
            if algorithm_history.muts:
                # The second condition is that, if and only if there are no unmutated subnetworks, there must be a
                # recorded score for each one
                if self.no_unmut_subnets():
                    if len(list(itertools.chain.from_iterable([this_thread.history.score for this_thread in allthreads]))) == len(list(itertools.chain.from_iterable([this_thread.history.muts for this_thread in allthreads]))):
                        return True
                    else:
                        return False
                return True
            else:
                return False
        else:
            return False

    def no_unmut_subnets(self):
        # Determine from algorithm_history whether there are any subnetworks not containing at least one single mutant
        subnetworks = self.get_subnetworks()
        if len(self.algorithm_history.muts) > len(subnetworks):    # > b/c of wt # todo: should/can I do better than this?
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
