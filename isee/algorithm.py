"""
Interface for Algorithm objects. New Algorithms can be implemented by constructing a new class that inherits from
Algorithm and implements its abstract methods.
"""

import abc
import mdtraj
import subprocess
import re
import time
from isee import utilities

class Algorithm(abc.ABC):
    """
    Abstract base class for isEE algorithms.

    Implements methods for all of the task manager-specific tasks that isEE might need.

    """

    @abc.abstractmethod
    def get_next_step(self, thread, settings):
        """
        Determine the next step for the thread, or return 'TER' if there is none.

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
            terminating

        """

        pass


class Script(Algorithm):
    """
    Adapter class for the "algorithm" that is simply following the "script" of mutations prescribed by the user in the
    settings.mutation_script list.

    """

    def get_next_step(self, thread, settings):
        untried = [item for item in settings.mutation_script if not item in thread.history.muts]
        try:
            return untried[0]   # first untried mutation
        except IndexError:  # no untried mutation remains
            return 'TER'


class CovarianceSaturation(Algorithm):
    """
    Adapter class for algorithm that finds the residue with the minimum RMSD of covariance profile within the enzyme and
    then performs saturation mutagenesis before repeating.

    RMSD reference is determined by settings.covariance_reference_resid

    """

    def get_next_step(self, thread, settings):
        all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

        if thread.history.muts == []:   # first ever mutation
            saturation = True   # saturation = True means: pick a new residue to mutate

        # Otherwise, saturation is defined by having tried all the residues in all_resnames (excluding the wild type)
        # elif (the last (len(all_resnames) - 1) mutation resnames are all unique) and\
        #   (all of the last (len(all_resnames) - 1) mutation resnames are in all_resnames) and\
        #   (all of the last (len(all_resnames) - 1) mutation resids are equal to the last resid):
        elif len(set([item[-3:] for item in thread.history.muts[-1 * (len(all_resnames) - 1):]])) == (len(all_resnames) - 1) and\
                set([item[-3:] for item in thread.history.muts[-1 * (len(all_resnames) - 1):]]).issubset(set(all_resnames))\
                and all([item[:-3] == thread.history.muts[-1][:-3] for item in thread.history.muts[-1 * (len(all_resnames) - 1):]]):
            saturation = True

        else:
            saturation = False

        if saturation:
            # Perform RMSD profile analysis
            rmsd_covar = utilities.covariance_profile(thread, -1, settings)     # -1: operate on most recent trajectory

            # Pick minimum RMSD residue that hasn't already been done
            already_done = set([int(item[:-3]) for item in thread.history.muts] + [settings.covariance_reference_resid])
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

            return str(int(resid)) + all_resnames[0]
        else:   # unsaturated, so pick an unused mutation on the same residue as the previous mutation
            # First, we need to generate a list of all the mutations that haven't been tried yet on this residue
            done = [item[-3:] for item in thread.history.muts if item[:-3] == thread.history.muts[-1][:-3]]
            todo = [item for item in all_resnames if not item in done]

            # Then, remove the wild type residue name from the list
            mtop = mdtraj.load_prmtop(thread.history.tops[0])   # 0th topology corresponds to the "wild type" here
            wt = mtop.residue(int(thread.history.muts[-1][:-3]) - 1)  # mdtraj topology is zero-indexed, so -1
            if str(wt)[:3] in todo:
                todo.remove(str(wt)[:3])   # wt is formatted as <three letter code><zero-indexed resid>

            return thread.history.muts[-1][:-3] + todo[0]
