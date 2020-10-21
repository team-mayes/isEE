"""
Interface for Algorithm objects. New Algorithms can be implemented by constructing a new class that inherits from
Algorithm and implements its abstract methods.
"""

import abc
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
        elif set([item[-3:] for item in thread.history.muts[-1 * (len(all_resnames) - 1):]]).issubset(set(all_resnames))\
                and all([item[:-3] == thread.history.muts[-1][:-3] for item in thread.history.muts[-1 * (len(all_resnames) - 1):]]):
            saturation = True
        else:
            saturation = False

        if saturation:
            # Perform RMSD profile analysis
            rmsd_covar = utilities.covariance_profile(thread, -1, settings)     # -1: operate on most recent trajectory

            # Pick minimum RMSD residue that hasn't already been done
            already_done = set([item[:-3] for item in thread.history.muts] + [settings.covariance_reference_resid])
            paired = [[i + 1, value] for value in rmsd_covar]   # paired resid, rmsd-of-covariance
            paired = sorted(paired, key=lambda x: x[1])         # sorted by ascending rmsd
            resid = settings.covariance_reference_resid
            this_index = 0
            while resid in already_done:    # iterate through paired in order of ascending rmsd until resid is unused
                resid = paired[this_index][0]
                this_index += 1
                if this_index >= len(paired):    # if there are no more residues to mutate
                    return 'TER'

            return all_resnames[0] + str(int(resid))
        else:   # unsaturated, so pick an unused mutation on the same residue as the previous mutation
            # First, we need to generate a list of all the mutations that haven't been tried yet on this residue
            done = [item[-3:] for item in thread.history.muts if item[:-3] == thread.history.muts[-1][:-3]]
            todo = [item for item in all_resnames if not item in done]

            return todo[0] + thread.history.muts[-1][:-3]
