"""
Interface for Algorithm objects. New Algorithms can be implemented by constructing a new class that inherits from
Algorithm and implements its abstract methods.
"""

import abc
import subprocess
import re
import time

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
            pass
