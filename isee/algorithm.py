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


class AdaptScript(Algorithm):
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
