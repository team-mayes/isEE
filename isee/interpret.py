"""
This portion of the program is responsible for handling update of the results, checking global termination criteria, and
implementing the calls to JobType methods to control the value of the thread.coordinates attribute for the next step.
"""

import pickle
from isee.infrastructure import factory

def interpret(thread, allthreads, running, settings):
    """
    The main function of interpret.py. Makes calls to JobType methods to update results, check termination criteria, and
    update thread.coordinates

    Parameters
    ----------
    thread : Thread
        The Thread object on which to act
    allthreads : list
        The list of all extant Thread objects
    running : list
        The list of all currently running Thread objects
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    termination: bool
        True if a global termination criterion has been met; False otherwise

    """

    jobtype = factory.jobtype_factory(settings.job_type)

    jobtype.analyze(thread, settings)    # analyze just-completed simulation
    termination = jobtype.algorithm(thread, settings)   # query algorithm to decide next move

    # Dump restart.pkl with updates from analysis and algorithm
    pickle.dump(allthreads, open('restart.pkl.bak', 'wb'))  # if the code crashes while dumping it could delete the contents of the pkl file
    if not os.path.getsize('restart.pkl.bak') == 0:
        shutil.copy('restart.pkl.bak', 'restart.pkl')       # copying after is safer

    return termination, running
