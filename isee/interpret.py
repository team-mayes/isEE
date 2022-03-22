"""
This portion of the program is responsible for handling update of the results, checking global termination criteria, and
implementing the calls to JobType methods to control the value of the thread.coordinates attribute for the next step.
"""

import os
import shutil
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

    if not thread.idle:     # only analyze if there's something to analyze, i.e., last step was not idle
        okay = jobtype.analyze(thread, settings)    # analyze just-completed simulation
        if not okay:    # simulation did not work for whatever reason
            try:
                null = thread.consec_fails
            except AttributeError:  # for backwards compatibility with older versions that lacked this attribute
                thread.consec_fails = 0
            if thread.consec_fails < settings.resubmit_on_failure:
                thread.consec_fails += 1
                file = thread.history.trajs[-1][0].replace('.nc', '.slurm')     # todo: kludge, generalize/cleanup
                taskmanager = factory.taskmanager_factory(settings.task_manager)
                thread.jobids[-1] = (taskmanager.submit_batch(file, settings))  # replace last jobid with new job
                return False, running   # exit without proceeding to termination
            else:
                raise RuntimeError('jobtype.analyze failed for thread with most recent trajectory file: ' +
                                   thread.history.trajs[-1][0] + '\nThe number of consecutive failures exceeded the'
                                   ' resubmit_on_failure setting (' + str(settings.resubmit_on_failure) +  '), '
                                   'so exiting.')
        else:   # this else not strictly necessary (if not okay always terminates), but added for clarity
            thread.consec_fails = 0

    thread.moves_this_time += 1     # after analysis, increment count of moves this thread has made

    termination = jobtype.algorithm(thread, allthreads, settings)   # query algorithm to decide next move

    # Dump restart.pkl with updates from analysis and algorithm
    pickle.dump(allthreads, open('restart.pkl.bak', 'wb'))  # if the code crashes while dumping it could delete the contents of the pkl file
    if not os.path.getsize('restart.pkl.bak') == 0:
        shutil.copy('restart.pkl.bak', 'restart.pkl')       # copying after is safer

    return termination, running
