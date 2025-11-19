"""
main.py
in silico Enzyme Evolution

This script handles the primary loop of building and submitting jobs in independent Threads, using the methods thereof 
to execute various interfaced/abstracted commands.
"""

import copy
import os
import dill as pickle
import shutil
import sys
import time
from isee import interpret, process, initialize_charges, utilities
from isee.infrastructure import factory, configure
import cProfile


class Thread(object):
    """
    Object representing a series of simulations and containing the relevant information to define its current state.

    Threads represent the level on which isEE is parallelized. This flexible object is used for every type of job
    performed by isEE.

    Parameters
    ----------
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    None

    """

    def __init__(self):
        self.name = ''
        self.jobids = []                # list of jobids associated with the present step of this thread
        self.terminated = False         # boolean indicating whether the thread has reached a termination criterion
        self.current_type = []          # list of job types for the present step of this thread
        self.current_name = []          # list of job names corresponding to the job types
        self.current_results = []       # results of each job, if applicable
        self.suffix = 0                 # index of current step
        self.total_moves = 0            # running total of "moves" attributable to this thread
        self.accept_moves = 0           # running total of "accepted" "moves", as defined by JobType.update_results
        self.status = 'fresh thread'    # tag for current status of a thread
        self.skip_update = False        # used by restart to resubmit jobs as they were rather than doing the next step
        self.idle = False               # flag for idle thread, to facilitate parallelization of algorithms
        self.mps_idle = False			# flag for idling caused not by algorithm but by mps_patient
        self.consec_fails = 0           # consecutive failures count
        self.moves_this_time = 0        # number of moves completed since thread was last started or restarted

    # Remember in implementations of Thread methods that 'self' is a thread object, even though they may make calls to
    # methods of other types of objects (that therefore use 'self' to refer to non-Thread objects)

    def process(self, running, allthreads, settings):
        return process.process(self, running, allthreads, settings)

    def interpret(self, allthreads, running, settings):
        return interpret.interpret(self, allthreads, running, settings)

    def gatekeeper(self, allthreads, settings):
        jobtype = factory.jobtype_factory(settings.job_type)
        return jobtype.gatekeeper(self, allthreads, settings)

    def get_next_step(self, settings):
        jobtype = factory.jobtype_factory(settings.job_type)
        self.current_name = jobtype.get_next_step(self, settings)
        return self.current_name

    def get_batch_template(self, settings):
        jobtype = factory.jobtype_factory(settings.job_type)
        return jobtype.get_batch_template(settings)

    def get_frame(self, traj, frame, settings):
        mdengine = factory.mdengine_factory(settings.md_engine)
        return mdengine.get_frame(traj, frame, settings)

    def get_status(self, job_index, settings):
        batchsystem = factory.batchsystem_factory(settings.batch_system)
        return batchsystem.get_status(self.jobids[job_index], settings)

    def cancel_job(self, job_index, settings):
        batchsystem = factory.batchsystem_factory(settings.batch_system)
        batchsystem.cancel_job(self.jobids[job_index], settings)

    def get_exit_statuses(self, settings):
        batchsystem = factory.batchsystem_factory(settings.batch_system)
        return batchsystem.get_exit_statuses(self.jobids)


def init_threads(settings):
    """
    Initialize all the Thread objects called for by the user input file.

    In the case where settings.restart == True, this involves unpickling restart.pkl; otherwise, brand new objects are
    produced in accordance with settings.job_type (aimless_shooting, committor_analysis, equilibrium_path_sampling, or
    isee).

    Parameters
    ----------
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    allthreads : list
        List of all Thread objects, including those for which no further tasks are scheduled.

    """

    if settings.restart:
        allthreads = pickle.load(open(settings.working_directory + '/restart.pkl', 'rb'))
        for thread in allthreads:
            thread.moves_this_time = 0  # reset count of moves since thread last started/restarted
            thread.consec_fails = 0     # reset count of consecutive failures since last started/restarted
            if not thread.current_type == []:
                thread.skip_update = True
        if settings.restart_terminated_threads:
            for thread in allthreads:
                thread.terminated = False
        running = allthreads.copy()     # just for interpret steps below as needed
        # Handle each thread
        for thread in allthreads:
            # Possibilities here:
            #  1) Last step finished, interpret was run, and thread terminated
            #  2) Last step was interrupted before interpret and is now...
            #     a) still running
            #     b) finished successfully
            #     c) cancelled prematurely

            # Start by checking for case (2a) and if so, wait until it's done and prepare next step
            if not thread.idle and not thread.gatekeeper(allthreads, settings):
                while not thread.idle and not thread.gatekeeper(allthreads, settings):
                    # Thread is still running; wait until thread.gatekeeper is True
                    time.sleep(30)
                thread.interpret(allthreads, running, settings)

            # Next check for cases (1) or (2b) and if so, prepare next step if necessary
            elif not thread.idle and all([status in ['timeout', 'completed'] for status in thread.get_exit_statuses(settings)]):
                # Check for case (1) by seeing if next step is set up yet
                # If interpret() has been called then thread.suffix + 1 > len(thread.history.trajs)
                if len(thread.history.trajs) < thread.suffix + 1:   # trajs hasn't been appended to since last interpret
                    pass    # it is set up, nothing to do
                else:   # trajs was appended to more recently than suffix was incremented, so need interpret
                    thread.interpret(allthreads, running, settings)

            # This leaves only case (2c), in which case we rebuild the last mutant to resubmit if it's not WT
            else:
                if not thread.history.muts[-1] == ['WT']:
                    initial_coordinates_to_mutate = settings.initial_coordinates[0]
                    if '/' in initial_coordinates_to_mutate:
                        initial_coordinates_to_mutate = initial_coordinates_to_mutate[initial_coordinates_to_mutate.rindex('/') + 1:]
                    new_inpcrd, new_top = utilities.mutate(initial_coordinates_to_mutate, settings.init_topology,
                                                           thread.history.muts[-1], initial_coordinates_to_mutate + '_' +
                                                           '_'.join(thread.history.muts[-1]), settings)
                    try:
                        assert thread.history.inpcrd[-1] == new_inpcrd
                        assert thread.history.tops[-1] == new_top
                    except AssertionError:
                        raise RuntimeError('Tried to rebuild mutant coordinates and topology for mutation: ' +
                                           thread.history.muts[-1] + ' but the resulting files did not match the names of '
                                           'the files in the corresponding thread.\n New filenames:' +
                                           '\n ' + new_inpcrd +
                                           '\n ' + new_top +
                                           'Expected filenames:' +
                                           '\n ' + thread.history.inpcrd[-1] +
                                           '\n ' + thread.history.tops[-1])

        return allthreads

    # If not restart:
    allthreads = []
    jobtype = factory.jobtype_factory(settings.job_type)

    # Set topology properly even if it's given as a path
    og_prmtop = settings.init_topology
    if '/' in settings.init_topology:
        settings.init_topology = settings.init_topology[settings.init_topology.rindex('/') + 1:]
    try:
        shutil.copy(og_prmtop, settings.working_directory + '/' + settings.init_topology)
    except shutil.SameFileError:
        pass

    for file in jobtype.get_initial_coordinates(settings):
        if '/' in file:
            file = file[file.rindex('/') + 1:]          # drop path to file from filename

        thread = Thread()   # initialize the thread object

        jobtype.update_history(thread, settings, **{'initialize': True, 'inpcrd': file})    # initialize thread.history

        thread.history.tops = [settings.init_topology]   # todo: settings.init_topology also needs to be a list now
        thread.history.timestamps = [time.time()]

        thread.name = file

        # todo: should the inpcrd and tops history saves above be replaced by saves to separate attributes (that is, not
        # todo: within the history attribute) to accommodate the fact that not all threads will actually do simulations
        # todo: using these files?

        allthreads.append(thread)

    for thread in allthreads:
        thread.moves_this_time = 0  # reset count of moves since thread last started/restarted

    return allthreads


def handle_loop_exception(running, exception, settings):
    """
    Handle cancellation of jobs after encountering an exception.

    Parameters
    ----------
    running : list
        List of Thread objects that are currently running. These are the threads that will be canceled if the isEE run
        cannot be rescued.
    exception : Exception
        The exception that triggered calling this function
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    None

    """

    print('\nCancelling currently running batch jobs belonging to this process in order to '
          'preserve resources.')
    for thread in running:
        try:
            for job_index in range(len(thread.jobids)):
                thread.cancel_job(job_index, settings)
        except Exception as little_e:
            print('\nEncountered an additional exception while attempting to cancel a job: ' + str(little_e) +
                  '\nIgnoring and continuing...')

    print('Job cancellation complete, isEE is now shutting down. The full exception that triggered this was: ')

    raise exception


def main_function(settings):
    """
    Perform the primary loop of building, submitting, monitoring, and analyzing jobs.

    This function works via a loop of calls to thread.process and thread.interpret for each thread that hasn't
    terminated, until either the global termination criterion is met or all the individual threads have completed.

    Parameters
    ----------
    settings : argparse.Namespace
        Settings namespace object
    rescue_running : list
        List of threads passed in from handle_loop_exception, containing running threads. If given, setup is skipped and
        the function proceeds directly to the main loop.

    Returns
    -------
    None

    """

    # Make working directory if it does not exist, handling overwrite and restart as needed
    if os.path.exists(settings.working_directory):
        if settings.overwrite and not settings.restart:
            shutil.rmtree(settings.working_directory)
            os.mkdir(settings.working_directory)
        elif not settings.restart:
            raise RuntimeError('Working directory ' + settings.working_directory + ' already exists, but overwrite '
                               '= False and restart = False. Either change one of these two settings or choose a '
                               'different working directory.')
    else:
        if not settings.restart:
            os.mkdir(settings.working_directory)
        else:
            raise RuntimeError('Working directory ' + settings.working_directory + ' does not yet exist, but '
                               'restart = True.')

    # if settings.shared_history_file and not settings.restart:
    #     if os.path.exists(settings.shared_history_file) and settings.overwrite:
    #         os.remove(settings.shared_history_file)
    #     elif os.path.exists(settings.shared_history_file):
    #         raise RuntimeError('the specified shared history file: ' + settings.shared_history_file + ' already exists,'
    #                            ' but neither overwrite nor restart are set to True. Change the appropriate setting or '
    #                            'else move or delete the shared history file.')

    # Store settings object in the working directory for compatibility with analysis/utility scripts
    if not settings.dont_dump:
        temp_settings = copy.deepcopy(settings)  # initialize temporary copy of settings to modify
        temp_settings.__dict__.pop('env')  # env attribute is not picklable
        pickle.dump(temp_settings, open(settings.working_directory + '/settings.pkl', 'wb'))

    # If desired, set appropriate charge distribution based on results from QM/MM simulation
    if settings.initialize_charges and not settings.restart and not (settings.DEBUG or settings.SPOOF):
        if any([not item == settings.initial_coordinates[0] for item in settings.initial_coordinates]):
            raise RuntimeError('initialize_charges does not currently support multiple threads with different initial '
                               'coordinates. All threads must have the same initial coordinates or initialize_charges '
                               'must be set to False.')

        # Have to do this stuff first even though it'll get repeated in init_threads... ew # todo: clean up
        # Set topology properly even if it's given as a path
        og_prmtop = settings.init_topology
        if '/' in settings.init_topology:
            settings.init_topology = settings.init_topology[settings.init_topology.rindex('/') + 1:]
        try:
            shutil.copy(og_prmtop, settings.working_directory + '/' + settings.init_topology)
        except shutil.SameFileError:
            pass
        # Copy initial coordinate files to working directory
        jobtype = factory.jobtype_factory(settings.job_type)
        jobtype.get_initial_coordinates(settings)

        current_dir = os.getcwd()               # store current directory
        os.chdir(settings.working_directory)    # move to working directory to do initialize_charges

        new_top = initialize_charges.main(settings)
        settings.init_topology = settings.working_directory + '/' + new_top
        # settings.initial_coordinates = [new_inpcrd for null in range(len(settings.initial_coordinates))]
        os.chdir(current_dir)                   # move back to previous directory to initialize threads

    # Build or load threads
    allthreads = init_threads(settings)

    # Move runtime to working directory
    os.chdir(settings.working_directory)

    running = allthreads.copy()     # to be pruned later by thread.process()
    termination_criterion = False   # initialize global termination criterion boolean
    jobtype = factory.jobtype_factory(settings.job_type)    # initialize jobtype

    # Initialize threads with first process step
    try:
        for thread in allthreads:
            if not thread.history.trajs:    # if there have been no steps in this thread yet
                jobtype.algorithm(thread, allthreads, settings)
            running = thread.process(running, allthreads, settings)
    except Exception as e:
        if settings.restart:
            print('The following error occurred while attempting to initialize threads from restart.pkl. It may be '
                  'corrupted.')
        raise e

    # Begin main loop
    # This whole thing is in a try-except block to handle cancellation of jobs when the code crashes in any way
    try:
        while (not termination_criterion) and running:
            for thread in running:
                if thread.gatekeeper(allthreads, settings):
                    termination_criterion, running = thread.interpret(allthreads, running, settings)
                    if termination_criterion:   # global termination
                        for thread in running:    # todo: should I replace this with something to finish up running jobs and just block submission of new ones?
                            for job_index in range(len(thread.jobids)):
                                thread.cancel_job(job_index, settings)
                        running = []
                        break
                    running = thread.process(running, allthreads, settings)
                else:
                    time.sleep(30)  # to prevent too-frequent calls to batch system by thread.gatekeeper

    except Exception as e:  # catch arbitrary exceptions in the main loop so we can cancel jobs
        print(str(e))
        handle_loop_exception(running, e, settings)

    if termination_criterion:
        return 'isEE run exiting normally (global termination criterion met)'
    else:
        return 'isEE run exiting normally (all threads ended individually)'


def run_main():
    # Obtain settings namespace, initialize threads, and move promptly into main.
    try:
        working_directory = sys.argv[2]
    except IndexError:
        working_directory = ''
    settings = configure.configure(sys.argv[1], working_directory)
    exit_message = main_function(settings)
    print(exit_message)

if __name__ == "__main__":
    cProfile.run('run_main()')
#if __name__ == '__main__':
#    import cProfile
#    # if check avoids hackery when not profiling
#    # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
#    if sys.modules['__main__'].__file__ == cProfile.__file__:
#        import isee  # Imports you again (does *not* use cache or execute as __main__)
#        globals().update(vars(isee))  # Replaces current contents with newly imported stuff
#        sys.modules['__main__'] = isee  # Ensures pickle lookups on __main__ find matching version
#    run_main()  # Or series of statements
