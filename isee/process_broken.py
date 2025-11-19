"""
This portion of the program is responsible for handling setup of the appropriate batch script(s) for the next step in a
Thread, passing them to a task manager to submit them, and updating the list of currently running threads accordingly.
"""

import jinja2
import os
import sys
import math
import time
import pickle
import shutil
import warnings
from isee.infrastructure import factory

def process(thread, running, allthreads, settings, inp_override=''):
    """
    Reads the thread to identify the next step, then builds and returns the batch file(s).

    Parameters
    ----------
    thread : Thread()
        The Thread object on which to act
    running : list
        The list of currently running threads, which will be updated as needed
    allthreads : list
        List of all extant threads, running or not
    settings : argparse.Namespace
        Settings namespace object
    inp_override : str
        Specified input file to use in place of the one from jobtype.get_input_file (used by initialize_charges.py)

    Returns
    -------
    running : list
        The updated list of currently running threads after this process step

    """
    # First, check if the previous call to jobtype.algorithm flagged an 'IDLE' step, and if so, skip processing
    if thread.idle:
        if thread not in running:  # idle threads are still running
            running.append(thread)
        return running

    # Determine next step and, if appropriate, build corresponding list of batch files
    if not thread.skip_update:
        thread.current_name = thread.get_next_step(settings)
    else:
        thread.skip_update = False

    if thread.terminated:
        if thread in running:
            running.remove(thread)
            if running is None:
                running = []  # to keep running as a list, even if empty
        return running

    batchfiles = []  # initialize list of batch files to submit
    while len(batchfiles) < settings.nvidia_mps:
        this_len = len(batchfiles)
        batchfiles += process_one(thread, settings, inp_override)
        if len(batchfiles) == this_len:     # failsafe for if process_one stops returning new batch files
            break
    running = submit(batchfiles, thread, running, allthreads, settings)
    return running

def process_one(thread, settings, inp_override=''):
    """
    Process a single step of the thread. This is called multiple times per process() call if settings.nvidia_mps > 1.

    Parameters
    ----------
    thread : Thread()
        The Thread object on which to act
    settings : argparse.Namespace
        Settings namespace object
    inp_override : str
        Specified input file to use in place of the one from jobtype.get_input_file (used by initialize_charges.py)

    Returns
    -------
    batchfiles : list
        A list of batch files created by this function.

    """

    batchfiles = []
    jobtype = factory.jobtype_factory(settings.job_type)    # get jobtype for calling jobtype.update_history
    this_inpcrd, this_top = jobtype.get_struct(thread)
    name = thread.current_name

    if not inp_override:
        inp = jobtype.get_input_file(thread, settings)
    else:
        inp = inp_override

    if settings.degeneracy and not name == 'ic':
        degeneracy = ['_' + str(ii) for ii in range(settings.degeneracy)]
    else:
        degeneracy = ['']

    for degen in degeneracy:
        template = settings.env.get_template(thread.get_batch_template(settings))

        these_kwargs = { 'name': thread.name + '_' + name + degen,
                         'nodes': eval('settings.nodes'),
                         'taskspernode': eval('settings.ppn'),
                         'walltime': eval('settings.walltime'),
                         'mem': eval('settings.mem'),
                         'solver': eval('settings.solver'),
                         'inp': inp,
                         'out': thread.name + '_' + name + degen + '.out',
                         'prmtop': this_top,
                         'inpcrd': this_inpcrd,
                         'rst': thread.name + '_' + name + degen + '.rst7',
                         'nc': thread.name + '_' + name + degen + '.nc',
                         'working_directory': settings.working_directory,
                         'extra': eval(settings.extra),
                         'degen': degen,
                         'mps': '{{ mps }}'}

        filled = template.render(these_kwargs)
        newfilename = thread.name + '_' + name + degen + '.' + settings.batch_system
        try:
            with open(newfilename, 'w') as newfile:
                newfile.write(filled)
                newfile.close()
        except OSError as e:    # todo: can I come up with a fix for this before it's too late this time? I suppose the only real fix would be to name batch files using some code scheme and store a map of which names correspond to what in the history attributes of the corresponding threads (and also probably elsewhere)
            if 'name too long' in str(e):
                warnings.warn('Encountered too-long filename: ' + newfilename + '. Skipping submission of this thread '
                              'to the task manager, terminating it, and continuing. Consider renaming a sample of the '
                              'initial coordinate files you like best and starting a new isEE run. If this limitation '
                              'is a problem for you, please raise an issue on GitHub detailing your use-case.')
                thread.terminated = True
                return []
            else:
                raise e

        batchfiles.append(newfilename)
        jobtype.update_history(thread, settings, **these_kwargs)
    return batchfiles

def submit(batchfiles, thread, running, allthreads, settings):
    """
    Submit batch files in a list, with support for NVIDIA MPS.

    Parameters
    ----------
    batchfiles : list
        A list of strings pointing to batch files ready to be submitted or combined and then submitted

    Returns
    -------
    None

    """

    # Support for NVIDIA MPS
    if settings.nvidia_mps > 1:
        old_batchfiles = batchfiles.copy()
        batchfiles = mps_combine(batchfiles, settings)
        for batchfile in old_batchfiles:    # cleanup
            os.remove(batchfile)

    ### Submit batch files to task manager ###
    taskmanager = factory.taskmanager_factory(settings.task_manager)
    thread.jobids = []      # to clear out previous jobids if any exist

    if settings.nvidia_mps > 1:
        jobids = []
        for file in batchfiles:
            jobids.append(taskmanager.submit_batch(file, settings))

        # Duplicate jobids as appropriate
        jobids = sum([[jobid for _ in range(settings.nvidia_mps)] for jobid in jobids], [])[:len(thread.current_type)]
        thread.jobids = jobids
    else:
        for file in batchfiles:
            thread.jobids.append(taskmanager.submit_batch(file, settings))

    # Dump restart.pkl with updates from process (if allthreads was passed)
    if allthreads:
        pickle.dump(allthreads, open('restart.pkl.bak', 'wb'))  # if the code crashes while dumping it could delete the contents of the pkl file
        if not os.path.getsize('restart.pkl.bak') == 0:
            shutil.copy('restart.pkl.bak', 'restart.pkl')       # copying after is safer

    if thread not in running:
        running.append(thread)
    return running

def mps_combine(batchfiles, settings):
    """
    Create new batch files by combining the files in batchfiles in chunks of settings.nvidia_mps. Also fill the
    {{ mps }} slot with a unique identifier.

    Does not add necessary lines to the batch files for spinning up an MPS daemon (or closing one). These lines must be
    present in the batch file template.

    Parameters
    ----------
    batchfiles : list
        List of strings, filenames of batch files to combine
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    combined : list
        List of strings, filename of newly created combined batch files

    """

    # Get lines of each file as a list of lists, ensure length parity
    lines = [open(file, 'r').readlines() for file in batchfiles]
    try:
        assert all([len(item) == len(lines[0]) for item in lines])
    except AssertionError:
        raise RuntimeError('Attempted to combine files for NVIDIA MPS, but they are not all of the same number of '
                           'lines. Offending files are: ' + str(batchfiles))

    combined_file_count = math.floor(len(batchfiles) / settings.nvidia_mps)     # number of fully combined files to make
    mod = len(batchfiles) % settings.nvidia_mps     # number of leftover files to fit into last combined file
    if mod:
        if_mod = 1
    else:
        if_mod = 0

    # Get batchsystem for future reference
    batchsystem = factory.batchsystem_factory(settings.batch_system)

    # Initialize list of files to return
    to_return = []

    # Build each new file with a loop
    for ii in range(combined_file_count + if_mod):
        # Set unique identifier for MPS system based on current system time
        uniqid = str(time.time())
        print('uniqid: ' + uniqid)

        # Determine subset of files to combine in this loop
        try:
            files = batchfiles[ii * settings.nvidia_mps : (ii + 1) * settings.nvidia_mps]
        except IndexError:  # leftover files
            files = batchfiles[ii * settings.nvidia_mps :]

        # Combine them by iterating through lines
        newfilename = files[0] + '_mps.' + settings.batch_system
        open(newfilename, 'w').close()
        with open(newfilename, 'a') as f:
            lines = [open(file, 'r').readlines() for file in files]
            for jj in range(len(lines[0])):     # if an ignore_string is present, write without modifying
                if any([ignore_string in lines[0][jj] for ignore_string in batchsystem.mps_ignore_strings()]):
                    f.write(lines[0][jj].replace('{{ mps }}', uniqid))
                elif len(set([lines[kk][jj] for kk in range(len(lines))])) == 1:    # elif all lines at this position are the same
                    f.write(lines[0][jj].replace('{{ mps }}', uniqid))
                else:   # neither of the above, so combine with ampersands to parallelize with MPS
                    joined = ' & '.join([lines[kk][jj].replace('\n', '') for kk in range(len(lines))]) + '\n'
                    f.write(joined.replace('{{ mps }}', uniqid))

        to_return.append(newfilename)

    return to_return