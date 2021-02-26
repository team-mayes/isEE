"""
Routine for getting and setting the appropriate charges for atoms in the transition state. Can be called by isEE as
needed, or used individually from the command line by providing an appropriate settings.pkl file.
"""

import argparse
import pickle
import sys
import time
import parmed
import re
import copy
from isee import main as iseemain
from isee.process import process

def main(settings):
    """
    Perform a QM/MM simulation to determine appropriate charge distribution for atoms in the specified QM region and
    write a new topology file with those charges.

    Parameters
    ----------
    settings: argparse.Namespace
        The settings namespace object.

    Returns
    -------
    new_top: str
        Path to the newly written topology file with the appropriate charge distribution

    """
    # Write new input file based on isEE input file, with added QM settings
    # todo: this is necessarily Amber-specific. There's no way to do this in OpenMM (that I know of). If I decide to
    # todo: publish isEE as a standalone package with extensibility to other packages I'm going to need to make large
    # todo: portions of this script methods of the MDEngine abc, or else take an entirely different approach.
    isee_input_file = settings.path_to_input_files + '/isee_amber.in'
    input_lines = open(isee_input_file, 'r').readlines()

    # Check to be sure this doesn't already contain qmmm stuff
    if any([('ifqnt' in line or '&qmmm' in line) for line in input_lines]):
        raise RuntimeError('The input file ' + isee_input_file + ' contains an ifqnt setting and/or a &qmmm namelist. '
                           'These should not be present (even if it\'s ifqnt=0).')

    # Add ifqnt and &qmmm namelist # todo: this is sloppy, not really suited for publication
    new_input_file = 'initialize_charges_amber.in'
    with open(new_input_file, 'w') as f:
        added = False
        for line in input_lines:
            if '&cntrl' in line:
                line = ' &cntrl\n  ifqnt=1,\n'
            elif 'nstlim=' in line:
                line = '  nstlim=1,\n'      # only a single step
            elif 'dt=' in line:
                line = '  dt=0.0001,\n'     # extremely small timestep
            elif 'ntpr=' in line:
                line = '  ntpr=1,\n'  # write to output every step
            elif 'noshakemask=' in line:
                continue    # todo: should in fact only remove QM atoms from noshakemask...
            f.write(line)

        f.write('\n &qmmm')
        f.write('\n  qmmask=\'' + str(settings.ic_qm_mask) + '\',')
        f.write('\n  qm_theory=\'' + str(settings.ic_qm_theory) + '\',')
        f.write('\n  qmcut=' + str(settings.ic_qm_cut) + ',')
        f.write('\n  qmcharge=' + str(settings.ic_qm_charge) + ',')
        f.write('\n  printcharges=1,')
        f.write('\n &end\n')

    # Make a thread to pass to process
    thread = iseemain.Thread()
    thread.name = 'initialize_charges'
    thread.skip_update = True
    thread.idle = False
    thread.current_name = 'prod'
    thread.terminated = False
    thread.history = argparse.Namespace()
    thread.history.inpcrd = settings.initial_coordinates
    thread.history.tops = [settings.init_topology]
    thread.history.trajs = []
    thread.history.muts = []
    thread.history.score = []
    thread.history.timestamps = []

    # Have to specify appropriate settings for a short sander run
    # todo: have to do better than this if I want to publish this code.
    temp_settings = copy.copy(settings)
    temp_settings.solver = 'sander'
    temp_settings.walltime = '00:05:00'
    temp_settings.ppn = 1
    temp_settings.nodes = 1
    temp_settings.mem = '1000mb'

    # Submit job with process
    running = process(thread, [], temp_settings, inp_override=new_input_file)

    # Use the same loop strategy as in main.main to wait for this job to finish
    while True:
        if thread.gatekeeper(running, settings):
            break
        else:
            time.sleep(30)  # to prevent too-frequent calls to batch system by thread.gatekeeper

    # Read atomic charges from output file, store them in a python-interpretable format
    # This is a little ugly because the information is stored across two separate tables: one that links a QM index to
    # the MM (standard) index, and one that actually gives the partial charge data by QM index.
    output_name = thread.name + '_' + thread.current_name + '.out'
    output = open(output_name)     # open output file as an iterator

    # First, extract lines containing first QM-index/MM-index table:
    first_table_flag = False
    first_table = []
    second_table_flag = False
    second_table = []
    try:
        while True:
            line = next(output)
            if first_table_flag:
                if '*' in line:     # indicates link atoms, these come last and we want to ignore them
                    first_table_flag = False
                    continue
                first_table.append(line)
            elif second_table_flag:
                if 'Total Mulliken Charge' in line:
                    second_table_flag = False
                    break
                second_table.append(line)
            if '  QMMM: QM_NO.   MM_NO.  ATOM         X         Y         Z' in line:
                first_table_flag = True
            elif '  Atom    Element       Mulliken Charge' in line:
                second_table_flag = True
    except StopIteration:
        raise RuntimeError('Search through initialize_charges output file ' + settings.working_directory + '/' +
                           output_name + ' terminated before the end of the charge table. It must be formatted '
                           'incorrectly or in an unexpected way.')

    # Next, extract relevant data from tables using regex
    pattern = re.compile('[0-9.-]+')
    first_table_frmt = []
    second_table_frmt = []
    for line in first_table:
        regex = pattern.findall(line)
        first_table_frmt.append([regex[0], regex[1]])   # QM number, MM number
    for line in second_table:
        regex = pattern.findall(line)
        second_table_frmt.append([regex[0], regex[1]])  # QM number, charge

    # Quick sanity check
    for i in range(len(first_table_frmt)):
        if not int(first_table_frmt[i][0]) == i + 1:
            raise RuntimeError('Unexpected QM indices in QM/MM mapping:\n' + str(first_table_frmt))

    # Combine data into third table by replacing QM number in second_table with corresponding MM number
    third_table = []
    for pair in second_table_frmt:
        try:
            new_pair = [first_table_frmt[int(pair[0]) - 1][1], pair[1]]   # MM number, charge
            third_table.append(new_pair)
        except IndexError:
            if not len(third_table) == len(first_table_frmt):
               raise RuntimeError('Encountered unexpected IndexError in trying to build MM index/charge mapping')

    # Finally, for each term in third_table, set the corresponding atom to the specified partial charge using parmed
    new_top = 'charge_initialized_' + settings.init_topology
    parmed_top = parmed.load_file(settings.init_topology)
    for atom_charge_pair in third_table:
        arg = [str(item) for item in atom_charge_pair]
        try:
            # change <property> <mask> <new_value>
            setcharge = parmed.tools.actions.change(parmed_top, 'CHARGE', '\'@' + arg[0] + '\'', arg[1])
            setcharge.execute()
        except parmed.tools.exceptions.SetParamError as e:
            raise RuntimeError('encountered parmed.tools.exceptions.SetParamError: ' + e + '\n'
                               'The offending atom/charge pair is: ' + str(arg))
    parmed_top.save(new_top, overwrite=True)

    return new_top

if __name__ == "__main__":
    # Load settings.pkl file given as command line argument
    settings = pickle.load(open(sys.argv[1], 'rb'))
    main(settings)
