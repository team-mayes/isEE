"""
Routine for getting and setting the appropriate charges for atoms in the transition state. Can be called by isEE as
needed, or used individually from the command line by providing an appropriate settings.pkl file.
"""

import argparse
import mdtraj
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
    thread.name = 'ic'
    thread.skip_update = True
    thread.idle = False
    thread.current_name = 'ic'
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
    temp_settings.walltime = '01:00:00'
    temp_settings.ppn = 1
    temp_settings.nodes = 1
    temp_settings.md_engine = 'amber_init_charges'

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

    # Now we want to get each index in third_table as a function of the index of one atom per residue and save that
    # information as a .pkl file for set_charges to interpret later.
    mtop = mdtraj.load_prmtop(settings.init_topology)
    residues = []           # list of residues accounted for
    first_index = []        # index for first atom come to in each residue
    get_index_strs = []     # :RES@NAME for first encountered index in residue in corresponding position in residues
    relative_indices = []   # strings to evaluate to obtain each atom index, in same order as third_table
    pattern = re.compile('[0-9]+')  # regex for getting resid integer out of mtop residue
    for atom_index in [int(item[0]) for item in third_table]:
        res = pattern.findall(str(mtop.atom(atom_index - 1).residue))[-1]   # - 1 converts from Amber to mdtraj
        if not res in residues:
            residues.append(res)
            first_index.append(atom_index)
            get_index_str = ':' + res + '@' + mtop.atom(atom_index).name
            get_index_strs.append(get_index_str)
            relative_indices.append(get_index_str)
        else:
            index_in_residues = residues.index(res)
            relative_distance = atom_index - first_index[index_in_residues]
            relative_indices.append(get_index_strs[index_in_residues] + ' + ' + str(relative_distance))
    relative_charge_table = [[relative_indices[i], third_table[i][1]] for i in range(len(relative_indices))]
    pickle.dump(relative_charge_table, open('relative_charge_table.pkl', 'wb'))

    # Finally, call set_charges to actually implement modifying the topology file
    new_top = set_charges(settings.init_topology)

    # Setup thread and temp_settings again for equilibration job with new topology file
    thread.current_name = 'equil'
    thread.skip_update = True
    thread.history.tops = [new_top]
    temp_settings.solver = 'pmemd.cuda'
    temp_settings.walltime = '02:00:00'
    temp_settings.md_engine = 'amber'

    new_input_file = 'ic_equil_amber.in'
    with open(new_input_file, 'w') as f:
        added = False
        for line in input_lines:
            if 'dt=' in line:
                line = '  dt=0.002,\n'          # 2 fs steps
            elif 'nstlim=' in line:
                line = '  nstlim=500000,\n'     # 500000 steps -> 1 ns run
            f.write(line)

    # Submit new equilibration job with process
    running = process(thread, [], temp_settings, inp_override=new_input_file)

    # Use the same loop strategy as in main.main to wait for this job to finish
    while True:
        if thread.gatekeeper(running, settings):
            break
        else:
            time.sleep(30)  # to prevent too-frequent calls to batch system by thread.gatekeeper

    return new_top, settings.working_directory + '/' + thread.name + '_' + thread.current_name + '.rst7'

def set_charges(top):
    """
    Implements setting the charges in a topology file 'top' to match the

    Parameters
    ----------
    settings: argparse.Namespace
        The settings namespace object.

    Returns
    -------
    new_top: str
        Path to the newly written topology file with the appropriate charge distribution

    """
    # First step: load relative_charge_table.pkl file and then calculate indices to produce absolute_charge_table
    relative_charge_table = pickle.load(open('relative_charge_table.pkl', 'rb'))
    evaluated = []
    pattern = re.compile('\:[0-9]+\@[A-Z0-9]+')
    mtop = mdtraj.load_prmtop(top)
    for item in relative_charge_table:
        relative_to = pattern.findall(item[0])[0]
        if not relative_to in [item[0] for item in evaluated]:
            parsed = relative_to.replace(':', 'resid ').replace('@', ' and name ')
            relative_to_index = str(mtop.select(parsed)[0])
        else:
            internal_index = [item[0] for item in evaluated].index(relative_to)
            relative_to_index = str(evaluated[internal_index][1])
        to_eval = item[0].replace(relative_to, relative_to_index)
        evaluated.append([relative_to, eval(to_eval)])

    absolute_charge_table = [[evaluated[i][1], relative_charge_table[i][1]] for i in range(len(relative_charge_table))]

    # If top is a path, dissect it into filename and path to that filename
    if '/' in top:
        path_to_top = top[:top.rindex('/') + 1]
        top = top[top.rindex('/') + 1:]
    else:
        path_to_top = ''

    parmed_top = parmed.load_file(path_to_top + top)

    new_top = 'ic_' + top
    for atom_charge_pair in absolute_charge_table:
        arg = [str(item) for item in atom_charge_pair]
        try:
            # change <property> <mask> <new_value>
            setcharge = parmed.tools.actions.change(parmed_top, 'CHARGE', '\'@' + arg[0] + '\'', arg[1])
            setcharge.execute()
        except parmed.tools.exceptions.SetParamError as e:
            raise RuntimeError('encountered parmed.tools.exceptions.SetParamError: ' + e + '\n'
                               'The offending atom/charge pair is: ' + str(arg))
    parmed_top.write_parm(path_to_top + new_top)

    return path_to_top + new_top

if __name__ == "__main__":
    # Load settings.pkl file given as command line argument
    settings = pickle.load(open(sys.argv[1], 'rb'))
    main(settings)
