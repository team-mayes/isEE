"""
Routine for getting and setting the appropriate charges for atoms in the transition state. Can be called by isEE as
needed, or used individually from the command line by providing an appropriate settings.pkl file.
"""

import argparse
import pickle
import sys
import time
import isee.main as iseemain
from isee.process import process

def main(settings):
    """
    Perform a QM/MM simulation to determine appropriate charge distribution for atoms in the specified QM region and
    write a new topology file with those charges.

    Parameters
    ----------
    settings: argparse.Namespace
        The settings namespace, containing values for ic_qm_mask, ic_qm_theory, ic_qm_cut, ic_qm_charge, init_topology,
        initial_coordinates, ts_bonds, path_to_input_files, md_engine, and path_to_templates.

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
            elif 'nstlim' in line:
                line = '  nstlim=1,\n'      # only a single step
            elif 'ntpr' in line:
                line = '  ntpr=1,\n'        # write to output every step
            f.write(line)

        f.write('\n &qmmm')
        f.write('\n  qmmask=' + str(settings.ic_qm_mask) + ',')
        f.write('\n  qm_theory=' + str(settings.ic_qm_theory) + ',')
        f.write('\n  qmcut=' + str(settings.ic_qm_cut) + ',')
        f.write('\n  qmcharge=' + str(settings.ic_qm_charge) + ',')
        f.write('\n  printcharges=1,')
        f.write('\n &end\n')

    sys.exit()

    # Make a thread to pass to process
    thread = iseemain.Thread()
    thread.name = 'initialize_charges'
    thread.skip_update = True
    thread.idle = False
    thread.current_name = 'prod'
    thread.terminated = False
    thread.history.inpcrd = settings.initial_coordinates
    thread.history.tops = [settings.init_topology]

    # Submit job with process
    running = process(thread, [], settings, inp_override=new_input_file)

    # Use the same loop strategy as in main.main to wait for this job to finish
    while True:
        if thread.gatekeeper(running, settings):
            break
        else:
            time.sleep(30)  # to prevent too-frequent calls to batch system by thread.gatekeeper

    # Read atomic charges from output file, store them in a python-interpretable format
    output = thread.name + '_' + name + '.out'

if __name__ == "__main__":
    # Load settings.pkl file given as command line argument
    settings = pickle.load(open(sys.argv[1], 'rb'))
    main(settings)
