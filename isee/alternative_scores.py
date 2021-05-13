"""
Some scripts to test out alternatives to LIE for scoring trajectories. If they are successful, I'll move them to other
part of isEE
"""

import os
import sys
import parmed
import pickle
import pytraj
import mdtraj
import argparse
import glob
import isee.utilities

def ts_bond_energy(traj, top):
    # Calculate the internal bond energy in the transition state bonds averaged over each frame of traj
    return 'not implemented'

def mmpbgbsa(traj, top):
    # Implement MMPBSA and/or MMGBSA using pytraj
    return 'not implemented'

def partial_lie(traj, top, settings):
    # Use LIE, but only on the part of the substrate indicated in settings.lie_mask, removing all the other atoms in
    # settings.ts_mask from the structure first.

    ptraj = pytraj.iterload(traj, top)
    diff = list(ptraj.top.select('(' + settings.ts_mask + ') & !(' + settings.lie_mask + ')'))  # atom indices to remove

    if diff:
        # Remove atoms in diff from topology w/ parmed
        parmed_top = parmed.load_file(top)
        action = parmed.tools.actions.strip(parmed_top, '@' + ','.join([str(item + 1) for item in diff]))
        action.execute()
        action = parmed.tools.actions.setMolecules(parmed_top)
        action.execute()

        # Save the topology file with the new bonds
        parmed_top.write_parm(top + '.temp.prmtop')

        # Remove atoms in diff from coordinates w/ pytraj
        ptraj = ptraj.strip('@' + ','.join([str(item + 1) for item in diff]))
        pytraj.write_traj(traj + '.temp.nc', ptraj, overwrite=True)

        # Do LIE
        result = isee.utilities.lie(traj + '.temp.nc', top + '.temp.prmtop', settings)

        # Remove temporary files and return
        # os.remove(traj + '.temp.nc')
        # os.remove(top + '.temp.prmtop')
    else:
        result = isee.utilities.lie(traj, top, settings)

    return result

if __name__ == "__main__":
    if os.path.exists('settings.pkl'):
        settings = pickle.load(open('settings.pkl', 'rb'))
    else:
        settings = argparse.Namespace()
        settings.lie_dry = True
        settings.lie_decomposed = True
        settings.ts_mask = ':442,443'
        settings.lie_mask = ':442 | :443@O4,H4O'
        settings.lie_alpha = 0.4 #0.18
        settings.lie_beta = -0.04 #0.33
    # print('ts_bond_energy: ' + ts_bond_energy(traj, top))
    # print('mmpbgbsa: ' + mmpbgbsa(traj, top))

    for file in glob.glob('../5steps/*.nc'):
        traj = file
        top = traj.replace('_dry.nc', '_tleap_dry.prmtop').replace('../5steps/', '../5steps/ic_') # sys.argv[2]
        print(traj)
        print(top)
        print('partial_lie: ' + str(partial_lie(traj, top, settings)))

    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/reactants/equil.rst7_64ASP_394ALA_386SER_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/reactants/equil.rst7_64ASP_394ALA_386SER_tleap_dry.prmtop' # sys.argv[2]

    # print('reactants')
    # print('partial_lie (M5): ' + str(partial_lie(traj, top, settings)))

    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/equil.rst7_WT_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/ic_equil.rst7_WT_tleap_dry.prmtop' # sys.argv[2]
    #
    # print(0)
    # print('partial_lie (WT): ' + str(partial_lie(traj, top, settings)))
    #
    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/equil.rst7_64ASP_394ALA_386SER_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/ic_equil.rst7_64ASP_394ALA_386SER_tleap_dry.prmtop' # sys.argv[2]
    #
    # print('partial_lie (M5): ' + str(partial_lie(traj, top, settings)))
    #
    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat1/equil.rst7_WT_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat1/ic_equil.rst7_WT_tleap_dry.prmtop' # sys.argv[2]
    #
    # print(1)
    # print('partial_lie (WT): ' + str(partial_lie(traj, top, settings)))
    #
    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat1/equil.rst7_64ASP_394ALA_386SER_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat1/ic_equil.rst7_64ASP_394ALA_386SER_tleap_dry.prmtop' # sys.argv[2]
    #
    # print('partial_lie (M5): ' + str(partial_lie(traj, top, settings)))
    #
    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat2/equil.rst7_WT_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat2/ic_equil.rst7_WT_tleap_dry.prmtop' # sys.argv[2]
    #
    # print(2)
    # print('partial_lie (WT): ' + str(partial_lie(traj, top, settings)))
    #
    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat2/equil.rst7_64ASP_394ALA_386SER_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat2/ic_equil.rst7_64ASP_394ALA_386SER_tleap_dry.prmtop' # sys.argv[2]
    #
    # print('partial_lie (M5): ' + str(partial_lie(traj, top, settings)))
    #
    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat3/equil.rst7_WT_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat3/ic_equil.rst7_WT_tleap_dry.prmtop' # sys.argv[2]
    #
    # print(3)
    # print('partial_lie (WT): ' + str(partial_lie(traj, top, settings)))
    #
    # traj = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat3/equil.rst7_64ASP_394ALA_386SER_dry.nc' # sys.argv[1]
    # top = '/Users/tburgin/Documents/PycharmProjects/isEE/repeat3/ic_equil.rst7_64ASP_394ALA_386SER_tleap_dry.prmtop' # sys.argv[2]
    #
    # print('partial_lie (M5): ' + str(partial_lie(traj, top, settings)))
