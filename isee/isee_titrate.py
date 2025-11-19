"""
A standalone script that titrates the input structure (given in the form of an .rst7 and a .prmtop) according to the
predictions from propka3 and the settings stored in the settings.pkl file in the local directory. This script modifies
the input files in place.
"""

import os
import re
import sys
import mdtraj
import pickle
import pytraj
import argparse
import warnings
import fileinput
from contextlib import contextmanager
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import proteinPrepare, systemPrepare
from isee.utilities import mutate


def main(rst, top):
    """
    Modify coordinate and topology files rst and top in place according to the titration predictions from moleculekit.

    Protons are added or removed as needed according to whether the pKa is above or below the titration pH indicated
    in the settings.pH value stored in the settings.pkl file stored in the local directory.

    The actual adding or removing of protons is accomplished by calling isee.utilities.mutate (with no mutations), which
    means that any relevant attributes of the loaded settings object (such as paths_to_forcefields, hmr, and min_steps)
    are passed along to that function.

    Parameters
    ----------
    rst : str
        Path to input coordinate file in rst7 format
    top : str
        Path to input topology file corresponding to rst in prmtop format

    Returns
    -------
    diff : bool
        True if changes were made to the titration state; False if no changes were made. Also outputted to the terminal.

    """
    # Load in the settings
    try:
        settings = pickle.load(open('settings.pkl', 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError('isee_titrate.py requires that a settings.pkl file created by isEE is present in the '
                                'directory it is called from, but one was not found.')

    # Record initial protein sequence of three-letter residue codes
    init_seq = [str(int(str(atom).replace('-CA', '')[3:]) + 1) + str(atom)[0:3] for atom in
                mdtraj.load_prmtop(top).atoms if (atom.residue.is_protein and (atom.name == 'CA'))]

    # Strip out non-protein and convert to pdb format for propka using pytraj
    protein_resnames_list = ['ARG', 'HIS', 'HID', 'HIE', 'HIP', 'LYS', 'ASP', 'ASH', 'GLU', 'GLH', 'SER', 'THR', 'ASN',
                              'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP', 'CYX',
                              'CYM', 'HYP'] + settings.treat_as_protein
    protein_resnames = (':' + ','.join(protein_resnames_list))
    traj = pytraj.load(rst, top)
    traj.strip('!(' + protein_resnames + ')')
    pdbname = rst.replace('.rst7', '') + '.pdb'  # removes .rst7 extension if present, adds .pdb either way
    pytraj.write_traj(pdbname, traj, 'pdb', overwrite=True)
    if not os.path.exists(pdbname):
        raise RuntimeError('failed to create PDB file: ' + pdbname)

    # Rename all ASH -> ASP, GLH -> GLU, and HIP, HID, and HIE -> HIS
    for line in fileinput.input(pdbname, inplace=True):
        print(line.replace(
            ' ASH ', ' ASP ').replace(
            ' GLH ', ' GLU ').replace(
            ' HIP ', ' HIS ').replace(
            ' HID ', ' HIS ').replace(
            ' HIE ', ' HIS '), end='')

    # Deprecated propka workflow (couldn't distinguish between HIE and HID)
    # # Run propka
    # optargs = [[pdbname]]
    # options = loadOptions(*optargs)
    # pdbfile = options.filenames[0]
    # parameters = read_parameter_file(options.parameters, Parameters())
    # my_molecule = MolecularContainer(parameters, options)
    # my_molecule = read_molecule_file(pdbfile, my_molecule)
    # my_molecule.calculate_pka()
    # my_molecule.write_pka()     # writes file with same names as pdbname, but with '.pdb' replaced with '.pka'
    #
    # if not os.path.exists(pdbname[:-3] + 'pka'):
    #     raise RuntimeError('propka failed to produce output file with expected name: ' + pdbname[:-3] + 'pka')
    #
    # # Pull desired information out of pka file
    # pkas = []
    # lines = open(pdbname[:-3] + 'pka', 'r').readlines()
    # are_we_there_yet = False
    # for line in lines:
    #     if are_we_there_yet:
    #         if '---------' in line:
    #             break
    #         else:
    #             summary = line.split()     # entries in the titration line; resname, resid, predicted pKa, model-pKa
    #             pkas.append([summary[0], summary[1], summary[3]])   # summary[2] is the chain id
    #     elif 'Group      pKa  model-pKa   ligand atom-type' in line:
    #         are_we_there_yet = True
    # os.remove(pdbname[:-3] + 'pka')

    # Run moleculekit.proteinPrepare and prepare titrations list
    mol = Molecule(pdbname)
    #mol = systemPrepare(mol, ignore_ns_errors=True)
    with pytraj.utils.context.capture_stdout() as out:  # handy pytraj utility catches C output streams
        molPrep, prepData = systemPrepare(mol, pH=settings.pH, return_details=True, ignore_ns_errors=True)

    # Extract desired information from produced prepData object
    titrations = []
    ii = 0
    while True:  # works around the fact that the length of prepData.data.loc isn't easy to get for some reason
        try:
            titrations.append([prepData.data.loc[ii]['resname'],
                               str(prepData.data.loc[ii]['resid']),
                               prepData.data.loc[ii]['protonation']])
        except:
            break
        ii += 1

    # Call mutate with titrations to apply the changes to the protonation state
    with pytraj.utils.context.capture_stdout() as out:  # handy pytraj utility catches C output streams
        new_rst, new_top = mutate(rst, top, [], pdbname[:-4], settings, titrations)

    # Finally, rename files produced by mutate to match filenames that this was called with
    os.rename(new_rst, rst)
    os.rename(new_top, top)

    # Get new (titrated) protein sequence of three-letter codes
    new_seq = [str(int(str(atom).replace('-CA', '')[3:]) + 1) + str(atom)[0:3] for atom in mdtraj.load_prmtop(top).atoms
               if (atom.residue.is_protein and (atom.name == 'CA'))]

    if not new_seq == init_seq:
        print(True)
        return True
    else:
        print(False)
        return False


if __name__ == "__main__":
    ## All this stuff until the last line just for testing
    # settings = argparse.Namespace()
    # settings.pH = 5
    # settings.SPOOF = False
    # settings.working_directory = './'
    # settings.topology = 'TmAfc_D224G_t200.prmtop'
    # settings.lie_alpha = 0.18
    # settings.lie_beta = 0.33
    # settings.hmr = False
    # settings.ts_mask = ':442,443'
    # settings.paths_to_forcefields = ['171116_FPA_4NP-Xyl_ff.leaprc']
    # settings.min_steps = 0
    # settings.immutable = [260]
    #
    # # Residue1:AtomName1; Residue2:AtomName2; weight in kcal/mol-Å**2; equilibrium bond length in Å
    # settings.ts_bonds = ([':260@OE2', ':***@O4', ':***@O4', ':***@N1'],
    #                      [':***@H4O', ':***@H4O', ':***@C1', ':***@C1'],
    #                      [200, 200, 200, 200],
    #                      [1.27, 1.23, 1.9, 2.4])
    #
    # pickle.dump(settings, open('settings.pkl', 'wb'))
    #
    # main('test_tleap_min.rst7', 'test_tleap.prmtop')
    # os.remove('settings.pkl')

    main(sys.argv[1], sys.argv[2])
