"""
Utility functions implemented here are clearly defined unit operations. They may only be called once in the code, but
are defined separately for cleanliness and legibility.
"""

import sys
import math
import pytraj
import numpy
import re
import argparse
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from paprika import tleap
import parmed

def update_progress(progress, message='Progress', eta=0, quiet=False):
    """
    Print a dynamic progress bar to stdout.

    Credit to Brian Khuu from stackoverflow, https://stackoverflow.com/questions/3160699/python-progress-bar

    Parameters
    ----------
    progress : float
        A number between 0 and 1 indicating the fractional completeness of the bar. A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%.
    message : str
        The string to precede the progress bar (so as to indicate what is progressing)
    eta : int
        Number of seconds to display as estimated completion time (converted into HH:MM:SS)
    quiet : bool
        If True, suppresses output entirely

    Returns
    -------
    None

    """

    if quiet:
        return None

    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done!          \r\n"
    block = int(round(barLength * progress))
    if eta:
        # eta is in seconds; convert into HH:MM:SS
        eta_h = str(math.floor(eta/3600))
        eta_m = str(math.floor((eta % 3600) / 60))
        eta_s = str(math.floor((eta % 3600) % 60)) + ' '
        if len(eta_m) == 1:
            eta_m = '0' + eta_m
        if len(eta_s) == 2:
            eta_s = '0' + eta_s
        eta_str = eta_h + ':' + eta_m + ':' + eta_s
        text = "\r" + message + ": [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 2), status) + " ETA: " + eta_str
    else:
        text = "\r" + message + ": [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def lie(trajectory, topology, settings):
    """
    Measure and return the linear interaction energy between the atoms in settings.ts_mask and the remainder of the
    system, using the weighting parameters settings.lie_alpha and settings.lie_beta.

    Parameters
    ----------
    trajectory : str
        Path to the trajectory file to analyze
    topology : str
        Path to the topology file corresponding to the trajectory file
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    energy : float
        Interaction energy in kcal/mol

    """

    # Load trajectory
    traj = pytraj.iterload(trajectory, topology)

    # Compute LIE
    EEL = []
    VDW = []
    update_progress(0, 'LIE')
    i = 0
    for frame in range(traj.n_frames):
        lie_temp = pytraj.energy_analysis.lie(traj, mask=settings.ts_mask, frame_indices=[frame])
        EEL = numpy.append(EEL, lie_temp['LIE[EELEC]'])
        VDW = numpy.append(VDW, lie_temp['LIE[EVDW]'])
        i += 1
        update_progress(i / traj.n_frames, 'LIE')

    return settings.lie_alpha * numpy.mean(VDW) + settings.lie_beta * numpy.mean(EEL)


def mutate(coords, topology, mutation, settings):
    """
    Apply the specified mutation to the structure given by coords and topology and return the names of the new coord
    and topology files.

    The strategy for this method is to export the solvent (including any ions) into a separate object, cast the
    remaining coordinates to the .pdb format, remove the sidechain atoms of the specified residue and rename it, then
    rebuild a new coordinate file and topology using AmberTools' tleap program, which will automatically build the
    missing sidechain for the appropriate residue and resolvate it in the exported solvent (with water molecules deleted
    where there is conflict with the new model). The resulting structure will then be minimized using OpenMM directly
    and the results outputted to a new .rst7 formatted coordinate file which is the output from this method.

    Parameters
    ----------
    coords : str
        Path to the coordinate file to mutate
    topology : str
        Path to the topology file corresponding to coords
    mutation : str
        Mutation to apply, given as "<resid><three-letter code>". For example, "70ASP" mutates residue 70 to aspartate
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    new_coords : str
        Path to the newly created, mutated coordinate file
    new_topology : str
        Path to the newly created, mutated topology file corresponding to new_coords

    """

    # todo: implement format checking on 'mutation' (and coords and topology while we're at it)

    from contextlib import contextmanager

    # Define helper function to suppress unwanted output from tleap
    @contextmanager
    def suppress_stderr():
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr

    ### Store box information for later
    # boxline = open(coords, 'r').readlines()[-1]
    box_line_index = open(topology, 'r').readlines().index('%FLAG BOX_DIMENSIONS\n')
    box_dimensions = ' '.join([str(float(item)) for item in open(topology, 'r').readlines()[box_line_index + 2].split()[1:]])

    ### Get all non-protein, store separately as mol2 to preserve explicit atom types and topology
    protein_resnames = ':ARG,HIS,HID,HIE,HIP,LYS,ASP,ASH,GLU,GLH,SER,THR,ASN,GLN,CYS,GLY,PRO,ALA,VAL,ILE,LEU,MET,PHE,TYR,TRP,CYX,HYP'   # todo: is this exhaustive? Is there a better way to do this?
    traj = pytraj.load(coords, topology)
    traj.strip(protein_resnames)
    pytraj.write_traj('nonprot.mol2', traj, overwrite=True)

    ### Remove all bond terms between atoms with non-standard bonds
    # Going FULL KLUDGE on this because it's starting to look like doing it "right" is extremely involved.
    open('nonprot_mod.mol2', 'w').close()
    with open('nonprot_mod.mol2', 'a') as f:
        atoms_yet = False
        bonds_yet = False
        substructure_yet = False
        index_name_list = []
        removed_lines = 0
        bond_count = 0
        for line in open('nonprot.mol2', 'r').readlines():
            if '@<TRIPOS>ATOM' in line:
                atoms_yet = True
                newline = line
                f.write(newline)
                continue
            if '@<TRIPOS>BOND' in line:
                bonds_yet = True
                atoms_yet = False
                index_name_list = list(map(list, zip(*index_name_list)))    # transpose index_name_list
                newline = line
                f.write(newline)
                continue
            if '@<TRIPOS>SUBSTRUCTURE' in line:
                substructure_yet = True
                bonds_yet = False
                newline = line
                f.write(newline)
                continue
            if substructure_yet:    # a substructure line
                newline = line
                f.write(newline)
                continue
            if atoms_yet and not 'WAT' in line: # an atom line not water
                index_name_list.append([line.split()[0], line.split()[1]])
            if bonds_yet == False:  # a water atom line, or preamble
                newline = line
                f.write(newline)
                continue
            else:   # a bond line
                atoms = line.split()[1:3]
                try:
                    [index_name_list[1][index_name_list[0].index(atom)] for atom in atoms]
                except ValueError:  # atom index not in list, so it's a WAT
                    bond_count += 1
                    newline = line.replace(line.split()[0], str(bond_count), 1)
                    f.write(newline)
                    continue
                if all([':***@' + index_name_list[1][index_name_list[0].index(atom)] in settings.ts_bonds[0] + settings.ts_bonds[1] for atom in atoms]):    # todo: can this be generalized?
                    removed_lines += 1
                    continue
                else:
                    bond_count += 1
                    newline = line.replace(line.split()[0], str(bond_count), 1)
                    f.write(newline)
                    continue
    open('nonprot_mod_2.mol2', 'w').close()
    with open('nonprot_mod_2.mol2', 'a') as f2:
        count = 0
        for line in open('nonprot_mod.mol2', 'r').readlines():
            if count == 2:
                numbers = line.split()
                newline = line.replace(numbers[1], str(int(numbers[1]) - removed_lines))
                f2.write(newline)
            else:
                f2.write(line)
            count += 1

    os.remove('nonprot_mod.mol2')
    os.remove('nonprot.mol2')
    os.rename('nonprot_mod_2.mol2', 'nonprot.mol2')

    ### Cast remainder to separate .pdb
    traj = pytraj.load(coords, topology)
    traj.strip('!(' + protein_resnames + ')')
    pytraj.write_traj('prot.pdb', traj, overwrite=True)

    ### Mutate  # todo: consider implementing multiple simultaneous mutations by looping this block
    pattern = re.compile('\s+[A-Z0-9]+\s+[A-Z]{3}\s+' + mutation[:-3])
    with open('mutated.pdb','w') as f:
        for line in open('prot.pdb', 'r').readlines():
            if not pattern.findall(line) == []:
                if pattern.findall(line)[0].split()[0] in ['C','N','O','CA']:
                    newline = line.replace(pattern.findall(line)[0].split()[1], mutation[-3:].upper())
                else:
                    newline = ''
            else:
                newline = line
            f.write(newline)

    new_name = 'mutant'     # todo: need a naming convention here

    ### Rebuild with tleap into .rst7 and .prmtop files
    # todo: there's no way to use tleap to do this without being forced into using Amber (or CHARMM?) force fields...
    # todo: I need to come up with a different strategy if I want to move away from Amber-only.
    system = tleap.System()
    system.pbc_type = None  # turn off automatic solvation
    system.neutralize = False
    system.output_prefix = new_name
    system.template_lines = [
        'source leaprc.protein.ff14SB',
        'source leaprc.water.tip3p'] + \
        ['source ' + item for item in settings.paths_to_forcefields] + \
        ['loadOff atomic_ions.lib',
        'mut = loadpdb mutated.pdb',
        'nonprot = loadmol2 nonprot.mol2',
        'model = combine { mut nonprot }',
        'set model box {' + box_dimensions + '}'
    ]
    with suppress_stderr():
        system.build()  # produces a ton of unwanted "WARNING" messages in stderr even when successful

    mutated_rst = new_name + '.rst7'
    mutated_top = new_name + '.prmtop'

    ### Add ts_bonds using parmed
    parmed_top = parmed.load_file('mutant.prmtop')

    ## KLUDGE KLUDGE KLUDGE ##
    # todo: kloooooj
    settings.ts_bonds = ([':260@OE2', ':443@O4',  ':443@O4', ':442@N1'],
                         [':443@H4O', ':443@H4O', ':442@C1', ':442@C1'],
                         [200,        200,        200,       200],
                         [1.27,       1.23,       1.9,       2.4])
    ## KLUDGE KLUDGE KLUDGE ##

    settings.ts_bonds = list(map(list, zip(*settings.ts_bonds)))
    for bond in settings.ts_bonds:
        arg = [str(item) for item in bond]
        setbond = parmed.tools.actions.setBond(parmed_top, arg[0], arg[1], arg[2], arg[3])
        setbond.execute()
    parmed_top.save(mutated_top, overwrite=True)

    ### Minimize with OpenMM
    # First, cast .prmtop to OpenMM topology todo: replace Amber-specific stuff with call to method of MDEngine that returns an OpenMM Simulation object
    openmm_top = AmberPrmtopFile(mutated_top)
    openmm_sys = openmm_top.createSystem(constraints=None, nonbondedMethod=PME, nonbondedCutoff=0.8*nanometer)
    openmm_rst = AmberInpcrdFile(mutated_rst)
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(openmm_top.topology, openmm_sys, integrator)
    simulation.context.setPositions(openmm_rst.positions)

    if openmm_rst.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*openmm_rst.boxVectors)

    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter('min.pdb', 500, enforcePeriodicBox=False))
    simulation.reporters.append(StateDataReporter(sys.stdout, 500, step=True, potentialEnergy=True, temperature=True))
    simulation.step(5000)

    ### Return results
    # First, cast minimization output .pdb back to .rst7
    traj = pytraj.load('min.pdb', mutated_top, frame_indices=[-1])
    pytraj.write_traj(new_name + '_min.rst7', traj)
    if os.path.exists(new_name + '_min.rst7.1'):
        os.rename(new_name + '_min.rst7.1', new_name + '_min.rst7')

    return new_name + '_min.rst7', mutated_top


def covariance_profile(thread, move_index, resid, settings):
    """
    Calculate and return the covariance profile for the residue given by resid, in reference to the one given by
    settings.covariance_reference_resid. The trajectory and topology are taken from the move_index'th step in thread.

    Parameters
    ----------
    thread : Thread
        Thread on which to operate
    move_index : int
        Index of step within thread to operate on. Negative values are supported, to read from the end.
    resid : int
        resid of residue within the trajectory to calculate the profile for
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    covariance_profile : list
        List of covariance between resid and each other protein residue corresponding to the list index

    """

    # todo: implement

    pass


if __name__ == '__main__':
    ### This stuff is all for testing, shouldn't ever be called during an isEE run ###
    settings = argparse.Namespace
    settings.topology = 'TmAfc_D224G_t200.prmtop'
    settings.lie_alpha = 0.18
    settings.lie_beta = 0.33
    settings.ts_mask = ':442,443'
    # print(lie(['3_equil_5ns.nc', '3_equil_10ns.nc'], settings))

    # Residue1:AtomName1; Residue2:AtomName2; weight in kcal/mol-Å**2; equilibrium bond length in Å
    settings.ts_bonds = ([':260@OE2', ':***@O4',  ':***@O4', ':***@N1'],
                         [':***@H4O', ':***@H4O', ':***@C1', ':***@C1'],
                         [200,        200,        200,       200],
                         [1.27,       1.23,       1.9,       2.4])
    coords = 'data/one_frame.rst7'
    topology = 'data/TmAfc_D224G_t200.prmtop'
    mutate(coords, topology, '70ASP', settings)
