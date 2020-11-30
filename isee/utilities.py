"""
Utility functions implemented here are clearly defined unit operations. They may only be called once in the code, but
are defined separately for cleanliness and legibility.
"""

import sys
import math
import pytraj
import mdtraj
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
    for frame in range(traj.n_frames):  # todo: this can take a very long time. What is the appropriate spacing between frames?
        lie_temp = pytraj.energy_analysis.lie(traj, mask=settings.ts_mask, frame_indices=[frame])
        EEL = numpy.append(EEL, lie_temp['LIE[EELEC]'])
        VDW = numpy.append(VDW, lie_temp['LIE[EVDW]'])
        i += 1
        update_progress(i / traj.n_frames, 'LIE')

    return settings.lie_alpha * numpy.mean(VDW) + settings.lie_beta * numpy.mean(EEL)


def mutate(coords, topology, mutation, name, settings):
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
    mutation : list
        List of mutations to apply, each given as "<resid><three-letter code>". For example, "70ASP" mutates residue 70
        to aspartate.
    name : str
        String to prepend to all filenames produced by this method
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    new_coords : str
        Path to the newly created, mutated coordinate file, named as name + '_min.rst7'
    new_topology : str
        Path to the newly created, mutated topology file corresponding to new_coords, named as name + '.prmtop'

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
    pytraj.write_traj(name + '_nonprot.mol2', traj, overwrite=True)

    ### Remove all bond terms between atoms with non-standard bonds
    # Going FULL KLUDGE on this because it's starting to look like doing it "right" is extremely involved.
    open(name + '_nonprot_mod.mol2', 'w').close()
    with open(name + '_nonprot_mod.mol2', 'a') as f:
        atoms_yet = False
        bonds_yet = False
        substructure_yet = False
        index_name_list = []
        removed_lines = 0
        bond_count = 0
        for line in open(name + '_nonprot.mol2', 'r').readlines():
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
    open(name + '_nonprot_mod_2.mol2', 'w').close()
    with open(name + '_nonprot_mod_2.mol2', 'a') as f2:
        count = 0
        for line in open(name + '_nonprot_mod.mol2', 'r').readlines():
            if count == 2:
                numbers = line.split()
                newline = line.replace(numbers[1], str(int(numbers[1]) - removed_lines))
                f2.write(newline)
            else:
                f2.write(line)
            count += 1

    os.remove(name + '_nonprot_mod.mol2')
    os.remove(name + '_nonprot.mol2')
    os.rename(name + '_nonprot_mod_2.mol2', name + '_nonprot.mol2')

    ### Cast remainder to separate .pdb
    traj = pytraj.load(coords, topology)
    traj.strip('!(' + protein_resnames + ')')
    pytraj.write_traj(name + '_prot.pdb', traj, overwrite=True)

    ### Mutate
    with open(name + '_mutated.pdb', 'w') as f:
        patterns = [re.compile('\s+[A-Z0-9]+\s+[A-Z]{3}\s+' + mutant[:-3] + '\s+') for mutant in mutation]
        for line in open(name + '_prot.pdb', 'r').readlines():
            if not all(pattern.findall(line) == [] for pattern in patterns):
                pat_index = 0
                for pattern in patterns:
                    try:
                        if pattern.findall(line)[0].split()[0] in ['C','N','O','CA']:
                            newline = line.replace(pattern.findall(line)[0].split()[1], mutation[pat_index][-3:].upper())
                            break
                        else:
                            newline = ''
                    except IndexError:  # happens when line doesn't match pattern
                        pass
                    pat_index += 1
            else:
                newline = line
            f.write(newline)


    ### Rebuild with tleap into .rst7 and .prmtop files
    # todo: there's no way to use tleap to do this without being forced into using Amber (or CHARMM?) force fields...
    # todo: I need to come up with a different strategy if I want to move away from Amber-only.
    system = tleap.System()
    system.pbc_type = None  # turn off automatic solvation
    system.neutralize = False
    system.output_prefix = name + '_tleap'
    system.template_lines = [
        'source leaprc.protein.ff14SB',
        'source leaprc.water.tip3p'] + \
        ['source ' + item + '\n' for item in settings.paths_to_forcefields if item] + \
        ['loadOff atomic_ions.lib',
        'mut = loadpdb ' + name + '_mutated.pdb',
        'nonprot = loadmol2 '  + name + '_nonprot.mol2',
        'model = combine { mut nonprot }',
        'set model box {' + box_dimensions + '}'
    ]
    with suppress_stderr():
        system.build()  # produces a ton of unwanted "WARNING" messages in stderr even when successful

    mutated_rst = name + '_tleap.rst7'
    mutated_top = name + '_tleap.prmtop'

    ### Add ts_bonds using parmed
    parmed_top = parmed.load_file(mutated_top)

    ## KLUDGE KLUDGE KLUDGE ##
    # todo: kloooooj
    temp_ts_bonds = copy.copy(settings.ts_bonds)
    settings.ts_bonds = ([':260@OE2', ':443@O4',  ':443@O4', ':442@N1'],
                         [':443@H4O', ':443@H4O', ':442@C1', ':442@C1'],
                         [200,        200,        200,       200],
                         [1.27,       1.23,       1.9,       2.4])
    ## KLUDGE KLUDGE KLUDGE ##

    settings.ts_bonds = list(map(list, zip(*settings.ts_bonds)))
    for bond in settings.ts_bonds:
        arg = [str(item) for item in bond]
        try:
            setbond = parmed.tools.actions.setBond(parmed_top, arg[0], arg[1], arg[2], arg[3])
            setbond.execute()
        except parmed.tools.exceptions.SetParamError as e:
            raise RuntimeError('encountered parmed.tools.exceptions.SetParamError: ' + e + '\n'
                               'The offending bond and topology are: ' + str(arg) + ' and ' + mutated_top)
    parmed_top.save(mutated_top, overwrite=True)

    settings.ts_bonds = temp_ts_bonds   # todo: k-k-k-kludge

    # if settings.TEST:   # skip minimization to save time
    #     os.rename(mutated_rst, name + '_min.rst7')
    #     return name + '_min.rst7', mutated_top

    ### Minimize with OpenMM
    # First, cast .prmtop to OpenMM topology todo: replace Amber-specific stuff with call to method of MDEngine that returns an OpenMM Simulation object
    openmm_top = AmberPrmtopFile(mutated_top)
    openmm_sys = openmm_top.createSystem(constraints=HBonds, nonbondedMethod=PME, nonbondedCutoff=0.8*nanometer)
    openmm_rst = AmberInpcrdFile(mutated_rst)
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(openmm_top.topology, openmm_sys, integrator)
    simulation.context.setPositions(openmm_rst.positions)

    if openmm_rst.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*openmm_rst.boxVectors)

    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter(name + '_min.pdb', int(settings.min_steps/10), enforcePeriodicBox=False))
    simulation.reporters.append(StateDataReporter(sys.stdout, int(settings.min_steps/10), step=True, potentialEnergy=True, temperature=True))
    simulation.step(settings.min_steps)

    ### Return results
    # First, cast minimization output .pdb back to .rst7
    traj = mdtraj.load_frame(name + '_min.pdb', -1)
    traj.save_amberrst7(name + '_min.rst7')

    ## Deprecated code to do casting with pytraj; fails to carry over box information
    # traj = pytraj.load('min.pdb', mutated_top, frame_indices=[-1])
    # pytraj.write_traj(new_name + '_min.rst7', traj)
    # if os.path.exists(new_name + '_min.rst7.1'):
    #     os.rename(new_name + '_min.rst7.1', new_name + '_min.rst7')

    # Finally, return the coordinate and topology files!
    return name + '_min.rst7', mutated_top


def covariance_profile(thread, move_index, settings):
    """
    Calculate and return the rmsd-of-covariance profile for the alpha carbons in a trajectory, in reference to the
    covariance profile for the residue with index given by settings.covariance_reference_resid. The trajectory and
    topology are taken from the move_index'th step in thread.

    Parameters
    ----------
    thread : Thread
        Thread on which to operate
    move_index : int
        Index of step within thread to operate on. Negative values are supported, to read from the end.
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    rmsd_covariance_profile : list
        List of RMSDs of covariances between the reference residue and each other protein residue corresponding to the
        list index

    """

    if settings.covariance_reference_resid == -1:
        raise RuntimeError('trying to assess a covariance profile without setting the covariance_reference_resid')

    traj = pytraj.iterload(thread.history.trajs[move_index], thread.history.tops[move_index])
    covar_3d = pytraj.matrix.covar(traj, '@CA')     # covariance matrix of alpha carbons; 3Nx3N matrix for N alpha carbons

    # Now we need to average together x-, y-, and z-components to get overall covariance
    N = len(covar_3d[0])
    try:
        assert N/3 % 1 == 0
    except AssertionError:
        raise RuntimeError('attempted to evaluate a covariance matrix, but the length of the sides of the square matrix'
                           ' returned by pytraj.matrix.covar() is not divisible by three as expected. The length was: '
                           + str(len(covar_3d[0])))
    covar = [[numpy.mean(covar_3d[i:i+2,j:j+2]) for i in range(0, N, 3)] for j in range(0, N, 3)]

    # Now, calculate the RMSD between each residue's covariance profile and that of the reference residue and return
    ref_profile = covar[settings.covariance_reference_resid - 1]    # - 1 because resids are 1-indexed
    rmsd_covariance_profile = [
        numpy.sqrt(sum([(ref_profile[j] - covar[resid][j]) ** 2 for j in range(len(covar[resid]))]) / len(covar[resid]))
        for resid in range(len(ref_profile))]

    return rmsd_covariance_profile


if __name__ == '__main__':
    ### This stuff is all for testing, shouldn't ever be called during an isEE run ###
    settings = argparse.Namespace()
    settings.topology = 'TmAfc_D224G_t200.prmtop'
    settings.lie_alpha = 0.18
    settings.lie_beta = 0.33
    settings.ts_mask = ':442,443'
    settings.paths_to_forcefields = ['171116_FPA_4NP-Xyl_ff.leaprc']
    # print(lie(['3_equil_5ns.nc', '3_equil_10ns.nc'], settings))

    # Residue1:AtomName1; Residue2:AtomName2; weight in kcal/mol-Å**2; equilibrium bond length in Å
    settings.ts_bonds = ([':260@OE2', ':***@O4',  ':***@O4', ':***@N1'],
                         [':***@H4O', ':***@H4O', ':***@C1', ':***@C1'],
                         [200,        200,        200,       200],
                         [1.27,       1.23,       1.9,       2.4])
    coords = 'data/one_frame.rst7'
    topology = 'data/TmAfc_D224G_t200.prmtop'

    mutate(coords, topology, ['64ASP','60ALA'], 'test', settings)

    thread = main.Thread()
    thread.history.trajs = ['']
