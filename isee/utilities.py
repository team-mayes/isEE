"""
Utility functions implemented here are clearly defined unit operations. They may only be called once in the code, but
are defined separately for cleanliness and legibility.
"""

import os
import re
import sys
import copy
import math
import numpy
import pytraj
import mdtraj
import parmed
import argparse
import fileinput
import dill as pickle   # I think this is kosher!
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from isee.initialize_charges import set_charges
from main import Thread

# Two different ways to import tleap depending on I think paprika version
try:
    from paprika.build.system import TLeap as tleap
except ModuleNotFoundError:
    from paprika import tleap

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
    lie_temp = pytraj.energy_analysis.lie(traj, mask=settings.ts_mask)
    EEL = lie_temp['LIE[EELEC]']
    VDW = lie_temp['LIE[EVDW]']

    ## Deprecated version. Pytraj has a bug where there's a limit of ~500 calls to pytraj.energy_analysis.lie before it
    ## crashes; there's a workaround on GitHub here, supposedly:
    ## https://github.com/Amber-MD/pytraj/issues/1498#issuecomment-558360392
    # EEL = []
    # VDW = []
    # update_progress(0, 'LIE')
    # i = 0
    # for frame in range(traj.n_frames):  # todo: this can take a very long time. What is the appropriate spacing between frames?
    #     lie_temp = pytraj.energy_analysis.lie(traj, mask=settings.ts_mask, frame_indices=[frame])
    #     EEL = numpy.append(EEL, lie_temp['LIE[EELEC]'])
    #     VDW = numpy.append(VDW, lie_temp['LIE[EVDW]'])
    #     i += 1
    #     update_progress(i / traj.n_frames, 'LIE')

    return settings.lie_alpha * numpy.mean(VDW) + settings.lie_beta * numpy.mean(EEL)


def mutate(coords, topology, mutation, name, settings, titrations=[]):
    """
    Apply the specified mutation to the structure given by coords and topology and return the names of the new coord
    and topology files.

    The strategy for this method is to export the solvent (including any ions) into a separate object, cast the
    remaining coordinates to the .pdb format, remove the sidechain atoms of the specified residue(s) and rename it, then
    rebuild a new coordinate file and topology using AmberTools' tleap program, which will automatically build the
    missing sidechain for the appropriate residue and resolvate it in the exported solvent (with water molecules deleted
    where there is conflict with the new model). The resulting structure will then be minimized using OpenMM directly
    and the results outputted to a new .rst7 formatted coordinate file and corresponding topology file.

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
    titrations : list
        List of entries [resname, resid, protonation_state] to use to protonate histidine, aspartate, and glutamate
        residues. The protonation_state is the resname for the appropriate protonation state (e.g., HIP, GLH)

    Returns
    -------
    new_coords : str
        Path to the newly created, mutated coordinate file, named as name + '_min.rst7'
    new_topology : str
        Path to the newly created, mutated topology file corresponding to new_coords, named as name + '.prmtop'

    """

    if settings.SPOOF:
        return name + '_min.rst7', name + '_tleap.prmtop'

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

    # Force runtime to working directory; should not be necessary, but might be anyway...
    os.chdir(settings.working_directory)

    ### Store box information for later
    boxline = ''
    lineindex = -1
    while not boxline:  # skip any number of blank lines
        boxline = open(coords, 'r').readlines()[lineindex]
        lineindex -= 1
    box_dimensions = ' '.join(boxline.split()[0:3])

    # These lines get box_dimensions from the topology file, which may be different; Amber reads box dimensions from coordinate files when present
    # box_line_index = open(topology, 'r').readlines().index('%FLAG BOX_DIMENSIONS\n')
    # box_dimensions = ' '.join([str(float(item)) for item in open(topology, 'r').readlines()[box_line_index + 2].split()[1:]])

    ### Get all non-protein, store separately as mol2 to preserve explicit atom types and topology
    protein_resnames = ':ARG,HIS,HID,HIE,HIP,LYS,ASP,ASH,GLU,GLH,SER,THR,ASN,GLN,CYS,GLY,PRO,ALA,VAL,ILE,LEU,MET,PHE,TYR,TRP,CYX,CYM,HYP'   # todo: is this exhaustive? Is there a better way to do this?
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

    # Adjust number of bonds
    open(name + '_nonprot_mod_2.mol2', 'w').close()
    with open(name + '_nonprot_mod_2.mol2', 'a') as f2:
        count = 0
        for line in open(name + '_nonprot_mod.mol2', 'r').readlines():
            if count == 2:
                numbers = line.split()
                newline = line[::-1].replace(numbers[1][::-1], str(int(numbers[1]) - removed_lines)[::-1], 1)[::-1] # replace once from right
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
    # First, reset all titrations if we have new ones to use
    if titrations:
        # Rename all ASH -> ASP, GLH -> GLU, and HIP, HID, and HIE -> HIS
        for line in fileinput.input(name + '_prot.pdb', inplace=True):
            print(line.replace(
                ' ASH ', ' ASP ').replace(
                ' GLH ', ' GLU ').replace(
                ' HIP ', ' HIS ').replace(
                ' HID ', ' HIS ').replace(
                ' HIE ', ' HIS '), end='')
    with open(name + '_mutated.pdb', 'w') as f:
        patterns = [re.compile('\s+[A-Z0-9]+\s+[A-Z]{3}\s+' + mutant[:-3] + '\s+') for mutant in mutation if mutant]
        for line in open(name + '_prot.pdb', 'r').readlines():
            if patterns and not all(pattern.findall(line) == [] for pattern in patterns):
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
            elif titrations:  # if we were passed titration results to use and this line wasn't already mutated
                try:
                    titration_index = -1
                    try:
                        resnameandindex = [line.split()[3], line.split()[4]]
                        titration_index = [[titration[0], titration[1]] for titration in titrations].index(
                            resnameandindex)
                    except IndexError:  # line doesn't have a residue listed
                        continue
                    if titration_index >= 0 and titrations[titration_index][0] in ['HIS', 'ASP', 'GLU'] and \
                            not int(titrations[titration_index][1]) in settings.immutable and \
                            not titrations[titration_index][0] == titrations[titration_index][2] and \
                            not line.split()[2][0] == 'H':  # last statement: if this is not a hydrogen
                        newline = line.replace(titrations[titration_index][0], titrations[titration_index][2])
                    elif titration_index >= 0 and titrations[titration_index][0] in ['HIS', 'ASP', 'GLU'] and \
                            not int(titrations[titration_index][1]) in settings.immutable and \
                            not titrations[titration_index][0] == titrations[titration_index][2] and \
                            line.split()[2][0] == 'H':  # same as above but it IS a hydrogen
                        newline = ''
                    else:
                        newline = line
                except ValueError:  # this line doesn't correspond to an entry in titrations
                    newline = line
            else:
                newline = line
            f.write(newline)

    ### Rebuild with tleap into .rst7 and .prmtop files
    # todo: there's no way to use tleap to do this without being forced into using Amber (or CHARMM?) force fields...
    # todo: I need to come up with a different strategy if I want to move away from Amber-only. Really this whole thing
    # todo: needs to be user-customizable in some way, fundamental as it is to the process as a whole.
    try:
        system = tleap()
    except TypeError:
        system = tleap.System()
    system.pbc_type = None  # turn off automatic solvation
    system.neutralize = False
    system.output_path = settings.working_directory
    system.output_prefix = name + '_tleap'
    system.template_lines = ['source ' + item + '\n' for item in settings.paths_to_forcefields if item] + \
        ['source leaprc.protein.ff19SB',
        'source leaprc.GLYCAM_06j-1',
        'source leaprc.water.opc',  # essential to load OPC last to avoid solvent model getting overwritten
        'WAT = OP3',
        'HOH = OP3',
        'mut = loadpdb ' + name + '_mutated.pdb',
        'nonprot = loadmol2 '  + name + '_nonprot.mol2',
        'model = combine { mut nonprot }',
        'set model box {' + box_dimensions + '}'
    ]
    with suppress_stderr():
        try:
            system.build(clean_files=False)  # produces a ton of unwanted "WARNING" messages in stderr even when successful
        except TypeError:   # older versions don't support clean_files argument
            system.build()

    mutated_rst = name + '_tleap.rst7'
    mutated_top = name + '_tleap.prmtop'

    # Add ts_bonds to mutated_top
    add_ts_bonds(mutated_top, settings)

    # If appropriate, apply calculated charges
    if settings.initialize_charges:
        mutated_top = set_charges(mutated_top)

    # if settings.TEST:   # skip minimization to save time
    #     os.rename(mutated_rst, name + '_min.rst7')
    #     return name + '_min.rst7', mutated_top

    if settings.min_steps > 0:
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
        traj.save_pdb()

        # Some file cleanup
        os.remove(name + '_min.pdb')
        os.remove(name + '_tleap.rst7')

        to_return = name + '_min.rst7'
    else:
        to_return = mutated_rst

    ## Deprecated code to do casting with pytraj; fails to carry over box information
    # traj = pytraj.load('min.pdb', mutated_top, frame_indices=[-1])
    # pytraj.write_traj(new_name + '_min.rst7', traj)
    # if os.path.exists(new_name + '_min.rst7.1'):
    #     os.rename(new_name + '_min.rst7.1', new_name + '_min.rst7')

    # Clean up unnecessary files and return the coordinate and topology files!
    os.remove(name + '_mutated.pdb')
    os.remove(name + '_prot.pdb')
    os.remove(name + '_nonprot.mol2')

    return to_return, mutated_top


def add_ts_bonds(top, settings):
    """
    Add settings.ts_bonds to the given topology file using parmed. The file will be modified in place.

    Also applied hydrogen mass repartitioning if settings.hmr = True

    Parameters
    ----------
    top : str
        Path to the topology file to modify
    settings : argparse.Namespace
        Settings namespace object

    Returns
    -------
    None

    """

    # Load topology into parmed
    try:
        parmed_top = parmed.load_file(top)
    except parmed.exceptions.FormatNotFound:
        raise RuntimeError('problem with topology file: ' + top + '\nDid something go wrong with tleap?')

    ## KLUDGE KLUDGE KLUDGE ##
    # todo: kloooooj
    temp_ts_bonds = copy.copy(settings.ts_bonds)
    settings.ts_bonds = ([':260@OE2', ':443@O4',  ':443@O4', ':442@N1'],
                         [':443@H4O', ':443@H4O', ':442@C1', ':442@C1'],
                         [1000,       1000,       400,       400],
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
                               'The offending bond and topology are: ' + str(arg) + ' and ' + top)

    if settings.hmr:
        action = parmed.tools.actions.HMassRepartition(parmed_top)
        action.execute()

    # Save the topology file with the new bonds
    parmed_top.write_parm(top)

    settings.ts_bonds = temp_ts_bonds   # todo: k-k-k-kludge


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


def score_spoof(seq, correl, settings):
    """
    Spoof a score for a given residue sequence according to some internal rules.

    This function is used only for testing purposes, when settings.SPOOF = True. It plays no role in production.

    Spoofing is done by assigning a random magnitude to each position proportional to the covariance score and a random
    type of residue (positive, negative, aromatic, hydrophobic, polar, bulky, small) that is preferred in that position.
    Then a random synergy score is applied for each pair of residues and this is multiplied by each individual score.
    Finally, a small amount of random noise is added.

    Parameters
    ----------
    seq : list
        The sequence of three-letter residue codes for this mutant
    correl : list
        seq-length list of correlation scores by residue
    settings : argparse.Namespace
        Settings namespace

    Returns
    -------
    score : float
        Spoofed score value for the given sequence

    """
    all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL',
                    'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']

    noise = 0.001  # weight factor for random noise (multiplied by random from normal distribution)
    syn_weight = 0.001    # weight factor for synergy terms

    # First, set up the latent algorithm if it hasn't already been
    if not settings.spoof_latent:
        mutability = 0.05   # likelihood that a residue will be assigned a different ideal type than its current type
        settings.spoof_latent = argparse.Namespace()    # initialize
        settings.spoof_latent.weights = []              # randomly chosen weights proportional to correl
        settings.spoof_latent.ideals = []               # randomly chosen ideal residue type, odds to be different from wild type are given by mutability
        settings.spoof_latent.synergies = numpy.ones([len(seq),len(seq)])  # random synergy map
        resid = -1      # initialize resid
        for res in seq:
            resid += 1  # increment resid
            weight = numpy.random.normal() * correl[resid]
            settings.spoof_latent.weights.append(weight)

            if res in ['ARG', 'LYS', 'HIS']:
                wt_type = 'positive'
            elif res in ['ASP', 'GLU']:
                wt_type = 'negative'
            elif res in ['TRP', 'PHE', 'TYR']:
                wt_type = 'aromatic'
            elif res in ['LEU', 'ILE', 'VAL', 'MET']:
                wt_type = 'hydrophobic'
            elif res in ['SER', 'THR', 'GLN', 'ASN']:
                wt_type = 'polar'
            elif res in ['ALA', 'GLY']:
                wt_type = 'small'
            elif res == 'CYS':
                wt_type = 'CYS'
            elif res == 'PRO':
                wt_type = 'PRO'
            else:
                raise RuntimeError('Unrecognized residue name: ' + res)

            if numpy.random.rand() > mutability:
                settings.spoof_latent.ideals.append(wt_type)
            else:
                settings.spoof_latent.ideals.append(['positive', 'negative', 'aromatic', 'hydrophobic', 'polar', 'small', 'CYS', 'PRO'][numpy.random.randint(0, 8)])

            for other_index in range(len(seq)):
                if other_index == resid:
                    continue
                settings.spoof_latent.synergies[resid, other_index] = numpy.random.normal()
                settings.spoof_latent.synergies[other_index, resid] = settings.spoof_latent.synergies[resid, other_index]

        if not settings.dont_dump:
            temp_settings = copy.copy(settings)  # initialize temporary copy of settings to modify
            temp_settings.__dict__.pop('env')  # env attribute is not picklable
            pickle.dump(temp_settings, open(settings.working_directory + '/settings.pkl', 'wb'))

    score = 0   # initialize score

    resid = -1
    for res in seq:
        resid += 1
        ideal = settings.spoof_latent.ideals[resid]
        if res in ['ARG', 'LYS', 'HIS'] and ideal == 'positive':
            ideal_type = True
        elif res in ['ASP', 'GLU'] and ideal == 'negative':
            ideal_type = True
        elif res in ['TRP', 'PHE', 'TYR'] and ideal == 'aromatic':
            ideal_type = True
        elif res in ['LEU', 'ILE', 'VAL', 'MET'] and ideal == 'hydrophobic':
            ideal_type = True
        elif res in ['SER', 'THR', 'GLN', 'ASN'] and ideal == 'polar':
            ideal_type = True
        elif res in ['ALA', 'GLY'] and ideal == 'small':
            ideal_type = True
        elif res == 'CYS' and ideal == 'CYS':
            ideal_type = True
        elif res == 'PRO' and ideal == 'PRO':
            ideal_type = True
        else:
            ideal_type = False

        this_score = 0

        if not ideal_type:
            this_score += settings.spoof_latent.weights[resid] * -1
        else:
            this_score += settings.spoof_latent.weights[resid]

        this_score += noise * numpy.random.normal()

        syn_score = 0
        for other_index in range(len(seq)):
            if other_index == resid:
                continue
            syn_score += syn_weight * settings.spoof_latent.synergies[resid, other_index] * this_score

        score += this_score + syn_score

    return score


def strip_and_store(traj, top, settings):
    """
    Strip water (':WAT') out of the given trajectory and topology files and store the "dry" versions in
    settings.storage_directory.

    The files are named the same as the input files, except for '_dry' inserted just before the file extension.

    Parameters
    ----------
    traj : str
        Path to trajectory file
    top : str
        Path to topology file
    settings : argparse.Namespace
        Settings namespace

    Returns
    -------
    None

    """

    if not os.path.exists(settings.storage_directory):
        os.mkdir(settings.storage_directory)

    # Trajectory
    ptraj = pytraj.iterload(traj, top)
    ptraj = pytraj.strip(ptraj, ':WAT')
    dry_traj_name = traj[:traj.rindex('.')] + '_dry' + traj[traj.rindex('.'):]  # insert '_dry'
    dry_traj_name = dry_traj_name[traj.rindex('/') + 1:]                        # remove path, leaving only filename
    pytraj.write_traj(settings.storage_directory + '/' + dry_traj_name, ptraj)  # save it to storage

    # Topology
    ptop = pytraj.load_topology(top)
    ptop = pytraj.strip(ptop, ':WAT')
    dry_top_name = top[:top.rindex('.')] + '_dry' + top[top.rindex('.'):]       # insert '_dry'
    dry_top_name = dry_top_name[traj.rindex('/') + 1:]                          # remove path, leaving only filename
    pytraj.write_parm(settings.storage_directory + '/' + dry_top_name, ptop)    # save it to storage


if __name__ == '__main__':
    ### This stuff is all for testing, shouldn't ever be called during an isEE run ###
    settings = argparse.Namespace()
    # settings.spoof_latent = False
    # settings.covariance_reference_resid = 260
    # settings.plural_penalty = 1
    #
    # mtraj = mdtraj.load('data/one_frame.rst7', top='data/TmAfc_D224G_t200.prmtop')
    # seq = [str(atom)[0:3] for atom in mtraj.topology.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]
    #
    # thread = Thread()
    # thread.history = argparse.Namespace()
    # thread.history.trajs = ['data/one_frame.rst7']
    # thread.history.tops = ['data/TmAfc_D224G_t200.prmtop']
    #
    # rmsd_covar = [2.409151127122221, 1.9845629910244407, 1.6409770071516554, 1.2069303829003601, 1.1774104996742552,
    #               1.1414646722509711, 1.6277495210688269, 1.7178594550668773, 1.5573849264923196, 1.6222616936921075,
    #               2.139257339920909, 2.0355790819694772, 2.0670293620816573, 2.0509967163136467, 2.3718218856798265,
    #               2.725089368986255, 2.6368147154575774, 2.2260307731037887, 2.5657250982226993, 2.8139534330304286,
    #               2.4156272283232267, 2.1185344711518828, 1.7966304212974096, 1.495615071178279, 1.0665349714934624,
    #               0.62622389357105, 0.41661834767956046, 0.33238540271523664, 0.5266911884945009, 0.8603863335688027,
    #               1.2961691493210963, 1.3149073997096503, 1.087848067605599, 1.3980104852851503, 1.7593319546319754,
    #               1.436783423351133, 1.6735766023866119, 1.659270934828769, 1.6463508914996159, 1.3603778641613482,
    #               1.546642862021431, 1.6936155445917012, 1.7574027783441544, 1.7342633556126716, 1.4121915763550301,
    #               1.6524978464246118, 1.6129345105308022, 1.7510921326857187, 1.6307382119143303, 2.154613585911121,
    #               2.156907141198287, 1.725686467726665, 1.8871688474992456, 2.288093265532053, 2.0877648683775574,
    #               1.6154540392514545, 1.2448718430424812, 1.1084758370905747, 1.2047520664808857, 1.1741113899674331,
    #               1.4872555630783437, 1.8148940365888924, 1.9624111544303016, 2.1034988697779524, 2.360337367399307,
    #               2.692259080940934, 2.8470775307056306, 2.950603784872049, 3.4151379698075583, 3.474764677558553,
    #               3.0042367509224386, 2.794936213963638, 2.5714411950890304, 3.0499310290409363, 3.2612488945150475,
    #               2.863811282105302, 2.925291191195256, 3.4715106088872867, 3.383187628094896, 2.9925172066994925,
    #               3.078550824373092, 3.5276035223482007, 3.6829112885280217, 3.4131480501006095, 2.975245985352162,
    #               2.571204510725302, 2.026526280899236, 2.1705092136262616, 2.326261646306838, 1.895536239749708,
    #               1.612370269197273, 1.7443935702592746, 1.6920798364775174, 1.2961090424218624, 1.1321792266686648,
    #               0.7915806960860717, 0.8425653659092112, 0.6276519979344761, 0.5508744129335861, 0.768327956507145,
    #               0.9515841437190209, 1.3683201822203757, 1.1712898660583293, 1.0250037039145583, 1.462105078021419,
    #               1.686335245815784, 1.4334984152413268, 1.5708144546831355, 2.052314708700172, 2.038735418403611,
    #               1.8131322137605297, 2.2361306675822448, 1.9378961600424507, 2.190625937133571, 1.787426435393106,
    #               1.2976323362080442, 0.8843715780240702, 0.5228643898438631, 0.2753030693786261, 0.2931480175456677,
    #               0.5604517800951593, 0.7368742584734022, 1.146118756018045, 1.3938059917073062, 1.243670202048456,
    #               1.0929851633070546, 0.6902545909682317, 0.4509524708374205, 0.4099084244249464, 0.48631771273118757,
    #               0.4715336762950448, 0.5811367041156211, 0.7975226590410984, 1.1169981318533977, 1.299278924163868,
    #               1.353778763719865, 0.9631797347498909, 0.7778446523137179, 0.7216003876497649, 1.0897635260493828,
    #               1.2557407427022462, 1.1505684470150626, 0.974161238371388, 0.765544297085648, 0.6202937686638598,
    #               0.6299491474448252, 0.533219470284319, 0.6634134980282882, 0.9140578548659407, 1.0212212614103604,
    #               1.0497564563853488, 1.3142009062245186, 1.5813282681733314, 1.5830934657785871, 1.6888525831972165,
    #               2.041600833871994, 2.202213598464107, 2.2170919890886727, 2.4894435986060954, 2.1440275518530654,
    #               2.011302037148799, 1.5824784649567594, 1.231793242869025, 0.7597221084756685, 0.36574467108503445,
    #               0.2426180788866027, 0.5627032962183899, 0.5795986303204472, 0.8688177925717158, 1.2011175714700462,
    #               1.4146732517566771, 1.9096741487027076, 1.9433687887027842, 1.9567698304844803, 2.1513283474511753,
    #               2.6395192768727735, 2.7210238754183904, 2.3910420851463057, 1.9554328922013766, 2.2069190451734633,
    #               1.8735633762533965, 1.3824968668748268, 1.484110887336474, 1.7264446496422705, 1.263819735591131,
    #               1.137969578171706, 1.6015427254654742, 1.855641517560868, 1.6201557567389, 1.7824029538859167,
    #               2.233427836014575, 1.972627246902554, 1.5540540530661726, 1.454894772726324, 1.2679617701951245,
    #               0.851167820885501, 0.7074277161313157, 0.8157599686830077, 0.5664890398789291, 0.2701987764217301,
    #               0.30107691606115244, 0.3163238299766273, 0.3685276068911975, 0.5869647407044045, 0.5324965531897539,
    #               0.6922229012169531, 1.075642431945148, 1.1037202279417786, 1.043566513472472, 1.285622065768631,
    #               1.5418846846427452, 1.441335928529488, 1.6879800474718196, 1.3087187529675477, 0.9074400812850182,
    #               0.606917811710234, 0.21780464531946842, 0.2931568214704138, 0.630420891795513, 0.47699966727896803,
    #               0.5547007426080225, 0.9079568930591633, 0.9576678785506575, 0.8452547790687734, 0.5146568706079566,
    #               0.25019025232020536, 0.12654961819877517, 0.19407457914767376, 0.39592379771816294,
    #               0.7609881678155445, 0.8226159250285602, 0.7910535454423352, 1.0979486806926377, 1.4144657484975933,
    #               1.4041643681220197, 1.4528880499993324, 1.8154599187476193, 2.057022967204504, 2.0486456185722006,
    #               2.128184414690382, 2.5047800212121523, 2.3410527214924435, 1.8597505389873465, 1.5375925603209826,
    #               1.1361808589684976, 0.8608179507169844, 0.566523921777011, 0.13140513506918064, 0.07643223807901155,
    #               0.10645400076481107, 0.27127477466451755, 0.6354730069052482, 0.9679355477112124, 1.0532284826810165,
    #               1.446919662451723, 1.2985703066987582, 0.9327201827220035, 0.6363473270869764, 0.37707143679148636,
    #               0.0, 0.11117381419196652, 0.5411263237797045, 0.5286961430415533, 0.22935841839544527,
    #               0.275447433171813, 0.7361389457070859, 1.177071043945897, 1.0809764298283535, 1.7106017113521783,
    #               1.4211480726764867, 1.745983440371482, 1.6773188079734158, 1.767584681909782, 1.3388175680809646,
    #               1.1410965551353385, 0.8290482834641163, 0.4444106899569443, 0.17953802145559583, 0.21121193561551252,
    #               0.22396349255208844, 0.47577322321209203, 0.7254588799332433, 0.7779900420863387, 0.5091329394865541,
    #               0.704930093038628, 0.7941977888666573, 1.0859653157623244, 1.3038849488772195, 1.3662859941822716,
    #               1.0967829645658635, 0.9715060851202962, 0.6464536128311025, 0.5987589389687789, 0.6077785965036145,
    #               0.3611553664445629, 0.28945654938493703, 0.6287773515956873, 1.0913176823927886, 1.2927752008730398,
    #               0.9581180586300749, 1.090772797990753, 1.6000260470249799, 1.4915620210429987, 1.2186495598768083,
    #               1.6242335959151546, 1.9598572280225335, 1.6580632496121226, 1.5876816490809949, 2.0526459274293436,
    #               2.205011437726407, 1.9124488341758386, 2.061931045203091, 1.7368555755010284, 1.6297440578280893,
    #               1.3103347862305699, 1.0151235722146335, 0.8059030226851512, 0.42043910934251677, 0.3848455687139601,
    #               0.3590899804844901, 0.4141687821613241, 0.6995170831116444, 0.9826975673263751, 0.8194318159601455,
    #               0.5092828602216573, 0.44155123531463325, 0.40661681691716456, 0.4052022342291145, 0.42814104434780625,
    #               0.3833280005477048, 0.48796059768077726, 0.7969440863216863, 0.9240711606558762, 0.9388317747586525,
    #               1.1852224388384098, 1.4941374463147943, 1.5350917933650963, 1.6461872916744764, 1.9756075334098921,
    #               2.2140541337950506, 2.2657139881707336, 2.4478347857196536, 2.7996263736786267, 3.0125933434981746,
    #               2.9837737832734907, 2.901736796335611, 3.328834576772897, 3.1183658185321623, 2.73079868631526,
    #               2.9248641891616955, 3.202118688821212, 3.1508518551592273, 3.0757607896187644, 2.9124350891931727,
    #               3.1606175487987334, 3.0644915574986578, 2.68605399609867, 2.3542348162145617, 2.1849715516617287,
    #               2.275185568753259, 2.096041955897397, 2.3356478796927447, 2.2685770315015517, 1.989714490289354,
    #               1.655795013026901, 1.8393096927883619, 1.8222212723938744, 2.2501436955325262, 2.34359750439921,
    #               2.7869629794529933, 2.994581454486375, 3.4884471102193877, 3.8101881861875384, 4.262079394553044,
    #               4.548642999489229, 4.238979005642821, 3.7733912293831864, 3.393830151211838, 3.1279857232921846,
    #               2.674127307105069, 2.4175490279309866, 1.8687369311056565, 1.8869166551159962, 2.2636483829618825,
    #               2.4023400423920345, 2.359951525860288, 2.8270711837342333, 3.1686124840254863, 3.4165226314876707,
    #               3.17498211022827, 2.972637284797088, 2.8439145950543123, 2.5982878703543366, 2.6852835930288164,
    #               3.138381512010268, 3.6813803788278388, 4.023955267760979, 4.448302859331009, 4.606420642764985,
    #               5.105594724983221, 4.858767352349876, 4.353596739482308, 4.284761409624414, 3.897795345829884,
    #               3.7587067732613675, 3.712495038883498, 4.157025759828298, 4.5783732940487525, 4.581539121601806,
    #               4.712942631238383, 4.418594026092707, 4.566824163913762, 4.32212482190961, 4.29595069676135,
    #               4.136138750950603, 3.9951211532035513, 3.7980376683223147, 3.3070358891562206, 3.2361462063231747,
    #               3.5892744274502157, 3.6740648671037346, 3.739058741077101, 3.8511284283921143, 3.5419133044670237,
    #               3.594028848231803, 3.196600532786006, 3.356354056977835, 3.402522717011724, 2.9375519598739452,
    #               2.7466527424402343, 2.963536795599139, 2.773755925825151, 2.2682060292216804, 1.9490444166797825,
    #               2.3281984080601004, 2.7692411688918215, 3.0911847946978086, 3.546368506384973, 3.9041936075483354,
    #               4.165817729092229, 4.584574723822292]
    #
    # if os.path.exists('temp_settings.pkl'):
    #     settings = pickle.load(open('temp_settings.pkl', 'rb'))
    #
    # for null in range(10):
    #     score = score_spoof(seq, rmsd_covar, settings)
    #     print(score)
    # for null in range(10):
    #     score = score_spoof(seq[:260] + ['ALA'] + seq[261:], rmsd_covar, settings)
    #     print(score)
    # for null in range(10):
    #     score = score_spoof(seq[:260] + ['SER'] + seq[261:], rmsd_covar, settings)
    #     print(score)
    #
    # pickle.dump(settings, open('temp_settings.pkl', 'wb'))
    #
    # # print(score)
    # sys.exit()

    settings.SPOOF = False
    settings.working_directory = './'
    settings.topology = 'TmAfc_D224G_t200.prmtop'
    settings.lie_alpha = 0.18
    settings.lie_beta = 0.33
    settings.hmr = False
    settings.ts_mask = ':442,443'
    settings.paths_to_forcefields = ['171116_FPA_4NP-Xyl_ff.leaprc']
    settings.min_steps = 0
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
