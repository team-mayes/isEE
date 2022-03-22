"""
Independent script for resampling existing isEE data that may be spread out among different subdirectories.

In the future this may be reworked as a user-facing utility script, but for now, I'm writing it with my particular need
to reanalyze extant trajectories with alternative scoring functions in mind.
"""

import os
import sys
import copy
import glob
import numpy as np
import pytraj
import mdtraj
import pickle
import shutil
import in_place
import warnings
import argparse

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


def main(dirs):
    """
    Iterate through directories in list dirs and resample trajectories therein, producing a new history pickle file in
    the directory from which this is called.

    Each directory in dirs should be a unique isEE working directory.

    Parameters
    ----------
    dirs : list
        List of directories to operate on

    Returns
    -------
    None

    """
    original_dir = os.getcwd()  # store for later
    for dir in dirs:
        os.chdir(dir)   # move to target directory

        # Load the local history file
        if not os.path.exists('algorithm_history.pkl'):
            raise FileNotFoundError('specified directory: ' + dir + ' does not contain an algorithm_history.pkl file. '
                                    'Are you sure this is an isEE working directory?')
        this_history = pickle.load(open('algorithm_history.pkl', 'rb'))
        # todo: if there's a shared history file, then loading "this_history" actually loads a copy of the shared history file... not a problem per se, but silly and inefficient, and produces a lot of warnings.

        # Load the local settings file
        if not os.path.exists('settings.pkl'):
            raise FileNotFoundError('specified directory: ' + dir + ' does not contain a settings.pkl file. '
                                    'Are you sure this is an isEE working directory?')
        settings = pickle.load(open('settings.pkl', 'rb'))

        length = len(this_history.trajs)
        update_progress(0, 'Resampling directory: ' + dir + '...')
        for ii in range(length):    # iterate through moves in the history file
            if this_history.trajs[ii] and this_history.tops[ii] and os.path.exists(this_history.trajs[ii][0]) and os.path.exists(this_history.tops[ii]):
                score = score_func(this_history.trajs[ii][0], this_history.tops[ii])
                this_history.score[ii] = score  # set new score value
            else:
                if this_history.score[ii]:
                    # warnings.warn('directory: ' + dir + ' contains a history file with a score entry for the following '
                    #               'trajectory: ' + this_history.trajs[ii][0] + ', but this file (and/or its corresponding '
                    #               'topology file) does not exist. Setting score to [] and moving on.')
                    this_history.score[ii] = []
            update_progress((ii + 1)/length, 'Resampling directory: ' + dir + '...')

        # Write a new history file for this directory in the original directory (we'll merge them all later)
        os.chdir(original_dir)
        pickle.dump(this_history, open('temp_history_' + dir.replace('/', '_slsh_') + '.pkl', 'wb'))

    # Merge all temp_history pickle files and clean up
    for file in glob.glob('temp_history_*.pkl'):
        this_history = pickle.load(open(file, 'rb'))
        try:
            for key in full_history.__dict__.keys():
                full_history.__dict__[key] += this_history.__dict__[key]
        except NameError:   # first one, full_history doesn't exist yet
            full_history = this_history
        #os.remove(file)

    pickle.dump(full_history, open('resampled_history.pkl', 'wb'))


def get_enm_weights(mtraj, type='fine'):
    # Calculate and return vector of elastic network model weights for the mdtraj trajectory object mtraj. If type ==
    # 'fine' then each coarse-grain bead is one protein residue; otherwise it's divided into 20 total beads.
    # First step: generate list of beads and corresponding masks
    beads = []
    if type == 'fine':
        for res in mtraj.topology.residues:
            beads.append(res)
    elif type == 'coarse':
        beads = [list(item) for item in np.array_split([res for res in mtraj.topology.residues if res.is_protein], 20)]

    # Calculate average position of each bead
    # For each bead get a list of atomic indices
    indices = []
    for bead in beads:
        this_indices = []
        for i in range(len(bead)):
            this_indices += [ato.index for ato in [atom for atom in [res.atoms for res in bead]][i]]
        indices.append(this_indices)

    # Then average by-atom xyz coordinates over each frame
    xyz = np.mean(mtraj.xyz, axis=0)

    # Calculate average position of each bead
    means = []
    for indice in indices:
        means.append(np.mean(xyz[indice], axis=0))

    # For each bead within a cutoff (on average), fit a spring constant to their fluctuations
    kij = np.zeros((len(beads), len(beads)))    # spring constants
    rij = np.zeros((len(beads), len(beads)))    # equilibrium distances
    for i in range(len(beads)):
        for j in range(len(beads)):
            if i == j:
                kij[i, j] = 0
                rij[i, j] = 0
            else:
                rij[i, j] = np.sqrt(np.sum([(means[i][k] - means[j][k])**2 for k in range(3)]))

    # Repeat on a by-atom basis

    # Do iterative Normal Modes fitting on CG model and return final weights

def score_func(traj, top):
    """
    Implement desired score function for resampling.

    Parameters
    ----------
    traj : str
        Path to trajectory file to score
    top : str
        Path to corresponding topology file

    Returns
    -------
    score : float
        Score (usually will be a float, but can be anything really)

    """

    # RMSF of the lie_mask
    ptraj = pytraj.load(traj, top)
    if ptraj.n_frames == 0:
        return []
    ptraj = ptraj[int(ptraj.n_frames/3):]

    mask = '(:442&!@N3)|:443@H4O,O4,C4|:260@OE2,CD' # Beware the kludge
    score = np.mean([item[1] for item in list(pytraj.rmsf(ptraj, mask))])

    # By-residue RMSF profile of protein
    rmsf = pytraj.rmsf(ptraj, options='byres')[:440]

    # By-residue covariance with catalytic residue
    covar = pytraj.matrix.covar(ptraj, '@CA')

    def avg3x3(array):
        # Average in 3x3 blocks across a square array to return another square array of length one third
        assert array.shape[0] == array.shape[1] and array.shape[0] % 3 == 0 and len(array.shape) == 2
        result = np.zeros([int(array.shape[0] / 3), int(array.shape[0] / 3)])
        for ii in range(int(array.shape[0] / 3)):
            for jj in range(int(array.shape[0] / 3)):
                result[ii, jj] = np.mean(array[ii * 3:ii * 3 + 3, jj * 3:jj * 3 + 3])

        return result

    covar = avg3x3(covar)[440]

    # I also want by-atom RMSF with the water and ions removed (experimentally)
    #ptraj = pytraj.strip(ptraj, ':WAT,Na+,Cl-')
    #byatom_rmsf = pytraj.rmsf(ptraj, options='byatom')

    # And throw in some structural parameters using mdtraj
    mtop = mdtraj.load_prmtop(top)
    seq = [str(atom)[0:3] for atom in mtop.atoms if (all([item in [ato.name for ato in atom.residue.atoms] for item in ['C', 'N', 'O']]) and (atom.name == 'CA'))]  # for later
    for resid in range(len(seq)):   # mdtraj renames HIP, HID, and HIE to HIS; we need to undo this
        if seq[resid] == 'HIS':
            names = [atom.name for atom in mtop.residue(resid).atoms]
            if 'HE2' in names and 'HD1' in names:
                seq[resid] = 'HIP'
            elif 'HE2' in names:
                seq[resid] = 'HIE'
            elif 'HD1' in names:
                seq[resid] = 'HID'
            else:
                warnings.warn('Got HIS with unknown protonation state; should be HID, HIE, or HIP. Skipping.')
                return []

    # Rename non-standard residues for analysis
    # table, bonds = mtop.to_dataframe()
    # for pair in [('ASH', 'ASP'), ('GLH', 'GLU'), ('HIP', 'HIS'), ('HIE', 'HIS'), ('HID', 'HIS')]:
    #     table[table['resName'] == pair[0]] = table[table['resName'] == pair[0]].replace(pair[0], pair[1])
    # mtop = mdtraj.Topology.from_dataframe(table, bonds)

    # This way of renaming residues is more than 10x faster
    shutil.copy(top, top + '_temp')
    with in_place.InPlace(top + '_temp') as file:
        for line in file:
            for pair in [('ASH', 'ASP'), ('GLH', 'GLU'), ('HIP', 'HIS'), ('HIE', 'HIS'), ('HID', 'HIS')]:
                line = line.replace(pair[0], pair[1])
            file.write(line)
    mtop = mdtraj.load_prmtop(top + '_temp')
    os.remove(top + '_temp')
    assert len([str(atom) for atom in mtop.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]) == len(seq)

    # Load trajectory in with modified topology
    mtraj = mdtraj.load(traj, top=mtop)
    mtraj = mtraj[int(mtraj.n_frames/3):]
    mtraj = mtraj.remove_solvent()

    # Elastic Network Model weights
    enm_fine = get_enm_weights(mtraj, 'fine')
    enm_coarse = get_enm_weights(mtraj, 'coarse')

    # By-residue number of hydrogen bonds
    hbonds = mdtraj.baker_hubbard(mtraj, periodic=False)
    num_hbonds = [0 for _ in range(mtraj.n_residues)]    # initialize hbond counts by residue
    for hbond in hbonds:
        if not str(mtraj.topology.atom(hbond[0])).split('-')[0] == str(mtraj.topology.atom(hbond[2])).split('-')[0]:
            for atom in [hbond[0], hbond[2]]:
                num_hbonds[int(str(mtraj.topology.atom(atom)).split('-')[0][3:]) - 1] += 1

    # Nematic order parameter
    nematic = np.mean(mdtraj.compute_nematic_order(mtraj, 'residues'))

    # Contact type map (441-length square matrix with each position indicating the type of interaction, if any)
    # First need to compute residue pairs; this code adapted from the compute_contacts() definition of contacts='all'
    # but modified to actually be for all of them and not just those separated by two or more positions
    residue_pairs = []
    for i in mdtraj.utils.six.moves.xrange(mtraj.n_residues):
        residue_i = mtraj.topology.residue(i)
        if not any(a for a in residue_i.atoms if a.name.lower() == 'ca'):
            continue
        for j in mdtraj.utils.six.moves.xrange(i + 1, mtraj.n_residues):
            residue_j = mtraj.topology.residue(j)
            if not any(a for a in residue_j.atoms if a.name.lower() == 'ca'):
                continue
            if residue_i.chain == residue_j.chain:
                residue_pairs.append((i, j))
    residue_pairs = np.array(residue_pairs)

    cutoff = 0.8  # specify cutoff distance in nm

    distances, pairs = mdtraj.compute_contacts(mtraj, contacts=residue_pairs, scheme='closest-heavy', ignore_nonprotein=True)
    contact_map = np.mean(mdtraj.geometry.squareform(distances, pairs), axis=0)
    raw_contact_map = copy.copy(contact_map)

    # Now I need to produce a vector of residue types using the seq vector stored above
    types = []
    assert len(seq) == contact_map.shape[0]
    for res in seq:
        if res.lower() in ['arg', 'hip', 'lys']:    # positively charged sidechains
            types.append('neg')
        elif res.lower() in ['asp', 'glu']:         # negatively charged sidechains
            types.append('pos')
        elif res.lower() in ['ser', 'thr', 'asn', 'gln', 'glh', 'ash', 'hie', 'hid']:   # polar uncharged sidechains
            types.append('pol')
        elif res.lower() in ['phe', 'trp', 'tyr']:  # aromatic sidechains
            types.append('aro')
        elif res.lower() in ['cys', 'gly', 'pro', 'ala', 'ile', 'leu', 'met', 'val']:   # non-aromatic hydrophobic sidechains
            types.append('pho')
        else:
            raise RuntimeError('unrecognized residue name: ' + res)

    interactions = [(1, {'neg', 'neg'}), (1, {'pos', 'pos'}),
                    (2, {'neg', 'pos'}),
                    (3, {'neg', 'pol'}), (3, {'pos', 'pol'}),
                    (4, {'neg', 'aro'}), (4, {'neg', 'pho'}), (4, {'pos', 'aro'}), (4, {'pos', 'pho'}),
                    (5, {'pol', 'pol'}),
                    (6, {'pol', 'aro'}), (6, {'pol', 'pho'}),
                    (7, {'aro', 'aro'}),
                    (8, {'aro', 'pho'}),
                    (9, {'pho', 'pho'})]

    # And now build a new contact map where each distance is replaced with a new integer corresponding to the type of
    # interaction between the residue types iff the distance is less than the cutoff
    for i in range(contact_map.shape[0]):
        for j in range(contact_map.shape[1]):
            if contact_map[i,j] <= cutoff:
                gotem = False
                for inter in interactions:
                    if inter[1] == {types[i], types[j]}:
                        contact_map[i,j] = inter[0]
                        gotem = True
                        break
                if not gotem:
                    raise RuntimeError('unrecognized interaction type pair: ' + types[i] + ',' + types[j])
            else:
                contact_map[i,j] = 0

    result = argparse.Namespace()
    result.rmsf_ts = score
    result.rmsf = rmsf
    result.covar = covar
    result.nematic = nematic
    #result.byatom_rmsf = byatom_rmsf
    result.num_hbonds = num_hbonds
    result.contact_map = contact_map
    result.raw_contact_map = raw_contact_map

    return result


if __name__ == "__main__":
    main(sys.argv[1:])    # todo: consider instead accepting a history file and letting this script identify the directories to work on based on which ones contain local history files that are a subset of the specified file?

