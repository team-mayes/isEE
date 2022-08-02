"""
Script for fitting an elastic network model (enm) to a protein from a trajectory.
"""

import sys
import mdtraj
import numpy as np

def fit_enm(traj, top):
    """
    Fit an elastic network model to a trajectory and return the matrix of spring weights between all alpha carbons.

    Parameters
    ----------
    traj : str
        Path to the trajectory file to fit to
    top : str
        Path to the corresponding topology file

    Returns
    -------
    enm_weights : np.array
        Square matrix of spring weights between all pairs of alpha carbons

    """
    traj = mdtraj.iterload(traj, top=top)

    # Get list of CA atoms in protein
    ca_atoms = [atom.index for atom in traj.top.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]

    # Initialize enm_weights as zeroes
    enm_weights = np.zeros(len(ca_atoms))

    # Iterate through each pair of CA atoms to fit spring weight



if __name__ == "__main__":
    fit_enm(sys.argv[1], sys.argv[2])