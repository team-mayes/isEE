"""
Interface for StabilityModel objects. New StabilityModels can be implemented by constructing a new class that inherits
from StabilityModel and implements its abstract methods.
"""

import os
import abc
import sys
import time
import numpy
import mdtraj
import pytraj
import fileinput
from subprocess import Popen, PIPE

class StabilityModel(abc.ABC):
    """
    Abstract base class for stability models.

    Implements methods for all of the stabilitymodel-specific tasks that isEE might need.

    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def predict(self, ref_rst, ref_top, mutations):
        """
        Predict the ∆∆G of folding for the specified mutant compared to the given reference structure.

        Negative values indicate destabilizing mutations and positive, stabilizing.

        All mutations indicated are applied together; if multiple distinct mutants are desired, predict should be called
        multiple times.

        Parameters
        ----------
        ref_rst : str
            Path to the reference coordinate file in .rst7 format
        ref_top : str
            Path to the reference topology file corresponding to ref_rst
        mutations : list
            List of mutations to apply in the same format as elsewhere in isEE, namely, each item in the list gives the
            residue index followed by the target residue's three-letter code.

        Returns
        -------
        prediction : float
            The predicted ∆∆G of folding for the mutant

        """

        pass


class DDGunMean(StabilityModel):
    """
    Adapter class for DDGunMean stability model. Uses a modified version of the code available at:
    https://github.com/biofold/ddgun.

    DDGunMean is the average of the two models, DDGunSeq and DDGun3D.

    """

    def predict(self, ref_rst, ref_top, mutations):
        # Prepare list to track residue identities
        all_resnames = [['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                         'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP', 'GLH', 'ASH', 'HIP', 'HIE', 'HID', 'CYX',
                         'CYM', 'HYP'],
                        ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y',
                         'W', 'E', 'D', 'H', 'H', 'H', 'C', 'C', 'P']]

        # Convert to one-letter codes
        mutations = [mut.replace(mut[-3:], all_resnames[1][all_resnames[0].index(mut[-3:])]) for mut in mutations]

        # Convert rst/top files to reference PDB file
        traj = pytraj.iterload(ref_rst, ref_top)                            # load as pytraj trajectory
        traj = pytraj.strip(traj, '!:' + ','.join(all_resnames[0]))         # strip out everything but protein
        pytraj.write_traj('stability.pdb', traj, 'pdb', overwrite=True)     # write pdb of just protein

        # Convert all protonated residue names in pdb file to standard names for compatibility with ddgun
        for line in fileinput.input('stability.pdb', inplace=True):
            print(line.replace(
                ' ASH ', ' ASP ').replace(
                ' GLH ', ' GLU ').replace(
                ' HIP ', ' HIS ').replace(
                ' HID ', ' HIS ').replace(
                ' HIE ', ' HIS '), end='')

        # Prepare FASTA sequence file
        mtraj = mdtraj.load(ref_rst, top=ref_top)
        seq = [str(atom)[0:3] for atom in mtraj.topology.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]
        fasta = ''
        for res in seq:
            fasta += all_resnames[1][all_resnames[0].index(res)]
        open('stability.fasta', 'w').write('>stability.pdb\n' + fasta)

        # Write mutations to an input file in the appropriate format for DDGun
        mutations = [fasta[int(mut[:-1]) - 1] + mut for mut in mutations]   # convert from 123Y to X123Y based on fasta
        open('stability.muts', 'w').write(','.join(mutations))

        # Call DDGun3D, store result
        command = sys.executable + ' /home/tburgin/ddgun/ddgun_3d.py stability.pdb _ stability.muts'    # _ -> all chains   # todo: fix call to ddgun_3d.py
        p = Popen(command.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        try:
            result_3d = numpy.sum([float(item) for item in output.decode().split()[-2].split(',')])
        except:
            raise RuntimeError('unexpected output from ddgun_3d: ' + output.decode() + '\noutput to stderr was: ' + err.decode())

        # Call DDGunSeq, store result
        command = sys.executable + ' /home/tburgin/ddgun/ddgun_seq.py stability.fasta stability.muts'    # _ -> all chains   # todo: fix call to ddgun_seq.py
        p = Popen(command.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        try:
            result_seq = numpy.sum([float(item) for item in output.decode().split()[-2].split(',')])
        except:
            raise RuntimeError('unexpected output from ddgun_seq: ' + output.decode() + '\noutput to stderr was: ' + err.decode())

        # Take average, return
        return numpy.mean([result_3d, result_seq])

class DDGun3D(StabilityModel):
    """
    Adapter class for DDGun3D stability model. Uses a modified version of the code available at:
    https://github.com/biofold/ddgun.

    """

    def predict(self, ref_rst, ref_top, mutations):
        # Prepare list to track residue identities
        all_resnames = [['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                         'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP', 'GLH', 'ASH', 'HIP', 'HIE', 'HID', 'CYX',
                         'CYM', 'HYP'],
                        ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y',
                         'W', 'E', 'D', 'H', 'H', 'H', 'C', 'C', 'P']]

        # Convert to one-letter codes
        mutations = [mut.replace(mut[-3:], all_resnames[1][all_resnames[0].index(mut[-3:])]) for mut in mutations]

        # Convert rst/top files to reference PDB file
        traj = pytraj.iterload(ref_rst, ref_top)                            # load as pytraj trajectory
        traj = pytraj.strip(traj, '!:' + ','.join(all_resnames[0]))         # strip out everything but protein
        pytraj.write_traj('stability.pdb', traj, 'pdb', overwrite=True)     # write pdb of just protein

        # Convert all protonated residue names in pdb file to standard names for compatibility with ddgun
        for line in fileinput.input('stability.pdb', inplace=True):
            print(line.replace(
                ' ASH ', ' ASP ').replace(
                ' GLH ', ' GLU ').replace(
                ' HIP ', ' HIS ').replace(
                ' HID ', ' HIS ').replace(
                ' HIE ', ' HIS '), end='')

        # Prepare FASTA sequence
        mtraj = mdtraj.load(ref_rst, top=ref_top)
        seq = [str(atom)[0:3] for atom in mtraj.topology.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]
        fasta = ''
        for res in seq:
            fasta += all_resnames[1][all_resnames[0].index(res)]

        # Write mutations to an input file in the appropriate format for DDGun
        mutations = [fasta[int(mut[:-1]) - 1] + mut for mut in mutations]   # convert from 123Y to X123Y based on fasta
        open('stability.muts', 'w').write(','.join(mutations))

        def call_ddgun3d():
            command = sys.executable + ' /home/tburgin/ddgun/ddgun_3d.py stability.pdb _ stability.muts'  # _ -> all chains   # todo: fix call to ddgun_3d.py
            p = Popen(command.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, err = p.communicate()
            result_3d = numpy.sum([float(item) for item in output.decode().split()[-2].split(',')])
            return result_3d

        # Call DDGun3D, store result
        try:
            result_3d = call_ddgun3d()
        except:
            time.sleep(30)  # wait 30 seconds
            try:    # try again (sometimes works, I dunno)
                result_3d = call_ddgun3d()
            except:
                raise RuntimeError('unexpected output from ddgun_3d: ' + output.decode() + '\noutput to stderr was: ' + err.decode())

        # Take average, return
        return result_3d

if __name__ == "__main__":
    # For testing only, should never be used
    model = DDGunMean()
    print(model.predict('data/one_frame.rst7', 'data/TmAfc_D224G_t200.prmtop', ['64ASP', '394ALA']))
