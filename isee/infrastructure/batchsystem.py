"""
Interface for BatchSystem objects. New BatchSystems can be implemented by constructing a new class that inherits from
BatchSystem and implements its abstract methods.
"""

import abc
import subprocess
import time
import warnings

class BatchSystem(abc.ABC):
    """
    Abstract base class for HPC cluster batch systems.

    Implements methods for all of the batch system-specific tasks that isEE might need.

    """

    @abc.abstractmethod
    def get_status(self, jobid, settings):  # todo: this should MAYBE be moved to a method of TaskManager
        """
        Query batch system for a status string for the given job.

        Parameters
        ----------
        jobid : str
            The jobid to query
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        status : str
            A one-character status string. Options are: 'R'unning, 'Q'ueued, and 'C'omplete/'C'anceled.

        """

        pass

    @abc.abstractmethod
    def cancel_job(self, jobid, settings):  # todo: this should DEFINITELY be moved to a method of TaskManager
        """
        Cancel the job given by jobid

        Parameters
        ----------
        jobid : str
            The jobid to cancel
        settings : argparse.Namespace
            Settings namespace object

        Returns
        -------
        output : str
            Raw output string from batch system, if any

        """

        pass
    
    @abc.abstractmethod
    def get_submit_command(self):
        """
        Return the appropriate terminal command for submitting a batch job, with '{file}' where the file to submit 
        should be indicated.

        Parameters
        ----------
        None

        Returns
        -------
        output : str
            Appropriate command including '{file}' substring

        """
        
        pass

    @abc.abstractmethod
    def get_exit_statuses(self, jobids):
        """
        Return a list of exit statuses for each jobid in the list jobids

        Parameters
        ----------
        None

        Returns
        -------
        statuses : list
            List of strings; each is either 'completed', 'running', 'cancelled', 'timeout', or 'other'

        """

        pass

    @abc.abstractmethod
    def mps_ignore_strings(self):
        """
        Return strings that, when contained in a line in a batch script file, indicate that the line should not be
        duplicated and separated with an ampersand for use with NVIDIA MPS

        Parameters
        ----------
        None

        Returns
        -------
        output : list
            List of strings

        """


class AdaptSlurm(BatchSystem):
    """
    Adapter class for Slurm BatchSystem.

    """

    def get_status(self, jobid, settings):
        if settings.DEBUG:
            return 'C'

        count = 1
        max_tries = 5
        output = 'first_attempt'
        errors = ['first_attempt', 'slurm_load_jobs', 'slurm_receive_msg', 'send/recv', 'connect failure']   # error messages to retry on

        command = 'squeue -o %t --job ' + str(jobid)
        while True in [error in output for error in errors] and count <= max_tries:
            if not output == 'first_attempt':
                time.sleep(30)      # wait 30 seconds before trying again
                count += 1
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       close_fds=True, shell=True)
            output = process.stdout.read().decode()     # decode converts from bytes-like to string
        if True in [error in output for error in errors] and not 'Invalid job id specified' in output:
            raise ValueError('Queried Slurm system status for jobid ' + str(jobid) + ' but got unexpected status: ' + output)
        try:
            status = output.split('\n')[-2] # [-1] is a blank line, counting from the start is dangerous
            try:
                assert output.split('\n')[-3] == 'ST'
            except AssertionError:
                if output.split('\n')[-2] == 'ST':
                    return 'C'
                else:
                    raise ValueError('Queried Slurm system status for jobid ' + str(jobid) + ' but got unexpected status: '
                                 + output)
        except IndexError:      # jobid either matches no jobs or 'invalid job id specified', which can mean same thing
            return 'C'          # job isn't in the queue; so it's 'C'omplete
        if status in ['PD', 'CF']:
            status = 'Q'
        elif status in ['CG', 'ST'] or 'Invalid job id' in output or not output:  # 'Invalid job id' or empty output returned when job is finished in Slurm
            status = 'C'
        elif status == 'R':
            status = 'R'            # just to be really explicit I guess
        else:
            raise ValueError('Queried Slurm system status for jobid ' + str(jobid) + ' but got unexpected status: ' +
                             str(output.split('\n')))

        return status

    def cancel_job(self, jobid, settings):
        command = 'scancel ' + str(jobid)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read().decode()     # decode converts from bytes-like to string
        return output
    
    def get_submit_command(self):
        return 'sbatch {file}'

    def get_exit_statuses(self, jobids):
        statuses = []
        for jobid in jobids:
            command = 'sacct -j ' + str(jobid) + ' -o State'
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       close_fds=True, shell=True)
            output = process.stdout.read().decode()     # decode converts from bytes-like to string
            try:
                status = output.split('\n')[2].lower().strip(' ')
            except IndexError:  # occurs if slurm accounting is not supported
                statuses.append('unknown')
                continue
            if any([item in status for item in ['completed', 'running', 'cancelled', 'timeout']]):
                statuses.append(status)
            else:
                statuses.append('other')

        return statuses

    def mps_ignore_strings(self):
        return ['#SBATCH']


class AdaptPBS(BatchSystem):
    """
    Adapter class for PBS/Torque BatchSystem.

    """

    def get_status(self, jobid, settings):
        if settings.DEBUG:
            return 'C'

        # Iterate through each status three times for robustness in case the status changes while being checked
        for status in ['CE', 'QHW', 'R', 'CE', 'QHW', 'R', 'CE', 'QHW', 'R']:
            command = 'qselect -u $USER -s ' + status
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       close_fds=True, shell=True)
            output = process.stdout.read().decode()     # decode converts from bytes-like to string

            # Some PBS-specific error handling to help handle common issues by simply resubmitting as necessary.
            while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           close_fds=True, shell=True)
                output = process.stdout.read()
            if 'Bad UID for job execution MSG=user does not exist in server password file' in str(output) or \
               'This stream has already been closed. End of File.' in str(output):
                time.sleep(60)
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           close_fds=True, shell=True)
                output = process.stdout.read()

            if str(jobid) in output:
                if status == 'CE':      # 'Complete' or 'Exiting'
                    return 'C'
                elif status == 'QHW':   # 'Queued', 'Waiting', or 'Hold'
                    return 'Q'
                elif status == 'R':     # 'Running'
                    return 'R'

        # If this code is reached, it means the jobid did not appear in any of the above searches
        process = subprocess.Popen('qstat -u $USER', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read().decode()  # decode converts from bytes-like to string
        raise RuntimeError('unexpected PBS status for jobid: ' + str(jobid) + '\nFull status string: ' + output)

    def cancel_job(self, jobid, settings):
        command = 'qdel ' + str(jobid)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   close_fds=True, shell=True)
        output = process.stdout.read().decode()     # decode converts from bytes-like to string
        while 'Pbs Server is currently too busy to service this request. Please retry this request.' in str(output):
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       close_fds=True, shell=True)
            output = process.stdout.read()
        return output

    def get_submit_command(self):
        return 'qsub {file}'

    def get_exit_statuses(self, jobids):
        # todo: implement (need access to a pbs system to test on)
        warnings.warn('get_exit_statuses is not yet implemented for PBS/Torque; returning [\'other\']')
        return ['other']

    def mps_ignore_strings(self):
        return ['#PBS']