"""
isEE
Automated in silico enzyme evolution based on optimizing transition state binding energy in unbiased MD simulations.
"""

from . import main
from . import interpret
from . import jobtype
from . import process
from . import utilities
from . import algorithm
from . import initialize_charges
from isee.infrastructure import configure
from isee.infrastructure import batchsystem
from isee.infrastructure import factory
from isee.infrastructure import mdengine
from isee.infrastructure import taskmanager

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
