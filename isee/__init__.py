"""
isEE
Automated in silico enzyme evolution based on optimizing transition state binding energy in unbiased MD simulations.
"""

from isee.infrastructure import configure
from . import interpret
from . import jobtype
# Add imports here
from . import main
from . import process
from . import utilities
from . import algorithm
# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
