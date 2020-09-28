"""
isEE
Automated in silico enzyme evolution based on optimizing transition state binding energy in unbiased MD simulations.
"""

# Add imports here
from .isee import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
