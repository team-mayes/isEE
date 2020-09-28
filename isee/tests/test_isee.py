"""
Unit and regression test for the isee package.
"""

# Import package, test suite, and other packages as needed
import isee
import pytest
import sys

def test_isee_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "isee" in sys.modules
