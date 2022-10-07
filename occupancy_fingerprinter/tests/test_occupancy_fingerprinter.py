"""
Unit and regression test for the occupancy_fingerprinter package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import occupancy_fingerprinter


def test_occupancy_fingerprinter_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "occupancy_fingerprinter" in sys.modules
