"""
Unit and regression test for the occupancy_fingerprinter package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import occupancy_fingerprinter
from occupancy_fingerprinter import BindingSite

import numpy as np


def test_occupancy_fingerprinter_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "occupancy_fingerprinter" in sys.modules

def test_binding_site_init():
    """Test binding site init"""
    center = np.array([10.,10.,10.])
    r = 5.
    spacing = np.array([1., 1., 1.])
    b = BindingSite(center, r, spacing)
    assert (b._center == center).all()
    assert b._r == r 
    assert (b._spacing == spacing).all()
    assert (b._counts == b.get_grid_counts()).all()
    assert (b._origin == b.get_origin()).all()
    assert (b._upper_most_corner_crd == (b._center + ((b._counts - 1) * b._spacing)/2)).all()
    assert (b._upper_most_corner == (b._counts - 1)).all()
    assert (b._size == np.prod(b._counts))

