"""
Unit and regression test for the occupancy_fingerprinter package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import occupancy_fingerprinter
from occupancy_fingerprinter import BindingSite
from occupancy_fingerprinter import Grid

import numpy as np
import mdtraj as md
import os


def test_occupancy_fingerprinter_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "occupancy_fingerprinter" in sys.modules

def test_binding_site_init():
    """Test binding site init"""
    center = np.array([10.,10.,10.])
    r = 5.
    spacing = np.array([1., 1., 1.])
    b = BindingSite(center, r, spacing)
    grid_x, grid_y, grid_z = b._cal_grid_coordinates()
    assert (b._center == center).all()
    assert b._r == r 
    assert (b._spacing == spacing).all()
    assert (b._counts == b.get_grid_counts()).all()
    assert (b._origin == b.get_origin()).all()
    assert (b._grid_x == grid_x).all()
    assert (b._grid_y == grid_y).all()
    assert (b._grid_z == grid_z).all()
    assert (b._upper_most_corner_crd == (b._center + ((b._counts - 1) * b._spacing)/2)).all()
    assert (b._upper_most_corner == (b._counts - 1)).all()
    assert (b._size == np.prod(b._counts))

def test_grid_init():
    print(os.getcwd())
    traj_path = "../data/CLONE0.xtc"
    top_path = "../data/prot_masses.pdb"
    t = md.load(traj_path, top=top_path)
    center = np.array([10., 10., 10.])
    r = 3.
    spacing = np.array([1., 1., 1.])
    g = Grid(t)
    assert g._n_sites == 0
    assertDictEqual(g._sites, {})
    g.add_binding_site(center, r, spacing)
    assert g._n_sites == 1


