"""Provide the primary functions."""
import mdtraj as md
import numpy as np
from _occupancy_fingerprinter import c_fingerprint

class Grid():
    def __init__(self, traj):
        self._n_sites = 0
        self._sites = {}
        self._atom_radii = np.array([md.geometry.sasa._ATOMIC_RADII[atom.element.symbol] for atom in traj.top.atoms], dtype=np.float64)

    def calc_fingerprint(self, frame, site):
        assert len(self._sites) != 0, "No binding sites set."
        return c_fingerprint(frame.xyz[0].astype(np.float64),
                             site._grid_x,
                             site._grid_y,
                             site._grid_z,
                             site._origin,
                             site._upper_most_corner_crd,
                             site._upper_most_corner,
                             site._spacing,
                             site._counts,
                             self._atom_radii)

    def add_binding_site(self, center, r, spacing):
        self._sites[self._n_sites] = BindingSite(center, r, spacing)
        self._n_sites += 1

    def process_trajectory(self):
        #TODO: add multiprocessing for this
        print("unimplemented")

class BindingSite():
    def __init__(self, center, r, spacing):
        self._center = center
        self._r = r
        self._spacing = spacing
        self._counts = self.get_grid_counts()
        self._origin = self.get_origin()
        self._grid_x, self._grid_y, self._grid_z = self._cal_grid_coordinates()
        self._upper_most_corner_crd = self._center + ((self._counts - 1) * self._spacing)/2
        self._upper_most_corner = (self._counts - 1)
        self._size = int(sum([(lambda x: x**3)(x) for x in self._counts]))

    def get_origin(self):
        origin_corner = ((self._counts-1)*self._spacing)/2
        print(origin_corner, self._center - origin_corner, self._center, self._center+origin_corner)
        return self._center - origin_corner

    def get_grid_counts(self):
        d = ((self._r)*2)
        print(d)
        counts = (np.array([d,d,d])/self._spacing)+1
        print("counts - 1", (counts-1)*self._spacing)
        return counts.astype(np.int64)

    def _cal_grid_coordinates(self):
        grid_x = np.linspace(
            self._origin[0],
            self._origin[0] + ((self._counts[0] - 1) * self._spacing[0]),
            num=self._counts[0]
        )
        grid_y = np.linspace(
            self._origin[1],
            self._origin[1] + ((self._counts[1] - 1) * self._spacing[1]),
            num=self._counts[1]
        )
        grid_z = np.linspace(
            self._origin[2],
            self._origin[2] + ((self._counts[2] - 1) * self._spacing[2]),
            num=self._counts[2]
        )

        return grid_x, grid_y, grid_z







if __name__ == "__main__":
    # Do something if this file is invoked on its own
    traj_path = "./data/CLONE0.xtc"
    top_path = "./data/prot_masses.pdb"
    t = md.load(traj_path, top=top_path)
    n_frames = t.n_frames


    grid = Grid(t)
    # center1 = np.array([58.390,73.130,27.410])
    center1 = np.array([10., 10., 10.])
    center2 = np.array([90.460,85.970,50.260])
    spacing = np.array([0.25,0.25,0.25])
    grid.add_binding_site(center1,8.,spacing)
    grid.add_binding_site(center2,8.,spacing)

    print("origin", grid._sites[0]._origin, "upper", grid._sites[0]._upper_most_corner_crd)

    # fingerprints = np.zeros((t.n_frames, grid._sites[0]._size), dtype=np.int64)
    # print(fingerprints.shape)
    print(grid._sites[0]._grid_x, grid._sites[0]._grid_y, grid._sites[0]._grid_z)
    for i, frame in enumerate(t):
    #     # print(frame.xyz[0].shape)
        c = grid.calc_fingerprint(frame, grid._sites[1])
    #     # fingerprints[i, :] = c.flatten()
        print(c.sum())
        # fingerprint = grid.c_fingerprint()
    # print(fingerprints[0,:])
    # from math import sqrt
    #
    # d1 = 0
    # d2 = 0
    # org = np.array([0., 0., 0.])
    # dest = np.array([16., 16., 16.])
    # print("calc'd upper most corner crd", grid._sites[0]._upper_most_corner * grid._sites[0]._spacing, "counts",
    #       grid._sites[0]._counts)
    # for i in range(3):
    #     tmp = grid._sites[0]._upper_most_corner_crd[i] - grid._sites[0]._origin[i]
    #     tmp2 = ((grid._sites[0]._counts[i] - 1) * grid._sites[0]._spacing[i]) - org[i]
    #     # tmp2 = dest[i]-1 - org[i]
    #     d1 += tmp * tmp
    #     d2 += tmp2 * tmp2
    # print(sqrt(d1), sqrt(d2))


