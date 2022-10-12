"""Provide the primary functions."""
import mdtraj as md
import numpy as np
# from occupancy_fingerprinter._occupancy_fingerprinter import c_fingerprint

class Grid():
    def __init__(self, traj):
        self._n_sites = 0
        self._sites = {}
        self._atom_radii = [md.geometry._ATOMIC_RADII[atom.element.symbol] for atom in traj.top.atoms]

    def calc_fingerprint(self, frame):
        assert len(self._sites) != 0, "No binding sites set."
        return c_fingerprint(frame.xyz, self._sites, self._atom_radii)

    def add_binding_site(self, crd, r, spacing):
        self._sites[self.n_sites] = BindingSite(crd, r, spacing)
        self._n_sites += 1

    def process_trajectory(self):
        #TODO: add multiprocessing for this
        print("unimplemented")

class BindingSite():
    def __init__(self, crd, r, spacing):
        self._crd = crd
        self._r = r
        self._spacing = spacing
        self._counts = self.get_grid_counts()
        self._origin = self.get_origin()
        self._grid_x, self._grid_y, self._grid_z = self._cal_grid_coordinates()

    def get_origin(self):
        print("unimplemented.")
        origin_corner = -(self._counts)/2
        return self._crd - origin_corner

    def get_grid_counts(self):
         return np.array([self._r,self._r,self._r])

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
    sites = np.array([[58.390,73.130,27.410,8.],[90.460,85.970,50.260,8.]])
    n_sites = sites.shape[0]
    n_frames = t.n_frames
    size = int(sum([(lambda x: x**3)(x) for x in sites[:,3]]))
    print(size)

    fingerprints = np.zeros((t.n_frames,size), dtype=np.int64)
    print(fingerprints.shape)
    for i, frame in enumerate(t):
        grid = Grid(frame.top, frame.xyz, sites)
        a = np.zeros((8,8,8))
        b = np.ones((8,8,8))
        c = np.concatenate([a,b])
        fingerprints[i, :] = c.flatten()

        # fingerprint = grid.c_fingerprint()
    print(fingerprints[0,:])



