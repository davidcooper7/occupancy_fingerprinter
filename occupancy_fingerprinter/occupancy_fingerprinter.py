"""Provide the primary functions."""
import mdtraj as md
import numpy as np
from _occupancy_fingerprinter import c_fingerprint

class Grid():
    def __init__(self, traj):
        self._n_sites = 0
        self._sites = {}
        self._atom_radii = np.array([md.geometry.sasa._ATOMIC_RADII[atom.element.symbol] for atom in traj.top.atoms], dtype=np.float64)*10.

    def calc_fingerprint(self, frame, site):
        assert len(self._sites) != 0, "No binding sites set."
        return c_fingerprint(frame.xyz[0].astype(np.float64)*10.,
                             site._grid_x,
                             site._grid_y,
                             site._grid_z,
                             site._origin,
                             site._upper_most_corner_crd,
                             site._upper_most_corner,
                             site._spacing,
                             site._counts,
                             self._atom_radii*.80)

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
        self._size = np.prod(self._counts)

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

    def write_dx(self, FN, data, grid):
        """
        Writes a grid in dx format
        """
        n_points = data['counts'][0] * data['counts'][1] * data['counts'][2]
        if FN.endswith('.dx'):
            F = open(FN, 'w')

        F.write("""object 1 class gridpositions counts {0[0]} {0[1]} {0[2]}
origin {1[0]} {1[1]} {1[2]}
delta {2[0]} 0.0 0.0
delta 0.0 {2[1]} 0.0
delta 0.0 0.0 {2[2]}
object 2 class gridconnections counts {0[0]} {0[1]} {0[2]}
object 3 class array type double rank 0 items {3} data follows
    """.format(data['counts'], data['origin'], data['spacing'], n_points))

        for start_n in range(0, len(grid.ravel()), 3):
            F.write(' '.join(['%6e' % c
                              for c in grid.ravel()[start_n:start_n + 3]]) + '\n')

        F.write('object 4 class field\n')
        F.write('component "positions" value 1\n')
        F.write('component "connections" value 2\n')
        F.write('component "data" value 3\n')
        F.close()

    def write(self, FN, grid):
        """
        Writes a grid in dx or netcdf format.
        The multiplier affects the origin and spacing.
        """
        print(self._origin, self._counts, self._spacing)
        data = {
            'origin': self._origin,
            'counts': self._counts,
            'spacing': self._spacing,
            'vals': grid
        }
        if FN.endswith('.nc'):
            print('skip')
        #       _write_nc(FN, data_n)
        elif FN.endswith('.dx') or FN.endswith('.dx.gz'):
            self.write_dx(FN, data, grid)
        else:
            raise Exception('File type not supported')







if __name__ == "__main__":
    # Do something if this file is invoked on its own
    traj_path = "./data/CLONE0.xtc"
    top_path = "./data/prot_masses.pdb"
    t = md.load(traj_path, top=top_path)
    n_frames = t.n_frames


    grid = Grid(t)
    center1 = np.array([58.390,73.130,27.410])
    # center1 = np.array([10., 10., 10.])
    center2 = np.array([90.460,85.970,50.260])
    # spacing = np.array([0.25,0.25,0.25])
    spacing = np.array([0.5, 0.5, 0.5])
    grid.add_binding_site(center1,8.,spacing)
    grid.add_binding_site(center2,8.,spacing)

    total_size = 0
    for site in grid._sites.values():
        total_size += site._size
    fingerprints = np.zeros((t.n_frames, total_size), dtype=np.int64)

    import time
    start_time = time.time()
    for i, frame in enumerate(t[:20]):
        c = grid.calc_fingerprint(t[i], grid._sites[0])
        d = grid.calc_fingerprint(t[i], grid._sites[1])
        flat_sites = np.concatenate([c,d]).flatten()
        fingerprints[i, :] = flat_sites
    # c = grid.calc_fingerprint(t[0], grid._sites[0])
    # d = grid.calc_fingerprint(t[0], grid._sites[1])
    print("--- %s seconds ---" % (time.time() - start_time))
    grid._sites[0].write("./data/site0.dx", c)
    grid._sites[1].write("./data/site1.dx", d)
    import h5py as h5
    f = h5.File('fingerprints.h5', 'w')
    f1 = h5.File('fingerprints_compressed.h5', 'w')
    dset = f.create_dataset("init", data=fingerprints)
    dset2 = f1.create_dataset("init", data=fingerprints, compression="gzip")


