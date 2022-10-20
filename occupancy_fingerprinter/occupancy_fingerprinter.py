"""Provide the primary functions."""
import mdtraj as md
import numpy as np
import concurrent.futures
import h5py as h5
import multiprocessing

try:
    from _occupancy_fingerprinter import c_fingerprint
except:
    from occupancy_fingerprinter._occupancy_fingerprinter import c_fingerprint


def process_trajectory(traj, sites, atom_radii):
    total_size = 0
    for site in sites.values():
        total_size += site._size
    fingerprints = np.zeros((traj.n_frames,total_size), dtype=np.int64)
    for i, frame in enumerate(traj):
        site_list = []
        for site in sites.values():
            site_list.append(c_fingerprint(frame.xyz[0].astype(np.float64) * 10.,
                                 site._grid_x,
                                 site._grid_y,
                                 site._grid_z,
                                 site._origin,
                                 site._upper_most_corner_crd,
                                 site._upper_most_corner,
                                 site._spacing,
                                 site._counts,
                                 atom_radii))
        fingerprints[i, :] = np.concatenate(site_list, axis=0).flatten()
    return fingerprints


class Grid():
    def __init__(self, traj):
        self._traj = traj
        self._n_sites = 0
        self._sites = {}
        self._atom_radii = np.array([md.geometry.sasa._ATOMIC_RADII[atom.element.symbol] for atom in traj.top.atoms], dtype=np.float64)*10.

    def cal_fingerprint(self, FN, n_tasks=0, return_array=False):
        assert len(self._sites) != 0, "No binding sites set."
        if n_tasks == 0:
            task_divisor = multiprocessing.cpu_count()
        else:
            if n_tasks > multiprocessing.cpu_count():
                task_divisor = multiprocessing.cpu_count()
            else:
                task_divisor = n_tasks
        print(
            f"Calculating fingerprint for {self._n_sites} sites for trajectory of {self._traj.n_frames} frames using {task_divisor} of {multiprocessing.cpu_count()} cpu cores")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result_list = []
            for i in range(task_divisor):
                n_frames = self._traj.n_frames
                n_frames_i = n_frames // task_divisor
                if i == task_divisor - 1:
                    n_frames_i += n_frames % task_divisor
                frame_ind = i * (self._traj.n_frames // task_divisor)
                result_list.append(executor.submit(
                    process_trajectory,
                    self._traj[frame_ind:frame_ind+n_frames_i],
                    self._sites,
                    self._atom_radii
                ))
            frames_list = []
            for i in range(task_divisor):
                partial_frames = result_list[i].result()
                frames_list.append(partial_frames)
            fingerprints = np.concatenate(tuple(frames_list), axis=0)
            with h5.File(FN, 'w') as f:
                f.create_dataset("frames", data=fingerprints, compression="gzip")
            if return_array:
                return fingerprints

    def add_binding_site(self, center, r, spacing):
        self._sites[self._n_sites] = BindingSite(center, r, spacing)
        self._n_sites += 1


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
        return self._center - origin_corner

    def get_grid_counts(self):
        d = ((self._r)*2)
        counts = (np.array([d,d,d])/self._spacing)+1
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
    t = t[:1]
    n_frames = t.n_frames


    grid = Grid(t)
    # real test
    # center1 = np.array([58.390,73.130,27.410])
    # center2 = np.array([90.460,85.970,50.260])
    # spacing = np.array([0.5, 0.5, 0.5])
    # r = 8.
    # grid.add_binding_site(center1,r,spacing)
    # grid.add_binding_site(center2,r,spacing)

    # quicker test
    center1 = np.array([58., 73., 27.])
    r = 3.
    spacing = np.array([1., 1., 1.])
    grid.add_binding_site(center1, r, spacing)

    import time
    start_time = time.time()
    a = grid.cal_fingerprint("./data/fingerprints.h5", n_tasks=1, return_array=True)
    print("--- %s seconds ---" % (time.time() - start_time))


