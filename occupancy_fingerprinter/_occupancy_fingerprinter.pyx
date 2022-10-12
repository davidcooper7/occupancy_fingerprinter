import cython
import numpy as np
cimport numpy as np
import math

cdef extern from "math.h":
    double sqrt(double)

@cython.boundscheck(False)
def cdistance(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    cdef int i, lmax
    cdef double d, tmp
    lmax = x.shape[0]
    d = 0.
    for i in range(lmax):
        tmp = x[i] - y[i]
        d += tmp*tmp
    return sqrt(d)

@cython.boundscheck(False)
def c_get_corner_crd(np.ndarray[np.int64_t, ndim=1] corner,
                     np.ndarray[np.float64_t, ndim=1] grid_x,
                     np.ndarray[np.float64_t, ndim=1] grid_y,
                     np.ndarray[np.float64_t, ndim=1] grid_z):
    cdef:
        int i, j, k
        np.ndarray[np.float64_t, ndim=1] crd

    i, j, k = corner
    crd = np.array([grid_x[i], grid_y[j], grid_z[k]] , dtype=float)
    return crd

@cython.boundscheck(False)
def c_is_in_grid(np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                 np.ndarray[np.float64_t, ndim=1] origin_crd,
                 np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd):
    cdef int i, lmax
    lmax = atom_coordinate.shape[0]
    for i in range(lmax):
        if (atom_coordinate[i] < origin_crd[i]) or (atom_coordinate[i] >= uper_most_corner_crd[i]):
            return False
    return True

@cython.boundscheck(False)
def c_lower_corner_of_containing_cube(  np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                                        np.ndarray[np.float64_t, ndim=1] origin_crd,
                                        np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                                        np.ndarray[np.float64_t, ndim=1] spacing):
    cdef:
        np.ndarray[np.float64_t, ndim=1] tmp
        np.ndarray[np.int64_t, ndim=1]   lower_corner

    if not c_is_in_grid(atom_coordinate, origin_crd, uper_most_corner_crd):
        return np.array([], dtype=int)

    tmp = atom_coordinate - origin_crd
    lower_corner = np.array(tmp / spacing, dtype=int)
    return lower_corner


@cython.boundscheck(False)
def c_corners_within_radius(np.ndarray[np.float64_t, ndim=1] atom_coordinate,
                            double radius,
                            np.ndarray[np.float64_t, ndim=1] origin_crd,
                            np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                            np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
                            np.ndarray[np.float64_t, ndim=1] spacing,
                            np.ndarray[np.float64_t, ndim=1] grid_x,
                            np.ndarray[np.float64_t, ndim=1] grid_y,
                            np.ndarray[np.float64_t, ndim=1] grid_z,
                            np.ndarray[np.int64_t, ndim=1]   grid_counts):
    cdef:
        list corners
        int count_i, count_j, count_k
        int i, j, k
        float r, R

        np.ndarray[np.int64_t, ndim=1] lower_corner
        np.ndarray[np.int64_t, ndim=1] corner

        np.ndarray[np.float64_t, ndim=1] lower_corner_crd
        np.ndarray[np.float64_t, ndim=1] corner_crd
        np.ndarray[np.float64_t, ndim=1] tmp
        np.ndarray[np.float64_t, ndim=1] lower_bound
        np.ndarray[np.float64_t, ndim=1] uper_bound
        np.ndarray[np.float64_t, ndim=1] dx2, dy2, dz2

    assert radius >= 0, "radius must be non-negative"
    if radius == 0:
        return []

    lower_corner = c_lower_corner_of_containing_cube(atom_coordinate, origin_crd, uper_most_corner_crd, spacing)
    if lower_corner.shape[0] > 0:

        lower_corner_crd = c_get_corner_crd(lower_corner, grid_x, grid_y, grid_z)
        r = radius + cdistance(lower_corner_crd, atom_coordinate)

        tmp = np.ceil(r / spacing)
        count_i, count_j, count_k = np.array(tmp, dtype=int)

        corners = []
        for i in range(-count_i, count_i + 1):
            for j in range(-count_j, count_j + 1):
                for k in range(-count_k, count_k + 1):

                    corner = lower_corner + np.array([i, j, k], dtype=int)

                    if np.all(corner >= 0) and np.all(corner <= uper_most_corner):
                        corner_crd = c_get_corner_crd(corner, grid_x, grid_y, grid_z)

                        if cdistance(corner_crd, atom_coordinate) <= radius:
                            corners.append(corner)
        return corners
    else:
        lower_bound = origin_crd - radius
        uper_bound = uper_most_corner_crd + radius
        if np.any(atom_coordinate < lower_bound) or np.any(atom_coordinate > uper_bound):
            return []
        else:
            dx2 = (grid_x - atom_coordinate[0]) ** 2
            dy2 = (grid_y - atom_coordinate[1]) ** 2
            dz2 = (grid_z - atom_coordinate[2]) ** 2

            corners = []
            count_i, count_j, count_k = grid_counts

            for i in range(count_i):
                for j in range(count_j):
                    for k in range(count_k):
                        R = dx2[i] + dy2[j] + dz2[k]
                        R = sqrt(R)
                        if R <= radius:
                            corners.append(np.array([i, j, k], dtype=int))
            return corners

@cython.boundscheck(False)
def c_fingerprint(   str name,
                            np.ndarray[np.float64_t, ndim=2] crd,
                            np.ndarray[np.float64_t, ndim=1] grid_x,
                            np.ndarray[np.float64_t, ndim=1] grid_y,
                            np.ndarray[np.float64_t, ndim=1] grid_z,
                            np.ndarray[np.float64_t, ndim=1] origin_crd,
                            np.ndarray[np.float64_t, ndim=1] uper_most_corner_crd,
                            np.ndarray[np.int64_t, ndim=1]   uper_most_corner,
                            np.ndarray[np.float64_t, ndim=1] spacing,
                            np.ndarray[np.int64_t, ndim=1]   grid_counts,
                            np.ndarray[np.float64_t, ndim=1] atom_radii):

    cdef:
        list corners
        Py_ssize_t natoms = crd.shape[0]
        int i_max = grid_x.shape[0]
        int j_max = grid_y.shape[0]
        int k_max = grid_z.shape[0]
        Py_ssize_t i, j, k
        Py_ssize_t atom_ind
        double radius
        np.ndarray[np.float64_t, ndim=3] grid = np.zeros([i_max, j_max, k_max], dtype=np.float)
        np.ndarray[np.float64_t, ndim=3] grid_tmp
        np.ndarray[np.float64_t, ndim=1] atom_coordinate

    grid_tmp = np.empty([i_max, j_max, k_max], dtype=np.float)
    for atom_ind in range(natoms):
        atom_coordinate = crd[atom_ind]
        radius = atom_radii[atom_ind]
        corners = c_corners_within_radius(atom_coordinate, radius, origin_crd, uper_most_corner_crd,
                                              uper_most_corner, spacing, grid_x, grid_y, grid_z, grid_counts)
        for i, j, k in corners:
            grid[i,j,k] = 1.
    return grid

@cython.boundscheck(False)
def c_fingerprint():
    print("TODO")