# Import package, test suite, and other packages as needed
import sys
import pytest
import occupancy_fingerprinter
from occupancy_fingerprinter import BindingSite
from occupancy_fingerprinter import Grid
import numpy as np
import mdtraj as md
import h5py as h5
import os
from copy import deepcopy
import py3Dmol
from datetime import datetime
from pathlib import Path
cwd = Path.cwd()
mod_path = Path(os.getcwd()).parent
print(mod_path)
# Methods for clustering
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from vina.vina import Vina
from os.path import expanduser
home = expanduser("~")
import multiprocessing as mp


"""
CLUSTERING METHODS
"""


def fp_to_distance_matrix(fp):
    # Get matrix
    return pairwise_distances(fp, metric='jaccard', n_jobs=40)


def cluster_matrix(matrix, n_clusters):
    # Get linkage
    sq = squareform(matrix, checks=True)
    Z = linkage(sq, method='complete')

    # Cluster
    return fcluster(Z, t=n_clusters, criterion='maxclust')


def compute_centroids(dist_matrix: np.array, assignments: np.array):
    # Assertions
    assert len(dist_matrix) == len(assignments), f"Length of dist_matrix {len(dist_matrix)} does not equal length of assignments {len(assignments)}."

    # Compute centroids
    centroids = np.empty(assignments.max(), int)
    for i, cluster_no in enumerate(range(1, assignments.max()+1)):
        inds = np.where(assignments == cluster_no)[0]
        cluster_dist_matrix = dist_matrix[inds]
        cluster_dist_matrix = cluster_dist_matrix[:,inds]
        centroids[i] = int(inds[np.where(cluster_dist_matrix.sum(axis=0) == cluster_dist_matrix.sum(axis=0).min())[0][0]])

    return np.array(centroids)


def get_centroids_traj(super_traj: md.Trajectory, centroids: np.array):
    # Get traj
    trajs = []
    for centroid in centroids:
        trajs.append(super_traj.slice(centroid))

    centroids_traj = md.join(trajs, discard_overlapping_frames=False, check_topology=True)

    return centroids_traj

def save_centroid_frames(centroid_traj, centroids, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Save as .pdb
    for frame in range(centroid_traj.n_frames):
        pdb = os.path.join(save_dir, f"{centroids[frame]}.pdb")
        centroid_traj[frame].save_pdb(pdb)


"""
DOCKING PREP METHODS
"""

def prep_ligand(lig_smiles, lig_name, save_dir):
    # Write smiles to file
    smiles_dir = os.path.join(save_dir, 'smiles')
    if not os.path.exists(smiles_dir):
        os.mkdir(smiles_dir)
    F = open(os.path.join(smiles_dir, lig_name +".smiles"),'w')
    F.write(lig_smiles)
    F.close()

    # Convert to mol2 w/ obabel
    mol2_dir = os.path.join(save_dir, 'mol2')
    if not os.path.exists(mol2_dir):
        os.mkdir(mol2_dir)
    os.system(f'obabel {os.path.join(smiles_dir, lig_name +".smiles")} -O {os.path.join(mol2_dir, lig_name +".mol2")} --gen3d --best --canonical --minimize --ff GAFF --steps 10000 --sd')

    # Convert to pdbqt w/ mgltools
    pdbqt_dir = os.path.join(save_dir, 'pdbqt')
    if not os.path.exists(pdbqt_dir):
        os.mkdir(pdbqt_dir)
    cwd = os.getcwd()
    os.chdir(mol2_dir)
    if os.path.exists(os.path.join(home, 'miniconda3/envs/vina/bin/prepare_ligand4.py')):
        os.system(f'python2 ~/miniconda3/envs/vina/bin/prepare_ligand4.py -l {lig_name +".mol2"} -o {os.path.join(pdbqt_dir, lig_name +".pdbqt")} -U nphs_lps')
    
    elif os.path.exists(os.path.join(home, 'anaconda3/envs/vina/bin/prepare_ligand4.py')):
        os.system(f'python2 ~/anaconda3/envs/vina/bin/prepare_ligand4.py -l {lig_name +".mol2"} -o {os.path.join(pdbqt_dir, lig_name +".pdbqt")} -U nphs_lps')
    else:
        raise FileNotFoundError(os.path.join(home, 'miniconda3/envs/vina/bin/prepare_ligand4.py'))
    
    os.chdir(cwd)

    return os.path.join(pdbqt_dir, lig_name +".pdbqt")


def prep_receptor(pdb_dir, pdbqt_dir):
    
    # Prepare each pdb w/ mgltools
    for pdb in os.listdir(pdb_dir):
        if pdb.endswith('.pdb'):
            if os.path.exists(os.path.join(home, 'miniconda3/envs/vina/bin/prepare_receptor4.py')):
                os.system(f'python2 ~/miniconda3/envs/vina/bin/prepare_receptor4.py -r {os.path.join(pdb_dir, pdb)} -o {os.path.join(pdbqt_dir, pdb.split('.')[0] + '.pdbqt')} -U nphs_lps')
            elif os.path.exists(os.path.join(home, 'anaconda3/envs/vina/bin/prepare_receptor4.py')):
                os.system(f'python2 ~/anaconda3/envs/vina/bin/prepare_receptor4.py -r {os.path.join(pdb_dir, pdb)} -o {os.path.join(pdbqt_dir, pdb.split('.')[0] + '.pdbqt')} -U nphs_lps')
            else:
                raise FileNotFoundError(os.path.join(home, 'anaconda3/envs/vina/bin/prepare_receptor4.py'))


"""
DOCKING METHODS
"""


def vina_dock(lig_dir, lig_pdbqt_list, receptor_pdbqt, center, box_size, lig_pdbqt_out_dir=None, n_poses=1, exhaustiveness=8):

    
    # Build vina obj
    try:
        n_threads = int(os.environ['NUM_THREADS'])
    except:
        n_threads = mp.cpu_count()
    v = Vina(cpu=n_threads, verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
    # Prepare
    v.compute_vina_maps(center, box_size, force_even_voxels=True)


    # Set ligands
    scores = np.zeros((len(lig_pdbqt_list), n_poses))
    for i, lig_pdbqt in enumerate(lig_pdbqt_list):

        print(os.path.join(lig_dir, lig_pdbqt))
        v.set_ligand_from_file(os.path.join(lig_dir, lig_pdbqt))
            
        # Dock
        v.dock(n_poses=n_poses, exhaustiveness=exhaustiveness)
        lig_scores = v.energies(n_poses=n_poses)[:,0]
        scores[i, :len(lig_scores)] = lig_scores.copy() 

        # Save pose, if specified
        if lig_pdbqt_out_dir is not None:
            lig_name = lig_pdbqt.split('/')[-1].split('.')[0]
            print(lig_pdbqt_out_dir, lig_name)
            print(os.path.join(lig_pdbqt_out_dir, lig_name+'.pdbqt'))
            v.write_poses(os.path.join(lig_pdbqt_out_dir, lig_name+'.pdbqt'), n_poses=n_poses, overwrite=True)


    return scores

    

def exp_avg(scores):
    beta = 1.0
    weights = np.exp(-beta * scores)
    return np.sum(weights * scores) / np.sum(weights)