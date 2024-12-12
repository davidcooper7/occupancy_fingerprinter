# Import package, test suite, and other packages as needed
import sys, pytest, os, py3Dmol, json, sys
import occupancy_fingerprinter
from occupancy_fingerprinter import BindingSite
from occupancy_fingerprinter import Grid
import numpy as np
import mdtraj as md
import h5py as h5
from copy import deepcopy
from datetime import datetime
from pathlib import Path
cwd = Path.cwd()
mod_path = Path(os.getcwd()).parent
import matplotlib.pyplot as plt
from stratification_sampling import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Load json
    json_fn = sys.argv[1]
    with open(json_fn, 'r') as f:
        params_dict = json.load(f)
        f.close()

    
    # Build trajectory
    traj_dir = params_dict['traj_dir']
    resSeqs_to_include = params_dict['resSeqs_to_include']
    binding_pocket_resSeqs = params_dict["binding_pocket_resSeqs"]
    super_traj_dir = params_dict['super_traj_dir']
    
    traj, frame_labels = combine_trajectories(traj_dir, resSeqs_to_include, binding_pocket_resSeqs, super_traj_dir)

    response = input(f'Slicing traj / 1000 with shape {traj.n_frames}, are you sure you want to continue? y/n?\n')
    if response != 'y':
        raise Exception(response)
    else:
        traj = traj[::1000]
    

    # Identify binding site from trajectory
    bp_resSeqs = params_dict['bp_resSeqs']
    sele = traj.topology.select('name CA and resSeq '+ ' '.join([str(resSeq) for resSeq in bp_resSeqs]))
    com = md.compute_center_of_mass(traj.atom_slice(sele)) * 10 # convert to Angstrom

    
    # Build Grid object with binding site
    g = Grid(traj)
    center = np.array([com[:,0].mean(), com[:,1].mean(), com[:,2].mean()])
    g.add_binding_site(center=center,
                       r=params_dict['radius'],
                       spacing=np.array(params_dict['spacing']))

    
    # Run fingerprinting
    start = datetime.now()
    fp_output_dir = params_dict['fp_output_dir']
    if not os.path.exists(os.path.join(fp_output_dir, 'fingerprinting.h5')):
        fp = g.cal_fingerprint(os.path.join(fp_output_dir, 'fingerprinting.h5'), n_tasks=params_dict['n_tasks'], return_array=True)
        np.save(os.path.join(fp_output_dir, 'fingerprinting.npy'), fp)
        print(datetime.now() - start, flush=True)

    
    # Load fingerprint into memory if not already there
    if 'fp' not in locals():
        fp = np.load(os.path.join(fp_output_dir, 'fingerprinting.npy'))
    if 'frame_labels' not in locals():
        frame_labels = np.load(os.path.join(fp_output_dir, 'frame_labels.npy'))

    
    # Get distance matrix
    stride=1
    dist_matrix = fp_to_distance_matrix(fp[::stride])
    del fp

    
    # Set directories
    docking_dir = params_dict['docking_dir']
    if not os.path.exists(docking_dir):
        os.mkdir(docking_dir)
    receptor_dir = os.path.join(docking_dir, 'receptors')
    if not os.path.exists(receptor_dir):
        os.mkdir(receptor_dir)
    receptor_pdb_dir = os.path.join(receptor_dir, 'pdb')
    if not os.path.exists(receptor_pdb_dir):
        os.mkdir(receptor_pdb_dir)
    receptor_pdbqt_dir = os.path.join(receptor_dir, 'pdbqt')
    if not os.path.exists(receptor_pdbqt_dir):
        os.mkdir(receptor_pdbqt_dir)
    save_dir = params_dict['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    
    # Prepare ligands
    lig_dir = os.path.join(docking_dir, 'ligands')
    if not os.path.exists(lig_dir):
        os.mkdir(lig_dir)
    ligands = params_dict['ligands']
    response = input('ARE YOU SURE BOZO?')
    if response != 'y':
        raise Exception()
    else:
        ligands = {'CP55490':  'CCCCCCC(C)(C)C1=CC(=C(C=C1)[C@@H]2C[C@@H](CC[C@H]2CCCO)O)O',
           'HU308': 'CCCCCCC(C)(C)C1=CC(=C(C(=C1)OC)[C@@H]2C=C([C@@H]3C[C@H]2C3(C)C)CO)OC'}
    lig_names = []
    lig_pdbqts = []
    for lig, smiles in ligands.items():
        lig_pdbqts.append(prep_ligand(smiles, lig, lig_dir))
        lig_names.append(lig)


    # Iterate through n_clusters
    n_clusters = params_dict['n_clusters']
    response = input('ARE YOU SURE BOZO?')
    if response != 'y':
        raise Exception()
    else:
        n_clusters = [2, 3, 4]
    
    
    exp_avgs = np.empty((len(n_clusters), len(list(ligands.keys()))))
    for i, n in enumerate(n_clusters):

        print(f'Testing {n} clusters')

        # Clear out or build receptors dirs
        os.system(f'rm {receptor_pdb_dir}/*')
        os.system(f'rm {receptor_pdbqt_dir}/*')
        if not os.path.exists(receptor_pdbqt_dir):
            os.mkdir(receptor_pdbqt_dir)
        if not os.path.exists(receptor_pdb_dir):
            os.mkdir(receptor_pdb_dir)

        # Cluster occupany fingerprint
        assignments = cluster_matrix(dist_matrix, n)
        centroids = compute_centroids(dist_matrix, assignments)
        centroid_traj = get_centroids_traj(traj[::stride], centroids)
        save_centroid_frames(centroid_traj, centroids, receptor_pdb_dir)

        # Prepare receptors
        prep_receptor(receptor_pdb_dir, receptor_pdbqt_dir)

        # Dock
        lig_scores = np.zeros((n, len(list(ligands.keys()))))
        for j, receptor_pdbqt in enumerate(os.listdir(receptor_pdbqt_dir)):
            print(f'Testing receptor {j} with {len(lig_pdbqts)} ligands')
            lig_scores[j] = vina_dock(lig_dir, lig_pdbqts, os.path.join(receptor_pdbqt_dir, receptor_pdbqt), center, box_size=[25,25,25])


        for k in range(lig_scores.shape[1]):
            exp_avgs[i,k] = exp_avg(lig_scores[:,k])

        # Save temporary results
        np.save(os.path.join(save_dir, f'exp_avgs_{n}.npy'), exp_avgs[i])


    # Plot results
    plt.plot(n_clusters, exp_avgs, label=list(ligands.keys()), alpha=0.5)
    plt.plot(n_clusters, exp_avgs.mean(axis=1), label='average', alpha=1, color='black')
    plt.ylabel('Exp. Avg. Docking Affinities')
    plt.xlabel('n_clusters')
    plt.legend()
    plt.show()

    # Save results
    plt.savefig(os.path.join(save_dir, 'final.png'))
    np.save(os.path.join(save_dir, 'exp_avgs.npy'), exp_avgs)
    np.save(os.path.join(save_dir, 'fp_dist_matrix.npy'), dist_matrix)