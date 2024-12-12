"""
Utils to power run.py
"""
import os, sys, json
import mdtraj as md
import numpy as np



def combine_trajectories(traj_dir: str, resSeqs_to_include: np.array, binding_pocket_resSeqs: np.array, super_traj_dir: str=None):
    """
    Parameters:
    -----------
        traj_dir (str): 
            Specify the working directory where subdirectories name 'pdb' and 'dcd' can be found with the appropriate topologies/trajectories to include for occupany fingerprinting
            
        resSeqs_to_include (np.array): 
            Array of resSeqs to include in the super_trajectory

        binding_pocket_resSeqs (np.array):
            Array of binding pocket residues to use for alignment
            
        super_traj_dir (str): 
            String path to output directory where super_traj will be stored


    Returns:
    --------
        super_traj (md.Trajectory):
            Super trajectory with all the trajectories combined and topology corrected from the traj_dir. 

        frame_labels (np.array):
            Array of frame labels in format ['NAME', frame_indice] for each frame in the super_traj to retain frame identity. 
    """

    # Check if trajectory exists
    if os.path.exists(super_traj_dir):
        try:
            super_traj = md.load(os.path.join(super_traj_dir, 'super_traj.dcd'), top=os.path.join(super_traj_dir, 'super_traj.pdb'))
            frame_labels = np.load(os.path.join(super_traj_dir, 'frame_labels.npy'))
            assert super_traj.n_frames == len(frame_labels)

            
            print('Loaded super_traj and frame_labels from disk.')
            return super_traj, frame_labels
        except:
            pass
    

    # Combine the trajectories
    pdb_dir = os.path.join(traj_dir, 'pdb')
    dcd_dir = os.path.join(traj_dir, 'dcd')
    
    # Select a reference trajectory
    ref_name = os.listdir(pdb_dir)[0].split('.pdb')[0]
    ref = md.load(os.path.join(dcd_dir, ref_name + '.dcd'), top=os.path.join(pdb_dir, ref_name + '.pdb'))
    sele_str = " ".join([str(resSeq) for resSeq in resSeqs_to_include])
    ref_sele = ref.topology.select(f'protein and resSeq {sele_str}')
    
    # Iterate through all trajectories
    trajs = []
    frame_labels = []
    for name in sorted(os.listdir(pdb_dir)):
    
        # Load files
        dcd = os.path.join(dcd_dir, name.split('.')[0] + '.dcd')
        pdb = os.path.join(pdb_dir, name)
        
        # Load traj
        traj = md.load(dcd, top=pdb)
        
        # Load selection
        traj_sele = traj.topology.select(f'protein and resSeq {sele_str}')
        print(name, traj_sele.shape)
    
        # Map positions to reference
        traj = map_positions(ref, ref_sele, traj, traj_sele, topology_correction=True)
    
        # Save
        trajs.append(traj)
        for frame in range(traj.n_frames):
            frame_labels.append([name, frame])
    
    # Combine trajectories
    super_traj = md.join(trajs, check_topology=True)
    bp_sele = super_traj.topology.select(f'protein and backbone and resSeq {" ".join([str(r) for r in binding_pocket_resSeqs])}')
    super_traj = super_traj.superpose(super_traj, atom_indices=bp_sele, ref_atom_indices=bp_sele)

    # Save, if specified
    if super_traj_dir is not None:
        super_traj[0].save_pdb(os.path.join(super_traj_dir, 'super_traj.pdb'))
        super_traj.save_dcd(os.path.join(super_traj_dir, 'super_traj.dcd'))
        np.save(os.path.join(super_traj_dir, 'frame_labels.npy'), frame_labels)

    return super_traj, np.array(frame_labels)





def build_atom_map(ref, traj):

    # Get proteins
    ref_sele = ref.topology.select('protein')
    traj_sele = traj.topology.select('protein')    
    
    # Iterate through atoms to build map
    atom_map = np.empty((ref_sele.shape[0]), dtype=int) # Indice of traj atom in ref
    for i, (atom1, atom2) in enumerate(zip(ref.topology.atoms, traj.topology.atoms)):
        if i in ref_sele and i in traj_sele:
            
            if atom1.name == atom2.name and atom1.residue.resSeq == atom2.residue.resSeq:
                atom_map[i] = i

            else:
                found = False
                for j, atom1 in enumerate(ref.topology.atoms):
                    if j in ref_sele:
                        if atom1.name == atom2.name and atom1.residue.resSeq == atom2.residue.resSeq:
                            found = True
                            atom_map[i] = j
                            break

        else:
            print(i, atom1, atom2)

    return atom_map



def map_positions(ref, ref_sele, traj, traj_sele, topology_correction):

    # Check selections
    assert ref_sele.shape == traj_sele.shape, f"{ref_sele.shape} {traj_sele.shape}"
    
    # Slice simulations
    traj = traj.atom_slice(traj_sele)
    ref = ref.atom_slice(ref_sele)
    assert traj.xyz.shape[1] == ref.xyz.shape[1], f"{ref.xyz.shape} {traj.xyz.shape}"
    
    # Build atom map if specified
    if topology_correction:
        atom_map = build_atom_map(ref, traj)
        assert atom_map.shape == ref_sele.shape, f"{atom_map.shape} {ref_sele.shape}"
    
    # Map positions
    if topology_correction:
        traj = _map_positions(ref, traj, atom_map)
    else: traj = _map_positions(ref, traj)


    return traj



def _map_positions(ref, traj, atom_map=None):

    
    # Align
    traj = traj.superpose(ref)

    # Change positions of reference
    assert ref.n_frames == traj.n_frames
    if atom_map is not None:
        ref.xyz[:,atom_map] = traj.xyz.copy()
    else:
        ref.xyz = traj.xyz.copy()

    return ref


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)



