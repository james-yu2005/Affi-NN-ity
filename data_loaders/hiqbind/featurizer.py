import torch
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
from src.features import (
    get_node_features, get_edge_features, 
    get_adjacency_info, get_morgan_fingerprint, 
    generate_combined_embedding
)

def process_hiqbind_dataset(dataset_dict):
    molecule_data = []
    
    for cid, entry in tqdm(dataset_dict.items(), desc="Featurizing HiQBind"):
        ligand_path = entry['ligand_path']
        protein_path = entry['protein_path']
        pocket_path = entry['pocket_path']
        y_val = entry['y']

        ligand_mol = Chem.MolFromMolFile(str(ligand_path))
        if ligand_mol is None: continue

        try:
            data = Data(
                x=get_node_features(ligand_mol),
                edge_index=get_adjacency_info(ligand_mol),
                edge_attr=get_edge_features(ligand_mol),
                y=torch.tensor([y_val], dtype=torch.float)
            )
            # Pass BOTH protein and pocket path
            data.target_features = generate_combined_embedding(protein_path, pocket_path).unsqueeze(0)
            data.morgan_fp = get_morgan_fingerprint(ligand_mol).unsqueeze(0)
            data.complex_id = cid
            molecule_data.append(data)
        except Exception:
            continue

    return molecule_data