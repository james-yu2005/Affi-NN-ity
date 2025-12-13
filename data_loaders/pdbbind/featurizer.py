import torch
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
from src.features import (
    get_node_features, get_edge_features, 
    get_adjacency_info, get_morgan_fingerprint, 
    generate_combined_embedding
)

def process_deepchem_dataset(data_list):
    molecule_data = []
    print(f"Featurizing {len(data_list)} samples...")
    
    for X, y, w, ids in tqdm(data_list):
        ligand_path = X[0][0]
        protein_path = X[0][1]

        try:
            if str(ligand_path).endswith('.pdb'):
                ligand_mol = Chem.MolFromPDBFile(str(ligand_path), sanitize=False)
            else:
                ligand_mol = Chem.MolFromMolFile(str(ligand_path), sanitize=False)
            
            if ligand_mol is None: continue
            try: Chem.SanitizeMol(ligand_mol)
            except: pass

            data = Data(
                x=get_node_features(ligand_mol),
                edge_index=get_adjacency_info(ligand_mol),
                edge_attr=get_edge_features(ligand_mol),
                y=torch.tensor(y, dtype=torch.float)
            )
            
            data.target_features = generate_combined_embedding(str(protein_path)).unsqueeze(0)
            data.morgan_fp = get_morgan_fingerprint(ligand_mol).unsqueeze(0)
            data.pdb_id = str(ids)

            molecule_data.append(data)
        except Exception:
            continue

    return molecule_data