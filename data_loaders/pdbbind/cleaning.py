import os
from rdkit import Chem
from tqdm import tqdm
from tdc.multi_pred import DTI
from src.features import extract_sequence_from_pdb

def get_canonical_smiles(mol_or_path):
    try:
        if isinstance(mol_or_path, str):
            if mol_or_path.endswith('.pdb'):
                mol = Chem.MolFromPDBFile(mol_or_path, sanitize=False)
            else:
                mol = Chem.MolFromMolFile(mol_or_path, sanitize=False)
        else:
            mol = mol_or_path

        if mol:
            try: Chem.SanitizeMol(mol)
            except: pass
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

def get_forbidden_pairs(core_dataset):
    forbidden_pairs = set()
    print(f"\nBuilding Forbidden Pairs (Leakage Prevention)...")

    # 1. TDC Benchmarks
    for name in ['DAVIS', 'KIBA']:
        try:
            data = DTI(name=name)
            df = data.get_data()
            for _, row in df.iterrows():
                try:
                    mol = Chem.MolFromSmiles(row['Drug'])
                    if mol:
                        can_smi = Chem.MolToSmiles(mol, canonical=True)
                        forbidden_pairs.add((can_smi, row['Target']))
                except: continue
        except Exception: pass

    # 2. PDBBind Core
    print("Extracting pairs from PDBBind Core Set...")
    iterator = core_dataset.iterbatches(batch_size=1, deterministic=True)
    for X, _, _, _ in tqdm(iterator, total=len(core_dataset)):
        ligand_path = X[0][0]
        protein_path = X[0][1]
        
        can_smi = get_canonical_smiles(str(ligand_path))
        
        prot_path_str = str(protein_path).replace("pocket", "protein")
        if not os.path.exists(prot_path_str): prot_path_str = str(protein_path)
        
        seq = extract_sequence_from_pdb(prot_path_str)

        if can_smi and seq:
            forbidden_pairs.add((can_smi, seq))

    return forbidden_pairs

def remove_invalid_molecules(dc_dataset, dataset_name, forbidden_pairs=None):
    valid_entries = []
    removed_invalid = 0
    removed_leakage = 0

    print(f"Cleaning {dataset_name}...")
    iterator = dc_dataset.iterbatches(batch_size=1, deterministic=True)
    
    for X, y, w, ids in tqdm(iterator, total=len(dc_dataset)):
        ligand_path = X[0][0]
        protein_path = X[0][1]

        if not os.path.exists(str(ligand_path)):
            removed_invalid += 1
            continue

        try:
            if str(ligand_path).endswith('.pdb'):
                ligand_mol = Chem.MolFromPDBFile(str(ligand_path), sanitize=False)
            else:
                ligand_mol = Chem.MolFromMolFile(str(ligand_path), sanitize=False)

            if ligand_mol is None:
                removed_invalid += 1
                continue

            if forbidden_pairs:
                try:
                    current_smi = Chem.MolToSmiles(ligand_mol, canonical=True)
                    prot_path_str = str(protein_path).replace("pocket", "protein")
                    if not os.path.exists(prot_path_str): prot_path_str = str(protein_path)
                    
                    current_seq = extract_sequence_from_pdb(prot_path_str)
                    if (current_smi, current_seq) in forbidden_pairs:
                        removed_leakage += 1
                        continue 
                except:
                    removed_invalid += 1
                    continue

            valid_entries.append((X, y, w, ids))

        except Exception:
            removed_invalid += 1
            continue

    print(f"Summary {dataset_name}: Removed {removed_invalid} invalid, {removed_leakage} leakage.")
    return valid_entries