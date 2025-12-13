import os
import pickle
import pandas as pd
import numpy as np
import random
import warnings
from tqdm import tqdm
from scipy.spatial import distance_matrix
from Bio.PDB import PDBParser, PDBIO, Select

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to match your local machine or server
BASE_ROOT = "./data/HiQBind"  
RAW_DATA_ROOT = os.path.join(BASE_ROOT, "raw") 
OUTPUT_DIR = os.path.join(BASE_ROOT, "processed")

FILTER_OPTS = {
    "exclude_nmr": True,
    "allowed_units": ["nM", "uM", "pM", "M"],
    "allowed_types": ["ki", "kd"]
}

# ==========================================
# POCKET GENERATION UTILS
# ==========================================
class PocketSelect(Select):
    def __init__(self, valid_residues):
        self.valid_residues = valid_residues

    def accept_residue(self, residue):
        return residue in self.valid_residues

def generate_pocket_file(structure, ligand_coords, output_path, cutoff=10.0):
    prot_atoms = []
    prot_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    prot_atoms.append(atom)
                    prot_coords.append(atom.get_coord())

    if not prot_coords: return False

    prot_coords = np.array(prot_coords)
    dm = distance_matrix(ligand_coords, prot_coords)
    min_dists = np.min(dm, axis=0) 
    mask = min_dists <= cutoff

    if not np.any(mask): return False

    valid_residues = set()
    for i, is_close in enumerate(mask):
        if is_close:
            valid_residues.add(prot_atoms[i].get_parent())

    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(output_path, select=PocketSelect(valid_residues))
        return True
    except Exception as e:
        print(f"Error saving pocket: {e}")
        return False

# ==========================================
# MAIN PIPELINE
# ==========================================
def process_dataset():
    if not os.path.exists(RAW_DATA_ROOT):
        print(f"❌ ERROR: Raw data folder not found at {RAW_DATA_ROOT}")
        return

    # 1. Load CSVs
    print("Reading CSVs...")
    dfs = []
    for name in ["hiqbind_sm_metadata.csv", "hiqbind_poly_metadata.csv"]:
        path = os.path.join(BASE_ROOT, name)
        if os.path.exists(path):
            print(f"  Loaded {name}")
            dfs.append(pd.read_csv(path))

    if not dfs:
        print(f"❌ ERROR: No metadata CSVs found in {BASE_ROOT}")
        return

    df = pd.concat(dfs, ignore_index=True)
    pockets_dir = os.path.join(OUTPUT_DIR, "pockets")
    os.makedirs(pockets_dir, exist_ok=True)

    dataset = {"train": {}, "valid": {}, "test": {}}
    stats = {"processed": 0, "skipped": 0, "error": 0}

    # 2. Iterate
    from rdkit import Chem 
    # (Import inside here to avoid top-level dependency if user just wants to check paths)

    print(f"🚀 Processing {len(df)} samples...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # -- Filters --
        if FILTER_OPTS["exclude_nmr"] and str(row.get("Resolution")).upper() == "NMR":
            continue
        
        meas_type = str(row.get("Binding Affinity Measurement", "")).lower()
        if meas_type not in FILTER_OPTS["allowed_types"]:
            continue

        try:
            val = float(row["Binding Affinity Value"])
            unit = row["Binding Affinity Unit"]
            if unit == "nM": val *= 1e-9
            elif unit == "uM": val *= 1e-6
            elif unit == "pM": val *= 1e-12
            if val <= 0: continue
            pKd = -np.log10(val)
        except:
            continue

        # -- Paths --
        pdbid = str(row["PDBID"]).lower()
        cid = f"{pdbid}_{row['Ligand Name']}_{row['Ligand Chain']}_{row['Ligand Residue Number']}"
        
        # Path logic (Customize this based on your actual folder structure)
        lig_fname = f"{cid}_ligand_refined.sdf"
        prot_fname = f"{cid}_protein_refined.pdb"
        
        # Check potential locations (SM vs Poly folders)
        found = False
        for sub in ["raw_data_hiq_sm", "raw_data_hiq_poly", "."]:
             l_p = os.path.join(RAW_DATA_ROOT, sub, pdbid, cid, lig_fname)
             p_p = os.path.join(RAW_DATA_ROOT, sub, pdbid, cid, prot_fname)
             if os.path.exists(l_p) and os.path.exists(p_p):
                 lig_path, prot_path = l_p, p_p
                 found = True
                 break
        
        if not found:
            stats["skipped"] += 1
            continue

        # -- Processing --
        try:
            lig_m = Chem.SDMolSupplier(lig_path)[0]
            if not lig_m: continue
            l_c = lig_m.GetConformer().GetPositions()

            parser = PDBParser(QUIET=True)
            s = parser.get_structure("p", prot_path)
            
            pocket_path = os.path.join(pockets_dir, f"{cid}_pocket_10A.pdb")
            if not os.path.exists(pocket_path):
                if not generate_pocket_file(s, l_c, pocket_path):
                    stats["error"] += 1
                    continue

            # Split Assignment (Random 80/10/10)
            r = random.random()
            sp = "train" if r < 0.8 else ("valid" if r < 0.9 else "test")

            dataset[sp][cid] = {
                "ligand_path": lig_path,
                "protein_path": prot_path,
                "pocket_path": pocket_path,
                "y": pKd
            }
            stats["processed"] += 1

        except Exception:
            stats["error"] += 1
            continue

    # 3. Save
    out_file = os.path.join(OUTPUT_DIR, "hiqbind_dataset_clean.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(dataset, f)

    print(f"🎉 Done! Processed: {stats['processed']}. Saved to {out_file}")

if __name__ == "__main__":
    process_dataset()