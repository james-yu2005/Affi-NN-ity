import torch
import esm
import os
from rdkit import Chem, AllChem
from Bio.PDB import PDBParser, PPBuilder

# --- ESM Model Singleton (Loads once) ---
_ESM_MODEL = None
_ESM_BATCH_CONVERTER = None

def load_esm_model():
    global _ESM_MODEL, _ESM_BATCH_CONVERTER
    if _ESM_MODEL is None:
        print("Loading ESM-2 Model...")
        _ESM_MODEL, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        _ESM_BATCH_CONVERTER = alphabet.get_batch_converter()
        _ESM_MODEL.eval()
    return _ESM_MODEL, _ESM_BATCH_CONVERTER

# --- Protein Features ---
def extract_sequence_from_pdb(pdb_path):
    if not pdb_path or not os.path.exists(pdb_path):
        return ""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("struct", pdb_path)
        ppb = PPBuilder()
        sequence = ""
        for pp in ppb.build_peptides(structure):
            sequence += str(pp.get_sequence())
        return sequence
    except Exception:
        return ""

def embed_sequence(seq):
    model, batch_converter = load_esm_model()
    if not seq: return torch.zeros(320)
    
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", seq)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
    # Mean pooling
    return results["representations"][6][0, 1:len(seq)+1].mean(0)

def generate_combined_embedding(full_path, pocket_path=None):
    """
    Handles both PDBBind (inferred pocket path) and HiQBind (explicit pocket path).
    """
    if pocket_path is None:
        # PDBBind Logic: infer pocket path if not provided
        if "pocket" in full_path:
            pocket_path = full_path
            full_path = full_path.replace("pocket", "protein")
        else:
            pocket_path = full_path # Fallback

    full_seq = extract_sequence_from_pdb(full_path)
    pocket_seq = extract_sequence_from_pdb(pocket_path)
    
    full_emb = embed_sequence(full_seq)
    pocket_emb = embed_sequence(pocket_seq)
    
    return torch.cat([full_emb, pocket_emb], dim=0) # 640 dim

# --- Molecule Features ---
def get_node_features(mol):
    all_node_feats = []
    for atom in mol.GetAtoms():
        node_feats = [
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), atom.GetIsAromatic(), atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(), atom.IsInRing(), int(atom.GetChiralTag())
        ]
        all_node_feats.append(node_feats)
    return torch.tensor(all_node_feats, dtype=torch.float)

def get_edge_features(mol):
    all_edge_feats = []
    for bond in mol.GetBonds():
        edge_feats = [bond.GetBondTypeAsDouble(), bond.IsInRing()]
        all_edge_feats += [edge_feats, edge_feats]
    return torch.tensor(all_edge_feats, dtype=torch.float)

def get_adjacency_info(mol):
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
    return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

def get_morgan_fingerprint(mol, radius=2, n_bits=1024):
    if mol is None: return torch.zeros(n_bits, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(list(fp), dtype=torch.float32)