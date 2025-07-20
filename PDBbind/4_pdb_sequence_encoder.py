from Bio import PDB
import torch
import esm

def extract_sequence_from_pdb(pdb_path):
    """Extracts the amino acid sequence from a PDB file using Biopython."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    ppb = PDB.PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += str(pp.get_sequence())

    return sequence

# define amino acid to index mapping
aa_to_idx = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
    'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18,
    'W': 19, 'Y': 20
}

def encode_sequence(seq, aa_to_idx, max_len=25):
    seq_idx = [aa_to_idx.get(aa, 0) for aa in seq]  # 0 for unknowns
    if len(seq_idx) < max_len:
        seq_idx += [0] * (max_len - len(seq_idx))
    else:
        seq_idx = seq_idx[:max_len]
    return seq_idx
