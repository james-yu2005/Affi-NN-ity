# Affi-NN-ity: Drug Protein Binding Affinity Prediction

Affi-NN-ity is an end-to-end system for predicting drug protein binding affinity using graph neural networks, protein language models, and multi-modal attention. The pipeline constructs ligand molecular graphs, extracts protein and pocket sequences from PDB files, embeds both using ESM-2, and trains a GIN-based model to regress pKd values.

---

## Features

- HiQBind preprocessing pipeline for ligands, proteins, and pockets  
- Automated pocket extraction using atom distance filtering  
- Protein and pocket embeddings using pretrained ESM-2  
- RDKit-based graph construction for ligand molecules  
- Graph Isomorphism Network (GIN) ligand encoder  
- Multi-modal fusion of ligand graphs, Morgan fingerprints, and protein embeddings  
- K-Fold cross-validation training  
- Full evaluation with RMSE, MAE, Pearson R, CI, and MSE  
- Reproducible data loaders and preprocessing utilities  

---

## Dataset Processing

### Pocket Extraction
The pipeline:
1. Loads ligand coordinates from SDF files  
2. Loads protein structures using BioPython  
3. Computes ligand-to-protein atom distances  
4. Selects residues within a 10 Å cutoff  
5. Outputs a refined PDB containing only pocket residues  

### Leakage Prevention
Train and validation samples are removed when their ligand-protein pairs overlap with:
- TDC DAVIS  
- TDC KIBA  
- The HiQBind test split  

This ensures no ligand-protein pairs leak across training and testing.

---

## Molecular Representation

### Ligand Representation
Using RDKit, the system extracts:

**Node features**
- atomic number  
- atom degree  
- formal charge  
- hybridization  
- aromaticity  
- total hydrogens  
- radical electrons  
- ring membership  
- chirality  

**Edge features**
- bond type  
- ring membership  

**Additional features**
- Morgan fingerprint (1024 bits)  
- Adjacency list for PyTorch Geometric  

### Protein Representation
For each complex:
- Full protein sequence is extracted  
- Pocket sequence is extracted  
- Both are embedded with ESM-2  
- Resulting embeddings are concatenated into a 640-dimensional feature vector  

---

## Graph Construction

Each entry is converted to a PyTorch Geometric `Data` object with:

- `x`: node features  
- `edge_index`: adjacency matrix  
- `edge_attr`: edge features  
- `morgan_fp`: fingerprint vector  
- `target_features`: protein+pocket embedding  
- `y`: binding affinity (pKd)  

These lists of graph objects are wrapped into standard PyG DataLoaders.

---

## Model Architecture

### Ligand Encoder
- MLP node embedder  
- Two GINConv layers with batch normalization  
- Residual connections  
- Combined mean and sum pooling  

### Protein Encoder
- Feed-forward network projecting the 640-dimensional embedding  

### Fingerprint Encoder
- MLP applied to the 1024-bit fingerprint  

### Multi-modal Fusion
- Concatenates ligand, protein, and fingerprint embeddings  
- Learns attention weights over modalities  
- Final regression MLP outputs a pKd prediction  

---

## Training

- Optimizer: AdamW  
- Loss: Mean Squared Error  
- ReduceLROnPlateau scheduler  
- Gradient clipping  
- Early stopping  
- Five-fold cross-validation  
- Training and validation metrics logged every epoch  

---

## Evaluation

The evaluation script reports:

- MSE  
- RMSE  
- MAE  
- Pearson correlation  
- Standard deviation of residuals  
- Concordance index  
