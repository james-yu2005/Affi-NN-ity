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

## Repository Structure

The codebase is organized into modular components for data handling, model definition, and training:

project_root/
│
├── train_pdbbind.py           # Main entry point for PDBBind experiments
├── train_hiqbind.py           # Main entry point for HiQBind experiments
├── requirements.txt           # Python dependencies
│
├── src/                       # SHARED CORE LOGIC
│   ├── models.py              # GIN architecture & Attention Fusion
│   ├── features.py            # ESM embedding & RDKit featurization
│   ├── engine.py              # Training loops & evaluation logic
│   └── utils.py               # Plotting helpers
│
├── data_loaders/              # DATASET HANDLING
│   ├── pdbbind/               # PDBBind loader (DeepChem integration)
│   └── hiqbind/               # HiQBind loader (Pickle-based loading)
│
└── scripts/                   # UTILITIES
    └── hiqbind_etl.py         # One-time ETL script for HiQBind pocket generation

---

## Usage Guide

### 1. Installation

First, ensure you have Python 3.8+ installed. Then install the dependencies:

    pip install -r requirements.txt

Key dependencies include: torch, torch-geometric, deepchem, rdkit, biopython, fair-esm, PyTDC, lifelines.

### 2. Data Preparation

#### PDBBind
PDBBind data is automatically handled via DeepChem. The `train_pdbbind.py` script will download the `core`, `refined`, and `general` sets automatically on first run. No manual setup is required.

#### HiQBind (ETL Pipeline)
For HiQBind, you must run the ETL script once to process raw PDB/SDF files into a clean pickle file.

1.  Place your raw data in `./data/HiQBind/raw` (or update `BASE_ROOT` in `scripts/hiqbind_etl.py`).
2.  Run the ETL script:

    python scripts/hiqbind_etl.py

    *This generates `hiqbind_dataset_clean.pkl` in your processed folder.*

### 3. Training

You can train the model on either dataset using the provided entry scripts.

**Train on PDBBind:**

    python train_pdbbind.py

*This loads PDBBind via DeepChem, filters for leakage against benchmarks (DAVIS/KIBA), trains the GIN model, and evaluates on the Core set.*

**Train on HiQBind:**

    # Ensure you ran the ETL script first!
    python train_hiqbind.py

*This loads the pre-processed pickle file, trains the GIN model, and evaluates on the test split.*

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