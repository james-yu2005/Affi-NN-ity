# Affi-NN-ity: Drug-Protein Binding Affinity Prediction

This repository provides a complete pipeline to predict drug-protein binding affinities using Graph Neural Networks (GIN) implemented with PyTorch and DeepChem.

## Overview

WATai.ca: https://watai.ca/projects

Predicting drug-protein binding affinity is vital for drug discovery. This project integrates:

* **DeepChem** for dataset handling.
* **RDKit and PyTorch Geometric** for molecular graph processing.
* **Graph Isomorphism Networks (GIN)** for predictive modeling.

## Dataset

The **PDBBind Refined** dataset from DeepChem is used, split into:

* Train: 3881 samples
* Validation: 485 samples
* Test: 486 samples

Binding affinities are normalized (zero-mean, unit-variance) via DeepChem transformers.

## Feature Extraction

### Drug Molecules

* **Node features**: Atomic number, degree, charge, hybridization, aromaticity, hydrogens, radical electrons, ring membership, chirality.
* **Edge features**: Bond type, ring membership.

### Protein Targets

* Amino acid sequences extracted from PDB files.
* Numerical encoding of sequences with padding for uniform length.

## Preprocessing

* Load and standardize data using DeepChem.
* Convert ligand molecule files (SDF) into graphs using RDKit.
* Protein sequences numerically encoded from PDB files.

## Model Architecture

### Graph Isomorphism Network (GIN)

* Embedding drug molecules using GIN convolutions.
* Protein sequences processed via dense layers.
* Embeddings combined to predict binding affinity.

## Training

* Optimizer: Adam
* Loss: Mean Squared Error (MSE)
* Epochs: 200 (early stopping with patience of 25)
* Learning rate: 0.001

## Evaluation Metrics

Performance on the test set:

* **MSE**: 0.7641
* **RMSE**: 0.8742
* **R² Score**: 0.2456
