# AffiNNity: Drug-Protein Binding Affinity Prediction
This repository contains a pipeline for predicting drug-protein binding affinities utilizing Graph Neural Networks (GIN) implemented in PyTorch and DeepChem.

## Overview
Predicting the binding affinity between drug molecules and target proteins is critical in drug discovery. This repository demonstrates an approach that combines:

* **DeepChem** for dataset loading and preprocessing.
* **RDKit and PyTorch Geometric** for molecule graph processing.
* **Graph Isomorphism Networks (GIN)** for predictive modeling.

## Dataset
Uses the **PDBBind Refined** dataset provided by DeepChem, divided into train, validation, and test subsets:
* Train: 3881 samples
* Validation: 485 samples
* Test: 486 samples

The binding affinity labels (y) are normalized to zero-mean and unit variance using DeepChem transformers.

## Feature Extraction
### Drug Molecules
* **Node features**: Atomic number, degree, charge, hybridization, aromaticity, number of hydrogens, radical electrons, ring membership, chirality.
* **Edge features**: Bond type, ring membership.

### Protein Targets
* Amino acid sequences are extracted directly from PDB files.
* Basic numeric encoding is applied by mapping amino acids to numeric indices, padding sequences to a fixed length to ensure consistent input dimensions.

## Preprocessing
* Data is loaded using DeepChem's `molnet.load_pdbbind` function.
* Labels are standardized using DeepChem transformers to achieve zero-mean and unit variance.
* Molecule structures from ligand files (SDF format) are converted into molecular graphs using RDKit.
* Protein sequences extracted from PDB files are encoded numerically to facilitate embedding in neural network models.

## Model Architecture
### Graph Isomorphism Network (GIN)
* Drug molecule graph embedding using GIN convolution layers.
* Protein numeric embeddings processed through fully connected layers.
* Combined embeddings passed through a predictor to estimate binding affinity.

## Training
Training employs Mean Squared Error (MSE) loss optimized via Adam optimizer:
* Epochs: 200
* Learning rate: 0.001

## Evaluation Metrics
Model performance evaluated on the test set:
* **Mean Squared Error (MSE)**: 0.0927
* **Root Mean Squared Error (RMSE)**: 0.3044
* **R² Score**: 0.6624

## Visualization
Predictions vs. Actual binding affinities visualization provided to assess model accuracy and distribution.

## Usage
Run scripts directly in a Colab environment or local Jupyter Notebook to replicate results. Modify hyperparameters and feature extraction methods as needed for experimentation.
