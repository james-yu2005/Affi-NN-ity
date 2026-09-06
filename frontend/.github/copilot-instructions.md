# Affi-NN-ity Project - Copilot Instructions

## Project Overview

This is **Affi-NN-ity**, a protein-drug binding affinity prediction visualization interface built with Next.js. The project combines Graph Neural Networks with ESM-2 protein language models for accurate binding affinity prediction.

## Architecture

- **Frontend**: Next.js 15 with TypeScript and Tailwind CSS
- **ML Model**: Dual-stream GNN + ESM-2 transformer architecture
- **Dataset**: PDBBind v2019 (4852 refined + 12800 general complexes)
- **Performance**: RMSE = 0.69, R² = 0.50
- **Generation**: ChemGAN for novel molecule synthesis

## Key Components

- `ProteinInput.tsx` - ESM-2 sequence processing interface
- `DrugCandidates.tsx` - ChemGAN generated compounds display
- `AffinityVisualization.tsx` - pKd prediction charts and ADMET analysis
- `MolecularViewer.tsx` - 3D protein structure visualization
- `AnalysisPipeline.tsx` - Real-time ML pipeline progress

## Development Guidelines

- Maintain biological accuracy in terminology and data structures
- Use scientific color palettes (DNA: blues, proteins: purples, binding: greens)
- Follow PDBBind dataset conventions for molecular properties
- Ensure responsive design for laboratory environments
- Keep SMILES notation and ADMET properties visible

## Sample Data

- COVID-19 Main Protease (Mpro), ACE2 Receptor, EGFR Kinase Domain
- Realistic pKd values, molecular weights, LogP calculations
- Binding pocket sequences for enhanced prediction accuracy
