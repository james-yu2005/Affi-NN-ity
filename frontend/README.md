# Affi-NN-ity: Protein-Ligand Screening UI

A Next.js 15 + TypeScript interface for screening ZINC ligands against a protein structure. The app extracts FASTA sequences from uploaded PDB files, streams drug candidates from a local CSV, and calls a Hugging Face Space for pKd estimates with optional OpenAI rationale text.

## What it does
- Upload or paste a PDB; SEQRES lines are converted to FASTA and displayed before screening.
- Choose how many ZINC candidates to load (pre-sorted by QED in `kaggle_zinc_filtered_sorted.csv`, falling back to `kaggle_zinc_filtered.csv`) and request the top N above a pKd threshold.
- Run `/api/screen-binding` to send SMILES plus FASTA to the configured HF Space (`/predict` by default) with progress feedback and ranked results.
- Optionally generate a short binding rationale for the top ligand when `OPENAI_API_KEY` is provided.
- A single-sample route `/api/hf-predict` exists for direct SMILES+FASTA inference and is used by the `AffiNNityPredictor` component (not mounted on the home page by default).

## Project structure
- `src/app/page.tsx`: Landing experience with PDB upload/paste, SEQRES extraction, dataset summary, and the `DrugScreening` workflow.
- `src/components/DrugScreening.tsx`: Client-side controller for candidate counts, thresholds, and progress; calls `/api/drug-csv` and `/api/screen-binding`.
- `src/app/api/drug-csv/route.ts`: Streams drug candidates from the local CSV files, capped at 227k rows and sorted by QED.
- `src/app/api/screen-binding/route.ts`: Batches SMILES through the HF Space, enforces top-N and pKd thresholding, and optionally calls OpenAI for rationale.
- `src/app/api/hf-predict/route.ts`: Passthrough to the HF Space for a single ligand/protein pair.
- `src/components/MolecularViewer3D*`: Optional 3D PDB viewer components available for embedding if needed.

## Getting started
1) Install dependencies
```bash
npm install
```
2) Run the dev server
```bash
npm run dev
```
3) Open http://localhost:3000 and upload a PDB file. The app requires SEQRES lines to extract FASTA; otherwise it will ask for a file that includes sequence data.

## Configuration
Create a `.env.local` with any overrides:
```
HF_SPACE_URL=https://sharanyabasu-affinnity.hf.space   # default used if unset
HF_PREDICT_ENDPOINT=/predict                           # default used if unset
OPENAI_API_KEY=...                                     # optional, enables rationale text
OPENAI_MODEL=gpt-4o-mini                               # optional, defaults to this when KEY is set
```
- `screen-binding` adds 7 to the model's returned value to map the normalized output back to pKd-scale numbers.
- The app reads ZINC candidates from `kaggle_zinc_filtered_sorted.csv` at the repo root; if absent it falls back to `kaggle_zinc_filtered.csv`.

## Usage notes
- Candidate limits: presets for 100k and 200k, or any custom value between 1 and 227,000.
- Results show pKd, QED, and any available MW/logP from the CSV; ranking is recalculated after thresholding.
- Network calls hit the configured HF Space and (optionally) OpenAI; ensure those services are reachable from your environment.

## Development scripts
```bash
npm run dev     # start the Next.js dev server
npm run build   # production build
npm run start   # start the built app
npm run lint    # lint with eslint-config-next
```
