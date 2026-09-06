# Connect front-end to a drug candidate nomination database


## Drug Nomination & Library Integration

- Added a live ChEMBL connector (`src/lib/drugLibrary.ts`) plus a `/api/drug-library` route so the frontend can pull SMILES-ready compounds with physicochemical metadata.

## 2024-XX-XX: Binding summary + GPT prompt template
- New postprocessing API at `src/app/api/postprocess/route.ts` that:
  - Accepts `{ sdf, bindingAffinity, proteinName, proteinAccession?, candidateName? }`
  - (Mock) resolves a drug name via NCBI Lit/Chem structure search endpoint and builds a GPT prompt template + `messages` array.
  - Returns `BindingPostProcessResult` with the prompt, messages, and provenance of the NCBI lookup (`mock` until live network is enabled).
- `src/types/prediction.ts` now includes `proteinName`/`proteinAccession` on `AnalysisJobRequest` and adds `BindingPostProcessRequest/Result` types for LLM-aware postprocessing.
- UI changes:
  - `ProteinInput` accepts `onProteinContextChange` to bubble protein name/accession upward (search import + manual name field).
  - Main page holds SDF textarea + `BindingReport` component to trigger the postprocess API and show the GPT prompt/metadata.
- Added PDB paste + 2D projection viewer (`PdbPreview`) on the main page to visualize ATOM/HETATM coordinates when a PDB is provided.
- Postprocess route now attempts live NCBI structure search when `NCBI_API_KEY` is set; otherwise falls back to mock naming.
- Postprocess route now also calls OpenAI Chat Completions when `OPENAI_API_KEY` is set (model defaults to `gpt-4o-mini`). Returns `llmOutput/llmModel/llmSource` and surfaces errors when the call is skipped or fails.
- Usage: after running analysis and pasting an SDF, click “Generate Summary” to produce the prompt that combines predicted pKd, protein identity, and the NCBI-resolved ligand name (mocked). Replace the stubbed fetch in `postprocess` route with a live NCBI call and wire the GPT client when networked execution is available.
- Introduced the `DrugLibraryBrowser` component to search ChEMBL, view ADMET cues, and queue up to 12 compounds for inference.
- Extended the analysis request/response types so queued compounds ride along with `/api/analyze` and show up ahead of the ChemGAN mock outputs, making it obvious when user-selected molecules are being evaluated.
- Documented the workflow additions in the main `README.md` (“Latest Updates” + new “Drug Nomination Library” usage section) for quicker onboarding.

> TODO: Expand the backend aggregator with DrugCentral/ZINC sources plus filters for target class, clinical phase, and Lipinski/Veber rules.
