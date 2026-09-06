// Core types for drug screening workflow

export interface DrugCandidate {
  smiles: string;
  qed: number;
  mw?: number;
  logP?: number;
  hbd?: number;
  hba?: number;
  rotatable_bonds?: number;
}

export interface ScreeningConfig {
  candidateCount: number;
  topN: number;
  minBindingAffinity: number;
}

export interface ScreenedDrug extends DrugCandidate {
  bindingAffinity: number;
  rank: number;
}

export interface ScreeningResult {
  topCandidates: ScreenedDrug[];
  totalScreened: number;
  passedThreshold: number;
  processingTimeMs: number;
  topRationale?: string | null;
}

export interface PDBData {
  content: string;
  filename: string;
}
