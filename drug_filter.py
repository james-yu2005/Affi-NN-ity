import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# RDKit Imports
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    SaltRemover,
    AllChem,
    FilterCatalog
)
from rdkit.Chem.MolStandardize import rdMolStandardize

# ============================================================
# CONFIGURATION
# ============================================================

class ScreeningConfig:
    MAX_MW = 600.0
    MIN_MW = 150.0
    MAX_LOGP = 5.0
    MIN_LOGP = -0.4
    MAX_HBD = 5
    MAX_HBA = 10
    MAX_ROTATABLE = 10
    MIN_QED = 0.3


# ============================================================
# MOLECULE FILTERING + STANDARDIZATION
# ============================================================

class MoleculeFilter:
    def __init__(self):
        self.salt_remover = SaltRemover.SaltRemover()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        self.pains_filter = FilterCatalog.FilterCatalog(params)

    def get_largest_fragment(self, mol: Chem.Mol) -> Chem.Mol:
        cleaned = self.salt_remover.StripMol(mol)

        if cleaned.GetNumAtoms() > 0:
            target_mol = cleaned
        else:
            target_mol = mol

        frags = Chem.GetMolFrags(target_mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return target_mol

        return max(frags, key=lambda m: m.GetNumHeavyAtoms())

    def canonicalize_tautomer(self, mol: Chem.Mol) -> Chem.Mol:
        return self.tautomer_enumerator.Canonicalize(mol)

    def check_pains(self, mol: Chem.Mol) -> List[str]:
        matches = []
        if self.pains_filter.HasMatch(mol):
            for entry in self.pains_filter.GetMatches(mol):
                matches.append(entry.GetDescription())
        return matches

    def evaluate(self, smiles_input: str) -> Dict[str, Any]:
        result = {
            "original_smiles": smiles_input,
            "cleaned_smiles": None,
            "is_viable": False,
            "reasons_for_failure": [],
            "properties": {},
        }

        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            result["reasons_for_failure"].append("Invalid SMILES string")
            return result

        try:
            core_mol = self.get_largest_fragment(mol)

            pains_hits = self.check_pains(core_mol)
            if pains_hits:
                result["reasons_for_failure"].append(
                    f"PAINS Alert: {', '.join(pains_hits)}"
                )
                return result

            final_mol = self.canonicalize_tautomer(core_mol)
            result["cleaned_smiles"] = Chem.MolToSmiles(final_mol, canonical=True)

            props = {
                "mw": Descriptors.MolWt(final_mol),
                "logp": Descriptors.MolLogP(final_mol),
                "hbd": Descriptors.NumHDonors(final_mol),
                "hba": Descriptors.NumHAcceptors(final_mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(final_mol),
                "qed": Descriptors.qed(final_mol),
            }
            result["properties"] = props

            if not (ScreeningConfig.MIN_MW <= props["mw"] <= ScreeningConfig.MAX_MW):
                result["reasons_for_failure"].append(f"MW {props['mw']:.1f} out of range")

            if not (ScreeningConfig.MIN_LOGP <= props["logp"] <= ScreeningConfig.MAX_LOGP):
                result["reasons_for_failure"].append(f"LogP {props['logp']:.1f} out of range")

            if props["hbd"] > ScreeningConfig.MAX_HBD:
                result["reasons_for_failure"].append(f"Too many H-bond donors ({props['hbd']})")

            if props["rotatable_bonds"] > ScreeningConfig.MAX_ROTATABLE:
                result["reasons_for_failure"].append("Too flexible (Rotatable bonds > 10)")

            if props["qed"] < ScreeningConfig.MIN_QED:
                result["reasons_for_failure"].append("Low QED score")

            if not result["reasons_for_failure"]:
                result["is_viable"] = True

        except Exception as e:
            result["reasons_for_failure"].append(f"Processing error: {str(e)}")

        return result


# ============================================================
# CONFORMER GENERATION (ETKDG + MMFF)
# ============================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def embed_conformers(mol, n_confs):
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.pruneRmsThresh = 0.1
    params.enforceChirality = True

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    return list(conf_ids)

def minimize_conformers(mol, conf_ids):
    try:
        AllChem.MMFFGetMoleculeProperties(mol)
        minimized = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    except:
        return None

    out = []
    for cid, (status, energy) in zip(conf_ids, minimized):
        out.append((cid, float(energy)))
    return out

def pick_best_conformer(minimized):
    return min(minimized, key=lambda x: x[1])[0]

def write_sdf(mol, conf_id, out_path):
    w = Chem.SDWriter(out_path)
    w.write(mol, confId=conf_id)
    w.close()

def generate_conformer(smiles, name, out_dir="sdf_out", n_confs=3):
    ensure_dir(out_dir)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    mol = Chem.AddHs(mol)

    conf_ids = embed_conformers(mol, n_confs)
    if len(conf_ids) == 0:
        print(f"ETKDG embedding failed: {name}")
        return None

    minimized = minimize_conformers(mol, conf_ids)
    if minimized is None:
        print(f"MMFF minimization failed: {name}")
        return None

    best_conf = pick_best_conformer(minimized)

    out_path = os.path.join(out_dir, f"{name}.sdf")
    write_sdf(mol, best_conf, out_path)

    print(f"Saved: {out_path}  best_conformer={best_conf}")
    return out_path


# ============================================================
# FASTAPI SERVICE
# ============================================================

app = FastAPI(title="Virtual Screening + Conformer Generation API")
filter_engine = MoleculeFilter()

class FilterRequest(BaseModel):
    smiles: str
    id: Optional[str] = "unknown"

class FilterResponse(BaseModel):
    id: str
    is_viable: bool
    cleaned_smiles: Optional[str]
    properties: Optional[Dict[str, float]]
    errors: List[str]

class ConformerRequest(BaseModel):
    smiles: str
    name: str
    n_confs: Optional[int] = 3


@app.get("/")
def health_check():
    return {"status": "Virtual Screening Server Active"}


@app.post("/filter", response_model=FilterResponse)
async def filter_molecule(payload: FilterRequest):
    analysis = filter_engine.evaluate(payload.smiles)
    return FilterResponse(
        id=payload.id,
        is_viable=analysis["is_viable"],
        cleaned_smiles=analysis["cleaned_smiles"],
        properties=analysis["properties"],
        errors=analysis["reasons_for_failure"],
    )


@app.post("/conformer")
async def api_generate_conformer(req: ConformerRequest):
    path = generate_conformer(req.smiles, req.name, n_confs=req.n_confs)
    if path is None:
        raise HTTPException(status_code=400, detail="Conformer generation failed")
    return {"sdf_path": path, "name": req.name}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Starting Virtual Screening Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
