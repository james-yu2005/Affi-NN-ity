import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# RDKit Imports
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import FilterCatalog

# --- Configuration ---

class ScreeningConfig:
    """Configuration thresholds for the virtual screening."""
    MAX_MW = 600.0          # Maximum Molecular Weight (Daltons)
    MIN_MW = 150.0          # Minimum Molecular Weight
    MAX_LOGP = 5.0          # Maximum LogP (Lipophilicity)
    MIN_LOGP = -0.4         # Minimum LogP
    MAX_HBD = 5             # Max Hydrogen Bond Donors (Lipinski)
    MAX_HBA = 10            # Max Hydrogen Bond Acceptors (Lipinski)
    MAX_ROTATABLE = 10      # Max Rotatable Bonds
    MIN_QED = 0.3           # Quantitative Estimation of Drug-likeness (0-1)

# --- Core Logic Class ---

class MoleculeFilter:
    def __init__(self):
        # Initialize Salt Remover
        self.salt_remover = SaltRemover.SaltRemover()
        
        # Initialize Tautomer Enumerator
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

        # Initialize PAINS filters
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        self.pains_filter = FilterCatalog.FilterCatalog(params)

    def get_largest_fragment(self, mol: Chem.Mol) -> Chem.Mol:
        """Removes salts and returns the largest organic fragment."""
        # 1. Try stripping salts defined in the catalog
        cleaned = self.salt_remover.StripMol(mol)
        
        # 2. Check if we have anything left
        # If StripMol removed everything (e.g. input was [Na+].[Acetate-]), 
        # cleaned.GetNumAtoms() will be 0. In this case, we revert to the 
        # original molecule (target_mol = mol) and rely on the fragment 
        # selector below to pick the largest piece (Acetate) over the small salt (Na).
        if cleaned.GetNumAtoms() > 0:
            target_mol = cleaned
        else:
            target_mol = mol

        # 3. Get Fragments
        frags = Chem.GetMolFrags(target_mol, asMols=True, sanitizeFrags=True)
        
        if not frags:
             return target_mol

        # 4. Return largest fragment based on Heavy Atom Count
        # This ensures that even if SaltRemover failed or stripped everything,
        # we still pick "Acetate" over "Sodium".
        return max(frags, key=lambda m: m.GetNumHeavyAtoms())

    def canonicalize_tautomer(self, mol: Chem.Mol) -> Chem.Mol:
        """Standardizes the tautomeric state of the molecule."""
        return self.tautomer_enumerator.Canonicalize(mol)

    def check_pains(self, mol: Chem.Mol) -> List[str]:
        """Returns a list of PAINS matches found."""
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
            "warnings": []
        }

        # 1. Parsing
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            result["reasons_for_failure"].append("Invalid SMILES string")
            return result

        try:
            # 2. Salt Removal & Fragment Selection
            core_mol = self.get_largest_fragment(mol)
            
            # 3. PAINS Filtering
            pains_matches = self.check_pains(core_mol)
            if pains_matches:
                result["reasons_for_failure"].append(f"PAINS Alert: {', '.join(pains_matches)}")
                return result

            # 4. Tautomer Canonicalization
            final_mol = self.canonicalize_tautomer(core_mol)
            result["cleaned_smiles"] = Chem.MolToSmiles(final_mol, canonical=True)

            # 5. Property Calculation
            props = {
                "mw": Descriptors.MolWt(final_mol),
                "logp": Descriptors.MolLogP(final_mol),
                "hbd": Descriptors.NumHDonors(final_mol),
                "hba": Descriptors.NumHAcceptors(final_mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(final_mol),
                "qed": Descriptors.qed(final_mol)
            }
            result["properties"] = props

            # 6. Rule-Based Filtering
            if not (ScreeningConfig.MIN_MW <= props["mw"] <= ScreeningConfig.MAX_MW):
                result["reasons_for_failure"].append(f"MW {props['mw']:.1f} out of range")

            if not (ScreeningConfig.MIN_LOGP <= props["logp"] <= ScreeningConfig.MAX_LOGP):
                result["reasons_for_failure"].append(f"LogP {props['logp']:.1f} out of range")
            
            if props["hbd"] > ScreeningConfig.MAX_HBD:
                result["reasons_for_failure"].append(f"Too many H-bond donors ({props['hbd']})")

            if props["rotatable_bonds"] > ScreeningConfig.MAX_ROTATABLE:
                result["reasons_for_failure"].append("Too flexible (Rotatable bonds > 10)")

            if props["qed"] < ScreeningConfig.MIN_QED:
                result["reasons_for_failure"].append("Low QED score (low drug-likeness)")

            # 7. Final Verdict
            if not result["reasons_for_failure"]:
                result["is_viable"] = True

        except Exception as e:
            result["reasons_for_failure"].append(f"Processing error: {str(e)}")

        return result

# --- API Service (FastAPI) ---

app = FastAPI(title="Virtual Screening Filter API")
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

@app.post("/filter", response_model=FilterResponse)
async def filter_molecule(payload: FilterRequest):
    analysis = filter_engine.evaluate(payload.smiles)
    return FilterResponse(
        id=payload.id,
        is_viable=analysis["is_viable"],
        cleaned_smiles=analysis["cleaned_smiles"],
        properties=analysis["properties"],
        errors=analysis["reasons_for_failure"]
    )

@app.get("/")
def health_check():
    return {"status": "Virtual Screening Pipeline Active"}

if __name__ == "__main__":
    print("Starting Virtual Screening Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
