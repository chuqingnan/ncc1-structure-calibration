from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(".")
WORK = ROOT / "work" / "validated_endpoint_sensitivity_2026_08_25"
OUT = ROOT / "outputs" / "validated_endpoint_sensitivity_2026_08_25"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK / "vendor"))

import freesasa  # noqa: E402


freesasa.setVerbosity(freesasa.nowarnings)

PAIR_PATH = ROOT / "outputs" / "candidate_background_unified" / "donor_pair_metrics_within_all_distances.csv.gz"
TARGETS = [
    ("USP-A", "MtrunR108HiC_003946.1", 116, "SG", 149, "SG"),
    ("SAM synthase CXXC", "MtrunR108HiC_007551.1", 44, "SG", 47, "SG"),
    ("SAM synthase C/M", "MtrunR108HiC_007551.1", 47, "SG", 54, "SD"),
    ("SAM synthase C/M secondary", "MtrunR108HiC_007551.1", 47, "SG", 52, "SD"),
]
PROBES = [1.2, 1.4, 1.6, 2.0]


def atom_area(structure, result, residue_number: int, atom_name: str) -> float:
    matches = [
        i for i in range(structure.nAtoms())
        if structure.residueNumber(i).strip() == str(residue_number)
        and structure.atomName(i).strip() == atom_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one atom for {residue_number} {atom_name}, found {len(matches)}")
    return float(result.atomArea(matches[0]))


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    rows = []
    for label, protein_id, residue_i, atom_i, residue_j, atom_j in TARGETS:
        models = pairs[pairs["protein_id"].eq(protein_id)][["rank", "pdb_path"]].drop_duplicates()
        if len(models) != 3:
            raise ValueError(f"Expected three models for {protein_id}, found {len(models)}")
        for model in models.itertuples(index=False):
            with tempfile.TemporaryDirectory() as temp_dir:
                ascii_path = Path(temp_dir) / "model.pdb"
                os.symlink(model.pdb_path, ascii_path)
                structure = freesasa.Structure(str(ascii_path))
                for probe in PROBES:
                    parameters = freesasa.Parameters({"probe-radius": probe})
                    result = freesasa.calc(structure, parameters)
                    sasa_i = atom_area(structure, result, residue_i, atom_i)
                    sasa_j = atom_area(structure, result, residue_j, atom_j)
                    rows.append({
                        "case": label,
                        "protein_id": protein_id,
                        "rank": int(model.rank),
                        "residue_i": residue_i,
                        "atom_i": atom_i,
                        "residue_j": residue_j,
                        "atom_j": atom_j,
                        "probe_radius_A": probe,
                        "freesasa_i_A2": sasa_i,
                        "freesasa_j_A2": sasa_j,
                        "freesasa_mean_A2": (sasa_i + sasa_j) / 2,
                        "pdb_path": model.pdb_path,
                    })
    details = pd.DataFrame(rows)
    details.to_csv(OUT / "freesasa_key_site_model_details.csv", index=False)
    summary = (details.groupby(["case", "protein_id", "residue_i", "residue_j", "probe_radius_A"], as_index=False)
               .agg(models=("rank", "nunique"),
                    donor_i_sasa_median_A2=("freesasa_i_A2", "median"),
                    donor_j_sasa_median_A2=("freesasa_j_A2", "median"),
                    mean_pair_sasa_median_A2=("freesasa_mean_A2", "median"),
                    mean_pair_sasa_min_A2=("freesasa_mean_A2", "min"),
                    mean_pair_sasa_max_A2=("freesasa_mean_A2", "max")))
    summary.to_csv(OUT / "freesasa_key_site_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
