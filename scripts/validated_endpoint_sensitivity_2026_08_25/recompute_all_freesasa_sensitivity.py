from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
WORK = ROOT / "work" / "validated_endpoint_sensitivity_2026_08_25"
OUT = ROOT / "outputs" / "validated_endpoint_sensitivity_2026_08_25"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK / "vendor"))
sys.path.insert(0, str(WORK))

import freesasa  # noqa: E402
from run_sensitivity_analysis import (  # noqa: E402
    SPECS,
    exact_binomial_two_sided,
    fisher_exact_two_sided,
    logistic_group_effect,
    positive_ids,
    ratio_intervals,
)


freesasa.setVerbosity(freesasa.nowarnings)

PAIR_PATH = ROOT / "outputs" / "candidate_background_unified" / "donor_pair_metrics_within_all_distances.csv.gz"
MASTER_PATH = ROOT / "outputs" / "candidate_background_unified" / "protein_level_master.csv"
MATCH_PATH = ROOT / "outputs" / "candidate_background_unified" / "matched_pairs_primary.csv"
FAMILY_PATH = ROOT / "outputs" / "matched_balance_family_sensitivity" / "family_assignments_all_thresholds.csv"


def sasa_lookup(pdb_path: str) -> dict[tuple[int, str], float]:
    with tempfile.TemporaryDirectory() as temp_dir:
        ascii_path = Path(temp_dir) / "model.pdb"
        os.symlink(pdb_path, ascii_path)
        structure = freesasa.Structure(str(ascii_path))
        result = freesasa.calc(structure, freesasa.Parameters({"probe-radius": 1.4}))
        lookup = {}
        for atom_index in range(structure.nAtoms()):
            key = (int(structure.residueNumber(atom_index).strip()), structure.atomName(atom_index).strip())
            lookup[key] = float(result.atomArea(atom_index))
        return lookup


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    model_paths = pairs[["group", "protein_id", "rank", "pdb_path"]].drop_duplicates().sort_values(["group", "protein_id", "rank"])
    area_rows = []
    total = len(model_paths)
    for count, model in enumerate(model_paths.itertuples(index=False), start=1):
        lookup = sasa_lookup(model.pdb_path)
        donor_rows = pairs[
            pairs["group"].eq(model.group)
            & pairs["protein_id"].eq(model.protein_id)
            & pairs["rank"].eq(model.rank)
        ]
        keys = set(zip(donor_rows["residue_i"], donor_rows["atom_i"])) | set(zip(donor_rows["residue_j"], donor_rows["atom_j"]))
        for residue, atom in keys:
            area_rows.append({
                "group": model.group,
                "protein_id": model.protein_id,
                "rank": int(model.rank),
                "residue": int(residue),
                "atom": atom,
                "freesasa_A2": lookup[(int(residue), atom)],
            })
        if count % 50 == 0 or count == total:
            print(f"FreeSASA models completed: {count}/{total}", flush=True)

    atom_areas = pd.DataFrame(area_rows)
    atom_areas.to_csv(OUT / "freesasa_all_model_donor_atom_areas.csv.gz", index=False, compression="gzip")
    left = atom_areas.rename(columns={"residue": "residue_i", "atom": "atom_i", "freesasa_A2": "freesasa_i_A2"})
    right = atom_areas.rename(columns={"residue": "residue_j", "atom": "atom_j", "freesasa_A2": "freesasa_j_A2"})
    keys_left = ["group", "protein_id", "rank", "residue_i", "atom_i"]
    keys_right = ["group", "protein_id", "rank", "residue_j", "atom_j"]
    fs_pairs = pairs.merge(left, on=keys_left, how="left", validate="many_to_one")
    fs_pairs = fs_pairs.merge(right, on=keys_right, how="left", validate="many_to_one")
    if fs_pairs[["freesasa_i_A2", "freesasa_j_A2"]].isna().any().any():
        raise ValueError("FreeSASA donor-area mapping is incomplete")
    fs_pairs["mean_donor_sasa_A2_approx"] = fs_pairs["mean_donor_sasa_A2"]
    fs_pairs["donor_sasa_i_A2_approx"] = fs_pairs["donor_sasa_i_A2"]
    fs_pairs["donor_sasa_j_A2_approx"] = fs_pairs["donor_sasa_j_A2"]
    fs_pairs["donor_sasa_i_A2"] = fs_pairs["freesasa_i_A2"]
    fs_pairs["donor_sasa_j_A2"] = fs_pairs["freesasa_j_A2"]
    fs_pairs["mean_donor_sasa_A2"] = (fs_pairs["freesasa_i_A2"] + fs_pairs["freesasa_j_A2"]) / 2
    fs_pairs.to_csv(OUT / "donor_pair_metrics_with_freesasa.csv.gz", index=False, compression="gzip")

    master = pd.read_csv(MASTER_PATH, low_memory=False)
    matched = pd.read_csv(MATCH_PATH)
    family = pd.read_csv(FAMILY_PATH, low_memory=False)[["protein_id", "family_40"]]
    complete = master[master["structure_analysis_included"].astype(bool)].merge(family, on="protein_id", validate="one_to_one")
    complete["candidate"] = complete["group"].eq("Candidate").astype(int)
    covariates = np.column_stack([
        np.log1p(complete["sequence_length"].astype(float).to_numpy()),
        np.log1p(complete["donor_residue_count"].astype(float).to_numpy()),
        complete["mean_fraction_plddt_ge70"].astype(float).to_numpy(),
    ])
    covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0, ddof=0)

    rows = []
    for endpoint, spec in SPECS.items():
        positives = positive_ids(fs_pairs, spec)
        cand = complete[complete["group"].eq("Candidate")]
        back = complete[complete["group"].eq("Background")]
        cp, bp = int(cand["protein_id"].isin(positives).sum()), int(back["protein_id"].isin(positives).sum())
        cn, bn = len(cand), len(back)
        y = complete["protein_id"].isin(positives).astype(int).to_numpy()
        adjusted = logistic_group_effect(y, complete["candidate"].to_numpy(), covariates, complete["family_40"].to_numpy())
        cm = matched["candidate_id"].isin(positives)
        bm = matched["background_id"].isin(positives)
        co, bo = int((cm & ~bm).sum()), int((~cm & bm).sum())
        rows.append({
            "endpoint": endpoint,
            "candidate_positive": cp,
            "candidate_total": cn,
            "candidate_fraction": cp / cn,
            "background_positive": bp,
            "background_total": bn,
            "background_fraction": bp / bn,
            "risk_difference": cp / cn - bp / bn,
            **ratio_intervals(cp, cn, bp, bn),
            "fisher_exact_p": fisher_exact_two_sided(cp, cn - cp, bp, bn - bp),
            **adjusted,
            "matched_candidate_positive": int(cm.sum()),
            "matched_background_positive": int(bm.sum()),
            "matched_candidate_only": co,
            "matched_background_only": bo,
            "mcnemar_exact_p": exact_binomial_two_sided(co, co + bo),
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "threshold_sensitivity_group_comparisons_freesasa.csv", index=False)
    print("\nFull FreeSASA sensitivity results")
    print(results[["endpoint", "candidate_positive", "background_positive", "risk_ratio", "fisher_exact_p", "adjusted_wald_p", "family40_adjusted_p", "mcnemar_exact_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
