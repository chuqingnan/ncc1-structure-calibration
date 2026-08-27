from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
INPUT = ROOT / "5 seeds 5 models"
OUTPUT = ROOT / "outputs/robustness_5seed5model_frozen6/complete_analysis_150models"
MANIFEST = ROOT / "outputs/robustness_5seed5model_frozen6/frozen6_manifest.csv"
SPECIFICATION = ROOT / "outputs/robustness_5seed5model_frozen6/analysis_specification.json"

import sys

sys.path.insert(0, str(ROOT / "work/unified_candidate_background"))
from unified_structure_analysis import VDW, atom_sasa, parse_pdb, symmetric_pae  # noqa: E402


FROZEN = {
    "MtrunR108HiC_012482.1": ((478, "M"), (501, "M")),
    "MtrunR108HiC_005650.1": ((469, "C"), (481, "C")),
    "MtrunR108HiC_013677.1": ((671, "M"), (720, "H")),
    "MtrunR108HiC_001767.1": ((27, "H"), (136, "C")),
    "MtrunR108HiC_008307.1": ((27, "H"), (136, "C")),
    "MtrunR108HiC_031716.1": ((197, "H"), (241, "M")),
}

MODEL_PATTERN = re.compile(r"model_(\d+)_seed_(\d{3})")
RANK_PATTERN = re.compile(r"_rank_(\d+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_key(path: Path) -> tuple[int, int]:
    match = MODEL_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse model/seed from {path}")
    return int(match.group(2)), int(match.group(1))


def score_for(pdb: Path) -> Path:
    seed, model = parse_key(pdb)
    candidates = [
        path
        for path in pdb.parent.glob("*scores*.json")
        if MODEL_PATTERN.search(path.name)
        and parse_key(path) == (seed, model)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one score JSON for {pdb}; found {candidates}")
    return candidates[0]


def discover_models() -> tuple[dict[tuple[str, int, int], Path], list[dict]]:
    selected: dict[tuple[str, int, int], Path] = {}
    audit: list[dict] = []
    for protein_id in FROZEN:
        protein_dirs = sorted(path for path in INPUT.iterdir() if path.is_dir() and protein_id in path.name)
        if not protein_dirs:
            raise FileNotFoundError(f"No extracted directory for {protein_id}")

        if protein_id == "MtrunR108HiC_012482.1":
            base = next(path for path in protein_dirs if path.name.endswith("_results"))
            supplement = next(path for path in protein_dirs if "seeds_03_04" in path.name)
            sources = [("original_seed_0_2", base), ("supplement_seed_3_4", supplement)]
            for source_label, source_dir in sources:
                for pdb in source_dir.rglob("*.pdb"):
                    if "unrelaxed" not in pdb.name:
                        continue
                    seed, model = parse_key(pdb)
                    allowed = seed <= 2 if source_label == "original_seed_0_2" else seed in (3, 4)
                    if allowed:
                        selected[(protein_id, seed, model)] = pdb
        else:
            source_dir = protein_dirs[0]
            sources = [("isolated_seed_run", source_dir)]
            for pdb in source_dir.rglob("*.pdb"):
                if "unrelaxed" not in pdb.name:
                    continue
                seed, model = parse_key(pdb)
                key = (protein_id, seed, model)
                if key in selected:
                    raise RuntimeError(f"Duplicate selected combination: {key}")
                selected[key] = pdb

        keys = sorted(key for key in selected if key[0] == protein_id)
        expected = [(protein_id, seed, model) for seed in range(5) for model in range(1, 6)]
        if keys != expected:
            missing = sorted(set(expected) - set(keys))
            extra = sorted(set(keys) - set(expected))
            raise RuntimeError(f"Incomplete grid for {protein_id}: missing={missing}, extra={extra}")

    for (protein_id, seed, model), pdb in sorted(selected.items()):
        score = score_for(pdb)
        source_policy = (
            "original archive (seeds 0-2)"
            if protein_id == "MtrunR108HiC_012482.1" and seed <= 2
            else "isolated supplement (seeds 3-4)"
            if protein_id == "MtrunR108HiC_012482.1"
            else "isolated five-seed archive"
        )
        audit.append(
            {
                "protein_id": protein_id,
                "seed": seed,
                "model": model,
                "source_policy": source_policy,
                "pdb_path": str(pdb),
                "score_json_path": str(score),
                "pdb_sha256": sha256(pdb),
                "score_json_sha256": sha256(score),
            }
        )
    return selected, audit


def quantile_iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def select_pair(donors: list[dict], pair_spec: tuple[tuple[int, str], tuple[int, str]]) -> tuple[dict, dict]:
    chosen = []
    for resseq, code in pair_spec:
        matches = [donor for donor in donors if donor["resseq"] == resseq and donor["code"] == code]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one donor residue {code}{resseq}; found {len(matches)}")
        chosen.append(matches[0])
    return chosen[0], chosen[1]


def analyze_model(protein_id: str, seed: int, model: int, pdb: Path) -> dict:
    score_path = score_for(pdb)
    score = json.loads(score_path.read_text())
    plddt = np.asarray(score["plddt"], dtype=float)
    pae = np.asarray(score["pae"], dtype=float)
    if pae.shape != (len(plddt), len(plddt)):
        raise RuntimeError(f"PAE shape mismatch in {score_path}")

    atoms, donors = parse_pdb(pdb)
    left, right = select_pair(donors, FROZEN[protein_id])
    choices = []
    for atom_left in left["atoms"]:
        for atom_right in right["atoms"]:
            distance = float(np.linalg.norm(atom_left["coord"] - atom_right["coord"]))
            choices.append((distance, atom_left, atom_right))
    distance, atom_left, atom_right = min(choices, key=lambda item: item[0])

    coords = np.stack([atom["coord"] for atom in atoms])
    radii = np.asarray([VDW.get(atom["element"], 1.7) + 1.4 for atom in atoms])
    sasa_left = float(atom_sasa(coords, radii, atom_left))
    sasa_right = float(atom_sasa(coords, radii, atom_right))
    mean_sasa = (sasa_left + sasa_right) / 2.0
    residue_a = FROZEN[protein_id][0][0]
    residue_b = FROZEN[protein_id][1][0]
    local_a = float(plddt[residue_a - 1])
    local_b = float(plddt[residue_b - 1])
    min_local = min(local_a, local_b)
    pair_pae = float(symmetric_pae(pae, residue_a, residue_b))

    pass_distance = 2.5 <= distance <= 5.0
    pass_local_plddt = min_local >= 70.0
    pass_pair_pae = pair_pae <= 10.0
    pass_sasa = mean_sasa >= 5.0
    qualifies = pass_distance and pass_local_plddt and pass_pair_pae and pass_sasa
    rank_match = RANK_PATTERN.search(pdb.name)

    return {
        "protein_id": protein_id,
        "frozen_pair": f"{FROZEN[protein_id][0][1]}{residue_a}-{FROZEN[protein_id][1][1]}{residue_b}",
        "seed": seed,
        "model": model,
        "rank_within_seed": int(rank_match.group(1)) if rank_match else np.nan,
        "sequence_length": int(len(plddt)),
        "global_mean_plddt": float(plddt.mean()),
        "global_fraction_plddt_ge70": float((plddt >= 70).mean()),
        "ptm": float(score.get("ptm", np.nan)),
        "residue_a": residue_a,
        "residue_a_type": FROZEN[protein_id][0][1],
        "residue_a_plddt": local_a,
        "residue_b": residue_b,
        "residue_b_type": FROZEN[protein_id][1][1],
        "residue_b_plddt": local_b,
        "min_pair_plddt": min_local,
        "symmetric_pair_pae_A": pair_pae,
        "donor_atom_a": atom_left["atom"],
        "donor_atom_b": atom_right["atom"],
        "donor_distance_A": distance,
        "donor_sasa_a_A2": sasa_left,
        "donor_sasa_b_A2": sasa_right,
        "mean_donor_sasa_A2": mean_sasa,
        "pass_distance_2p5_to_5A": pass_distance,
        "pass_both_residue_plddt_ge70": pass_local_plddt,
        "pass_symmetric_pair_pae_le10A": pass_pair_pae,
        "pass_mean_donor_sasa_ge5A2": pass_sasa,
        "qualifies_all_predefined_criteria": qualifies,
        "failed_criterion_count": 4 - sum([pass_distance, pass_local_plddt, pass_pair_pae, pass_sasa]),
        "pdb_path": str(pdb),
        "score_json_path": str(score_path),
    }


def seed_summary(model_df: pd.DataFrame) -> pd.DataFrame:
    grouped = model_df.groupby(["protein_id", "frozen_pair", "seed"], sort=False)
    out = grouped.agg(
        n_models=("model", "count"),
        n_qualifying=("qualifies_all_predefined_criteria", "sum"),
        qualifying_fraction=("qualifies_all_predefined_criteria", "mean"),
        median_global_plddt=("global_mean_plddt", "median"),
        median_ptm=("ptm", "median"),
        median_min_pair_plddt=("min_pair_plddt", "median"),
        median_pair_pae_A=("symmetric_pair_pae_A", "median"),
        median_donor_distance_A=("donor_distance_A", "median"),
        median_mean_donor_sasa_A2=("mean_donor_sasa_A2", "median"),
    ).reset_index()
    out["seed_has_any_qualifying_model"] = out["n_qualifying"] >= 1
    out["seed_has_majority_qualifying_models"] = out["n_qualifying"] >= 3
    return out


def candidate_summary(model_df: pd.DataFrame, seed_df: pd.DataFrame, frozen_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protein_id, group in model_df.groupby("protein_id", sort=False):
        seed_group = seed_df[seed_df["protein_id"] == protein_id]
        qualifying = group[group["qualifies_all_predefined_criteria"]]
        medians = {
            "distance": float(group["donor_distance_A"].median()),
            "min_plddt": float(group["min_pair_plddt"].median()),
            "pair_pae": float(group["symmetric_pair_pae_A"].median()),
            "sasa": float(group["mean_donor_sasa_A2"].median()),
        }
        representative_pool = qualifying if not qualifying.empty else group
        scale = {
            "distance": max(quantile_iqr(group["donor_distance_A"]), 0.1),
            "min_plddt": max(quantile_iqr(group["min_pair_plddt"]), 1.0),
            "pair_pae": max(quantile_iqr(group["symmetric_pair_pae_A"]), 0.2),
            "sasa": max(quantile_iqr(group["mean_donor_sasa_A2"]), 0.5),
        }
        representative_pool = representative_pool.copy()
        representative_pool["representative_deviation"] = (
            abs(representative_pool["donor_distance_A"] - medians["distance"]) / scale["distance"]
            + abs(representative_pool["min_pair_plddt"] - medians["min_plddt"]) / scale["min_plddt"]
            + abs(representative_pool["symmetric_pair_pae_A"] - medians["pair_pae"]) / scale["pair_pae"]
            + abs(representative_pool["mean_donor_sasa_A2"] - medians["sasa"]) / scale["sasa"]
        )
        representative = representative_pool.sort_values(
            ["representative_deviation", "min_pair_plddt"], ascending=[True, False]
        ).iloc[0]

        pass_rate = float(group["qualifies_all_predefined_criteria"].mean())
        min_seed_rate = float(seed_group["qualifying_fraction"].min())
        if pass_rate >= 0.80 and min_seed_rate >= 0.60:
            reproducibility_tier = "High (descriptive)"
        elif pass_rate >= 0.50 and int(seed_group["seed_has_any_qualifying_model"].sum()) >= 4:
            reproducibility_tier = "Intermediate (descriptive)"
        else:
            reproducibility_tier = "Low (descriptive)"

        rows.append(
            {
                "protein_id": protein_id,
                "frozen_pair": group["frozen_pair"].iloc[0],
                "n_models_expected": 25,
                "n_models_analyzed": int(len(group)),
                "n_seeds_analyzed": int(group["seed"].nunique()),
                "n_qualifying": int(group["qualifies_all_predefined_criteria"].sum()),
                "qualifying_fraction": pass_rate,
                "seeds_with_any_qualifying_model": int(seed_group["seed_has_any_qualifying_model"].sum()),
                "seeds_with_majority_qualifying_models": int(seed_group["seed_has_majority_qualifying_models"].sum()),
                "minimum_seed_qualifying_fraction": min_seed_rate,
                "maximum_seed_qualifying_fraction": float(seed_group["qualifying_fraction"].max()),
                "between_seed_qualifying_fraction_range": float(seed_group["qualifying_fraction"].max() - min_seed_rate),
                "median_global_plddt": float(group["global_mean_plddt"].median()),
                "iqr_global_plddt": quantile_iqr(group["global_mean_plddt"]),
                "median_ptm": float(group["ptm"].median()),
                "iqr_ptm": quantile_iqr(group["ptm"]),
                "median_min_pair_plddt": medians["min_plddt"],
                "minimum_min_pair_plddt": float(group["min_pair_plddt"].min()),
                "median_pair_pae_A": medians["pair_pae"],
                "maximum_pair_pae_A": float(group["symmetric_pair_pae_A"].max()),
                "median_donor_distance_A": medians["distance"],
                "iqr_donor_distance_A": quantile_iqr(group["donor_distance_A"]),
                "all_model_donor_distance_sd_A": float(group["donor_distance_A"].std(ddof=0)),
                "between_seed_median_distance_sd_A": float(seed_group["median_donor_distance_A"].std(ddof=0)),
                "between_seed_median_distance_range_A": float(seed_group["median_donor_distance_A"].max() - seed_group["median_donor_distance_A"].min()),
                "median_mean_donor_sasa_A2": medians["sasa"],
                "pass_distance_fraction": float(group["pass_distance_2p5_to_5A"].mean()),
                "pass_local_plddt_fraction": float(group["pass_both_residue_plddt_ge70"].mean()),
                "pass_pair_pae_fraction": float(group["pass_symmetric_pair_pae_le10A"].mean()),
                "pass_sasa_fraction": float(group["pass_mean_donor_sasa_ge5A2"].mean()),
                "reproducibility_tier": reproducibility_tier,
                "representative_seed": int(representative["seed"]),
                "representative_model": int(representative["model"]),
                "representative_distance_A": float(representative["donor_distance_A"]),
                "representative_min_pair_plddt": float(representative["min_pair_plddt"]),
                "representative_pair_pae_A": float(representative["symmetric_pair_pae_A"]),
                "representative_mean_donor_sasa_A2": float(representative["mean_donor_sasa_A2"]),
                "representative_pdb_path": representative["pdb_path"],
            }
        )

    summary = pd.DataFrame(rows)
    annotations = frozen_manifest[
        ["frozen_order", "protein_id", "r108_annotation", "localization", "integrated_score_0_100"]
    ].copy()
    summary = summary.merge(annotations, on="protein_id", how="left", validate="one_to_one")
    summary = summary.sort_values("frozen_order").reset_index(drop=True)
    columns = [
        "frozen_order", "protein_id", "r108_annotation", "localization", "integrated_score_0_100",
        *[column for column in summary.columns if column not in {
            "frozen_order", "protein_id", "r108_annotation", "localization", "integrated_score_0_100"
        }],
    ]
    return summary[columns]


def build_notebook(summary: pd.DataFrame) -> None:
    notebook_path = OUTPUT / "frozen6_5seed5model_robustness_analysis.ipynb"
    summary_view = summary[
        [
            "protein_id", "frozen_pair", "n_qualifying", "qualifying_fraction",
            "minimum_seed_qualifying_fraction", "median_donor_distance_A",
            "median_min_pair_plddt", "median_pair_pae_A", "median_mean_donor_sasa_A2",
            "reproducibility_tier",
        ]
    ].copy()
    display_text = summary_view.to_string(index=False, float_format=lambda value: f"{value:.3f}")
    high = summary.loc[summary["reproducibility_tier"] == "High (descriptive)", "protein_id"].tolist()
    medium = summary.loc[summary["reproducibility_tier"] == "Intermediate (descriptive)", "protein_id"].tolist()
    low = summary.loc[summary["reproducibility_tier"] == "Low (descriptive)", "protein_id"].tolist()
    takeaway = (
        f"High reproducibility: {', '.join(high) if high else 'none'}.  \n"
        f"Intermediate reproducibility: {', '.join(medium) if medium else 'none'}.  \n"
        f"Low reproducibility: {', '.join(low) if low else 'none'}."
    )
    script_path = ROOT / "work/robustness_5seed5model/analyze_frozen6_complete.py"
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Frozen six candidates: 5-seed × 5-model robustness analysis\n",
                    "\n",
                    "## tl;dr\n",
                    "Six pre-frozen residue pairs were evaluated without post-hoc pair replacement across 150 predictions. The candidate summary below reports global confidence, local confidence, pair PAE, donor geometry, solvent accessibility, qualifying fraction, and between-seed stability.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Context & Methods\n",
                    "Each protein contributed five AlphaFold2-ptm models for each of five seeds (25 predictions). The frozen-pair rule was fixed before this rerun: both residues pLDDT ≥70; symmetric pair PAE ≤10 Å; nearest chemically plausible donor-atom distance 2.5–5.0 Å; mean donor-atom SASA ≥5 Å². Cys SG, Met SD, and the nearer His ND1/NE2 were used. The descriptive reproducibility tier is secondary: High requires ≥80% overall qualification and ≥60% in every seed; Intermediate requires ≥50% overall and at least four seeds with any qualifying model; otherwise Low.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Data\n",
                    "The source folder contains extracted Kaggle predictions. Candidate 1 uses the original archive for seeds 0–2 and the isolated supplement for seeds 3–4; all other candidates use their isolated five-seed archives. SHA-256 hashes and file paths are retained in `input_audit.csv`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["Analysis completed: 150 unique predictions (6 candidates × 5 seeds × 5 models).\n"],
                    }
                ],
                "source": [
                    "# Re-run the complete analysis from the frozen specification and extracted PDB/JSON files.\n",
                    "import runpy\n",
                    f"runpy.run_path({str(script_path)!r}, run_name='__main__')\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Results\n"],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [
                    {"name": "stdout", "output_type": "stream", "text": [display_text + "\n"]}
                ],
                "source": [
                    "import pandas as pd\n",
                    f"summary = pd.read_csv({str(OUTPUT / 'candidate_summary.csv')!r})\n",
                    "summary[['protein_id','frozen_pair','n_qualifying','qualifying_fraction','minimum_seed_qualifying_fraction','median_donor_distance_A','median_min_pair_plddt','median_pair_pae_A','median_mean_donor_sasa_A2','reproducibility_tier']]\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Takeaways\n",
                    takeaway + "  \n",
                    "These are computational robustness screens, not evidence of Cu binding or direct NCC1 client status. The next structural-visualization step should use the preselected representative model while showing the full 25-model distribution in the paper/supplement.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected, audit_rows = discover_models()
    if len(selected) != 150:
        raise RuntimeError(f"Expected 150 unique predictions, found {len(selected)}")

    model_rows = []
    for (protein_id, seed, model), pdb in sorted(selected.items()):
        model_rows.append(analyze_model(protein_id, seed, model, pdb))
    model_df = pd.DataFrame(model_rows)
    if model_df.groupby("protein_id").size().ne(25).any():
        raise RuntimeError("Not every candidate has 25 models")
    if model_df.groupby(["protein_id", "seed"]).size().ne(5).any():
        raise RuntimeError("Not every candidate/seed has five models")

    seed_df = seed_summary(model_df)
    frozen_manifest = pd.read_csv(MANIFEST)
    summary_df = candidate_summary(model_df, seed_df, frozen_manifest)
    audit_df = pd.DataFrame(audit_rows)

    representative_keys = set(
        zip(summary_df["protein_id"], summary_df["representative_seed"], summary_df["representative_model"])
    )
    model_df["selected_representative_for_pymol"] = [
        (row.protein_id, row.seed, row.model) in representative_keys for row in model_df.itertuples()
    ]

    model_df.to_csv(OUTPUT / "model_level_metrics.csv", index=False)
    seed_df.to_csv(OUTPUT / "seed_level_summary.csv", index=False)
    summary_df.to_csv(OUTPUT / "candidate_summary.csv", index=False)
    audit_df.to_csv(OUTPUT / "input_audit.csv", index=False)
    workbook_payload = {
        "candidate_summary": json.loads(summary_df.to_json(orient="records")),
        "seed_summary": json.loads(seed_df.to_json(orient="records")),
        "model_metrics": json.loads(model_df.to_json(orient="records")),
        "input_audit": json.loads(audit_df.to_json(orient="records")),
    }
    (OUTPUT / "workbook_payload.json").write_text(json.dumps(workbook_payload, ensure_ascii=False))

    spec = json.loads(SPECIFICATION.read_text())
    qa = {
        "analysis_completed": True,
        "candidate_count": int(model_df["protein_id"].nunique()),
        "unique_prediction_count": int(len(model_df)),
        "models_per_candidate": model_df.groupby("protein_id").size().astype(int).to_dict(),
        "models_per_candidate_seed_all_equal_5": bool(model_df.groupby(["protein_id", "seed"]).size().eq(5).all()),
        "duplicate_protein_seed_model_rows": int(model_df.duplicated(["protein_id", "seed", "model"]).sum()),
        "missing_metric_counts": model_df[
            [
                "global_mean_plddt", "ptm", "min_pair_plddt", "symmetric_pair_pae_A",
                "donor_distance_A", "mean_donor_sasa_A2",
            ]
        ].isna().sum().astype(int).to_dict(),
        "qualification_rule": spec["qualification_rule"],
        "candidate1_source_policy": "Original archive seeds 0-2; isolated supplement seeds 3-4; duplicate partial seed-3 files excluded.",
        "representative_selection": "Among qualifying models (or all if none qualify), minimize scaled absolute deviation from the candidate medians across distance, local pLDDT, pair PAE, and donor SASA.",
        "interpretation_limit": "Descriptive structural robustness only; not a probability of Cu binding or NCC1 client status.",
    }
    (OUTPUT / "qa_report.json").write_text(json.dumps(qa, indent=2))
    build_notebook(summary_df)

    print("Analysis completed: 150 unique predictions (6 candidates × 5 seeds × 5 models).")
    print(summary_df[["protein_id", "frozen_pair", "n_qualifying", "qualifying_fraction", "reproducibility_tier"]].to_string(index=False))


if __name__ == "__main__":
    main()
