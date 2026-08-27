from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
INPUT = ROOT / "5 seeds 5 models/PRP1like_008596_008599_5seed5model/results"
OUTPUT = ROOT / "outputs/watchlist3_5seed5model_analysis"
UTILITY = ROOT / "work/unified_candidate_background/unified_structure_analysis.py"
MANIFEST = ROOT / "outputs/unbiased_scope16_screen/scope16_unbiased_manifest.csv"

PROTEINS = ["MtrunR108HiC_008596.1", "MtrunR108HiC_008599.1"]
FROZEN_PAIR = "H25-H68"
MODEL_RE = re.compile(r"_rank_(\d+)_alphafold2_ptm_model_(\d+)_seed_(\d{3})\.pdb$")

spec = importlib.util.spec_from_file_location("unified_structure_analysis", UTILITY)
usa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usa)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_identity(path: Path) -> tuple[int, int, int]:
    match = MODEL_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse rank/model/seed from {path}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def quantile_iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def tier(overall_fraction: float, minimum_seed_fraction: float, seeds_with_any: int) -> str:
    if overall_fraction >= 0.80 and minimum_seed_fraction >= 0.60:
        return "High (descriptive)"
    if overall_fraction >= 0.50 and seeds_with_any >= 4:
        return "Intermediate (descriptive)"
    return "Low (descriptive)"


def check_inputs() -> tuple[dict[str, dict], pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["prediction_id"].isin(PROTEINS)].copy()
    if set(manifest["prediction_id"]) != set(PROTEINS):
        raise RuntimeError("The two proteins are not both present in the frozen scope16 manifest")
    meta = {
        row.prediction_id: {
            "sequence_length": int(row.sequence_length_verified),
            "sequence": str(row.sequence),
        }
        for row in manifest.itertuples()
    }

    audit_rows = []
    for protein_id in PROTEINS:
        files = sorted((INPUT / protein_id).glob("*_unrelaxed_rank_*.pdb"))
        seen = set()
        for pdb in files:
            rank, model, seed = parse_identity(pdb)
            key = (seed, model)
            if key in seen:
                raise RuntimeError(f"Duplicate seed/model pair for {protein_id}: {key}")
            seen.add(key)
            score = usa.score_path_for(pdb)
            if not score.exists():
                raise FileNotFoundError(score)
            audit_rows.append({
                "protein_id": protein_id,
                "rank": rank,
                "model": model,
                "seed": seed,
                "pdb_path": str(pdb),
                "score_json_path": str(score),
                "pdb_sha256": sha256(pdb),
                "score_json_sha256": sha256(score),
            })
        expected = {(seed, model) for seed in range(5) for model in range(1, 6)}
        if seen != expected:
            raise RuntimeError(
                f"Incomplete 5x5 grid for {protein_id}: missing={sorted(expected-seen)}, "
                f"extra={sorted(seen-expected)}"
            )
    audit = pd.DataFrame(audit_rows).sort_values(["protein_id", "seed", "model"])
    return meta, audit


def analyze_all_models(meta: dict[str, dict], audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    pair_rows = []
    for row in audit.itertuples():
        pdb = Path(row.pdb_path)
        model_record, pairs = usa.analyze_model(
            "PRP1like_robustness", row.protein_id, row.rank, pdb, meta[row.protein_id]
        )
        model_record.update({"seed": row.seed, "model": row.model})
        model_rows.append(model_record)
        for pair in pairs:
            pair.update({"seed": row.seed, "model": row.model})
            pair["pass_all_cmh_frozen_endpoint"] = usa.endpoint_pass(
                pair, distance=5, plddt=70, separation=10, sasa=5, scope="all"
            )
            pair["pass_sulfur_primary_endpoint"] = usa.endpoint_pass(
                pair, distance=5, plddt=70, separation=10, sasa=5, scope="sulfur"
            )
            pair_rows.append(pair)
    models = pd.DataFrame(model_rows).sort_values(["protein_id", "seed", "model"])
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["protein_id", "seed", "model", "residue_i", "residue_j"]
    )
    if len(models) != 50:
        raise RuntimeError(f"Expected 50 models, found {len(models)}")
    return models, pairs


def summarize_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    passing = pairs[pairs["pass_all_cmh_frozen_endpoint"]].copy()
    rows = []
    seed_rows = []
    for (protein_id, pair_key, combo), group in passing.groupby(
        ["protein_id", "pair_key", "donor_combo"], sort=False
    ):
        seed_counts = group.groupby("seed").size().reindex(range(5), fill_value=0)
        seed_fractions = seed_counts / 5.0
        support = int(len(group))
        overall = support / 25.0
        rows.append({
            "protein_id": protein_id,
            "pair_key": pair_key,
            "donor_combo": combo,
            "sulfur_anchored": bool(group["sulfur_anchored"].iloc[0]),
            "support_models": support,
            "repeat_fraction": overall,
            "seeds_with_any_qualifying_model": int((seed_counts > 0).sum()),
            "seeds_with_majority_qualifying_models": int((seed_counts >= 3).sum()),
            "minimum_seed_repeat_fraction": float(seed_fractions.min()),
            "maximum_seed_repeat_fraction": float(seed_fractions.max()),
            "median_distance_A": float(group["donor_distance_A"].median()),
            "iqr_distance_A": quantile_iqr(group["donor_distance_A"]),
            "distance_range_A": float(group["donor_distance_A"].max() - group["donor_distance_A"].min()),
            "median_min_pair_plddt": float(group["min_local_plddt"].median()),
            "median_pair_pae_A": float(group["pair_pae_A"].median()),
            "median_mean_donor_sasa_A2": float(group["mean_donor_sasa_A2"].median()),
            "reproducibility_tier": tier(overall, float(seed_fractions.min()), int((seed_counts > 0).sum())),
        })
        for seed in range(5):
            seed_rows.append({
                "protein_id": protein_id,
                "pair_key": pair_key,
                "donor_combo": combo,
                "seed": seed,
                "qualifying_models": int(seed_counts.loc[seed]),
                "models": 5,
                "repeat_fraction": float(seed_fractions.loc[seed]),
            })
    pair_summary = pd.DataFrame(rows)
    if not pair_summary.empty:
        pair_summary = pair_summary.sort_values(
            ["protein_id", "support_models", "median_pair_pae_A"],
            ascending=[True, False, True],
        )
    return pair_summary, pd.DataFrame(seed_rows)


def summarize_frozen_pair(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pairs[pairs["pair_key"].eq(FROZEN_PAIR)].copy()
    if len(target) != 50:
        raise RuntimeError(f"Expected H25-H68 in all 50 models, found {len(target)}")
    rows = []
    seed_rows = []
    for protein_id, group in target.groupby("protein_id", sort=False):
        pass_mask = group["pass_all_cmh_frozen_endpoint"].astype(bool)
        by_seed = group.groupby("seed")["pass_all_cmh_frozen_endpoint"].agg(["sum", "mean"])
        overall = float(pass_mask.mean())
        min_seed = float(by_seed["mean"].min())
        rows.append({
            "protein_id": protein_id,
            "frozen_pair": FROZEN_PAIR,
            "models_analyzed": int(len(group)),
            "qualifying_models": int(pass_mask.sum()),
            "qualifying_fraction": overall,
            "seeds_with_any_qualifying_model": int((by_seed["sum"] > 0).sum()),
            "seeds_with_majority_qualifying_models": int((by_seed["sum"] >= 3).sum()),
            "minimum_seed_qualifying_fraction": min_seed,
            "maximum_seed_qualifying_fraction": float(by_seed["mean"].max()),
            "median_donor_distance_A": float(group["donor_distance_A"].median()),
            "iqr_donor_distance_A": quantile_iqr(group["donor_distance_A"]),
            "distance_sd_A": float(group["donor_distance_A"].std(ddof=0)),
            "distance_range_A": float(group["donor_distance_A"].max() - group["donor_distance_A"].min()),
            "median_min_pair_plddt": float(group["min_local_plddt"].median()),
            "minimum_min_pair_plddt": float(group["min_local_plddt"].min()),
            "median_pair_pae_A": float(group["pair_pae_A"].median()),
            "maximum_pair_pae_A": float(group["pair_pae_A"].max()),
            "median_mean_donor_sasa_A2": float(group["mean_donor_sasa_A2"].median()),
            "minimum_mean_donor_sasa_A2": float(group["mean_donor_sasa_A2"].min()),
            "pass_distance_fraction": float(group["donor_distance_A"].between(2.5, 5.0).mean()),
            "pass_local_plddt_fraction": float((group["min_local_plddt"] >= 70).mean()),
            "pass_pair_pae_fraction": float((group["pair_pae_A"] <= 10).mean()),
            "pass_sasa_fraction": float((group["mean_donor_sasa_A2"] >= 5).mean()),
            "reproducibility_tier": tier(overall, min_seed, int((by_seed["sum"] > 0).sum())),
        })
        for seed, sub in group.groupby("seed"):
            seed_rows.append({
                "protein_id": protein_id,
                "frozen_pair": FROZEN_PAIR,
                "seed": int(seed),
                "qualifying_models": int(sub["pass_all_cmh_frozen_endpoint"].sum()),
                "qualifying_fraction": float(sub["pass_all_cmh_frozen_endpoint"].mean()),
                "median_distance_A": float(sub["donor_distance_A"].median()),
                "median_min_pair_plddt": float(sub["min_local_plddt"].median()),
                "median_pair_pae_A": float(sub["pair_pae_A"].median()),
                "median_mean_donor_sasa_A2": float(sub["mean_donor_sasa_A2"].median()),
            })
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def family_replication(
    meta: dict[str, dict], pairs: pd.DataFrame, pair_summary: pd.DataFrame, frozen_summary: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    a, b = PROTEINS
    seq_a, seq_b = meta[a]["sequence"], meta[b]["sequence"]
    if len(seq_a) != len(seq_b):
        raise RuntimeError("Paralog sequences differ in length; direct residue-number comparison is invalid")
    identity = sum(x == y for x, y in zip(seq_a, seq_b)) / len(seq_a)

    target = pairs[pairs["pair_key"].eq(FROZEN_PAIR)].copy()
    keep = [
        "protein_id", "seed", "model", "pass_all_cmh_frozen_endpoint", "donor_distance_A",
        "min_local_plddt", "pair_pae_A", "mean_donor_sasa_A2",
    ]
    pa = target[target["protein_id"].eq(a)][keep].drop(columns="protein_id")
    pb = target[target["protein_id"].eq(b)][keep].drop(columns="protein_id")
    paired = pa.merge(pb, on=["seed", "model"], suffixes=("_008596", "_008599"), validate="one_to_one")
    qa = paired["pass_all_cmh_frozen_endpoint_008596"].astype(bool)
    qb = paired["pass_all_cmh_frozen_endpoint_008599"].astype(bool)
    paired["both_qualify"] = qa & qb
    paired["qualification_agrees"] = qa.eq(qb)
    paired["absolute_distance_difference_A"] = abs(
        paired["donor_distance_A_008596"] - paired["donor_distance_A_008599"]
    )

    union_positive = int((qa | qb).sum())
    binary_jaccard = float((qa & qb).sum() / union_positive) if union_positive else 1.0

    profiles = []
    for protein_id in PROTEINS:
        sub = pair_summary[pair_summary["protein_id"].eq(protein_id)][["pair_key", "repeat_fraction"]]
        profiles.append(sub.rename(columns={"repeat_fraction": protein_id}))
    profile = profiles[0].merge(profiles[1], on="pair_key", how="outer").fillna(0)
    profile_spearman = (
        float(profile[PROTEINS].corr(method="spearman").iloc[0, 1])
        if len(profile) > 1 else None
    )

    passing = pairs[pairs["pass_all_cmh_frozen_endpoint"]].copy()
    sets = {
        (pid, seed, model): set(group["pair_key"])
        for (pid, seed, model), group in passing.groupby(["protein_id", "seed", "model"])
    }
    jaccards = []
    for seed in range(5):
        for model in range(1, 6):
            sa = sets.get((a, seed, model), set())
            sb = sets.get((b, seed, model), set())
            union = sa | sb
            jaccards.append(len(sa & sb) / len(union) if union else 1.0)

    high_sets = {}
    majority_sets = {}
    for protein_id in PROTEINS:
        sub = pair_summary[pair_summary["protein_id"].eq(protein_id)]
        high_sets[protein_id] = set(sub.loc[sub["repeat_fraction"] >= 0.80, "pair_key"])
        majority_sets[protein_id] = set(sub.loc[sub["repeat_fraction"] >= 0.50, "pair_key"])

    def set_jaccard(left: set, right: set) -> float:
        return len(left & right) / len(left | right) if left | right else 1.0

    summary = {
        "sequence_length_each": len(seq_a),
        "direct_positionwise_sequence_identity": identity,
        "H25_conserved": seq_a[24] == seq_b[24] == "H",
        "H68_conserved": seq_a[67] == seq_b[67] == "H",
        "paired_seed_model_comparisons": int(len(paired)),
        "both_H25_H68_qualify": int(paired["both_qualify"].sum()),
        "both_H25_H68_qualify_fraction": float(paired["both_qualify"].mean()),
        "H25_H68_binary_qualification_agreement_fraction": float(paired["qualification_agrees"].mean()),
        "H25_H68_binary_positive_jaccard": binary_jaccard,
        "H25_H68_paired_mean_absolute_distance_difference_A": float(paired["absolute_distance_difference_A"].mean()),
        "H25_H68_paired_median_absolute_distance_difference_A": float(paired["absolute_distance_difference_A"].median()),
        "H25_H68_distance_spearman": float(paired[["donor_distance_A_008596", "donor_distance_A_008599"]].corr(method="spearman").iloc[0, 1]),
        "all_CMH_pair_frequency_profile_spearman": profile_spearman,
        "per_model_all_CMH_pair_set_jaccard_median": float(np.median(jaccards)),
        "per_model_all_CMH_pair_set_jaccard_iqr": float(np.quantile(jaccards, 0.75) - np.quantile(jaccards, 0.25)),
        "high_repeat_pair_set_008596": sorted(high_sets[a]),
        "high_repeat_pair_set_008599": sorted(high_sets[b]),
        "high_repeat_pair_set_jaccard": set_jaccard(high_sets[a], high_sets[b]),
        "majority_repeat_pair_set_008596": sorted(majority_sets[a]),
        "majority_repeat_pair_set_008599": sorted(majority_sets[b]),
        "majority_repeat_pair_set_jaccard": set_jaccard(majority_sets[a], majority_sets[b]),
        "interpretation_guardrail": (
            "Family replication assesses predicted monomer geometry only. H25-H68 is H-H, "
            "not sulfur anchored, and does not establish Cu occupancy, Cu transfer, or direct NCC1 binding."
        ),
    }

    frozen_lookup = frozen_summary.set_index("protein_id")
    summary["both_meet_existing_high_reproducibility_rule"] = bool(
        frozen_lookup.loc[a, "reproducibility_tier"] == "High (descriptive)"
        and frozen_lookup.loc[b, "reproducibility_tier"] == "High (descriptive)"
    )
    return paired, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    meta, audit = check_inputs()
    models, pairs = analyze_all_models(meta, audit)
    pair_summary, pair_seed_summary = summarize_pairs(pairs)
    frozen_summary, frozen_seed_summary = summarize_frozen_pair(pairs)
    paired_family, family = family_replication(meta, pairs, pair_summary, frozen_summary)

    audit.to_csv(OUTPUT / "input_audit_50models.csv", index=False)
    models.to_csv(OUTPUT / "model_level_global_metrics.csv", index=False)
    pairs.to_csv(OUTPUT / "all_model_CMH_pair_metrics.csv.gz", index=False, compression="gzip")
    pair_summary.to_csv(OUTPUT / "CMH_pair_repeat_summary.csv", index=False)
    pair_seed_summary.to_csv(OUTPUT / "CMH_pair_seed_repeat_summary.csv", index=False)
    frozen_summary.to_csv(OUTPUT / "H25_H68_geometry_summary.csv", index=False)
    frozen_seed_summary.to_csv(OUTPUT / "H25_H68_seed_summary.csv", index=False)
    paired_family.to_csv(OUTPUT / "H25_H68_paired_paralog_comparison.csv", index=False)
    (OUTPUT / "family_replication_summary.json").write_text(
        json.dumps(family, indent=2, ensure_ascii=False)
    )
    (OUTPUT / "analysis_specification.json").write_text(json.dumps({
        "frozen_pair": FROZEN_PAIR,
        "model_grid": "5 seeds x 5 AlphaFold2-ptm models per protein",
        "per_model_endpoint": {
            "donors": "Cys SG, Met SD, nearer His ND1/NE2",
            "donor_distance_A": [2.5, 5.0],
            "minimum_sequence_separation_aa": 10,
            "minimum_pair_residue_plddt": 70,
            "maximum_symmetric_pair_pae_A": 10,
            "minimum_mean_selected_donor_atom_sasa_A2": 5,
        },
        "high_reproducibility_rule": "overall qualifying fraction >=0.80 and every seed >=0.60",
        "intermediate_rule": "overall >=0.50 and at least four seeds with any qualifying model",
        "primary_vs_sensitivity": "Cys/Met anchored pairs are primary; H-H is sensitivity only",
    }, indent=2))

    print("\nH25-H68 SUMMARY")
    print(frozen_summary.to_string(index=False))
    print("\nTOP CMH PAIRS")
    print(pair_summary.groupby("protein_id").head(10).to_string(index=False))
    print("\nFAMILY REPLICATION")
    print(json.dumps(family, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
