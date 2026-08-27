from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
RAW = ROOT / "Figure 2 kaggle raw data"
MANIFEST_JSON = ROOT / "work" / "figure2" / "Figure2_candidate_manifest.json"
OUT = ROOT / "outputs" / "Figure2_structure_analysis"

LOCAL_PLDDT_MIN = 70.0
PAIR_PAE_MAX = 10.0
DISTANCE_LOWER = 3.0
THRESHOLDS = (6.0, 8.0, 10.0)
DISULFIDE_MAX = 2.5
TECHNICAL_EXCLUSIONS = {
    "MtrunR108HiC_000928.1": {
        "sequence_length": 2194,
        "reason": "GPU memory exhaustion in repeated ColabFold attempts; no structure produced",
        "classification": "technical_missing_not_negative",
    }
}

VDW = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "SE": 1.90, "FE": 1.80, "ZN": 1.39, "CU": 1.40,
}


def fibonacci_sphere(n: int = 240) -> np.ndarray:
    i = np.arange(n, dtype=float)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    return np.column_stack((np.cos(theta) * radius, y, np.sin(theta) * radius))


SPHERE = fibonacci_sphere()


def parse_pdb(path: Path):
    atoms = []
    cysteine_sg = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            atom_name = line[12:16].strip()
            residue = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                plddt = float(line[60:66])
            except ValueError:
                continue
            key = (chain, resseq, atom_name)
            if key in seen:
                continue
            seen.add(key)
            element = line[76:78].strip().upper() or re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()
            atom = {
                "chain": chain, "resseq": resseq, "residue": residue, "atom": atom_name,
                "coord": xyz, "plddt": plddt, "element": element,
            }
            atoms.append(atom)
            if residue == "CYS" and atom_name == "SG":
                cysteine_sg.append(atom)
    return atoms, cysteine_sg


def sg_sasa(atoms, target, probe: float = 1.4) -> float:
    """Approximate sulfur-atom SASA (A^2) with a Shrake-Rupley point test."""
    center = target["coord"]
    target_radius = VDW.get(target["element"], 1.7) + probe
    points = center + SPHERE * target_radius
    occluded = np.zeros(len(points), dtype=bool)
    for atom in atoms:
        if atom is target:
            continue
        blocker_radius = VDW.get(atom["element"], 1.7) + probe
        if np.linalg.norm(atom["coord"] - center) > target_radius + blocker_radius:
            continue
        squared = np.sum((points - atom["coord"]) ** 2, axis=1)
        occluded |= squared < blocker_radius ** 2
        if occluded.all():
            break
    accessible_fraction = 1.0 - float(occluded.mean())
    return accessible_fraction * 4.0 * math.pi * target_radius ** 2


def rank_from_name(path: Path) -> int:
    match = re.search(r"_rank_(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def model_name_from_path(path: Path) -> str:
    match = re.search(r"(alphafold2[^_]*_model_\d+_seed_\d+)", path.name)
    return match.group(1) if match else "unknown"


def paired_score_path(pdb_path: Path) -> Path:
    return pdb_path.with_name(pdb_path.name.replace("_unrelaxed_", "_scores_").replace(".pdb", ".json"))


def pair_pae(pae, i: int, j: int) -> float:
    return float((pae[i - 1][j - 1] + pae[j - 1][i - 1]) / 2.0)


def analyze_model(pdb_path: Path, manifest_row: dict):
    score_path = paired_score_path(pdb_path)
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score JSON for {pdb_path.name}")
    score = json.loads(score_path.read_text())
    atoms, cysteines = parse_pdb(pdb_path)
    pae = score["pae"]
    plddt = score["plddt"]
    if len(pae) != len(plddt):
        raise ValueError(f"PAE/pLDDT length mismatch: {pdb_path.name}")
    if len(plddt) != int(manifest_row["sequence_length"]):
        raise ValueError(
            f"Sequence/model length mismatch for {pdb_path.name}: "
            f"{len(plddt)} vs {manifest_row['sequence_length']}"
        )

    for atom in cysteines:
        atom["sasa"] = sg_sasa(atoms, atom)

    pairs = []
    for left_index in range(len(cysteines)):
        for right_index in range(left_index + 1, len(cysteines)):
            left = cysteines[left_index]
            right = cysteines[right_index]
            i, j = sorted((left["resseq"], right["resseq"]))
            distance = float(np.linalg.norm(left["coord"] - right["coord"]))
            pae_value = pair_pae(pae, i, j)
            min_local_plddt = min(float(plddt[i - 1]), float(plddt[j - 1]))
            qc = min_local_plddt >= LOCAL_PLDDT_MIN and pae_value <= PAIR_PAE_MAX
            pairs.append({
                "sequence_id": manifest_row["prediction_id"],
                "rank": rank_from_name(pdb_path),
                "model_name": model_name_from_path(pdb_path),
                "cys_i": i,
                "cys_j": j,
                "pair_key": f"C{i}-C{j}",
                "sequence_separation": j - i,
                "is_cxxc_pair": (j - i == 3),
                "sg_distance_A": distance,
                "plddt_i": float(plddt[i - 1]),
                "plddt_j": float(plddt[j - 1]),
                "min_local_plddt": min_local_plddt,
                "pair_pae_A": pae_value,
                "sg_sasa_i_A2": float(left["sasa"]),
                "sg_sasa_j_A2": float(right["sasa"]),
                "mean_pair_sg_sasa_A2": float((left["sasa"] + right["sasa"]) / 2.0),
                "passes_local_qc": qc,
                "possible_disulfide": distance <= DISULFIDE_MAX,
                "passes_6A": qc and DISTANCE_LOWER <= distance <= 6.0,
                "passes_8A": qc and DISTANCE_LOWER <= distance <= 8.0,
                "passes_10A": qc and DISTANCE_LOWER <= distance <= 10.0,
            })

    qc_pairs = [p for p in pairs if p["passes_local_qc"]]
    best = min(qc_pairs, key=lambda p: p["sg_distance_A"]) if qc_pairs else None
    return {
        "sequence_id": manifest_row["prediction_id"],
        "batch_folder": pdb_path.parent.name,
        "rank": rank_from_name(pdb_path),
        "model_name": model_name_from_path(pdb_path),
        "sequence_length": int(manifest_row["sequence_length"]),
        "cysteine_count_manifest": int(manifest_row["cysteine_count"]),
        "cysteine_sg_count_model": len(cysteines),
        "cxxc_count": int(manifest_row["cxxc_count"]),
        "ptm": float(score["ptm"]),
        "mean_plddt": float(np.mean(plddt)),
        "fraction_plddt_ge70": float(np.mean(np.asarray(plddt) >= LOCAL_PLDDT_MIN)),
        "n_cys_pairs": len(pairs),
        "n_qc_pairs": len(qc_pairs),
        "best_qc_pair": best["pair_key"] if best else "",
        "best_qc_distance_A": best["sg_distance_A"] if best else np.nan,
        "best_qc_pair_pae_A": best["pair_pae_A"] if best else np.nan,
        "best_qc_min_local_plddt": best["min_local_plddt"] if best else np.nan,
        "best_qc_pair_mean_sg_sasa_A2": best["mean_pair_sg_sasa_A2"] if best else np.nan,
        "any_possible_disulfide": any(p["possible_disulfide"] for p in pairs),
        "any_passes_6A": any(p["passes_6A"] for p in pairs),
        "any_passes_8A": any(p["passes_8A"] for p in pairs),
        "any_passes_10A": any(p["passes_10A"] for p in pairs),
        "pdb_path": str(pdb_path),
        "score_json_path": str(score_path),
    }, pairs


def summarize_protein(group: pd.DataFrame, pairs: pd.DataFrame, manifest_row: dict) -> dict:
    sequence_id = group.iloc[0]["sequence_id"]
    protein_pairs = pairs[pairs["sequence_id"] == sequence_id]
    pair_support = {}
    nonlocal_pair_support = {}
    for threshold in THRESHOLDS:
        flag = f"passes_{int(threshold)}A"
        counts = protein_pairs[protein_pairs[flag]].groupby("pair_key")["rank"].nunique()
        pair_support[threshold] = int(counts.max()) if not counts.empty else 0
        nonlocal_counts = protein_pairs[
            protein_pairs[flag] & (protein_pairs["sequence_separation"] > 3)
        ].groupby("pair_key")["rank"].nunique()
        nonlocal_pair_support[threshold] = int(nonlocal_counts.max()) if not nonlocal_counts.empty else 0

    passing_8 = protein_pairs[protein_pairs["passes_8A"]]
    if not passing_8.empty:
        consensus = (
            passing_8.groupby(["pair_key", "cys_i", "cys_j"], as_index=False)
            .agg(
                support_models=("rank", "nunique"),
                median_distance_A=("sg_distance_A", "median"),
                distance_range_A=("sg_distance_A", lambda s: float(s.max() - s.min())),
                median_pair_pae_A=("pair_pae_A", "median"),
                median_min_local_plddt=("min_local_plddt", "median"),
                median_pair_sg_sasa_A2=("mean_pair_sg_sasa_A2", "median"),
                is_cxxc_pair=("is_cxxc_pair", "max"),
            )
            .sort_values(["support_models", "median_pair_pae_A", "median_distance_A"], ascending=[False, True, True])
        )
        best = consensus.iloc[0]
    else:
        best = None

    passing_nonlocal_8 = protein_pairs[
        protein_pairs["passes_8A"] & (protein_pairs["sequence_separation"] > 3)
    ]
    if not passing_nonlocal_8.empty:
        nonlocal_consensus = (
            passing_nonlocal_8.groupby(["pair_key", "cys_i", "cys_j"], as_index=False)
            .agg(
                support_models=("rank", "nunique"),
                median_distance_A=("sg_distance_A", "median"),
                distance_range_A=("sg_distance_A", lambda s: float(s.max() - s.min())),
                median_pair_pae_A=("pair_pae_A", "median"),
                median_min_local_plddt=("min_local_plddt", "median"),
                median_pair_sg_sasa_A2=("mean_pair_sg_sasa_A2", "median"),
            )
            .sort_values(
                ["support_models", "median_pair_sg_sasa_A2", "median_pair_pae_A"],
                ascending=[False, False, True],
            )
        )
        best_nonlocal = nonlocal_consensus.iloc[0]
    else:
        best_nonlocal = None

    support8 = pair_support[8.0]
    if support8 == 3:
        stability = "3/3 stable"
    elif support8 == 2:
        stability = "2/3 supported"
    elif support8 == 1:
        stability = "1/3 unstable"
    else:
        stability = "0/3 negative/unresolved"

    has_cxxc = int(manifest_row["cxxc_count"]) > 0
    primary_call = support8 >= 2
    if primary_call and not has_cxxc:
        candidate_class = "Motif-negative structure-positive"
    elif primary_call and has_cxxc:
        candidate_class = "Motif-positive structure-positive"
    elif support8 == 1:
        candidate_class = "Geometry unstable"
    elif group["n_qc_pairs"].max() == 0:
        candidate_class = "Low-confidence/unresolved"
    else:
        candidate_class = "Geometry-negative"

    return {
        "sequence_id": sequence_id,
        "r108_gene_id": manifest_row["r108_gene_id"],
        "uniprot_accessions": manifest_row["uniprot_accessions"],
        "mapping_statuses": manifest_row["mapping_statuses"],
        "r108_annotation": manifest_row["r108_annotation"],
        "sequence_length": int(manifest_row["sequence_length"]),
        "cysteine_count": int(manifest_row["cysteine_count"]),
        "cxxc_count": int(manifest_row["cxxc_count"]),
        "models_available": int(group["rank"].nunique()),
        "mean_ptm": float(group["ptm"].mean()),
        "mean_model_plddt": float(group["mean_plddt"].mean()),
        "mean_fraction_plddt_ge70": float(group["fraction_plddt_ge70"].mean()),
        "models_with_any_qc_pair": int((group["n_qc_pairs"] > 0).sum()),
        "max_same_pair_support_6A": pair_support[6.0],
        "max_same_pair_support_8A": pair_support[8.0],
        "max_same_pair_support_10A": pair_support[10.0],
        "max_nonlocal_same_pair_support_6A": nonlocal_pair_support[6.0],
        "max_nonlocal_same_pair_support_8A": nonlocal_pair_support[8.0],
        "max_nonlocal_same_pair_support_10A": nonlocal_pair_support[10.0],
        "stability_8A": stability,
        "primary_8A_consensus_call": primary_call,
        "candidate_class": candidate_class,
        "consensus_pair_8A": best["pair_key"] if best is not None else "",
        "consensus_pair_support_models": int(best["support_models"]) if best is not None else 0,
        "consensus_pair_median_distance_A": float(best["median_distance_A"]) if best is not None else np.nan,
        "consensus_pair_distance_range_A": float(best["distance_range_A"]) if best is not None else np.nan,
        "consensus_pair_median_pae_A": float(best["median_pair_pae_A"]) if best is not None else np.nan,
        "consensus_pair_median_min_plddt": float(best["median_min_local_plddt"]) if best is not None else np.nan,
        "consensus_pair_median_sg_sasa_A2": float(best["median_pair_sg_sasa_A2"]) if best is not None else np.nan,
        "consensus_pair_is_cxxc": bool(best["is_cxxc_pair"]) if best is not None else False,
        "nonlocal_consensus_pair_8A": best_nonlocal["pair_key"] if best_nonlocal is not None else "",
        "nonlocal_consensus_pair_support_models": int(best_nonlocal["support_models"]) if best_nonlocal is not None else 0,
        "nonlocal_consensus_pair_median_distance_A": float(best_nonlocal["median_distance_A"]) if best_nonlocal is not None else np.nan,
        "nonlocal_consensus_pair_distance_range_A": float(best_nonlocal["distance_range_A"]) if best_nonlocal is not None else np.nan,
        "nonlocal_consensus_pair_median_pae_A": float(best_nonlocal["median_pair_pae_A"]) if best_nonlocal is not None else np.nan,
        "nonlocal_consensus_pair_median_min_plddt": float(best_nonlocal["median_min_local_plddt"]) if best_nonlocal is not None else np.nan,
        "nonlocal_consensus_pair_median_sg_sasa_A2": float(best_nonlocal["median_pair_sg_sasa_A2"]) if best_nonlocal is not None else np.nan,
        "possible_disulfide_in_any_model": bool(group["any_possible_disulfide"].any()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(MANIFEST_JSON.read_text())
    manifest = {row["prediction_id"]: row for row in payload["records"]}

    pdb_paths = sorted({
        *RAW.glob("Figure2_ColabFold_batch_*_results*/*_unrelaxed_rank_*.pdb"),
        *RAW.glob("Figure2_ColabFold_final_10_results/*/*_unrelaxed_rank_*.pdb"),
    })
    by_id = defaultdict(list)
    for path in pdb_paths:
        sequence_id = path.name.split("_unrelaxed_rank_")[0]
        by_id[sequence_id].append(path)

    expected_completed = set()
    for path in RAW.glob("Figure2_ColabFold_batch_*_results*/*.done.txt"):
        expected_completed.add(path.name.removesuffix(".done.txt"))
    for path in RAW.glob("Figure2_ColabFold_final_10_results/*/*.done.txt"):
        expected_completed.add(path.name.removesuffix(".done.txt"))

    eligible_ids = {
        row["prediction_id"] for row in payload["records"]
        if row["cys_pair_screen_eligible"] == "Yes"
    }

    qc = {
        "pdb_files_found": len(pdb_paths),
        "unique_sequence_ids_with_pdb": len(by_id),
        "unique_done_markers": len(expected_completed),
        "ids_with_exactly_3_models": sum(len(paths) == 3 for paths in by_id.values()),
        "done_without_pdb": sorted(expected_completed - set(by_id)),
        "pdb_without_manifest": sorted(set(by_id) - set(manifest)),
        "manifest_eligible_not_yet_downloaded": sorted(eligible_ids - set(by_id)),
        "technical_exclusions": TECHNICAL_EXCLUSIONS,
        "analyzable_coverage_fraction": len(by_id) / len(eligible_ids),
    }
    (OUT / "input_file_audit.json").write_text(json.dumps(qc, indent=2) + "\n")
    if qc["pdb_without_manifest"] or qc["done_without_pdb"]:
        raise RuntimeError(f"Input audit failed: {qc}")

    model_rows = []
    pair_rows = []
    for sequence_id, paths in sorted(by_id.items()):
        if len(paths) != 3:
            raise RuntimeError(f"Expected 3 models for {sequence_id}, found {len(paths)}")
        for path in sorted(paths, key=rank_from_name):
            model, pairs = analyze_model(path, manifest[sequence_id])
            model_rows.append(model)
            pair_rows.extend(pairs)

    models = pd.DataFrame(model_rows).sort_values(["sequence_id", "rank"])
    pairs = pd.DataFrame(pair_rows).sort_values(["sequence_id", "rank", "cys_i", "cys_j"])
    proteins = pd.DataFrame([
        summarize_protein(group, pairs, manifest[sequence_id])
        for sequence_id, group in models.groupby("sequence_id", sort=True)
    ]).sort_values("sequence_id")

    sensitivity_rows = []
    for threshold in THRESHOLDS:
        for analysis_set, column in [
            ("All Cys pairs", f"max_same_pair_support_{int(threshold)}A"),
            ("Nonlocal pairs (>3 residues apart)", f"max_nonlocal_same_pair_support_{int(threshold)}A"),
        ]:
            for motif_group, subset in proteins.groupby(np.where(proteins["cxxc_count"] > 0, "CXXC-positive", "CXXC-negative")):
                sensitivity_rows.append({
                    "analysis_set": analysis_set,
                    "upper_distance_A": threshold,
                    "motif_group": motif_group,
                    "protein_count": int((subset[column] >= 2).sum()),
                    "protein_denominator": int(len(subset)),
                    "protein_fraction": float((subset[column] >= 2).mean()),
                })
    sensitivity = pd.DataFrame(sensitivity_rows)

    models.to_csv(OUT / "model_level_metrics.csv", index=False)
    pairs.to_csv(OUT / "all_cys_pair_metrics.csv", index=False)
    proteins.to_csv(OUT / "protein_level_summary.csv", index=False)
    sensitivity.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    final_batch_ids = {
        "MtrunR108HiC_005592.1", "MtrunR108HiC_008618.1", "MtrunR108HiC_009165.1",
        "MtrunR108HiC_009426.1", "MtrunR108HiC_014450.1", "MtrunR108HiC_017009.1",
        "MtrunR108HiC_020073.1", "MtrunR108HiC_026991.1", "MtrunR108HiC_029401.1",
    }
    proteins[proteins["sequence_id"].isin(final_batch_ids)].to_csv(
        OUT / "final_batch_9_protein_summary.csv", index=False
    )
    pd.DataFrame([
        {"sequence_id": sequence_id, **details}
        for sequence_id, details in TECHNICAL_EXCLUSIONS.items()
    ]).to_csv(OUT / "technical_exclusion.csv", index=False)
    workbook_payload = {
        "protein_level_summary": json.loads(proteins.to_json(orient="records")),
        "model_level_metrics": json.loads(models.to_json(orient="records")),
        "threshold_sensitivity": json.loads(sensitivity.to_json(orient="records")),
    }
    (OUT / "workbook_payload.json").write_text(json.dumps(workbook_payload, indent=2) + "\n")

    unresolved = sorted(eligible_ids - set(by_id) - set(TECHNICAL_EXCLUSIONS))
    if unresolved:
        raise RuntimeError(f"Unexpected unresolved eligible proteins: {unresolved}")

    summary = {
        "analysis_state": "Final analyzable set: 93 of 94 eligible proteins; one prespecified technical exclusion",
        "eligible_proteins": len(eligible_ids),
        "proteins_analyzed": int(len(proteins)),
        "models_analyzed": int(len(models)),
        "analyzable_coverage_fraction": len(proteins) / len(eligible_ids),
        "technical_exclusions": TECHNICAL_EXCLUSIONS,
        "proteins_pending_final_batch": len(unresolved),
        "all_models_have_pairwise_pae": bool(models["score_json_path"].notna().all()),
        "proteins_with_same_pair_support_2of3_at_6A": int((proteins["max_same_pair_support_6A"] >= 2).sum()),
        "proteins_with_same_pair_support_2of3_at_8A": int((proteins["max_same_pair_support_8A"] >= 2).sum()),
        "proteins_with_same_pair_support_2of3_at_10A": int((proteins["max_same_pair_support_10A"] >= 2).sum()),
        "proteins_with_nonlocal_same_pair_support_2of3_at_6A": int((proteins["max_nonlocal_same_pair_support_6A"] >= 2).sum()),
        "proteins_with_nonlocal_same_pair_support_2of3_at_8A": int((proteins["max_nonlocal_same_pair_support_8A"] >= 2).sum()),
        "proteins_with_nonlocal_same_pair_support_2of3_at_10A": int((proteins["max_nonlocal_same_pair_support_10A"] >= 2).sum()),
        "motif_negative_consensus_candidates_at_8A": int(((proteins["max_same_pair_support_8A"] >= 2) & (proteins["cxxc_count"] == 0)).sum()),
        "motif_positive_consensus_candidates_at_8A": int(((proteins["max_same_pair_support_8A"] >= 2) & (proteins["cxxc_count"] > 0)).sum()),
        "proteins_with_only_1of3_support_at_8A": int((proteins["max_same_pair_support_8A"] == 1).sum()),
        "proteins_without_local_qc_pair_in_any_model": int((proteins["models_with_any_qc_pair"] == 0).sum()),
        "proteins_with_possible_disulfide_geometry": int(proteins["possible_disulfide_in_any_model"].sum()),
        "operational_rules": {
            "both_cys_plddt_min": LOCAL_PLDDT_MIN,
            "pair_pae_max_A": PAIR_PAE_MAX,
            "distance_lower_A": DISTANCE_LOWER,
            "distance_upper_sensitivity_A": list(THRESHOLDS),
            "primary_consensus": "same residue pair passes in at least 2 of 3 models at 8 A upper bound",
        },
    }
    (OUT / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
