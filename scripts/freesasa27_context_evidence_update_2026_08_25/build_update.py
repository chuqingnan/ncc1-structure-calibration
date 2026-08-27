from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "outputs" / "freesasa27_context_evidence_update_2026_08_25"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work" / "validated_endpoint_sensitivity_2026_08_25"))
from run_sensitivity_analysis import SPECS, positive_ids  # noqa: E402


PAIR_PATH = ROOT / "outputs" / "validated_endpoint_sensitivity_2026_08_25" / "donor_pair_metrics_with_freesasa.csv.gz"
MASTER_PATH = ROOT / "outputs" / "candidate_background_unified" / "protein_level_master.csv"
OLD_CONTEXT_PATH = ROOT / "outputs" / "functional_context_quantification_31" / "functional_context_31_quantified.csv"
FINAL_MATRIX_PATH = ROOT / "outputs" / "final_evidence_hierarchy_matrix" / "final_evidence_hierarchy_matrix_110_scope_completed.csv"

CONTEXT_ORDER = [
    "Canonical/conserved enzyme-fold coincidence",
    "Native metal/catalytic-site alternative",
    "Scaffold/proteostasis context",
    "Compartment-incompatible/indirect",
    "Annotation conflict/QC",
]

CORE = {
    "MtrunR108HiC_001767.1",
    "MtrunR108HiC_008307.1",
    "MtrunR108HiC_009426.1",
}
WATCH = {
    "MtrunR108HiC_008596.1",
    "MtrunR108HiC_008599.1",
}
CALIBRATION_CANDIDATE = {
    "MtrunR108HiC_012482.1",
    "MtrunR108HiC_005650.1",
    "MtrunR108HiC_009165.1",
    "MtrunR108HiC_020175.1",
}


def wilson(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = x / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, center - half), min(1, center + half)


def recurrent_site_table(pairs: pd.DataFrame) -> pd.DataFrame:
    spec = SPECS["Frozen"]
    passing = pairs[
        pairs["sulfur_anchored"].astype(bool)
        & pairs["donor_distance_A"].between(2.5, 5.0, inclusive="both")
        & pairs["min_local_plddt"].ge(70)
        & pairs["pair_pae_A"].le(10)
        & pairs["sequence_separation"].ge(spec["sequence_separation_min"])
        & pairs["mean_donor_sasa_A2"].ge(spec["mean_sasa_min_A2"])
    ].copy()
    site = (
        passing.groupby(["group", "protein_id", "pair_key", "donor_combo", "sequence_separation"], as_index=False)
        .agg(
            support_models=("rank", "nunique"),
            median_distance_A=("donor_distance_A", "median"),
            min_distance_A=("donor_distance_A", "min"),
            max_distance_A=("donor_distance_A", "max"),
            median_min_plddt=("min_local_plddt", "median"),
            median_pair_pae_A=("pair_pae_A", "median"),
            median_mean_freesasa_A2=("mean_donor_sasa_A2", "median"),
            min_mean_freesasa_A2=("mean_donor_sasa_A2", "min"),
            max_mean_freesasa_A2=("mean_donor_sasa_A2", "max"),
        )
    )
    return site[site["support_models"].ge(2)].sort_values(
        ["group", "protein_id", "support_models", "median_mean_freesasa_A2", "median_pair_pae_A"],
        ascending=[True, True, False, False, True],
    )


def added_020073_row(columns: list[str], master: pd.DataFrame) -> dict:
    m = master.loc[master["protein_id"].eq("MtrunR108HiC_020073.1")].iloc[0]
    row = {c: np.nan for c in columns}
    row.update({
        "group": "Candidate",
        "protein_id": m["protein_id"],
        "uniprot_accessions": m["uniprot_accessions"],
        "r108_annotation": m["r108_annotation"],
        "sequence_length": int(m["sequence_length"]),
        "cysteine_count": int(m["cysteine_count"]),
        "methionine_count": int(m["methionine_count"]),
        "histidine_count": int(m["histidine_count"]),
        "bait_psm": m["bait_psm"],
        "control_psm": m["control_psm"],
        "pull_down_log2_psm_ratio": m["pull_down_log2_psm_ratio"],
        "mtexpress_median_log2fc_nodule_vs_root": m["median_project_log2fc_nodule_vs_root"],
        "primary_class": "Macromolecular scaffold/WD40-fold coincidence",
        "site_interpretation": "The recurrent sulfur-anchored geometry lies in the DDB1/RSE1 WD40 beta-propeller scaffold; it is a fold-context coincidence and not evidence for a dedicated Cu-receiving module.",
        "native_alternative": "Conserved DDB1/RSE1 WD40 scaffold architecture",
        "final_disposition": "Exclude from direct-client interpretation; retain as a large WD40-scaffold false-positive calibration case.",
        "confidence": "Moderate-high",
        "rationale": "Exact R108 InterProScan identifies three DDB1/RSE1 beta-propeller regions; localization is compatible but residual pull-down specificity is background-like (52.7th background percentile).",
        "location": m["deeploc2_1_localizations"],
        "location_compatibility": m["location_compatibility_preliminary"],
        "signalp6_prediction": m["signalp6_prediction"],
        "deeptmhmm_tm_count": m["deeptmhmm_tm_count"],
        "domain_context": m["exact_interpro_all_matches"],
        "annotation_conflict": m["annotation_conflict_status"],
        "context_source": "Exact R108 DeepLoc/SignalP/DeepTMHMM/InterProScan plus frozen residual-specificity model",
        "source_urls": m["expression_source_url"],
        "bait_enriched": bool(m["bait_psm"] > m["control_psm"]),
        "control_detected": bool(m["control_found"]),
        "direct_client_supported": False,
        "client_claim_status": "No direct-client claim; geometry is compatible but explained by native scaffold context.",
        "disposition_bucket": "Exclude from direct-client interpretation",
        "context_bucket": "Scaffold/proteostasis context",
        "context_category": "Scaffold/proteostasis context",
        "context_label": "Scaffold/proteostasis",
        "flag_native_metal_or_catalytic": False,
        "flag_conserved_or_canonical_fold": False,
        "flag_compartment_incompatible_or_indirect": False,
        "flag_scaffold_or_proteostasis": True,
        "flag_annotation_conflict": False,
        "retained_secondary_interaction_hypothesis": False,
        "direct_NCC1_Cu_client_established": False,
    })
    return row


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    old = pd.read_csv(OLD_CONTEXT_PATH)
    sites = recurrent_site_table(pairs)
    positives = positive_ids(pairs, SPECS["Frozen"])
    complete = master[master["structure_analysis_included"].astype(bool)]
    expected = set(complete.loc[complete["protein_id"].isin(positives), "protein_id"])
    assert expected == positives
    assert len(positives) == 27

    old_ids = set(old["protein_id"])
    transition_ids = sorted(old_ids | positives)
    transition = pd.DataFrame({"protein_id": transition_ids})
    group_map = master.set_index("protein_id")["group"].to_dict()
    transition["group"] = transition["protein_id"].map(group_map)
    transition["approximate_endpoint_31"] = transition["protein_id"].isin(old_ids)
    transition["freesasa_endpoint_27"] = transition["protein_id"].isin(positives)
    transition["transition"] = np.select(
        [
            transition["approximate_endpoint_31"] & transition["freesasa_endpoint_27"],
            transition["approximate_endpoint_31"] & ~transition["freesasa_endpoint_27"],
            ~transition["approximate_endpoint_31"] & transition["freesasa_endpoint_27"],
        ],
        ["Retained", "Removed by FreeSASA", "Added by FreeSASA"],
        default="Unchanged negative",
    )
    transition.to_csv(OUT / "endpoint_31_to_27_transition.csv", index=False)

    context = old[old["protein_id"].isin(positives)].copy()
    add = pd.DataFrame([added_020073_row(list(old.columns), master)])
    context = pd.concat([context, add], ignore_index=True)
    site_groups = {pid: x.copy() for pid, x in sites.groupby("protein_id")}
    for idx, row in context.iterrows():
        x = site_groups[row["protein_id"]]
        best = x.iloc[0]
        context.loc[idx, "all_primary_sites"] = "; ".join(
            f"{r.pair_key} ({r.donor_combo}; {int(r.support_models)}/3; d={r.median_distance_A:.2f} A; FreeSASA={r.median_mean_freesasa_A2:.2f} A2)"
            for r in x.itertuples(index=False)
        )
        context.loc[idx, "best_primary_pair"] = best["pair_key"]
        context.loc[idx, "best_primary_donor_combo"] = best["donor_combo"]
        context.loc[idx, "best_primary_support_models"] = int(best["support_models"])
        context.loc[idx, "best_primary_distance_A"] = best["median_distance_A"]
        context.loc[idx, "best_primary_min_plddt"] = best["median_min_plddt"]
        context.loc[idx, "best_primary_pae_A"] = best["median_pair_pae_A"]
        context.loc[idx, "best_primary_mean_sasa_A2"] = best["median_mean_freesasa_A2"]
    context["context_order"] = context["context_category"].map({x: i + 1 for i, x in enumerate(CONTEXT_ORDER)})
    disposition_order = [
        "Secondary interaction candidate (transfer unproven)",
        "Negative-calibration/native-site benchmark",
        "Exclude from direct-client interpretation",
        "Background benchmark",
    ]
    context["disposition_order"] = context["disposition_bucket"].map({x: i + 1 for i, x in enumerate(disposition_order)})
    context = context.sort_values(["group", "context_order", "disposition_order", "protein_id"])
    context.to_csv(OUT / "formal27_freesasa_functional_context.csv", index=False)
    sites[sites["protein_id"].isin(positives)].to_csv(OUT / "formal27_recurrent_sites_freesasa.csv", index=False)

    context_rows = []
    for cat in CONTEXT_ORDER:
        row = {"context_category": cat}
        for group, denom in [("Candidate", 17), ("Background", 10), ("Overall", 27)]:
            mask = context["context_category"].eq(cat)
            if group != "Overall":
                mask &= context["group"].eq(group)
            n = int(mask.sum())
            lo, hi = wilson(n, denom)
            prefix = group.lower()
            row.update({f"{prefix}_n": n, f"{prefix}_total": denom, f"{prefix}_percent": 100*n/denom,
                        f"{prefix}_wilson_low_percent": 100*lo, f"{prefix}_wilson_high_percent": 100*hi})
        context_rows.append(row)
    pd.DataFrame(context_rows).to_csv(OUT / "formal27_functional_context_summary.csv", index=False)

    disp = (
        context.groupby(["group", "disposition_bucket"], as_index=False)
        .size().rename(columns={"size": "n"})
    )
    disp["group_total"] = disp["group"].map({"Candidate": 17, "Background": 10})
    disp["percent"] = 100 * disp["n"] / disp["group_total"]
    disp.to_csv(OUT / "formal27_disposition_summary.csv", index=False)

    final = pd.read_csv(FINAL_MATRIX_PATH, low_memory=False)
    frozen94 = final["population"].eq("Frozen94")
    frozen_positive = final["protein_id"].isin(positives)
    final.loc[frozen94, "sulfur_primary_geometry"] = frozen_positive[frozen94]
    best_lookup = sites.sort_values(["protein_id", "support_models", "median_mean_freesasa_A2"], ascending=[True, False, False]).drop_duplicates("protein_id").set_index("protein_id")
    for idx, row in final[frozen94].iterrows():
        pid = row["protein_id"]
        if pid in positives:
            b = best_lookup.loc[pid]
            final.loc[idx, "best_pair"] = b["pair_key"]
            final.loc[idx, "model_support"] = int(b["support_models"])
        elif bool(row["sulfur_primary_geometry"]) is False:
            final.loc[idx, "best_pair"] = np.nan
            final.loc[idx, "model_support"] = 0

    idx_020073 = final["protein_id"].eq("MtrunR108HiC_020073.1")
    final.loc[idx_020073, "integrated_class_code"] = "N1"
    final.loc[idx_020073, "integrated_class"] = "Geometry-positive with stronger alternative explanation"
    final.loc[idx_020073, "functional_context"] = "DDB1/RSE1 WD40 beta-propeller scaffold; recurrent sulfur geometry is a native fold-context coincidence."
    final.loc[idx_020073, "final_decision"] = "Use as a scaffold-fold false-positive calibration example; exclude from the direct-client set."
    final.loc[idx_020073, "claim_allowed"] = "Recurrent sulfur-anchored geometry within a DDB1/RSE1 WD40 scaffold; NCC1 pull-down association observed."
    final.loc[idx_020073, "claim_prohibited"] = "Do not call a Cu-binding site, Cu-transfer acceptor, validated NCC1 client, or direct NCC1 interface."

    annexin = final["protein_id"].eq("MtrunR108HiC_020175.1")
    final.loc[annexin, "core_status"] = "Background calibration; not watchlist"
    final.loc[annexin, "functional_context"] = "Annexin native Ca2+/membrane-binding fold provides a stronger alternative explanation; Cu specificity is untested."
    final.loc[annexin, "final_decision"] = "Use as a native-structure/background calibration example; do not prioritize for further structure-only work."

    def four_class(pid: str) -> str:
        if pid in CORE:
            return "Core candidate"
        if pid in WATCH:
            return "Observation/watchlist"
        if pid in CALIBRATION_CANDIDATE:
            return "Background calibration"
        return "Exclude/no promotion"

    final["frozen_four_class"] = final["protein_id"].map(four_class)
    final["freesasa_endpoint_version"] = np.where(final["population"].eq("Frozen94"), "FreeSASA 2.2.1; probe 1.4 A; endpoint updated 2026-08-25", "Scope16 result retained; outside 167-protein formal endpoint")
    class_order = {"P1": 1, "P2": 2, "S1": 3, "S2": 4, "N1": 5, "N2": 6, "Q1": 7, "S3": 8, "S4": 9, "N3": 10}
    final["_order"] = final["integrated_class_code"].map(class_order).fillna(99)
    final = final.sort_values(["_order", "integrated_rank", "protein_id"]).drop(columns="_order").reset_index(drop=True)
    final["integrated_rank"] = np.arange(1, len(final) + 1)
    final.to_csv(OUT / "final_evidence_hierarchy_matrix_110_freesasa27.csv", index=False)

    formal_four = context[["group", "protein_id", "r108_annotation", "disposition_bucket", "context_category"]].copy()
    formal_four["frozen_four_class"] = np.select(
        [
            formal_four["protein_id"].isin(CORE),
            formal_four["protein_id"].isin(CALIBRATION_CANDIDATE),
            formal_four["group"].eq("Background"),
        ],
        ["Core candidate", "Background calibration", "Background calibration"],
        default="Exclude/no promotion",
    )
    scope = final[final["protein_id"].isin(WATCH | {"MtrunR108HiC_020175.1"})][["population", "protein_id", "annotation", "frozen_four_class"]].copy()
    scope = scope.rename(columns={"population": "group", "annotation": "r108_annotation"})
    scope["disposition_bucket"] = np.where(scope["protein_id"].isin(WATCH), "Observation/watchlist", "Background calibration")
    scope["context_category"] = np.where(scope["protein_id"].isin(WATCH), "NTF2-like family sensitivity-only H-H geometry", "Native annexin/Ca-binding alternative")
    frozen_lists = pd.concat([formal_four, scope], ignore_index=True).sort_values(["frozen_four_class", "group", "protein_id"])
    frozen_lists.to_csv(OUT / "refrozen_four_class_lists.csv", index=False)

    audit = {
        "date": "2026-08-25",
        "endpoint": "sulfur-anchored donor pair; 2.5-5.0 A; pLDDT>=70; pair PAE<=10 A; sequence separation>=10; mean donor FreeSASA>=5 A2; same pair in >=2/3 models",
        "software": "FreeSASA 2.2.1",
        "probe_radius_A": 1.4,
        "old_positive_n": len(old_ids),
        "new_positive_n": len(positives),
        "candidate_positive_n": int((context["group"] == "Candidate").sum()),
        "background_positive_n": int((context["group"] == "Background").sum()),
        "removed": transition.loc[transition["transition"].eq("Removed by FreeSASA"), "protein_id"].tolist(),
        "added": transition.loc[transition["transition"].eq("Added by FreeSASA"), "protein_id"].tolist(),
        "core_ids": sorted(CORE),
        "watchlist_ids": sorted(WATCH),
        "annexin_status": "background calibration; not watchlist",
    }
    (OUT / "update_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    payload = {
        "audit": audit,
        "formal27": json.loads(context.replace({np.nan: None}).to_json(orient="records")),
        "sites": json.loads(sites[sites["protein_id"].isin(positives)].replace({np.nan: None}).to_json(orient="records")),
        "context_summary": json.loads(pd.read_csv(OUT / "formal27_functional_context_summary.csv").replace({np.nan: None}).to_json(orient="records")),
        "disposition_summary": json.loads(disp.replace({np.nan: None}).to_json(orient="records")),
        "transition": json.loads(transition.replace({np.nan: None}).to_json(orient="records")),
        "final110": json.loads(final.replace({np.nan: None}).to_json(orient="records")),
        "frozen_lists": json.loads(frozen_lists.replace({np.nan: None}).to_json(orient="records")),
        "group_sensitivity": json.loads(pd.read_csv(ROOT / "outputs" / "validated_endpoint_sensitivity_2026_08_25" / "threshold_sensitivity_group_comparisons_freesasa.csv").replace({np.nan: None}).to_json(orient="records")),
        "validated_recall": json.loads(pd.read_csv(ROOT / "outputs" / "validated_endpoint_sensitivity_2026_08_25" / "validated_case_endpoint_matrix.csv").replace({np.nan: None}).to_json(orient="records")),
    }
    (OUT / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
