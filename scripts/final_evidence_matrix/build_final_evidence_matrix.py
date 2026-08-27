#!/usr/bin/env python3
"""Build the final evidence-hierarchy matrix for the frozen NCC1 candidate set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "outputs" / "final_evidence_hierarchy_matrix"
OUT.mkdir(parents=True, exist_ok=True)

RANKING = ROOT / "outputs" / "sulfur_context_residual_specificity" / "candidate_residual_specificity_ranking.csv"
CONTEXT = ROOT / "outputs" / "formal31_functional_context_triage" / "formal31_functional_context_triage.csv"
ROBUST = ROOT / "outputs" / "robustness_5seed5model_frozen6" / "complete_analysis_150models" / "candidate_summary.csv"
MANIFEST = ROOT / "work" / "figure2" / "Figure2_candidate_manifest.json"
RAW_S3 = ROOT / "NCC1 supplemental tables" / "Table S3.xlsx"
QC_S3 = ROOT / "Table S3_R108_mapping_QC.xlsx"


CLASS_RULES = [
    {
        "code": "P1",
        "label": "Secondary interaction candidate; transfer unproven",
        "rule": "Formal context audit disposition is 'Secondary interaction candidate (transfer unproven)'.",
        "manuscript_role": "Main-text secondary candidate; never call an NCC1 Cu client",
    },
    {
        "code": "P2",
        "label": "Pull-down-prioritized; formal geometry unresolved",
        "rule": "Residual tier R1/R2 after removing P1, technical/QC, compartment-incompatible and formal geometry-alternative proteins.",
        "manuscript_role": "Supplementary association-prioritized candidate",
    },
    {
        "code": "N1",
        "label": "Geometry-positive with stronger alternative explanation",
        "rule": "Frozen formal geometry positive, but context audit supports native/catalytic, conserved-fold, scaffold or failed-interface explanation.",
        "manuscript_role": "Negative-calibration or false-positive benchmark",
    },
    {
        "code": "N2",
        "label": "Compartment-incompatible or indirect",
        "rule": "Modeled localization is incompatible with direct cytosolic NCC1 transfer, regardless of pull-down or geometry.",
        "manuscript_role": "Exclude from direct-client interpretation",
    },
    {
        "code": "Q1",
        "label": "Technical or annotation uncertainty",
        "rule": "Frozen structure unavailable/failed or formal annotation-conflict/QC class.",
        "manuscript_role": "QC/uncertainty record; do not treat as negative",
    },
    {
        "code": "N3",
        "label": "Background-like or insufficient current evidence",
        "rule": "All remaining Table S4 candidates: no elevated residual and no frozen formal geometry/context support.",
        "manuscript_role": "Full matrix only; no client or metal-site claim",
    },
]


def corrected_proteomics() -> pd.DataFrame:
    raw = pd.read_excel(RAW_S3, sheet_name="Proteins", header=1)
    qc = pd.read_excel(QC_S3, sheet_name="Proteins", header=1)
    for col in ["Unnamed: 0", "Data Base", "Accession", "Description"]:
        if not raw[col].fillna("").astype(str).eq(qc[col].fillna("").astype(str)).all():
            raise RuntimeError(f"Table S3 identity mismatch in {col}")
    mapped = qc[["Data Base", "Accession", "R108 protein ID", "Mapping status"]].copy()
    mapped["bait"] = pd.to_numeric(raw["# PSMs:\nBait"], errors="coerce")
    mapped["control"] = pd.to_numeric(raw["# PSMs:\nControl (no bait)"], errors="coerce")
    mapped = mapped[
        mapped["Data Base"].astype(str).str.contains("Medicago", case=False, na=False)
        & mapped["R108 protein ID"].notna()
    ].copy()
    mapped["protein_id"] = mapped["R108 protein ID"].astype(str).str.strip()
    rows = []
    for protein_id, group in mapped.groupby("protein_id", sort=False):
        control = group["control"].max()
        rows.append(
            {
                "protein_id": protein_id,
                "bait_psm_corrected": float(group["bait"].max()),
                "control_psm_corrected": float(0 if pd.isna(control) else control),
                "control_detected": bool(pd.notna(control) and control > 0),
                "table_s3_accessions": "; ".join(sorted(set(group["Accession"].dropna().astype(str)))),
            }
        )
    return pd.DataFrame(rows)


ranking = pd.read_csv(RANKING, low_memory=False)
if len(ranking) != 94 or ranking["protein_id"].nunique() != 94:
    raise RuntimeError("Expected 94 unique candidates in the frozen residual-ranking population")

context = pd.read_csv(CONTEXT)
context = context[context["group"].eq("Candidate")].copy()
if len(context) != 18 or context["protein_id"].nunique() != 18:
    raise RuntimeError("Expected 18 candidate rows in the formal 31-protein context audit")
context_columns = [
    "protein_id", "primary_class", "site_interpretation", "native_alternative",
    "final_disposition", "confidence", "rationale", "location", "location_compatibility",
    "domain_context", "annotation_conflict", "context_source", "source_urls",
    "direct_client_supported", "client_claim_status", "disposition_bucket", "context_bucket",
]
context = context[context_columns].rename(
    columns={
        "primary_class": "formal_primary_class",
        "disposition_bucket": "formal_disposition_bucket",
        "context_bucket": "formal_context_bucket",
    }
)

robust = pd.read_csv(ROBUST)
if len(robust) != 6 or robust["protein_id"].nunique() != 6:
    raise RuntimeError("Expected six frozen 5-seed/5-model candidates")
robust = robust[
    [
        "protein_id", "n_models_analyzed", "n_seeds_analyzed", "n_qualifying",
        "qualifying_fraction", "seeds_with_any_qualifying_model", "reproducibility_tier",
        "median_donor_distance_A", "median_min_pair_plddt", "median_pair_pae_A",
        "median_mean_donor_sasa_A2",
    ]
].rename(columns={"reproducibility_tier": "monomer_reproducibility_tier"})

data = ranking.merge(context, on="protein_id", how="left", validate="one_to_one", suffixes=("", "_formal"))
data = data.merge(robust, on="protein_id", how="left", validate="one_to_one")

data["strict_sequence_motif"] = data["motif_mxcxxc"].fillna(False) | data["motif_cxxc"].fillna(False)
data["broad_sequence_window"] = data["motif_any_frozen"].fillna(False)
data["formal_context_reviewed"] = data["formal_context_bucket"].notna()
data["residual_elevated"] = data["residual_priority_tier"].str.startswith(("R1", "R2"), na=False)
data["nodule_expression_context"] = np.select(
    [
        data["expression_mapping_status"].str.startswith("No single", na=False),
        data["nodule_median_tmm"].lt(1),
        data["median_project_log2fc_nodule_vs_root"].ge(1),
        data["median_project_log2fc_nodule_vs_root"].ge(0),
    ],
    [
        "Mapping unavailable",
        "Very low nodule abundance (<1 TMM)",
        "Nodule-enriched (median project log2FC >=1)",
        "Nodule-detected; median project log2FC >=0",
    ],
    default="Nodule-detected; median project log2FC <0",
)

secondary = data["formal_disposition_bucket"].eq("Secondary interaction candidate (transfer unproven)")
technical = (~data["structure_analysis_included"].fillna(False)) | data["formal_context_bucket"].eq("Annotation conflict/QC")
incompatible = data["compartment_model"].eq("Incompatible")
geometry = data["primary_geometry_positive"].fillna(False)
elevated = data["residual_elevated"]

data["final_evidence_class_code"] = np.select(
    [secondary, technical, incompatible, geometry, elevated],
    ["P1", "Q1", "N2", "N1", "P2"],
    default="N3",
)
rules = pd.DataFrame(CLASS_RULES).set_index("code")
data["final_evidence_class"] = data["final_evidence_class_code"].map(rules["label"])
data["manuscript_role"] = data["final_evidence_class_code"].map(rules["manuscript_role"])
data["classification_trigger"] = data["final_evidence_class_code"].map(rules["rule"])

data["monomer_25model_status"] = "Not selected for 5-seed/5-model monomer robustness"
has_robust = data["n_models_analyzed"].notna()
data.loc[has_robust, "monomer_25model_status"] = data.loc[has_robust].apply(
    lambda row: (
        f"{int(row['n_qualifying'])}/{int(row['n_models_analyzed'])} qualifying; "
        f"{int(row['n_seeds_analyzed'])} seeds; {row['monomer_reproducibility_tier']}"
    ),
    axis=1,
)
data["ncc1_multimer_status"] = "Not tested; monomer geometry cannot establish an NCC1 interface"
data.loc[data["protein_id"].eq("MtrunR108HiC_012482.1"), "ncc1_multimer_status"] = (
    "Negative computational triage: 25/25 models complete; median ipTM 0.150; high cross-chain PAE; nonreproducible candidate interface"
)

data["claim_allowed"] = data["final_evidence_class_code"].map(
    {
        "P1": "NCC1-associated secondary interaction candidate with reproducible monomer geometry; Cu binding/transfer unproven",
        "P2": "Pull-down-prioritized NCC1-associated protein; no frozen formal Cu-compatible geometry",
        "N1": "Reproducible donor geometry that calibrates native-site/fold/scaffold false positives",
        "N2": "Pull-down association may be indirect; direct cytosolic NCC1 transfer is not supported",
        "Q1": "Evidence unresolved because of technical exclusion or annotation conflict",
        "N3": "Table S4-associated protein with insufficient current evidence for structural or client prioritization",
    }
)
data["claim_prohibited"] = (
    "Do not call a validated NCC1 client, Cu-binding protein, Cu-transfer acceptor, or direct NCC1 interface"
)
data["next_action"] = data["final_evidence_class_code"].map(
    {
        "P1": "Carry to main-text candidate matrix; use functional-context caveat; orthogonal experiment only if future resources permit",
        "P2": "Retain in supplementary priority set; refine nodule coexpression/function before any structural follow-up",
        "N1": "Use as negative-calibration example showing stable geometry is not client evidence",
        "N2": "Exclude from direct-client set; retain only as compartment/indirect co-purification example",
        "Q1": "Resolve annotation or structure coverage before reclassification; never impute a negative result",
        "N3": "Retain in full matrix; no further structure computation is currently justified",
    }
)

class_order = {"P1": 1, "P2": 2, "N1": 3, "N2": 4, "Q1": 5, "N3": 6}
data["class_order"] = data["final_evidence_class_code"].map(class_order)
data = data.sort_values(
    ["class_order", "candidate_rank_median_models", "protein_id"],
    na_position="last",
    kind="stable",
).reset_index(drop=True)
data.insert(0, "matrix_rank", np.arange(1, len(data) + 1))

matrix_columns = [
    "matrix_rank", "final_evidence_class_code", "final_evidence_class", "manuscript_role", "classification_trigger",
    "protein_id", "uniprot_accessions", "r108_annotation", "sequence_length",
    "cysteine_count", "methionine_count", "histidine_count",
    "bait_psm_corrected", "control_psm_corrected", "pull_down_log2_psm_ratio_corrected",
    "residual_priority_tier", "residual_specificity_log2_primary_counts_context",
    "background_residual_percentile_primary_counts_context",
    "empirical_upper_tail_p_primary_counts_context", "empirical_bh_q_primary_counts_context",
    "candidate_rank_median_models", "strict_sequence_motif", "motif_mxcxxc", "motif_cxxc",
    "broad_sequence_window", "structure_analysis_included", "structure_exclusion_reason",
    "primary_geometry_positive", "best_primary_pair", "best_primary_donor_combo",
    "best_primary_support_models", "best_primary_distance_A", "best_primary_min_plddt",
    "best_primary_pae_A", "best_primary_mean_sasa_A2", "monomer_25model_status",
    "qualifying_fraction", "median_donor_distance_A", "median_min_pair_plddt",
    "median_pair_pae_A", "median_mean_donor_sasa_A2", "ncc1_multimer_status",
    "deeploc2_1_localizations", "compartment_model", "location_compatibility_preliminary",
    "signalp6_prediction", "deeptmhmm_tm_count", "nodule_expression_context",
    "nodule_median_tmm", "median_project_log2fc_nodule_vs_root", "projects_log2fc_gt0",
    "projects_log2fc_ge1", "formal_context_reviewed", "formal_context_bucket",
    "formal_primary_class", "site_interpretation", "native_alternative", "domain_context",
    "annotation_conflict", "final_disposition", "confidence", "rationale",
    "direct_client_supported", "client_claim_status", "claim_allowed", "claim_prohibited",
    "next_action", "source_urls",
]
matrix = data[matrix_columns].copy()
matrix.to_csv(OUT / "final_evidence_hierarchy_matrix_94.csv", index=False)

# Audit the 16 Table S4 R108 proteins that were outside the frozen >=2-Cys structure population.
manifest = json.loads(MANIFEST.read_text())
manifest_records = pd.DataFrame(manifest["records"])
if len(manifest_records) != 110 or manifest_records["prediction_id"].nunique() != 110:
    raise RuntimeError("Expected 110 unique R108 IDs in the Table S4 mapping manifest")
outside = manifest_records[~manifest_records["prediction_id"].isin(set(ranking["protein_id"]))].copy()
if len(outside) != 16:
    raise RuntimeError(f"Expected 16 Table S4 IDs outside the frozen 94, found {len(outside)}")
outside = outside.merge(corrected_proteomics(), left_on="prediction_id", right_on="protein_id", how="left", validate="one_to_one")
outside["scope_status"] = "Outside frozen 94-protein structure/residual population"
outside["interpretation"] = (
    "Excluded by the early >=2-Cys eligibility rule; structure geometry and residual specificity are unassessed, not negative"
)
outside["annotation_qc_note"] = ""
outside.loc[outside["r108_annotation"].str.contains("NTF2", case=False, na=False), "annotation_qc_note"] = (
    "NTF2-like domain superfamily protein; do not label as PR1/CAP"
)
outside["recommended_handling"] = (
    "Retain in scope audit; a future unbiased Cys/Met/His structure-first screen must not require >=2 cysteines"
)
outside_columns = [
    "prediction_id", "uniprot_accessions", "r108_annotation", "sequence_length", "cysteine_count",
    "mapping_statuses", "bait_psm_corrected", "control_psm_corrected", "control_detected",
    "scope_status", "exclusion_reason", "interpretation", "annotation_qc_note", "recommended_handling",
]
outside[outside_columns].sort_values("prediction_id").to_csv(OUT / "table_s4_outside_frozen94_scope_16.csv", index=False)

class_counts = (
    matrix.groupby(["final_evidence_class_code", "final_evidence_class", "manuscript_role"], sort=False)
    .size().reset_index(name="n_proteins")
)
class_counts["fraction_of_94"] = class_counts["n_proteins"] / 94
class_counts["sort"] = class_counts["final_evidence_class_code"].map(class_order)
class_counts = class_counts.sort_values("sort").drop(columns="sort")
class_counts.to_csv(OUT / "final_evidence_class_counts.csv", index=False)

strict = data["strict_sequence_motif"]
broad = data["broad_sequence_window"]
geo = data["primary_geometry_positive"].fillna(False)
resid = data["residual_elevated"]
compat = data["compartment_model"].eq("Compatible")
expr_enriched = data["nodule_expression_context"].str.startswith("Nodule-enriched")
layer_rows = [
    ("Frozen Table S4 candidate population", 94, "Residual/structure analysis population; conditioned on early >=2-Cys eligibility"),
    ("Strict sequence motif (MXCXXC or CXXC)", int(strict.sum()), "Narrow motif-only screen"),
    ("Broad frozen sequence window", int(broad.sum()), "Broad sequence heuristic; expected low specificity"),
    ("Frozen formal geometry positive", int(geo.sum()), "Protein-level consensus endpoint"),
    ("Geometry positive but strict motif negative", int((geo & ~strict).sum()), "Structure-first additions beyond strict motif"),
    ("Strict motif positive but geometry negative", int((strict & ~geo).sum()), "Motif hits not retained structurally"),
    ("Residual tier R1 or R2", int(resid.sum()), "Exploratory pull-down residual"),
    ("Compartment modeled compatible", int(compat.sum()), "Localization support; unknown is not counted as compatible"),
    ("Nodule-enriched expression", int(expr_enriched.sum()), "Median project log2FC >=1"),
    ("Geometry + compatible compartment", int((geo & compat).sum()), "Context-compatible formal geometry"),
    ("Geometry + compatible + elevated residual", int((geo & compat & resid).sum()), "Combined computational evidence before native-site audit"),
    ("Final P1 secondary candidates", int(data["final_evidence_class_code"].eq("P1").sum()), "Transfer remains unproven"),
    ("Direct NCC1 Cu clients established", 0, "No Cu binding/transfer/client-metalation evidence"),
]
layer_summary = pd.DataFrame(layer_rows, columns=["layer", "n_proteins", "definition"])
layer_summary["fraction_of_94"] = layer_summary["n_proteins"] / 94
layer_summary.to_csv(OUT / "evidence_layer_summary.csv", index=False)

overlap = pd.crosstab(
    [data["strict_sequence_motif"].map({True: "Strict motif +", False: "Strict motif -"}),
     data["primary_geometry_positive"].map({True: "Geometry +", False: "Geometry -"})],
    data["residual_priority_tier"],
).reset_index()
overlap.to_csv(OUT / "motif_geometry_residual_overlap.csv", index=False)

summary = {
    "as_of": "2026-08-14",
    "main_population": 94,
    "table_s4_unique_r108_ids": 110,
    "outside_frozen94_scope": 16,
    "class_counts": dict(zip(class_counts["final_evidence_class_code"], class_counts["n_proteins"])),
    "strict_motif_positive": int(strict.sum()),
    "broad_sequence_positive": int(broad.sum()),
    "formal_geometry_positive": int(geo.sum()),
    "geometry_missed_by_strict_motif": int((geo & ~strict).sum()),
    "residual_elevated": int(resid.sum()),
    "compartment_compatible": int(compat.sum()),
    "nodule_enriched": int(expr_enriched.sum()),
    "final_secondary_candidates": matrix.loc[matrix["final_evidence_class_code"].eq("P1"), "protein_id"].tolist(),
    "direct_clients_established": 0,
    "scope_warning": "The frozen 94-protein population was selected by an early >=2-Cys rule. Sixteen reliable Table S4 R108 proteins, including two NTF2-like proteins, remain unassessed rather than structure-negative.",
}
(OUT / "final_evidence_matrix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

report_lines = [
    "# Final evidence-hierarchy matrix for NCC1-associated candidates",
    "",
    "## Decision",
    "",
    "The final matrix contains **94 frozen structure/residual candidates** and a separate scope audit of **16 reliable Table S4 R108 proteins** excluded by the early >=2-Cys eligibility rule. The matrix deliberately avoids an additive total score. It assigns mutually exclusive evidence classes through frozen sequential rules, while preserving every raw evidence axis.",
    "",
    "No candidate reaches direct Cu-client evidence. Three proteins remain P1 secondary interaction candidates with transfer unproven: `MtrunR108HiC_001767.1`, `MtrunR108HiC_008307.1`, and `MtrunR108HiC_009426.1`.",
    "",
    "## Final class counts",
    "",
    "| Class | N | Meaning |",
    "|---|---:|---|",
]
for row in class_counts.itertuples(index=False):
    report_lines.append(f"| {row.final_evidence_class_code} | {row.n_proteins} | {row.final_evidence_class} |")
report_lines += [
    "",
    "## Main methodological findings",
    "",
    f"- Strict MXCXXC/CXXC motifs identify {int(strict.sum())}/94 proteins, whereas the frozen formal geometry endpoint identifies {int(geo.sum())}/94.",
    f"- Structure-first analysis adds {int((geo & ~strict).sum())} geometry-positive proteins missed by the strict motif, but context review shows that these additions are dominated by conserved folds, native sites and compartment conflicts rather than validated Cu clients.",
    f"- {int((strict & ~geo).sum())} strict-motif proteins do not pass the frozen formal structural endpoint.",
    f"- Residual tier R1/R2 contains {int(resid.sum())}/94 proteins, yet only {int((geo & compat & resid).sum())} simultaneously have a formal geometry, compatible compartment and elevated residual before native-site review.",
    "- The final evidence ceiling is P1 secondary association; Cu binding, contact-dependent transfer and client metalation are absent.",
    "",
    "## Scope correction",
    "",
    "The original Table S4 mapping contains 110 unique reliable R108 protein IDs. Sixteen were excluded from the frozen 94-protein structure population because they contain fewer than two cysteines. This is a historical Cys-pair eligibility rule, not a biological negative result. In particular, `MtrunR108HiC_008596.1` and `MtrunR108HiC_008599.1` are NTF2-like domain proteins and must not be described as PR1/CAP proteins. A future fully unbiased Cys/Met/His screen would need to predict these 16 proteins without requiring two cysteines; they are retained in the audit sheet rather than silently discarded.",
    "",
    "## Publication-safe use",
    "",
    "- P1 proteins may be described as secondary NCC1-associated interaction candidates with reproducible monomer geometry; transfer is unproven.",
    "- P2 proteins are pull-down-prioritized but lack the frozen formal geometry and should remain supplementary.",
    "- N1/N2 proteins are methodological negative-calibration cases, not failed experiments.",
    "- Q1 proteins are unresolved, not negative.",
    "- No class supports the phrases validated NCC1 client, Cu-binding site, Cu-transfer acceptor or direct NCC1 interface.",
]
(OUT / "final_evidence_hierarchy_report.md").write_text("\n".join(report_lines) + "\n")

payload = {
    "summary": summary,
    "class_rules": CLASS_RULES,
    "matrix": json.loads(matrix.replace({np.nan: None}).to_json(orient="records")),
    "outside_scope": json.loads(outside[outside_columns].replace({np.nan: None}).to_json(orient="records")),
    "class_counts": json.loads(class_counts.to_json(orient="records")),
    "layer_summary": json.loads(layer_summary.to_json(orient="records")),
    "overlap": json.loads(overlap.to_json(orient="records")),
}
(OUT / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

print(json.dumps(summary, ensure_ascii=False, indent=2))
