from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
INPUT = ROOT / "outputs" / "candidate_background_unified"
SCOPE16 = ROOT / "outputs" / "unbiased_scope16_screen"
OUT = ROOT / "outputs" / "validated_endpoint_sensitivity_2026_08_25"
OUT.mkdir(parents=True, exist_ok=True)

PAIR_PATH = INPUT / "donor_pair_metrics_within_all_distances.csv.gz"
MASTER_PATH = INPUT / "protein_level_master.csv"
MATCH_PATH = INPUT / "matched_pairs_primary.csv"
SCOPE16_PAIR_PATH = SCOPE16 / "scope16_all_donor_pair_metrics.csv.gz"
FAMILY_PATH = ROOT / "outputs" / "matched_balance_family_sensitivity" / "family_assignments_all_thresholds.csv"

# All specifications retain the frozen distance, confidence, PAE, sulfur-anchor,
# same-pair recurrence, and complete-structure rules. Only the named gates change.
SPECS = {
    "Frozen": {"sequence_separation_min": 10, "mean_sasa_min_A2": 5.0},
    "No_SASA_gate": {"sequence_separation_min": 10, "mean_sasa_min_A2": 0.0},
    "Separation_ge3": {"sequence_separation_min": 3, "mean_sasa_min_A2": 5.0},
    "No_SASA_and_separation_ge3": {"sequence_separation_min": 3, "mean_sasa_min_A2": 0.0},
}

VALIDATED = {
    "USP-A": {
        "protein_id": "MtrunR108HiC_003946.1",
        "a17_gene": "Medtr1g088640",
        "uniprot": "G7IF74",
        "evidence": "BiFC interaction; Cu binding and contact-dependent Cu transfer",
        "gold_standard": "Cu-transfer-positive",
        "formal_cohort_eligible": True,
    },
    "SAM synthase": {
        "protein_id": "MtrunR108HiC_007551.1",
        "a17_gene": "Medtr2g046710",
        "uniprot": "A0A072V8Q4",
        "evidence": "BiFC interaction only",
        "gold_standard": "BiFC-associated",
        "formal_cohort_eligible": True,
    },
    "Peroxiredoxin": {
        "protein_id": "MtrunR108HiC_032912.1",
        "a17_gene": "Medtr7g105830",
        "uniprot": "G7ZUV5",
        "evidence": "BiFC interaction only",
        "gold_standard": "BiFC-associated",
        "formal_cohort_eligible": True,
    },
    "PR/NTF2-like": {
        "protein_id": "MtrunR108HiC_008596.1",
        "a17_gene": "Medtr2g076010",
        "uniprot": "G7ITG5",
        "evidence": "BiFC interaction only",
        "gold_standard": "BiFC-associated",
        "formal_cohort_eligible": False,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def hypergeom_probability(a: int, row1: int, col1: int, n: int) -> float:
    return math.comb(col1, a) * math.comb(n - col1, row1 - a) / math.comb(n, row1)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    row1, col1 = a + b, a + c
    p_obs = hypergeom_probability(a, row1, col1, n)
    low, high = max(0, row1 - (n - col1)), min(row1, col1)
    return min(1.0, sum(
        hypergeom_probability(x, row1, col1, n)
        for x in range(low, high + 1)
        if hypergeom_probability(x, row1, col1, n) <= p_obs + 1e-15
    ))


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    p_obs = math.comb(n, k) / (2 ** n)
    return min(1.0, sum(
        math.comb(n, x) / (2 ** n)
        for x in range(n + 1)
        if math.comb(n, x) / (2 ** n) <= p_obs + 1e-15
    ))


def wilson_interval(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = x / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def ratio_intervals(a: int, n1: int, c: int, n0: int) -> dict[str, float]:
    # Counts are nonzero in all four analysis specifications, so uncorrected
    # log-scale Wald intervals are defined and match the formal analysis convention.
    b, d = n1 - a, n0 - c
    rr = (a / n1) / (c / n0)
    se_log_rr = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    odds = (a * d) / (b * c)
    se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = 1.959963984540054
    return {
        "risk_ratio": rr,
        "risk_ratio_ci_low": math.exp(math.log(rr) - z * se_log_rr),
        "risk_ratio_ci_high": math.exp(math.log(rr) + z * se_log_rr),
        "odds_ratio": odds,
        "odds_ratio_ci_low": math.exp(math.log(odds) - z * se_log_or),
        "odds_ratio_ci_high": math.exp(math.log(odds) + z * se_log_or),
    }


def logistic_group_effect(
    y: np.ndarray,
    group: np.ndarray,
    covariates: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict[str, float]:
    # Stable Newton/IRLS implementation of the prespecified adjusted model:
    # endpoint ~ candidate + log1p(length) + log1p(C/M/H count) + pLDDT>=70 coverage.
    x = np.column_stack([np.ones(len(y)), group, covariates]).astype(float)
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(200):
        eta = np.clip(x @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-8, None)
        hessian = x.T @ (w[:, None] * x)
        score = x.T @ (y - mu)
        step = np.linalg.pinv(hessian) @ score
        beta_new = beta + step
        if np.max(np.abs(step)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(x @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-8, None)
    covariance = np.linalg.pinv(x.T @ (w[:, None] * x))
    se = math.sqrt(max(covariance[1, 1], 0))
    z_value = beta[1] / se
    p_value = 2 * (1 - normal_cdf(abs(z_value)))
    z = 1.959963984540054
    result = {
        "adjusted_odds_ratio": math.exp(beta[1]),
        "adjusted_or_ci_low": math.exp(beta[1] - z * se),
        "adjusted_or_ci_high": math.exp(beta[1] + z * se),
        "adjusted_wald_p": p_value,
    }
    if clusters is not None:
        bread = np.linalg.pinv(x.T @ (w[:, None] * x))
        meat = np.zeros_like(bread)
        unique_clusters = np.unique(clusters)
        residual = y - mu
        for cluster in unique_clusters:
            use = clusters == cluster
            score_g = x[use].T @ residual[use]
            meat += np.outer(score_g, score_g)
        n, p = x.shape
        g = len(unique_clusters)
        correction = (g / (g - 1)) * ((n - 1) / (n - p))
        robust_covariance = correction * bread @ meat @ bread
        robust_se = math.sqrt(max(robust_covariance[1, 1], 0))
        robust_z = beta[1] / robust_se
        result.update({
            "family40_adjusted_or": math.exp(beta[1]),
            "family40_adjusted_or_ci_low": math.exp(beta[1] - z * robust_se),
            "family40_adjusted_or_ci_high": math.exp(beta[1] + z * robust_se),
            "family40_adjusted_p": 2 * (1 - normal_cdf(abs(robust_z))),
            "family40_clusters": g,
        })
    return result


def positive_ids(pair_table: pd.DataFrame, spec: dict[str, float]) -> set[str]:
    passing = pair_table[
        pair_table["sulfur_anchored"].astype(bool)
        & pair_table["donor_distance_A"].between(2.5, 5.0, inclusive="both")
        & pair_table["min_local_plddt"].ge(70.0)
        & pair_table["pair_pae_A"].le(10.0)
        & pair_table["sequence_separation"].ge(spec["sequence_separation_min"])
        & pair_table["mean_donor_sasa_A2"].ge(spec["mean_sasa_min_A2"])
    ]
    support = passing.groupby(["group", "protein_id", "pair_key"])["rank"].nunique()
    return set(support[support.ge(2)].reset_index()["protein_id"])


def pair_summary(pair_table: pd.DataFrame, protein_id: str, residue_i: int, residue_j: int) -> dict[str, float | int | str]:
    x = pair_table[
        pair_table["protein_id"].eq(protein_id)
        & pair_table["residue_i"].eq(residue_i)
        & pair_table["residue_j"].eq(residue_j)
    ].copy()
    if x.empty:
        return {"protein_id": protein_id, "pair_key": f"{residue_i}-{residue_j}", "models": 0}
    return {
        "protein_id": protein_id,
        "pair_key": x.iloc[0]["pair_key"],
        "donor_combo": x.iloc[0]["donor_combo"],
        "sequence_separation": int(x.iloc[0]["sequence_separation"]),
        "models": int(x["rank"].nunique()),
        "distance_median_A": float(x["donor_distance_A"].median()),
        "distance_min_A": float(x["donor_distance_A"].min()),
        "distance_max_A": float(x["donor_distance_A"].max()),
        "min_local_plddt_median": float(x["min_local_plddt"].median()),
        "pair_pae_median_A": float(x["pair_pae_A"].median()),
        "mean_donor_sasa_median_A2": float(x["mean_donor_sasa_A2"].median()),
        "donor_i_sasa_median_A2": float(x["donor_sasa_i_A2"].median()),
        "donor_j_sasa_median_A2": float(x["donor_sasa_j_A2"].median()),
    }


def main() -> None:
    pairs = pd.read_csv(PAIR_PATH)
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    matched = pd.read_csv(MATCH_PATH)
    scope16_pairs = pd.read_csv(SCOPE16_PAIR_PATH)
    families = pd.read_csv(FAMILY_PATH, low_memory=False)[["protein_id", "family_40"]]

    complete = master[master["structure_analysis_included"].astype(bool)].copy()
    complete = complete.merge(families, on="protein_id", how="left", validate="one_to_one")
    if complete["family_40"].isna().any():
        raise ValueError("Missing family_40 assignment for a complete-case protein")
    complete["candidate"] = complete["group"].eq("Candidate").astype(int)
    covariates = np.column_stack([
        np.log1p(complete["sequence_length"].astype(float).to_numpy()),
        np.log1p(complete["donor_residue_count"].astype(float).to_numpy()),
        complete["mean_fraction_plddt_ge70"].astype(float).to_numpy(),
    ])
    covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0, ddof=0)

    group_rows = []
    positives_by_spec: dict[str, set[str]] = {}
    for endpoint, spec in SPECS.items():
        positives = positive_ids(pairs, spec)
        positives_by_spec[endpoint] = positives
        cand = complete[complete["group"].eq("Candidate")]
        back = complete[complete["group"].eq("Background")]
        cp = int(cand["protein_id"].isin(positives).sum())
        bp = int(back["protein_id"].isin(positives).sum())
        cn, bn = len(cand), len(back)
        cand_ci = wilson_interval(cp, cn)
        back_ci = wilson_interval(bp, bn)
        rd = cp / cn - bp / bn
        rd_se = math.sqrt((cp / cn) * (1 - cp / cn) / cn + (bp / bn) * (1 - bp / bn) / bn)
        ratios = ratio_intervals(cp, cn, bp, bn)
        y = complete["protein_id"].isin(positives).astype(int).to_numpy()
        adjusted = logistic_group_effect(
            y,
            complete["candidate"].to_numpy(),
            covariates,
            complete["family_40"].astype(str).to_numpy(),
        )

        cand_matched = matched["candidate_id"].isin(positives)
        back_matched = matched["background_id"].isin(positives)
        cand_only = int((cand_matched & ~back_matched).sum())
        back_only = int((~cand_matched & back_matched).sum())
        group_rows.append({
            "endpoint": endpoint,
            "sequence_separation_min": spec["sequence_separation_min"],
            "mean_sasa_min_A2": spec["mean_sasa_min_A2"],
            "candidate_positive": cp,
            "candidate_total": cn,
            "candidate_fraction": cp / cn,
            "candidate_ci_low": cand_ci[0],
            "candidate_ci_high": cand_ci[1],
            "background_positive": bp,
            "background_total": bn,
            "background_fraction": bp / bn,
            "background_ci_low": back_ci[0],
            "background_ci_high": back_ci[1],
            "risk_difference": rd,
            "risk_difference_ci_low": rd - 1.959963984540054 * rd_se,
            "risk_difference_ci_high": rd + 1.959963984540054 * rd_se,
            **ratios,
            "fisher_exact_p": fisher_exact_two_sided(cp, cn - cp, bp, bn - bp),
            **adjusted,
            "matched_pairs": len(matched),
            "matched_candidate_positive": int(cand_matched.sum()),
            "matched_background_positive": int(back_matched.sum()),
            "matched_candidate_only": cand_only,
            "matched_background_only": back_only,
            "matched_risk_difference": float(cand_matched.mean() - back_matched.mean()),
            "mcnemar_exact_p": exact_binomial_two_sided(cand_only, cand_only + back_only),
        })
    group_results = pd.DataFrame(group_rows)
    group_results.to_csv(OUT / "threshold_sensitivity_group_comparisons.csv", index=False)

    # The later scope-completion model for PR/NTF2-like is used only to audit
    # end-to-end sensitivity. It is not inserted into the frozen 93-vs-74 cohort.
    pr_pairs = scope16_pairs[scope16_pairs["protein_id"].eq("MtrunR108HiC_008596.1")]
    validation_pairs = pd.concat([pairs, pr_pairs], ignore_index=True)
    validation_rows = []
    for endpoint, spec in SPECS.items():
        positives = positive_ids(validation_pairs, spec)
        for label, meta in VALIDATED.items():
            validation_rows.append({
                "endpoint": endpoint,
                "case": label,
                **meta,
                "endpoint_positive": meta["protein_id"] in positives,
            })
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "validated_case_endpoint_matrix.csv", index=False)

    recall_rows = []
    for endpoint in SPECS:
        x = validation[validation["endpoint"].eq(endpoint)]
        interaction_recalled = int(x["endpoint_positive"].sum())
        usp = x[x["gold_standard"].eq("Cu-transfer-positive")]
        recall_rows.extend([
            {
                "endpoint": endpoint,
                "gold_standard": "Four downstream BiFC-associated proteins",
                "recalled": interaction_recalled,
                "total": 4,
                "recall": interaction_recalled / 4,
                "denominator_note": "MtCOPT1 excluded because it is the upstream Cu input, not a downstream client",
            },
            {
                "endpoint": endpoint,
                "gold_standard": "Biochemically demonstrated Cu transfer",
                "recalled": int(usp["endpoint_positive"].sum()),
                "total": 1,
                "recall": float(usp["endpoint_positive"].mean()),
                "denominator_note": "USP-A is the only downstream protein with direct Cu-transfer evidence",
            },
        ])
    recall = pd.DataFrame(recall_rows)
    recall.to_csv(OUT / "validated_case_recall_summary.csv", index=False)

    site_rows = [
        {"case": "USP-A", "interpretation": "Validated Cu-transfer protein; geometry passes but apo-state SASA gate fails", **pair_summary(pairs, "MtrunR108HiC_003946.1", 116, 149)},
        {"case": "SAM synthase", "interpretation": "Conserved CXXC motif; motif cysteines are not spatially paired", **pair_summary(pairs, "MtrunR108HiC_007551.1", 44, 47)},
        {"case": "SAM synthase", "interpretation": "Recurrent exposed short-range Cys/Met geometry recovered by separation >=3", **pair_summary(pairs, "MtrunR108HiC_007551.1", 47, 54)},
        {"case": "SAM synthase", "interpretation": "Cys/Met neighbor is <=5 A in only one of three models and is not recurrent", **pair_summary(pairs, "MtrunR108HiC_007551.1", 47, 52)},
    ]
    site_metrics = pd.DataFrame(site_rows)
    site_metrics.to_csv(OUT / "validated_site_three_model_metrics.csv", index=False)

    summary_lines = [
        "# Validated-case endpoint sensitivity analysis",
        "",
        "The original experimental hierarchy was retained: four downstream proteins were BiFC-associated, but only USP-A was shown to bind Cu and receive Cu from NCC1. MtCOPT1 is an upstream input protein and is not part of downstream-client recall.",
        "",
        "## Group comparison",
        "",
        "| Endpoint | Candidate | Background | RR (95% CI) | Fisher P | Adjusted OR (95% CI) | Adjusted P | Family-aware P | Matched P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in group_results.itertuples(index=False):
        summary_lines.append(
            f"| {r.endpoint} | {r.candidate_positive}/{r.candidate_total} ({100*r.candidate_fraction:.1f}%) | "
            f"{r.background_positive}/{r.background_total} ({100*r.background_fraction:.1f}%) | "
            f"{r.risk_ratio:.2f} ({r.risk_ratio_ci_low:.2f}-{r.risk_ratio_ci_high:.2f}) | {r.fisher_exact_p:.3f} | "
            f"{r.adjusted_odds_ratio:.2f} ({r.adjusted_or_ci_low:.2f}-{r.adjusted_or_ci_high:.2f}) | "
            f"{r.adjusted_wald_p:.3f} | {r.family40_adjusted_p:.3f} | {r.mcnemar_exact_p:.3f} |"
        )
    summary_lines += [
        "",
        "## Validated-case recall",
        "",
        "| Endpoint | BiFC-associated recall | Cu-transfer recall |",
        "|---|---:|---:|",
    ]
    for endpoint in SPECS:
        x = recall[recall["endpoint"].eq(endpoint)]
        bifc = x[x["gold_standard"].str.startswith("Four")].iloc[0]
        cu = x[x["gold_standard"].str.startswith("Biochemically")].iloc[0]
        summary_lines.append(f"| {endpoint} | {bifc.recalled}/{bifc.total} | {cu.recalled}/{cu.total} |")
    summary_lines += [
        "",
        "## Interpretation",
        "",
        "- Removing only the SASA gate recovers USP-A, the sole direct Cu-transfer positive, while the candidate-background comparison remains null.",
        "- Relaxing sequence separation to >=3 recovers the recurrent SAM-synthase C47-M54 geometry, while the group comparison remains null.",
        "- Relaxing both gates recovers USP-A and SAM synthase (2/4 BiFC cases), but not peroxiredoxin or PR/NTF2-like; all unadjusted, adjusted, and matched comparisons remain non-significant.",
        "- Therefore the absence of candidate enrichment is robust across the rule family, whereas the frozen endpoint is not a sensitive detector of experimentally supported Cu-client chemistry.",
        "- These rules should be presented as calibration/prioritization endpoints, not as a diagnostic definition of Cu-binding sites or NCC1 clients.",
    ]
    (OUT / "analysis_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    audit = {
        "analysis_date": "2026-08-25",
        "formal_group_unit": "protein",
        "formal_candidate_total_complete": int((complete["group"] == "Candidate").sum()),
        "formal_background_total_complete": int((complete["group"] == "Background").sum()),
        "matched_pairs": len(matched),
        "endpoint_constants": {
            "distance_A": [2.5, 5.0],
            "local_plddt_min": 70,
            "pair_pae_max_A": 10,
            "same_pair_support_models_min": 2,
            "sulfur_anchor_required": True,
            "primary_pipeline_sasa_implementation": "240-point Shrake-Rupley approximation with 1.4-A probe (not FreeSASA)",
        },
        "sensitivity_specs": SPECS,
        "input_sha256": {str(p): sha256(p) for p in [PAIR_PATH, MASTER_PATH, MATCH_PATH, SCOPE16_PAIR_PATH, FAMILY_PATH]},
        "original_paper_verification": {
            "main_pdf": str(ROOT / "Cu参考文献" / "NCC1 2023.pdf"),
            "supporting_pdf": str(ROOT / "Cu参考文献" / "NCC1 SI.pdf"),
            "Fig7": "Four downstream BiFC interactors: Medtr7g105830, Medtr2g046710, Medtr1g088640, Medtr2g076010",
            "Fig8": "Direct Cu binding/transfer assayed only for USP-A",
            "FigS13": "AlphaFold + PyMOL thiol distance 4.5 A compared with CopZ 4.2 A",
        },
        "caveat": "Sensitivity estimates are descriptive and very imprecise because the direct Cu-transfer gold standard contains one protein.",
        "methods_correction_required": "The current manuscript says FreeSASA, but the unified 167-protein endpoint used a 240-point Shrake-Rupley approximation. Correct the manuscript or recompute all 167 proteins with FreeSASA before retaining that wording.",
        "key_site_freesasa_validation": {
            "software": "FreeSASA 2.2.1 Python package",
            "probe_radii_A": [1.2, 1.4, 1.6, 2.0],
            "summary_file": str(OUT / "freesasa_key_site_summary.csv"),
            "interpretation": "Independent key-case validation only; group endpoint calls retain the frozen unified SASA implementation",
        },
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(group_results[[
        "endpoint", "candidate_positive", "candidate_total", "background_positive", "background_total",
        "risk_ratio", "fisher_exact_p", "adjusted_odds_ratio", "adjusted_wald_p", "family40_adjusted_p", "mcnemar_exact_p"
    ]].to_string(index=False))
    print("\nValidated-case recall")
    print(recall[["endpoint", "gold_standard", "recalled", "total", "recall"]].to_string(index=False))
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
