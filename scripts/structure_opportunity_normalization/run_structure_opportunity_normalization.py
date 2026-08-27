#!/usr/bin/env python3
"""Opportunity-normalized comparison of NCC1 candidates and pull-down background.

The analysis keeps the protein as the inferential unit. It distinguishes:
1. sequence-search opportunities: unique, nonlocal Cys/Met/His pairs with >=1 S donor;
2. structure-evaluable opportunities: model-pairs passing confidence/exposure QC,
   before applying the distance criterion;
3. recurrent pair-events: the same qualifying pair in >=2 of 3 models;
4. recurrent pockets: connected components of recurrent residue-pair edges.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(".")
INPUT_DIR = ROOT / "outputs" / "candidate_background_unified"
OUTPUT_DIR = ROOT / "outputs" / "structure_opportunity_normalization"
PAIR_PATH = INPUT_DIR / "donor_pair_metrics_within_all_distances.csv.gz"
PROTEIN_PATH = INPUT_DIR / "protein_level_master.csv"
SEED = 20260824
BOOTSTRAP_ITERATIONS = 30_000
PERMUTATION_ITERATIONS = 100_000


def connected_pocket_count(edges: pd.DataFrame) -> int:
    """Collapse recurrent pair edges sharing a residue into connected pockets."""
    if edges.empty:
        return 0
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in edges.itertuples(index=False):
        union(int(row.residue_i), int(row.residue_j))
    return len({find(x) for x in parent})


def build_protein_metrics(pairs: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    key = ["group", "protein_id"]
    sequence_eligible = pairs[
        pairs["sulfur_anchored"] & (pairs["sequence_separation"] >= 10)
    ].copy()

    # Confirm that sequence opportunity sets are invariant across the three ranks.
    opportunity_by_rank = (
        sequence_eligible.groupby(key + ["rank"])["pair_key"].nunique().unstack("rank")
    )
    if opportunity_by_rank.shape[1] != 3 or not opportunity_by_rank.nunique(axis=1).eq(1).all():
        raise RuntimeError("Eligible sequence-pair opportunities are not invariant across ranks")

    sequence_opportunities = (
        sequence_eligible.groupby(key)["pair_key"].nunique().rename("eligible_sequence_pairs")
    )

    evaluable = sequence_eligible[
        (sequence_eligible["min_local_plddt"] >= 70)
        & (sequence_eligible["pair_pae_A"] <= 10)
        & (sequence_eligible["mean_donor_sasa_A2"] >= 5)
    ]
    evaluable_opportunities = (
        evaluable.groupby(key).size().rename("evaluable_model_pairs")
    )

    passing = pairs[pairs["primary_model_pass"]].copy()
    model_pass_events = passing.groupby(key).size().rename("qualifying_model_pair_events")
    support = (
        passing.groupby(key + ["pair_key", "residue_i", "residue_j"])["rank"]
        .nunique()
        .rename("support_models")
        .reset_index()
    )
    recurrent = support[support["support_models"] >= 2].copy()
    recurrent_pair_count = recurrent.groupby(key).size().rename("recurrent_pair_count")
    recurrent_pocket_count = recurrent.groupby(key).apply(
        connected_pocket_count, include_groups=False
    ).rename("recurrent_pocket_count")

    keep = proteins.loc[
        proteins["structure_analysis_included"].astype(bool),
        [
            "group",
            "protein_id",
            "sequence_length",
            "donor_residue_count",
            "models_available",
            "mean_fraction_plddt_ge70",
            "primary_geometry_positive",
        ],
    ].copy()
    metrics = keep.set_index(key).join(
        [
            sequence_opportunities,
            evaluable_opportunities,
            model_pass_events,
            recurrent_pair_count,
            recurrent_pocket_count,
        ]
    ).fillna(
        {
            "evaluable_model_pairs": 0,
            "qualifying_model_pair_events": 0,
            "recurrent_pair_count": 0,
            "recurrent_pocket_count": 0,
        }
    ).reset_index()

    integer_columns = [
        "eligible_sequence_pairs",
        "evaluable_model_pairs",
        "qualifying_model_pair_events",
        "recurrent_pair_count",
        "recurrent_pocket_count",
    ]
    metrics[integer_columns] = metrics[integer_columns].astype(int)
    if len(metrics) != 167 or metrics["protein_id"].nunique() != 167:
        raise RuntimeError("Expected 167 unique structure-included proteins")
    if (metrics["models_available"] != 3).any():
        raise RuntimeError("Every included protein must have exactly three models")
    if (metrics["eligible_sequence_pairs"] <= 0).any():
        raise RuntimeError("All included proteins must have at least one sequence opportunity")

    metrics["candidate"] = (metrics["group"] == "Candidate").astype(int)
    metrics["recurrent_positive"] = (metrics["recurrent_pair_count"] > 0).astype(int)
    metrics["recurrent_pair_rate_per_10000"] = (
        10_000 * metrics["recurrent_pair_count"] / metrics["eligible_sequence_pairs"]
    )
    metrics["recurrent_pocket_rate_per_10000"] = (
        10_000 * metrics["recurrent_pocket_count"] / metrics["eligible_sequence_pairs"]
    )
    metrics["model_event_rate_per_10000"] = (
        10_000
        * metrics["qualifying_model_pair_events"]
        / metrics["evaluable_model_pairs"].replace(0, np.nan)
    )
    return metrics


def group_rate(frame: pd.DataFrame, outcome: str, exposure: str, group: str) -> float:
    subset = frame[frame["group"] == group]
    return float(subset[outcome].sum() / subset[exposure].sum())


def stratified_bootstrap_rate_ratio(
    frame: pd.DataFrame, outcome: str, exposure: str, iterations: int
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + sum(map(ord, outcome)))
    candidate = frame[frame["group"] == "Candidate"].reset_index(drop=True)
    background = frame[frame["group"] == "Background"].reset_index(drop=True)
    ratios = np.empty(iterations)
    for index in range(iterations):
        c = candidate.iloc[rng.integers(0, len(candidate), len(candidate))]
        b = background.iloc[rng.integers(0, len(background), len(background))]
        c_rate = c[outcome].sum() / c[exposure].sum()
        b_rate = b[outcome].sum() / b[exposure].sum()
        ratios[index] = c_rate / b_rate if b_rate > 0 else np.nan
    ratios = ratios[np.isfinite(ratios)]
    return tuple(np.quantile(ratios, [0.025, 0.975]))


def protein_label_permutation_p(
    frame: pd.DataFrame, outcome: str, exposure: str, iterations: int
) -> float:
    rng = np.random.default_rng(SEED + 10_000 + sum(map(ord, outcome)))
    outcomes = frame[outcome].to_numpy(float)
    exposures = frame[exposure].to_numpy(float)
    candidate_n = int(frame["candidate"].sum())
    observed = math.log(
        group_rate(frame, outcome, exposure, "Candidate")
        / group_rate(frame, outcome, exposure, "Background")
    )
    exceed = 0
    all_indices = np.arange(len(frame))
    for _ in range(iterations):
        candidate_indices = rng.choice(all_indices, size=candidate_n, replace=False)
        candidate_mask = np.zeros(len(frame), dtype=bool)
        candidate_mask[candidate_indices] = True
        c_rate = outcomes[candidate_mask].sum() / exposures[candidate_mask].sum()
        b_rate = outcomes[~candidate_mask].sum() / exposures[~candidate_mask].sum()
        if c_rate > 0 and b_rate > 0 and abs(math.log(c_rate / b_rate)) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (iterations + 1)


def normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def robust_glm(
    outcome: np.ndarray,
    design: np.ndarray,
    family: str,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a small GLM by IRLS and return HC3 sandwich standard errors."""
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(design, dtype=float)
    linear_offset = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    coefficients = np.zeros(x.shape[1], dtype=float)
    if family == "poisson":
        coefficients[0] = math.log((y.sum() + 0.5) / np.exp(linear_offset).sum())
    elif family == "binomial":
        probability = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        coefficients[0] = math.log(probability / (1 - probability))
    else:
        raise ValueError(f"Unsupported family: {family}")

    for _ in range(100):
        eta = linear_offset + x @ coefficients
        if family == "poisson":
            mean = np.exp(np.clip(eta, -30, 30))
            weights = np.clip(mean, 1e-10, None)
        else:
            mean = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            weights = np.clip(mean * (1 - mean), 1e-10, None)
        fisher_information = x.T @ (weights[:, None] * x)
        step = np.linalg.pinv(fisher_information) @ (x.T @ (y - mean))
        coefficients_new = coefficients + step
        if np.max(np.abs(coefficients_new - coefficients)) < 1e-10:
            coefficients = coefficients_new
            break
        coefficients = coefficients_new

    eta = linear_offset + x @ coefficients
    if family == "poisson":
        mean = np.exp(np.clip(eta, -30, 30))
        weights = np.clip(mean, 1e-10, None)
    else:
        mean = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        weights = np.clip(mean * (1 - mean), 1e-10, None)
    bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
    leverage = np.einsum("ij,jk,ik->i", x, bread, x) * weights
    adjusted_residual = (y - mean) / np.clip(1 - leverage, 1e-6, None)
    meat = x.T @ ((adjusted_residual ** 2)[:, None] * x)
    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return coefficients, standard_errors


def robust_poisson_rate_ratio(
    frame: pd.DataFrame, outcome: str, exposure: str, label: str
) -> dict[str, float | str]:
    design = np.column_stack([np.ones(len(frame)), frame["candidate"].to_numpy(float)])
    coefficients, standard_errors = robust_glm(
        frame[outcome].to_numpy(float), design, "poisson", np.log(frame[exposure].to_numpy(float))
    )
    coefficient = float(coefficients[1])
    standard_error = float(standard_errors[1])
    return {
        "analysis": label,
        "estimand": "rate_ratio",
        "effect_candidate_vs_background": math.exp(coefficient),
        "ci_low": math.exp(coefficient - 1.96 * standard_error),
        "ci_high": math.exp(coefficient + 1.96 * standard_error),
        "p_value": normal_two_sided_p(coefficient / standard_error),
        "method": "Poisson log-rate model with protein-level HC3 robust SE and log exposure offset",
    }


def adjusted_logistic_odds_ratio(frame: pd.DataFrame) -> dict[str, float | str]:
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["candidate"].to_numpy(float),
            np.log(frame["eligible_sequence_pairs"].to_numpy(float)),
            frame["mean_fraction_plddt_ge70"].to_numpy(float),
        ]
    )
    coefficients, standard_errors = robust_glm(
        frame["recurrent_positive"].to_numpy(float), design, "binomial"
    )
    coefficient = float(coefficients[1])
    standard_error = float(standard_errors[1])
    return {
        "analysis": "Any recurrent pair, adjusted for opportunity and structure coverage",
        "estimand": "odds_ratio",
        "effect_candidate_vs_background": math.exp(coefficient),
        "ci_low": math.exp(coefficient - 1.96 * standard_error),
        "ci_high": math.exp(coefficient + 1.96 * standard_error),
        "p_value": normal_two_sided_p(coefficient / standard_error),
        "method": "Logistic GLM with HC3 robust SE; log eligible pairs and pLDDT>=70 coverage covariates",
    }


def summarize_groups(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, subset in metrics.groupby("group", sort=False):
        rows.append(
            {
                "group": group,
                "proteins": len(subset),
                "positive_proteins": int(subset["recurrent_positive"].sum()),
                "eligible_sequence_pairs": int(subset["eligible_sequence_pairs"].sum()),
                "evaluable_model_pairs": int(subset["evaluable_model_pairs"].sum()),
                "qualifying_model_pair_events": int(subset["qualifying_model_pair_events"].sum()),
                "recurrent_pair_count": int(subset["recurrent_pair_count"].sum()),
                "recurrent_pocket_count": int(subset["recurrent_pocket_count"].sum()),
                "median_eligible_pairs_per_protein": float(subset["eligible_sequence_pairs"].median()),
                "recurrent_pairs_per_10000_opportunities": float(
                    10_000 * subset["recurrent_pair_count"].sum() / subset["eligible_sequence_pairs"].sum()
                ),
                "recurrent_pockets_per_10000_opportunities": float(
                    10_000 * subset["recurrent_pocket_count"].sum() / subset["eligible_sequence_pairs"].sum()
                ),
                "model_events_per_10000_evaluable_opportunities": float(
                    10_000
                    * subset["qualifying_model_pair_events"].sum()
                    / subset["evaluable_model_pairs"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, results: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    """Create a compact, dependency-light review figure with Pillow."""
    image = Image.new("RGB", (2400, 820), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=20)
    title_font = ImageFont.load_default(size=34)
    bold = ImageFont.load_default(size=27)
    colors = {"Candidate": "#B24A4A", "Background": "#4C78A8"}
    draw.text((50, 25), "Structural-opportunity normalization does not reveal candidate enrichment",
              fill="#111111", font=title_font)

    # Panel A: opportunity distributions as median/IQR/range.
    left = 70
    draw.text((left, 105), "A  Search opportunities per protein", fill="#111111", font=bold)
    draw.text((left, 140), "log10 eligible sulfur-anchored donor pairs", fill="#555555", font=small)
    axis_y0, axis_y1 = 650, 180
    draw.line((left + 110, axis_y0, left + 110, axis_y1), fill="#555555", width=2)
    all_log = np.log10(metrics["eligible_sequence_pairs"].to_numpy(float))
    minimum, maximum = float(all_log.min()), float(all_log.max())
    scale = lambda value: axis_y0 - (value - minimum) / (maximum - minimum) * (axis_y0 - axis_y1)
    for tick in np.linspace(math.ceil(minimum * 2) / 2, math.floor(maximum * 2) / 2, 5):
        y = scale(float(tick))
        draw.line((left + 100, y, left + 110, y), fill="#555555", width=2)
        draw.text((left + 48, y - 10), f"{tick:.1f}", fill="#555555", font=small)
    for group, x in [("Background", left + 300), ("Candidate", left + 550)]:
        values = np.log10(metrics.loc[metrics["group"] == group, "eligible_sequence_pairs"].to_numpy(float))
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = values.min(), values.max()
        draw.line((x, scale(low), x, scale(high)), fill=colors[group], width=5)
        draw.rectangle((x - 55, scale(q3), x + 55, scale(q1)), outline=colors[group], fill="#EFEFEF", width=5)
        draw.line((x - 55, scale(median), x + 55, scale(median)), fill=colors[group], width=6)
        draw.text((x - 85, 680), group, fill="#222222", font=small)

    # Panel B: normalized rates.
    left = 830
    draw.text((left, 105), "B  Events per 10,000 opportunities", fill="#111111", font=bold)
    rate_specs = [
        ("Recurrent pairs", "recurrent_pair_count", "eligible_sequence_pairs"),
        ("Collapsed pockets", "recurrent_pocket_count", "eligible_sequence_pairs"),
        ("Model-pair events", "qualifying_model_pair_events", "evaluable_model_pairs"),
    ]
    rates = {(group, label): 10_000 * group_rate(metrics, outcome, exposure, group)
             for label, outcome, exposure in rate_specs for group in ["Background", "Candidate"]}
    max_rate = max(rates.values()) * 1.15
    baseline = 650
    for index, (label, _, _) in enumerate(rate_specs):
        center = left + 150 + index * 230
        for offset, group in [(-45, "Background"), (45, "Candidate")]:
            value = rates[(group, label)]
            height = int(400 * value / max_rate)
            draw.rectangle((center + offset - 34, baseline - height, center + offset + 34, baseline),
                           fill=colors[group])
            draw.text((center + offset - 38, baseline - height - 28), f"{value:.1f}", fill="#222222", font=small)
        draw.text((center - 85, 675), label, fill="#222222", font=small)
    draw.rectangle((left + 40, 745, left + 62, 767), fill=colors["Background"])
    draw.text((left + 72, 742), "Background", fill="#222222", font=small)
    draw.rectangle((left + 245, 745, left + 267, 767), fill=colors["Candidate"])
    draw.text((left + 277, 742), "Candidate", fill="#222222", font=small)

    # Panel C: forest plot.
    left = 1640
    draw.text((left, 105), "C  Candidate / background effect", fill="#111111", font=bold)
    plot_rows = results.iloc[:4]
    x0, x1 = left + 250, 2320
    log_min, log_max = math.log(0.15), math.log(4.5)
    x_scale = lambda value: x0 + (math.log(value) - log_min) / (log_max - log_min) * (x1 - x0)
    draw.line((x_scale(1), 170, x_scale(1), 650), fill="#999999", width=3)
    labels = ["Recurrent pairs", "Collapsed pockets", "Model-pair events", "Any recurrent pair"]
    for index, (_, row) in enumerate(plot_rows.iterrows()):
        y = 220 + index * 110
        draw.text((left, y - 15), labels[index], fill="#222222", font=small)
        draw.line((x_scale(float(row.ci_low)), y, x_scale(float(row.ci_high)), y), fill="#555555", width=5)
        draw.ellipse((x_scale(float(row.effect_candidate_vs_background)) - 8, y - 8,
                      x_scale(float(row.effect_candidate_vs_background)) + 8, y + 8), fill="#222222")
        draw.text((x1 - 120, y - 45), f"P={float(row.p_value):.3f}", fill="#333333", font=small)
    for tick in [0.25, 0.5, 1, 2, 4]:
        x = x_scale(tick)
        draw.line((x, 650, x, 660), fill="#333333", width=2)
        draw.text((x - 14, 668), str(tick), fill="#333333", font=small)
    image.save(OUTPUT_DIR / "Figure_structure_opportunity_normalization.png", dpi=(300, 300))
    image.save(OUTPUT_DIR / "Figure_structure_opportunity_normalization.pdf", resolution=300)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIR_PATH, low_memory=False)
    proteins = pd.read_csv(PROTEIN_PATH, low_memory=False)
    metrics = build_protein_metrics(pairs, proteins)
    group_summary = summarize_groups(metrics)

    analyses = [
        robust_poisson_rate_ratio(
            metrics,
            "recurrent_pair_count",
            "eligible_sequence_pairs",
            "Recurrent residue-pair events per eligible sequence pair",
        ),
        robust_poisson_rate_ratio(
            metrics,
            "recurrent_pocket_count",
            "eligible_sequence_pairs",
            "Connected recurrent pockets per eligible sequence pair",
        ),
        robust_poisson_rate_ratio(
            metrics[metrics["evaluable_model_pairs"] > 0].copy(),
            "qualifying_model_pair_events",
            "evaluable_model_pairs",
            "Qualifying model-pair events per evaluable model-pair",
        ),
        adjusted_logistic_odds_ratio(metrics),
    ]
    results = pd.DataFrame(analyses)

    for row_index, (outcome, exposure) in enumerate(
        [
            ("recurrent_pair_count", "eligible_sequence_pairs"),
            ("recurrent_pocket_count", "eligible_sequence_pairs"),
            ("qualifying_model_pair_events", "evaluable_model_pairs"),
        ]
    ):
        inference_frame = metrics[metrics[exposure] > 0].copy()
        bootstrap_low, bootstrap_high = stratified_bootstrap_rate_ratio(
            inference_frame, outcome, exposure, BOOTSTRAP_ITERATIONS
        )
        results.loc[row_index, "bootstrap_ci_low"] = bootstrap_low
        results.loc[row_index, "bootstrap_ci_high"] = bootstrap_high
        results.loc[row_index, "protein_label_permutation_p"] = protein_label_permutation_p(
            inference_frame, outcome, exposure, PERMUTATION_ITERATIONS
        )

    metrics.to_csv(OUTPUT_DIR / "protein_opportunity_metrics.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "group_opportunity_summary.csv", index=False)
    results.to_csv(OUTPUT_DIR / "opportunity_normalized_effects.csv", index=False)

    make_figure(metrics, results, group_summary)

    candidate_summary = group_summary[group_summary["group"] == "Candidate"].iloc[0]
    background_summary = group_summary[group_summary["group"] == "Background"].iloc[0]
    primary = results.iloc[0]
    report = f"""# Structural-opportunity normalization

## Result

Candidate proteins contained {int(candidate_summary.eligible_sequence_pairs):,} eligible sequence-pair opportunities and {int(candidate_summary.recurrent_pair_count)} recurrent qualifying residue-pair events ({candidate_summary.recurrent_pairs_per_10000_opportunities:.2f} per 10,000). Background proteins contained {int(background_summary.eligible_sequence_pairs):,} opportunities and {int(background_summary.recurrent_pair_count)} recurrent events ({background_summary.recurrent_pairs_per_10000_opportunities:.2f} per 10,000).

The protein-level robust Poisson rate ratio for candidate versus background was {primary.effect_candidate_vs_background:.2f} (95% CI {primary.ci_low:.2f}-{primary.ci_high:.2f}; P={primary.p_value:.3f}). The stratified protein-bootstrap interval was {primary.bootstrap_ci_low:.2f}-{primary.bootstrap_ci_high:.2f}, and the two-sided protein-label permutation P value was {primary.protein_label_permutation_p:.3f}. Thus, normalization for the actual number of searchable donor pairs did not reveal candidate enrichment.

## Frozen definitions

- Eligible sequence opportunity: unique Cys/Met/His residue pair with sequence separation >=10 aa and at least one Cys or Met.
- Structure-evaluable model-pair: eligible pair with minimum local pLDDT >=70, symmetric pair PAE <=10 A, and mean selected-donor SASA >=5 A2, before applying the distance criterion.
- Qualifying model-pair event: structure-evaluable pair with selected donor atoms 2.5-5.0 A apart.
- Recurrent pair-event: the same qualifying pair in at least two of three models.
- Recurrent pocket: connected component of recurrent pair-events sharing donor residues.

## Interpretation

The analysis estimates structural-opportunity-normalized geometry rates, not Cu occupancy, metal identity, or NCC1-mediated transfer. Pair-events within a protein are chemically correlated; therefore inference and resampling were performed with proteins, rather than individual pairs or models, as the sampling unit.

One candidate protein (MtrunR108HiC_021033.1) had sequence-level opportunities but no model-pair passing the non-distance confidence/exposure criteria. It remained in the primary recurrent-pair and binary analyses and was excluded only from the secondary evaluable-model-pair rate analysis.
"""
    (OUTPUT_DIR / "structure_opportunity_normalization_report.md").write_text(report, encoding="utf-8")

    audit = {
        "input_pair_rows": int(len(pairs)),
        "input_protein_rows": int(len(proteins)),
        "structure_included_proteins": int(len(metrics)),
        "candidate_proteins": int((metrics["group"] == "Candidate").sum()),
        "background_proteins": int((metrics["group"] == "Background").sum()),
        "models_per_protein": sorted(metrics["models_available"].unique().astype(int).tolist()),
        "zero_evaluable_model_pair_proteins": metrics.loc[
            metrics["evaluable_model_pairs"] == 0, "protein_id"
        ].tolist(),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "permutation_iterations": PERMUTATION_ITERATIONS,
        "random_seed": SEED,
        "status": "PASS",
    }
    (OUTPUT_DIR / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
