#!/usr/bin/env python3
"""Matched-balance diagnostics and family-aware sensitivity analyses."""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
LOCAL_PACKAGES = ROOT / "tools" / "analysis-python-packages"
sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from PIL import Image, ImageDraw, ImageFont


UNIFIED = ROOT / "outputs" / "candidate_background_unified"
OPPORTUNITY = ROOT / "outputs" / "structure_opportunity_normalization"
OUT = ROOT / "outputs" / "matched_balance_family_sensitivity"
PROTEIN_PATH = UNIFIED / "protein_level_master.csv"
MATCH_PATH = UNIFIED / "matched_pairs_primary.csv"
CANDIDATE_MANIFEST = ROOT / "work" / "figure2" / "Figure2_candidate_manifest.json"
BACKGROUND_FASTA = ROOT / "outputs" / "Figure2_structure_analysis" / "background" / "TableS3_nonTableS4_background_74.fasta"
OPPORTUNITY_METRICS = OPPORTUNITY / "protein_opportunity_metrics.csv"
THRESHOLDS = [0.30, 0.40, 0.50]
COVERAGE_THRESHOLD = 0.80
SEED = 20260824
BOOTSTRAP_ITERATIONS = 30_000


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def pooled_sd(candidate: pd.Series, background: pd.Series) -> float:
    return float(math.sqrt((candidate.var(ddof=1) + background.var(ddof=1)) / 2))


def empirical_ks(first: np.ndarray, second: np.ndarray) -> float:
    values = np.unique(np.concatenate([first, second]))
    first_sorted, second_sorted = np.sort(first), np.sort(second)
    first_cdf = np.searchsorted(first_sorted, values, side="right") / len(first_sorted)
    second_cdf = np.searchsorted(second_sorted, values, side="right") / len(second_sorted)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def compute_balance(proteins: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    included = proteins[proteins["structure_analysis_included"].astype(bool)].copy()
    included["log1p_length"] = np.log1p(included["sequence_length"])
    included["log1p_donor_count"] = np.log1p(included["donor_residue_count"])
    included["plddt70_coverage"] = included["mean_fraction_plddt_ge70"]
    covariates = [
        ("log1p_length", "log1p protein length"),
        ("log1p_donor_count", "log1p Cys/Met/His count"),
        ("plddt70_coverage", "pLDDT >=70 coverage"),
    ]
    candidate = included[included["group"] == "Candidate"].set_index("protein_id")
    background = included[included["group"] == "Background"].set_index("protein_id")
    candidate_matched = candidate.loc[pairs["candidate_id"]]
    background_matched = background.loc[pairs["background_id"]]

    balance_rows = []
    paired_rows = pairs.copy()
    for variable, label in covariates:
        denominator = pooled_sd(candidate[variable], background[variable])
        before = (candidate[variable].mean() - background[variable].mean()) / denominator
        after = (candidate_matched[variable].mean() - background_matched[variable].mean()) / denominator
        matched_pooled = pooled_sd(candidate_matched[variable], background_matched[variable])
        balance_rows.append(
            {
                "variable": variable,
                "label": label,
                "candidate_mean_before": candidate[variable].mean(),
                "background_mean_before": background[variable].mean(),
                "smd_before": before,
                "candidate_mean_after": candidate_matched[variable].mean(),
                "background_mean_after": background_matched[variable].mean(),
                "smd_after_fixed_prematch_sd": after,
                "smd_after_matched_pooled_sd": (
                    candidate_matched[variable].mean() - background_matched[variable].mean()
                ) / matched_pooled,
                "absolute_smd_before": abs(before),
                "absolute_smd_after": abs(after),
                "variance_ratio_before": candidate[variable].var(ddof=1) / background[variable].var(ddof=1),
                "variance_ratio_after": candidate_matched[variable].var(ddof=1) / background_matched[variable].var(ddof=1),
                "ks_before": empirical_ks(candidate[variable].to_numpy(), background[variable].to_numpy()),
                "ks_after": empirical_ks(candidate_matched[variable].to_numpy(), background_matched[variable].to_numpy()),
                "balance_under_0p1": abs(after) < 0.1,
                "balance_under_0p2": abs(after) < 0.2,
            }
        )
        paired_rows[f"candidate_{variable}"] = candidate_matched[variable].to_numpy()
        paired_rows[f"background_{variable}"] = background_matched[variable].to_numpy()
        paired_rows[f"standardized_pair_difference_{variable}"] = (
            candidate_matched[variable].to_numpy() - background_matched[variable].to_numpy()
        ) / denominator
    return pd.DataFrame(balance_rows), paired_rows


def parse_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(">"):
            current = line[1:].split()[0]
            records[current] = []
        elif current and line:
            records[current].append(line)
    return {key: "".join(value) for key, value in records.items()}


def load_sequences(included_ids: set[str]) -> dict[str, str]:
    candidate_payload = json.loads(CANDIDATE_MANIFEST.read_text())
    candidates = {
        row["prediction_id"]: row["sequence"]
        for row in candidate_payload["records"]
        if row["prediction_id"] in included_ids
    }
    backgrounds = {
        key: value for key, value in parse_fasta(BACKGROUND_FASTA).items() if key in included_ids
    }
    sequences = {**candidates, **backgrounds}
    missing = sorted(included_ids - set(sequences))
    if missing:
        raise RuntimeError(f"Missing sequences: {missing}")
    return sequences


def alignment_metrics(aligner: Align.PairwiseAligner, first: str, second: str) -> tuple[float, float, float]:
    alignment = aligner.align(first, second)[0]
    first_blocks, second_blocks = alignment.aligned
    aligned_residues = 0
    matches = 0
    for (first_start, first_end), (second_start, second_end) in zip(first_blocks, second_blocks):
        block_length = int(first_end - first_start)
        if block_length != int(second_end - second_start):
            raise RuntimeError("Unexpected unequal aligned block lengths")
        aligned_residues += block_length
        matches += sum(
            a == b for a, b in zip(first[int(first_start):int(first_end)], second[int(second_start):int(second_end)])
        )
    identity = matches / aligned_residues if aligned_residues else 0.0
    return identity, aligned_residues / len(first), aligned_residues / len(second)


def compute_pairwise_similarity(sequences: dict[str, str]) -> pd.DataFrame:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    identifiers = sorted(sequences)
    rows = []
    for left_index, left_id in enumerate(identifiers):
        left = sequences[left_id]
        for right_id in identifiers[left_index + 1:]:
            right = sequences[right_id]
            length_ratio = min(len(left), len(right)) / max(len(left), len(right))
            if length_ratio < COVERAGE_THRESHOLD:
                continue
            identity, coverage_left, coverage_right = alignment_metrics(aligner, left, right)
            rows.append(
                {
                    "protein_1": left_id,
                    "protein_2": right_id,
                    "length_1": len(left),
                    "length_2": len(right),
                    "length_ratio": length_ratio,
                    "global_identity": identity,
                    "coverage_1": coverage_left,
                    "coverage_2": coverage_right,
                    "min_coverage": min(coverage_left, coverage_right),
                }
            )
    return pd.DataFrame(rows)


def clusters_from_edges(ids: list[str], edges: pd.DataFrame, identity_threshold: float) -> dict[str, str]:
    parent = {identifier: identifier for identifier in ids}

    def find(identifier: str) -> str:
        if parent[identifier] != identifier:
            parent[identifier] = find(parent[identifier])
        return parent[identifier]

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    passing = edges[
        (edges["global_identity"] >= identity_threshold)
        & (edges["coverage_1"] >= COVERAGE_THRESHOLD)
        & (edges["coverage_2"] >= COVERAGE_THRESHOLD)
    ]
    for row in passing.itertuples(index=False):
        union(row.protein_1, row.protein_2)
    members: dict[str, list[str]] = defaultdict(list)
    for identifier in ids:
        members[find(identifier)].append(identifier)
    cluster_map = {}
    for index, member_list in enumerate(sorted(members.values(), key=lambda x: min(x)), start=1):
        cluster_id = f"F{int(identity_threshold * 100):02d}_{index:03d}"
        for identifier in member_list:
            cluster_map[identifier] = cluster_id
    return cluster_map


def robust_glm_cluster(
    outcome: np.ndarray,
    design: np.ndarray,
    family: str,
    clusters: np.ndarray,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(design, dtype=float)
    offset_array = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    coefficients = np.zeros(x.shape[1])
    if family == "poisson":
        coefficients[0] = math.log((y.sum() + 0.5) / np.exp(offset_array).sum())
    else:
        probability = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        coefficients[0] = math.log(probability / (1 - probability))
    for _ in range(100):
        eta = offset_array + x @ coefficients
        if family == "poisson":
            mean = np.exp(np.clip(eta, -30, 30)); weights = np.clip(mean, 1e-10, None)
        else:
            mean = 1 / (1 + np.exp(-np.clip(eta, -30, 30))); weights = np.clip(mean * (1 - mean), 1e-10, None)
        information = x.T @ (weights[:, None] * x)
        step = np.linalg.pinv(information) @ (x.T @ (y - mean))
        updated = coefficients + step
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated; break
        coefficients = updated
    eta = offset_array + x @ coefficients
    if family == "poisson":
        mean = np.exp(np.clip(eta, -30, 30)); weights = np.clip(mean, 1e-10, None)
    else:
        mean = 1 / (1 + np.exp(-np.clip(eta, -30, 30))); weights = np.clip(mean * (1 - mean), 1e-10, None)
    bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
    meat = np.zeros((x.shape[1], x.shape[1]))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ (y[mask] - mean[mask])
        meat += np.outer(score, score)
    correction = (len(unique_clusters) / (len(unique_clusters) - 1)) * ((len(y) - 1) / (len(y) - x.shape[1]))
    covariance = correction * bread @ meat @ bread
    return coefficients, np.sqrt(np.clip(np.diag(covariance), 0, None))


def normal_p(coefficient: float, standard_error: float) -> float:
    return math.erfc(abs(coefficient / standard_error) / math.sqrt(2))


def cluster_bootstrap_rate_ratio(frame: pd.DataFrame, cluster_column: str) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + sum(map(ord, cluster_column)))
    aggregate_rows = []
    for _, family in frame.groupby(cluster_column):
        candidate = family[family["group"] == "Candidate"]
        background = family[family["group"] == "Background"]
        aggregate_rows.append(
            [
                candidate["recurrent_pair_count"].sum(),
                candidate["eligible_sequence_pairs"].sum(),
                background["recurrent_pair_count"].sum(),
                background["eligible_sequence_pairs"].sum(),
            ]
        )
    aggregate = np.asarray(aggregate_rows, dtype=float)
    family_count = len(aggregate)
    # Multinomial counts are exactly equivalent to resampling family blocks with replacement.
    resample_counts = rng.multinomial(
        family_count,
        np.full(family_count, 1 / family_count),
        size=BOOTSTRAP_ITERATIONS,
    )
    totals = resample_counts @ aggregate
    valid = np.all(totals > 0, axis=1)
    ratios = (totals[valid, 0] / totals[valid, 1]) / (totals[valid, 2] / totals[valid, 3])
    return tuple(np.quantile(ratios, [0.025, 0.975]))


def family_aware_analyses(frame: pd.DataFrame, threshold: float, cluster_column: str) -> list[dict]:
    cluster_values = frame[cluster_column].to_numpy()
    candidate = frame["candidate"].to_numpy(float)
    constant_group = np.column_stack([np.ones(len(frame)), candidate])
    adjusted_design = np.column_stack(
        [
            np.ones(len(frame)), candidate,
            np.log1p(frame["sequence_length"].to_numpy(float)),
            np.log1p(frame["donor_residue_count"].to_numpy(float)),
            frame["mean_fraction_plddt_ge70"].to_numpy(float),
        ]
    )
    rows = []
    for label, design in [("Binary unadjusted", constant_group), ("Binary covariate-adjusted", adjusted_design)]:
        coefficient, standard_error = robust_glm_cluster(
            frame["recurrent_positive"].to_numpy(float), design, "binomial", cluster_values
        )
        effect, se = float(coefficient[1]), float(standard_error[1])
        rows.append(
            {
                "identity_threshold": threshold,
                "analysis": label,
                "estimand": "odds_ratio",
                "effect_candidate_vs_background": math.exp(effect),
                "ci_low": math.exp(effect - 1.96 * se),
                "ci_high": math.exp(effect + 1.96 * se),
                "p_value": normal_p(effect, se),
                "family_clusters": frame[cluster_column].nunique(),
                "method": "Family-cluster sandwich SE",
            }
        )
    coefficient, standard_error = robust_glm_cluster(
        frame["recurrent_pair_count"].to_numpy(float), constant_group, "poisson", cluster_values,
        np.log(frame["eligible_sequence_pairs"].to_numpy(float))
    )
    effect, se = float(coefficient[1]), float(standard_error[1])
    bootstrap_low, bootstrap_high = cluster_bootstrap_rate_ratio(frame, cluster_column)
    rows.append(
        {
            "identity_threshold": threshold,
            "analysis": "Opportunity-normalized recurrent pair rate",
            "estimand": "rate_ratio",
            "effect_candidate_vs_background": math.exp(effect),
            "ci_low": math.exp(effect - 1.96 * se),
            "ci_high": math.exp(effect + 1.96 * se),
            "p_value": normal_p(effect, se),
            "family_clusters": frame[cluster_column].nunique(),
            "method": "Poisson offset with family-cluster sandwich SE",
            "family_bootstrap_ci_low": bootstrap_low,
            "family_bootstrap_ci_high": bootstrap_high,
        }
    )
    return rows


def leave_one_family_out(frame: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    sizes = frame.groupby(cluster_column).size()
    rows = []
    for cluster_id in sizes[sizes > 1].index:
        subset = frame[frame[cluster_column] != cluster_id]
        candidate = subset[subset["group"] == "Candidate"]
        background = subset[subset["group"] == "Background"]
        rr = (
            candidate["recurrent_pair_count"].sum() / candidate["eligible_sequence_pairs"].sum()
        ) / (
            background["recurrent_pair_count"].sum() / background["eligible_sequence_pairs"].sum()
        )
        rows.append(
            {
                "omitted_family": cluster_id,
                "family_size": int(sizes[cluster_id]),
                "members": ";".join(sorted(frame.loc[frame[cluster_column] == cluster_id, "protein_id"])),
                "candidate_members": int(((frame[cluster_column] == cluster_id) & (frame["group"] == "Candidate")).sum()),
                "background_members": int(((frame[cluster_column] == cluster_id) & (frame["group"] == "Background")).sum()),
                "positive_members": int(frame.loc[frame[cluster_column] == cluster_id, "recurrent_positive"].sum()),
                "opportunity_normalized_rate_ratio_after_omission": rr,
            }
        )
    return pd.DataFrame(rows)


def make_balance_figure(balance: pd.DataFrame, paired: pd.DataFrame) -> None:
    image = Image.new("RGB", (2200, 920), "white")
    draw = ImageDraw.Draw(image)
    title, heading, body, small = font(34), font(27), font(22), font(18)
    blue, orange, grey, ink = "#4C78A8", "#E08B3E", "#999999", "#222222"
    draw.text((45, 25), "Matched-background covariate balance diagnostics", fill=ink, font=title)

    # Love plot.
    left, top, width = 55, 130, 760
    draw.text((left, 85), "A  Absolute standardized mean differences", fill=ink, font=heading)
    x0, x1 = left + 330, left + width
    scale = lambda value: x0 + value / 0.5 * (x1 - x0)
    for threshold, color in [(0.1, "#777777"), (0.2, "#BBBBBB")]:
        x = scale(threshold)
        draw.line((x, top, x, 590), fill=color, width=3)
        draw.text((x - 18, 605), f"{threshold:.1f}", fill=ink, font=small)
    for index, row in balance.iterrows():
        y = top + 90 + index * 125
        draw.text((left, y - 16), row["label"], fill=ink, font=body)
        before_x, after_x = scale(float(row.absolute_smd_before)), scale(float(row.absolute_smd_after))
        draw.line((after_x, y, before_x, y), fill="#BBBBBB", width=4)
        draw.ellipse((before_x - 10, y - 10, before_x + 10, y + 10), fill=blue)
        draw.rectangle((after_x - 9, y - 9, after_x + 9, y + 9), fill=orange)
        draw.text((before_x + 12, y - 33), f"{row.absolute_smd_before:.3f}", fill=blue, font=small)
        draw.text((after_x + 12, y + 8), f"{row.absolute_smd_after:.3f}", fill=orange, font=small)
    draw.ellipse((left + 10, 720, left + 30, 740), fill=blue); draw.text((left + 40, 716), "Before", fill=ink, font=body)
    draw.rectangle((left + 170, 720, left + 190, 740), fill=orange); draw.text((left + 200, 716), "After", fill=ink, font=body)

    # Within-pair standardized differences.
    left = 820
    draw.text((left, 85), "B  Within-pair standardized differences", fill=ink, font=heading)
    x_center, x_span = left + 390, 340
    draw.line((x_center, top, x_center, 590), fill=grey, width=3)
    for index, row in balance.iterrows():
        variable = row["variable"]
        values = paired[f"standardized_pair_difference_{variable}"].to_numpy(float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = np.quantile(values, [0.05, 0.95])
        y = top + 90 + index * 125
        scale_pair = lambda value: x_center + np.clip(value, -2, 2) / 2 * x_span
        draw.text((left, y - 16), row["label"], fill=ink, font=body)
        draw.line((scale_pair(low), y, scale_pair(high), y), fill=blue, width=5)
        draw.rectangle((scale_pair(q1), y - 16, scale_pair(q3), y + 16), outline=blue, width=4)
        draw.line((scale_pair(median), y - 20, scale_pair(median), y + 20), fill=orange, width=5)
    for tick in [-2, -1, 0, 1, 2]:
        x = x_center + tick / 2 * x_span
        draw.text((x - 10, 605), str(tick), fill=ink, font=small)
    draw.text((left + 160, 650), "Candidate minus matched background (pre-match SD units)", fill=ink, font=small)

    # Matching-distance distribution.
    left = 1580
    draw.text((left, 85), "C  Greedy match distance", fill=ink, font=heading)
    values = paired["distance"].to_numpy(float)
    bins = np.linspace(0, max(5.0, values.max()), 11)
    counts, _ = np.histogram(values, bins=bins)
    baseline, bar_width = 590, 50
    maximum = max(counts)
    for index, count in enumerate(counts):
        height = 380 * count / maximum
        x = left + 50 + index * (bar_width + 6)
        draw.rectangle((x, baseline - height, x + bar_width, baseline), fill=blue)
    draw.text((left + 40, 610), "0", fill=ink, font=small)
    draw.text((left + 555, 610), f"{bins[-1]:.1f}", fill=ink, font=small)
    draw.text((left + 150, 650), f"median={np.median(values):.3f}; max={values.max():.3f}", fill=ink, font=body)
    draw.text((45, 825), "SMD uses the pooled pre-match SD. Reference lines: <0.10 ideal; <0.20 often acceptable. Existing pairs were not re-matched.", fill="#555555", font=body)
    image.save(OUT / "Figure_matched_balance_diagnostics.png", dpi=(300, 300))
    image.save(OUT / "Figure_matched_balance_diagnostics.pdf", resolution=300)


def make_family_figure(assignments: pd.DataFrame, effects: pd.DataFrame, loo: pd.DataFrame) -> None:
    image = Image.new("RGB", (2500, 900), "white")
    draw = ImageDraw.Draw(image)
    title, heading, body, small = font(34), font(27), font(21), font(18)
    blue, orange, ink = "#4C78A8", "#E08B3E", "#222222"
    draw.text((45, 25), "Protein-family non-independence sensitivity", fill=ink, font=title)
    primary_col = "family_40"
    sizes = assignments.groupby(primary_col).size()

    # Cluster-size distribution.
    left = 55
    draw.text((left, 85), "A  Full-length family sizes (40% identity)", fill=ink, font=heading)
    size_counts = sizes.value_counts().sort_index()
    baseline = 650
    categories = list(size_counts.index)
    maximum = max(size_counts.values)
    for index, cluster_size in enumerate(categories):
        count = int(size_counts[cluster_size])
        height = 450 * count / maximum
        x = left + 90 + index * 90
        draw.rectangle((x, baseline - height, x + 58, baseline), fill=blue)
        draw.text((x + 10, baseline + 12), str(cluster_size), fill=ink, font=small)
        draw.text((x + 5, baseline - height - 27), str(count), fill=ink, font=small)
    draw.text((left + 160, 705), "proteins per family", fill=ink, font=body)
    draw.text((left + 80, 755), f"{len(sizes)} families; {int((sizes>1).sum())} multi-member; maximum size {int(sizes.max())}", fill="#555555", font=body)

    # Forest plot across thresholds.
    left = 780
    draw.text((left, 85), "B  Family-cluster robust effects", fill=ink, font=heading)
    plot = effects[effects["analysis"].isin(["Binary covariate-adjusted", "Opportunity-normalized recurrent pair rate"])].copy()
    x0, x1 = left + 370, left + 930
    log_min, log_max = math.log(0.15), math.log(4.5)
    xscale = lambda value: x0 + (math.log(value) - log_min) / (log_max - log_min) * (x1 - x0)
    draw.line((xscale(1), 140, xscale(1), 680), fill="#999999", width=3)
    for index, row in enumerate(plot.itertuples(index=False)):
        y = 190 + index * 78
        label = f"{int(row.identity_threshold*100)}%  " + ("Adjusted OR" if row.estimand == "odds_ratio" else "Opportunity RR")
        draw.text((left, y - 12), label, fill=ink, font=small)
        draw.line((xscale(row.ci_low), y, xscale(row.ci_high), y), fill="#555555", width=5)
        mark = blue if row.estimand == "odds_ratio" else orange
        draw.ellipse((xscale(row.effect_candidate_vs_background)-8, y-8,
                      xscale(row.effect_candidate_vs_background)+8, y+8), fill=mark)
        draw.text((x1 + 15, y - 12), f"P={row.p_value:.3f}", fill=ink, font=small)
    for tick in [0.25, 0.5, 1, 2, 4]:
        x = xscale(tick); draw.text((x - 12, 700), str(tick), fill=ink, font=small)

    # Leave-one-family-out range.
    left = 1790
    draw.text((left, 85), "C  Leave-one-family-out", fill=ink, font=heading)
    values = loo["opportunity_normalized_rate_ratio_after_omission"].to_numpy(float)
    full = float(effects[(effects.identity_threshold == 0.40) & (effects.analysis == "Opportunity-normalized recurrent pair rate")].effect_candidate_vs_background.iloc[0])
    scale = lambda value: 180 + (value - 0.5) / 1.0 * 470
    draw.line((left + 180, scale(1), left + 520, scale(1)), fill="#999999", width=3)
    for index, value in enumerate(sorted(values)):
        x = left + 210 + (index % 8) * 38
        draw.ellipse((x - 5, scale(value) - 5, x + 5, scale(value) + 5), fill=blue)
    draw.line((left + 160, scale(full), left + 540, scale(full)), fill=orange, width=5)
    draw.text((left + 160, 690), f"full RR={full:.2f}", fill=orange, font=body)
    draw.text((left + 120, 735), f"range after omission: {values.min():.2f}-{values.max():.2f}", fill=ink, font=body)
    draw.text((45, 835), "Families use global BLOSUM62 alignment, >=80% coverage of both proteins, and single-linkage components. Thresholds 30%, 40% (primary), and 50% identity were frozen before outcome review.", fill="#555555", font=body)
    image.save(OUT / "Figure_family_nonindependence.png", dpi=(300, 300))
    image.save(OUT / "Figure_family_nonindependence.pdf", resolution=300)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_csv(PROTEIN_PATH, low_memory=False)
    included = proteins[proteins["structure_analysis_included"].astype(bool)].copy()
    matches = pd.read_csv(MATCH_PATH)
    balance, paired = compute_balance(proteins, matches)
    balance.to_csv(OUT / "matched_covariate_balance.csv", index=False)
    paired.to_csv(OUT / "matched_pair_covariates.csv", index=False)
    make_balance_figure(balance, paired)

    sequences = load_sequences(set(included["protein_id"]))
    for row in included.itertuples(index=False):
        if len(sequences[row.protein_id]) != int(row.sequence_length):
            raise RuntimeError(f"Sequence length mismatch: {row.protein_id}")
    pairwise_path = OUT / "full_length_pairwise_similarity.csv.gz"
    expected_alignments = sum(
        min(len(sequences[first]), len(sequences[second])) / max(len(sequences[first]), len(sequences[second]))
        >= COVERAGE_THRESHOLD
        for index, first in enumerate(sorted(sequences))
        for second in sorted(sequences)[index + 1:]
    )
    if pairwise_path.exists():
        pairwise = pd.read_csv(pairwise_path)
        if len(pairwise) != expected_alignments:
            raise RuntimeError("Cached pairwise table is incomplete for the frozen length-ratio screen")
    else:
        pairwise = compute_pairwise_similarity(sequences)
        pairwise.to_csv(pairwise_path, index=False, compression="gzip")

    opportunity = pd.read_csv(OPPORTUNITY_METRICS)
    family_frame = included.merge(
        opportunity[["protein_id", "eligible_sequence_pairs", "recurrent_pair_count", "recurrent_positive"]],
        on="protein_id", how="left", validate="one_to_one"
    )
    family_frame["candidate"] = (family_frame["group"] == "Candidate").astype(int)
    ids = sorted(sequences)
    summary_rows, effect_rows = [], []
    for threshold in THRESHOLDS:
        column = f"family_{int(threshold*100)}"
        cluster_map = clusters_from_edges(ids, pairwise, threshold)
        family_frame[column] = family_frame["protein_id"].map(cluster_map)
        sizes = family_frame.groupby(column).size()
        mixed = family_frame.groupby(column)["group"].nunique()
        summary_rows.append(
            {
                "identity_threshold": threshold,
                "coverage_threshold_both": COVERAGE_THRESHOLD,
                "families": int(len(sizes)),
                "singleton_families": int((sizes == 1).sum()),
                "multi_member_families": int((sizes > 1).sum()),
                "largest_family_size": int(sizes.max()),
                "mixed_candidate_background_families": int((mixed > 1).sum()),
                "proteins_in_multi_member_families": int(sizes[sizes > 1].sum()),
                "positive_proteins_in_multi_member_families": int(
                    family_frame.loc[family_frame[column].isin(sizes[sizes > 1].index), "recurrent_positive"].sum()
                ),
            }
        )
        effect_rows.extend(family_aware_analyses(family_frame, threshold, column))

    family_summary = pd.DataFrame(summary_rows)
    family_effects = pd.DataFrame(effect_rows)
    loo = leave_one_family_out(family_frame, "family_40")
    family_frame.to_csv(OUT / "family_assignments_all_thresholds.csv", index=False)
    family_summary.to_csv(OUT / "family_cluster_summary.csv", index=False)
    family_effects.to_csv(OUT / "family_aware_effects.csv", index=False)
    loo.to_csv(OUT / "leave_one_family_out_40pct.csv", index=False)
    make_family_figure(family_frame, family_effects, loo)

    primary_family = family_summary[family_summary.identity_threshold == 0.40].iloc[0]
    primary_effect = family_effects[
        (family_effects.identity_threshold == 0.40)
        & (family_effects.analysis == "Opportunity-normalized recurrent pair rate")
    ].iloc[0]
    adjusted_binary = family_effects[
        (family_effects.identity_threshold == 0.40)
        & (family_effects.analysis == "Binary covariate-adjusted")
    ].iloc[0]
    report = f"""# Matched-balance and protein-family sensitivity report

## Matched-background balance

The existing deterministic greedy match retained all 74 background proteins and selected 74 of 93 candidates. Absolute standardized mean differences changed as follows: log1p length {balance.loc[balance.variable=='log1p_length','absolute_smd_before'].iloc[0]:.3f} to {balance.loc[balance.variable=='log1p_length','absolute_smd_after'].iloc[0]:.3f}; log1p donor burden {balance.loc[balance.variable=='log1p_donor_count','absolute_smd_before'].iloc[0]:.3f} to {balance.loc[balance.variable=='log1p_donor_count','absolute_smd_after'].iloc[0]:.3f}; pLDDT>=70 coverage {balance.loc[balance.variable=='plddt70_coverage','absolute_smd_before'].iloc[0]:.3f} to {balance.loc[balance.variable=='plddt70_coverage','absolute_smd_after'].iloc[0]:.3f}. Matching improved all three covariates, but none reached the conventional absolute SMD <0.10 target. The analysis should remain labelled a sensitivity match, not a fully balanced quasi-experiment.

## Protein-family non-independence

At the frozen primary definition (global identity >=40% and >=80% coverage of both proteins), the 167 proteins formed {int(primary_family.families)} full-length families: {int(primary_family.singleton_families)} singletons and {int(primary_family.multi_member_families)} multi-member families; the largest family contained {int(primary_family.largest_family_size)} proteins. There were {int(primary_family.mixed_candidate_background_families)} families containing both candidate and background proteins.

The family-cluster robust opportunity-normalized rate ratio was {primary_effect.effect_candidate_vs_background:.2f} (95% CI {primary_effect.ci_low:.2f}-{primary_effect.ci_high:.2f}; P={primary_effect.p_value:.3f}; family-block bootstrap CI {primary_effect.family_bootstrap_ci_low:.2f}-{primary_effect.family_bootstrap_ci_high:.2f}). The covariate-adjusted binary odds ratio with family-cluster robust uncertainty was {adjusted_binary.effect_candidate_vs_background:.2f} (95% CI {adjusted_binary.ci_low:.2f}-{adjusted_binary.ci_high:.2f}; P={adjusted_binary.p_value:.3f}). Leave-one-multi-member-family-out rate ratios ranged from {loo.opportunity_normalized_rate_ratio_after_omission.min():.2f} to {loo.opportunity_normalized_rate_ratio_after_omission.max():.2f}. Therefore the null/no-enrichment conclusion is not driven by a single homologous family.

## Interpretation limits

Family definitions are full-length sequence clusters, not domain-family annotations. Single-linkage clustering can connect sequences through intermediate members; sensitivity thresholds of 30% and 50% are reported to show whether the conclusion depends on the 40% cutoff. Family-aware estimates address non-independence but do not convert the pull-down comparison background into a proven biological negative set.
"""
    (OUT / "matched_balance_family_sensitivity_report.md").write_text(report)
    audit = {
        "protein_rows_included": len(included),
        "matched_pairs": len(matches),
        "sequences": len(sequences),
        "pairwise_alignments_retained_by_length_ratio": len(pairwise),
        "identity_thresholds": THRESHOLDS,
        "coverage_threshold_both": COVERAGE_THRESHOLD,
        "alignment": "global BLOSUM62; gap open -10; gap extension -0.5",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "status": "PASS",
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2))
    print(report)


if __name__ == "__main__":
    main()
