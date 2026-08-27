from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "outputs" / "sulfur_context_residual_specificity"
MASTER = ROOT / "outputs" / "candidate_background_unified" / "protein_level_master.csv"
RAW_S3 = ROOT / "NCC1 supplemental tables" / "Table S3.xlsx"
QC_S3 = ROOT / "Table S3_R108_mapping_QC.xlsx"
CONTEXT31 = ROOT / "outputs" / "formal31_functional_context_triage" / "formal31_functional_context_triage.csv"

ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
N_BOOT = 1000
RNG_SEED = 20260814


def rebuild_corrected_proteomics() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(RAW_S3, sheet_name="Proteins", header=1)
    qc = pd.read_excel(QC_S3, sheet_name="Proteins", header=1)
    identity_cols = ["Unnamed: 0", "Data Base", "Accession", "Description"]
    identity_checks = {}
    for col in identity_cols:
        same = raw[col].fillna("").astype(str).eq(qc[col].fillna("").astype(str))
        identity_checks[col] = {"matched": int(same.sum()), "total": int(len(same)), "rate": float(same.mean())}
        if not same.all():
            raise RuntimeError(f"Raw and QC Table S3 row identities diverge in {col}")

    control_col = "# PSMs:\nControl (no bait)"
    bait_col = "# PSMs:\nBait"
    control_mascot_col = "Score Mascot:  \nControl (no bait)"
    bait_mascot_col = "Score Mascot: \nBait"
    mapped = qc[["Data Base", "Accession", "R108 protein ID", "Mapping status"]].copy()
    mapped["bait_psm_raw"] = pd.to_numeric(raw[bait_col], errors="coerce")
    mapped["control_psm_raw"] = pd.to_numeric(raw[control_col], errors="coerce")
    mapped["bait_mascot_raw"] = pd.to_numeric(raw[bait_mascot_col], errors="coerce")
    mapped["control_mascot_raw"] = pd.to_numeric(raw[control_mascot_col], errors="coerce")
    mapped = mapped[mapped["Data Base"].astype(str).str.contains("Medicago", case=False, na=False)]
    mapped = mapped[mapped["R108 protein ID"].notna()].copy()
    mapped["protein_id"] = mapped["R108 protein ID"].astype(str).str.strip()

    rows = []
    for protein_id, group in mapped.groupby("protein_id", sort=False):
        raw_control = group["control_psm_raw"].max()
        raw_control_mascot = group["control_mascot_raw"].max()
        rows.append(
            {
                "protein_id": protein_id,
                "bait_psm_corrected": float(group["bait_psm_raw"].max()),
                "control_psm_corrected": float(0 if pd.isna(raw_control) else raw_control),
                "control_psm_missing_in_raw": bool(pd.isna(raw_control)),
                "bait_mascot_corrected": float(group["bait_mascot_raw"].max()),
                "control_mascot_corrected": float(0 if pd.isna(raw_control_mascot) else raw_control_mascot),
                "control_mascot_missing_in_raw": bool(pd.isna(raw_control_mascot)),
                "table_s3_row_count_corrected": int(len(group)),
                "table_s3_accessions_corrected": "; ".join(sorted(set(group["Accession"].dropna().astype(str)))),
            }
        )
    corrected = pd.DataFrame(rows)
    corrected["pull_down_log2_psm_ratio_corrected"] = np.log2(
        (corrected["bait_psm_corrected"] + 1) / (corrected["control_psm_corrected"] + 1)
    )
    audit = {
        "raw_rows": int(len(raw)),
        "qc_rows": int(len(qc)),
        "row_identity_checks": identity_checks,
        "raw_control_psm_missing_rows": int(pd.to_numeric(raw[control_col], errors="coerce").isna().sum()),
        "qc_control_psm_missing_rows": int(pd.to_numeric(qc[control_col], errors="coerce").isna().sum()),
        "qc_rows_where_missing_raw_control_was_replaced_by_15": int(
            (
                pd.to_numeric(raw[control_col], errors="coerce").isna()
                & pd.to_numeric(qc[control_col], errors="coerce").eq(15)
            ).sum()
        ),
        "mapped_r108_proteins": int(corrected["protein_id"].nunique()),
    }
    return corrected, audit


def create_model_dataset() -> tuple[pd.DataFrame, dict]:
    master = pd.read_csv(MASTER, low_memory=False)
    if len(master) != 168 or master["protein_id"].nunique() != 168:
        raise RuntimeError("Expected the frozen 168-protein master at one row per protein")
    corrected, raw_audit = rebuild_corrected_proteomics()
    data = master.merge(corrected, on="protein_id", how="left", validate="one_to_one")
    if data["bait_psm_corrected"].isna().any():
        raise RuntimeError("Corrected Table S3 quantities do not cover all 168 model proteins")

    data["master_control_psm_was_corrupted"] = ~np.isclose(
        data["control_psm"].astype(float), data["control_psm_corrected"].astype(float), equal_nan=True
    )
    data["master_bait_psm_matches_raw"] = np.isclose(
        data["bait_psm"].astype(float), data["bait_psm_corrected"].astype(float), equal_nan=True
    )
    data["observed_log2_bait_plus1"] = np.log2(data["bait_psm_corrected"] + 1)
    data["log2_control_plus1"] = np.log2(data["control_psm_corrected"] + 1)
    data["control_detected"] = data["control_psm_corrected"].gt(0)
    data["log2_length"] = np.log2(data["sequence_length"])
    data["log2_cys_plus1"] = np.log2(data["cysteine_count"] + 1)
    data["log2_met_plus1"] = np.log2(data["methionine_count"] + 1)
    data["cys_per_100aa"] = data["cysteine_count"] / data["sequence_length"] * 100
    data["met_per_100aa"] = data["methionine_count"] / data["sequence_length"] * 100
    data["expression_missing"] = data["nodule_median_tmm"].isna()
    data["log2_nodule_tmm_plus1"] = np.log2(data["nodule_median_tmm"] + 1)

    loc = data["location_compatibility_preliminary"].fillna("")
    data["compartment_model"] = "Unknown"
    data.loc[loc.eq("NCC1-compartment compatible"), "compartment_model"] = "Compatible"
    data.loc[loc.str.startswith("Compartment-incompatible"), "compartment_model"] = "Incompatible"
    data.loc[loc.str.startswith("Mixed"), "compartment_model"] = "Mixed"

    context = pd.read_csv(CONTEXT31)
    context_keep = context[["protein_id", "context_bucket", "disposition_bucket", "primary_class"]].copy()
    data = data.merge(context_keep, on="protein_id", how="left", validate="one_to_one")
    data["context_bucket_model"] = data["context_bucket"].fillna("No formal structure-positive context")
    data["primary_geometry_positive"] = data["primary_geometry_positive"].astype("boolean").fillna(False).astype(bool)
    # Incorporate the manually audited compartment flag for the one background protein where it is explicit.
    data.loc[data["primary_class"].fillna("").str.contains("compartment-incompatible", case=False), "compartment_model"] = "Incompatible"

    audit = {
        "n_rows": int(len(data)),
        "n_unique_proteins": int(data["protein_id"].nunique()),
        "groups": data["group"].value_counts().to_dict(),
        "corrected_bait_missing": int(data["bait_psm_corrected"].isna().sum()),
        "corrected_control_missing_after_zero_encoding": int(data["control_psm_corrected"].isna().sum()),
        "control_detected": data.groupby("group")["control_detected"].sum().astype(int).to_dict(),
        "expression_missing": data.groupby("group")["expression_missing"].sum().astype(int).to_dict(),
        "compartment_categories": {
            f"{group} | {category}": int(count)
            for (group, category), count in data.groupby(["group", "compartment_model"]).size().items()
        },
        "context_categories": {
            f"{group} | {category}": int(count)
            for (group, category), count in data.groupby(["group", "context_bucket_model"]).size().items()
        },
        "master_control_values_changed": int(data["master_control_psm_was_corrupted"].sum()),
        "master_bait_values_matched": int(data["master_bait_psm_matches_raw"].sum()),
        "raw_table_s3_audit": raw_audit,
    }
    return data, audit


COMPARTMENT_LEVELS = ["Unknown", "Compatible", "Incompatible", "Mixed"]
CONTEXT_LEVELS = [
    "No formal structure-positive context",
    "Canonical/conserved enzyme-fold coincidence",
    "Compartment-incompatible/indirect",
    "Native metal/catalytic-site alternative",
    "Scaffold/proteostasis context",
    "Annotation conflict/QC",
]


MODEL_SPECS = {
    "primary_counts_context": {
        "numeric": [
            "log2_length",
            "log2_cys_plus1",
            "log2_met_plus1",
            "log2_control_plus1",
            "control_detected",
            "log2_nodule_tmm_plus1",
            "expression_missing",
        ],
        "categorical": True,
        "description": "Primary background-only ridge model using total Cys/Met burden plus length, corrected control PSM, expression, compartment and formal functional context.",
    },
    "sensitivity_density_context": {
        "numeric": [
            "log2_length",
            "cys_per_100aa",
            "met_per_100aa",
            "log2_control_plus1",
            "control_detected",
            "log2_nodule_tmm_plus1",
            "expression_missing",
        ],
        "categorical": True,
        "description": "Sensitivity model replacing total Cys/Met counts with sequence-length-normalized densities.",
    },
    "sensitivity_chemistry_only": {
        "numeric": [
            "log2_length",
            "log2_cys_plus1",
            "log2_met_plus1",
            "log2_control_plus1",
            "control_detected",
            "log2_nodule_tmm_plus1",
            "expression_missing",
        ],
        "categorical": False,
        "description": "Sensitivity model omitting sparse compartment and functional-context labels.",
    },
    "sensitivity_no_sulfur_context": {
        "numeric": [
            "log2_length",
            "log2_control_plus1",
            "control_detected",
            "log2_nodule_tmm_plus1",
            "expression_missing",
        ],
        "categorical": True,
        "description": "Negative-control sensitivity model omitting both Cys and Met features while retaining length, control, expression, compartment and context.",
    },
}


def design_matrix(data: pd.DataFrame, training_index: np.ndarray, spec: dict) -> tuple[np.ndarray, list[str], dict]:
    numeric = spec["numeric"]
    x_parts = []
    names = []
    medians = {}
    for col in numeric:
        values = data[col].astype(float).copy()
        if values.isna().any():
            median = float(values.iloc[training_index].median())
            values = values.fillna(median)
            medians[col] = median
        x_parts.append(values.to_numpy(float)[:, None])
        names.append(col)
    if spec["categorical"]:
        for level in COMPARTMENT_LEVELS[1:]:
            x_parts.append(data["compartment_model"].eq(level).astype(float).to_numpy()[:, None])
            names.append(f"compartment={level}")
        for level in CONTEXT_LEVELS[1:]:
            x_parts.append(data["context_bucket_model"].eq(level).astype(float).to_numpy()[:, None])
            names.append(f"context={level}")
    return np.hstack(x_parts), names, medians


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict:
    mean_x = x_train.mean(axis=0)
    sd_x = x_train.std(axis=0, ddof=0)
    sd_x = np.where(sd_x < 1e-12, 1.0, sd_x)
    z = (x_train - mean_x) / sd_x
    mean_y = float(y_train.mean())
    centered_y = y_train - mean_y
    coef = np.linalg.pinv(z.T @ z + alpha * np.eye(z.shape[1])) @ z.T @ centered_y
    return {"mean_x": mean_x, "sd_x": sd_x, "mean_y": mean_y, "coef": coef, "alpha": float(alpha)}


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    return model["mean_y"] + ((x - model["mean_x"]) / model["sd_x"]) @ model["coef"]


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    n = len(y)
    ranks = pd.Series(y).rank(method="first").to_numpy()
    strata = np.minimum(3, ((ranks - 1) / max(1, n) * 4).astype(int))
    rng = np.random.default_rng(seed)
    fold_ids = np.full(n, -1, dtype=int)
    for stratum in range(4):
        idx = np.flatnonzero(strata == stratum)
        rng.shuffle(idx)
        for j, row in enumerate(idx):
            fold_ids[row] = j % k
    return [np.flatnonzero(fold_ids == fold) for fold in range(k)]


def select_alpha(x: np.ndarray, y: np.ndarray, seed: int, k: int = 5, repeats: int = 10) -> tuple[float, pd.DataFrame]:
    repeat_rmse = np.zeros((repeats, len(ALPHAS)), dtype=float)
    for repeat in range(repeats):
        folds = stratified_folds(y, min(k, len(y)), seed + repeat * 97)
        preds = np.zeros((len(ALPHAS), len(y)), dtype=float)
        for test in folds:
            train = np.setdiff1d(np.arange(len(y)), test)
            for a_i, alpha in enumerate(ALPHAS):
                preds[a_i, test] = predict_ridge(fit_ridge(x[train], y[train], alpha), x[test])
        repeat_rmse[repeat] = np.sqrt(np.mean((preds - y[None, :]) ** 2, axis=1))
    mean_rmse = repeat_rmse.mean(axis=0)
    se_rmse = repeat_rmse.std(axis=0, ddof=1) / math.sqrt(repeats)
    best = int(np.argmin(mean_rmse))
    threshold = mean_rmse[best] + se_rmse[best]
    eligible = np.flatnonzero(mean_rmse <= threshold)
    selected = int(eligible[-1])  # one-standard-error rule, favoring more shrinkage
    table = pd.DataFrame(
        {
            "alpha": ALPHAS,
            "cv_rmse_mean": mean_rmse,
            "cv_rmse_se": se_rmse,
            "minimum_rmse_alpha": ALPHAS[best],
            "one_se_threshold": threshold,
            "selected": np.arange(len(ALPHAS)) == selected,
        }
    )
    return float(ALPHAS[selected]), table


def nested_oof(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    folds = stratified_folds(y, 10, seed)
    predictions = np.zeros(len(y), dtype=float)
    selected = []
    for fold_no, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(y)), test)
        alpha, _ = select_alpha(x[train], y[train], seed + 1000 + fold_no * 31, k=5, repeats=5)
        predictions[test] = predict_ridge(fit_ridge(x[train], y[train], alpha), x[test])
        selected.append({"outer_fold": fold_no + 1, "n_train": len(train), "n_test": len(test), "selected_alpha": alpha})
    return predictions, pd.DataFrame(selected)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    residual = y - pred
    sst = float(np.sum((y - y.mean()) ** 2))
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 else float("nan")
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual**2) / sst) if sst > 0 else float("nan"),
        "pearson_observed_predicted": corr,
        "mean_residual": float(np.mean(residual)),
    }


def empirical_tail_p(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.array([(1 + np.sum(reference >= value)) / (len(reference) + 1) for value in values], dtype=float)


def empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.array(
        [100 * (np.sum(reference < value) + 0.5 * np.sum(reference == value)) / len(reference) for value in values],
        dtype=float,
    )


def bh_adjust(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p), dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def run_model(data: pd.DataFrame, spec_name: str, bootstrap: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    spec = MODEL_SPECS[spec_name]
    train_mask = data["group"].eq("Background").to_numpy()
    train_idx = np.flatnonzero(train_mask)
    x_all, feature_names, medians = design_matrix(data, train_idx, spec)
    x_bg = x_all[train_idx]
    y_all = data["observed_log2_bait_plus1"].to_numpy(float)
    y_bg = y_all[train_idx]

    alpha, cv_table = select_alpha(x_bg, y_bg, RNG_SEED + len(spec_name), k=10, repeats=20)
    oof_bg, outer_alpha = nested_oof(x_bg, y_bg, RNG_SEED + 300 + len(spec_name))
    full_model = fit_ridge(x_bg, y_bg, alpha)
    full_predictions = predict_ridge(full_model, x_all)
    expected = full_predictions.copy()
    expected[train_idx] = oof_bg
    residual = y_all - expected

    bg_residual = residual[train_idx]
    empirical_p = empirical_tail_p(residual, bg_residual)
    percentile = empirical_percentile(residual, bg_residual)
    candidate_mask = data["group"].eq("Candidate").to_numpy()
    empirical_q = np.full(len(data), np.nan)
    empirical_q[candidate_mask] = bh_adjust(empirical_p[candidate_mask])

    result = pd.DataFrame(
        {
            "protein_id": data["protein_id"],
            f"expected_log2_bait_plus1_{spec_name}": expected,
            f"expected_bait_psm_{spec_name}": np.maximum(0, 2**expected - 1),
            f"residual_specificity_log2_{spec_name}": residual,
            f"background_residual_percentile_{spec_name}": percentile,
            f"empirical_upper_tail_p_{spec_name}": empirical_p,
            f"empirical_bh_q_{spec_name}": empirical_q,
        }
    )

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_coefficient": full_model["coef"],
            "training_mean": full_model["mean_x"],
            "training_sd": full_model["sd_x"],
        }
    )
    coefficient_table["model"] = spec_name
    coefficient_table["selected_alpha"] = alpha

    boot_table = None
    if bootstrap:
        rng = np.random.default_rng(RNG_SEED + 700)
        boot_predictions = np.empty((N_BOOT, len(data)), dtype=np.float32)
        for b in range(N_BOOT):
            sample = rng.integers(0, len(train_idx), size=len(train_idx))
            model = fit_ridge(x_bg[sample], y_bg[sample], alpha)
            boot_predictions[b] = predict_ridge(model, x_all)
        boot_residual_parameter = y_all[None, :] - boot_predictions
        # Predictive uncertainty must include irreducible background residual variation,
        # not only uncertainty in the fitted mean. Sample from nested-OOF background residuals.
        predictive_noise = rng.choice(bg_residual, size=(N_BOOT, len(data)), replace=True)
        boot_residual_predictive = y_all[None, :] - (boot_predictions + predictive_noise)
        boot_table = pd.DataFrame(
            {
                "protein_id": data["protein_id"],
                "expected_bait_psm_boot_median": np.maximum(0, 2 ** np.median(boot_predictions, axis=0) - 1),
                "expected_bait_psm_boot_lo95": np.maximum(0, 2 ** np.percentile(boot_predictions, 2.5, axis=0) - 1),
                "expected_bait_psm_boot_hi95": np.maximum(0, 2 ** np.percentile(boot_predictions, 97.5, axis=0) - 1),
                "residual_specificity_parameter_boot_median": np.median(boot_residual_parameter, axis=0),
                "residual_specificity_parameter_boot_lo95": np.percentile(boot_residual_parameter, 2.5, axis=0),
                "residual_specificity_parameter_boot_hi95": np.percentile(boot_residual_parameter, 97.5, axis=0),
                "residual_specificity_predictive_lo95": np.percentile(boot_residual_predictive, 2.5, axis=0),
                "residual_specificity_predictive_hi95": np.percentile(boot_residual_predictive, 97.5, axis=0),
                "predictive_probability_residual_gt0": np.mean(boot_residual_predictive > 0, axis=0),
            }
        )

    metadata = {
        "model": spec_name,
        "description": spec["description"],
        "outcome": "log2(corrected bait PSM + 1)",
        "training_population": "74 Table S3 non-Table S4 background proteins",
        "scoring_population": "94 candidate proteins; background rows use nested out-of-fold expectations",
        "selected_alpha": alpha,
        "alpha_selection": "Repeated stratified CV with one-standard-error rule",
        "nested_oof_metrics_background": regression_metrics(y_bg, oof_bg),
        "feature_medians_used_for_missing_values": medians,
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    return result, metadata, cv_table.assign(model=spec_name), outer_alpha.assign(model=spec_name), boot_table


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, data_audit = create_model_dataset()
    data.to_csv(OUT / "corrected_model_input_168_proteins.csv", index=False)
    (OUT / "data_quality_audit.json").write_text(json.dumps(data_audit, indent=2, ensure_ascii=False, default=str))

    model_results = []
    model_meta = {}
    cv_tables = []
    outer_tables = []
    coefficient_tables = []
    boot_table = None
    for model_name in MODEL_SPECS:
        result, metadata, cv_table, outer_table, bootstrap_result = run_model(
            data, model_name, bootstrap=model_name == "primary_counts_context"
        )
        model_results.append(result)
        model_meta[model_name] = metadata
        cv_tables.append(cv_table)
        outer_tables.append(outer_table)
        # Rebuild coefficients once with the same frozen design and selected alpha for export.
        train_idx = np.flatnonzero(data["group"].eq("Background").to_numpy())
        x_all, names, _ = design_matrix(data, train_idx, MODEL_SPECS[model_name])
        full = fit_ridge(x_all[train_idx], data.loc[data.group.eq("Background"), "observed_log2_bait_plus1"].to_numpy(), metadata["selected_alpha"])
        coef = pd.DataFrame({"model": model_name, "feature": names, "standardized_coefficient": full["coef"], "training_mean": full["mean_x"], "training_sd": full["sd_x"], "selected_alpha": metadata["selected_alpha"]})
        coefficient_tables.append(coef)
        if bootstrap_result is not None:
            boot_table = bootstrap_result

    scoring = data.copy()
    for result in model_results:
        scoring = scoring.merge(result, on="protein_id", how="left", validate="one_to_one")
    scoring = scoring.merge(boot_table, on="protein_id", how="left", validate="one_to_one")

    candidate_mask = scoring["group"].eq("Candidate")
    residual_cols = [f"residual_specificity_log2_{name}" for name in MODEL_SPECS]
    for name, col in zip(MODEL_SPECS, residual_cols):
        scoring[f"candidate_rank_{name}"] = np.nan
        scoring.loc[candidate_mask, f"candidate_rank_{name}"] = scoring.loc[candidate_mask, col].rank(ascending=False, method="min")
    rank_cols = [f"candidate_rank_{name}" for name in MODEL_SPECS]
    scoring["candidate_rank_median_models"] = scoring[rank_cols].median(axis=1)
    scoring["candidate_rank_spread_models"] = scoring[rank_cols].max(axis=1) - scoring[rank_cols].min(axis=1)
    scoring["robust_top10_all_models"] = candidate_mask & scoring[rank_cols].le(10).all(axis=1)
    scoring["residual_priority_tier"] = "Background training row"
    primary_pct = scoring["background_residual_percentile_primary_counts_context"]
    boot_prob = scoring["predictive_probability_residual_gt0"]
    predictive_lo = scoring["residual_specificity_predictive_lo95"]
    scoring.loc[candidate_mask, "residual_priority_tier"] = "R3: background-like residual"
    scoring.loc[candidate_mask & primary_pct.ge(80) & boot_prob.ge(0.80), "residual_priority_tier"] = "R2: elevated but exploratory residual"
    scoring.loc[candidate_mask & primary_pct.ge(95) & predictive_lo.gt(0) & scoring[rank_cols].le(20).all(axis=1), "residual_priority_tier"] = "R1: robust high residual (exploratory)"
    scoring["interpretation"] = "Residual specificity is an exploratory deviation from the empirical Table S3 background expectation, not a probability of specific binding, Cu binding, or client status."
    scoring = scoring.sort_values(["group", "candidate_rank_median_models", "protein_id"], na_position="last")
    scoring.to_csv(OUT / "residual_specificity_scores_168_proteins.csv", index=False)
    scoring.loc[candidate_mask].sort_values("candidate_rank_median_models").to_csv(OUT / "candidate_residual_specificity_ranking.csv", index=False)
    pd.concat(cv_tables, ignore_index=True).to_csv(OUT / "ridge_alpha_cross_validation.csv", index=False)
    pd.concat(outer_tables, ignore_index=True).to_csv(OUT / "nested_outer_fold_alphas.csv", index=False)
    pd.concat(coefficient_tables, ignore_index=True).to_csv(OUT / "model_coefficients.csv", index=False)

    candidates = scoring[scoring["group"].eq("Candidate")].copy()
    bg = scoring[scoring["group"].eq("Background")].copy()
    correlation = candidates[residual_cols].corr(method="spearman")
    correlation.to_csv(OUT / "candidate_residual_rank_sensitivity_correlation.csv")
    top = candidates.nsmallest(20, "candidate_rank_median_models")
    rng = np.random.default_rng(RNG_SEED + 900)
    candidate_resid = candidates["residual_specificity_log2_primary_counts_context"].to_numpy()
    background_resid = bg["residual_specificity_log2_primary_counts_context"].to_numpy()
    null_means = rng.choice(background_resid, size=(100000, len(candidate_resid)), replace=True).mean(axis=1)
    aggregate_shift = {
        "candidate_mean_residual": float(candidate_resid.mean()),
        "candidate_median_residual": float(np.median(candidate_resid)),
        "background_oof_mean_residual": float(background_resid.mean()),
        "background_oof_median_residual": float(np.median(background_resid)),
        "one_sided_empirical_p_candidate_mean_gt_background": float((1 + np.sum(null_means >= candidate_resid.mean())) / (len(null_means) + 1)),
    }

    summary = {
        "as_of": "2026-08-14",
        "models": model_meta,
        "data_quality": data_audit,
        "candidate_tier_counts": candidates["residual_priority_tier"].value_counts().to_dict(),
        "robust_top10_all_models": candidates.loc[candidates["robust_top10_all_models"], "protein_id"].tolist(),
        "candidate_residual_rank_spearman": correlation.to_dict(),
        "minimum_empirical_p": float(candidates["empirical_upper_tail_p_primary_counts_context"].min()),
        "minimum_bh_q": float(candidates["empirical_bh_q_primary_counts_context"].min()),
        "background_oof_residual_mean": float(bg["residual_specificity_log2_primary_counts_context"].mean()),
        "aggregate_candidate_residual_shift": aggregate_shift,
        "top20_candidate_ids": top["protein_id"].tolist(),
    }
    (OUT / "model_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
