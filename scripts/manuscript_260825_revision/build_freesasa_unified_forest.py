from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(".")
OUT = ROOT / "outputs" / "manuscript_260825_revision"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ops = load_module("opportunity", ROOT / "work" / "structure_opportunity_normalization" / "run_structure_opportunity_normalization.py")


def robust_glm_cluster(outcome, design, family, clusters, offset=None):
    y, x = np.asarray(outcome, float), np.asarray(design, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    beta = np.zeros(x.shape[1])
    beta[0] = math.log((y.sum()+0.5)/np.exp(off).sum()) if family == "poisson" else math.log(np.clip(y.mean(),1e-5,1-1e-5)/(1-np.clip(y.mean(),1e-5,1-1e-5)))
    for _ in range(100):
        eta = off + x @ beta
        if family == "poisson":
            mu = np.exp(np.clip(eta,-30,30)); w = np.clip(mu,1e-10,None)
        else:
            mu = 1/(1+np.exp(-np.clip(eta,-30,30))); w = np.clip(mu*(1-mu),1e-10,None)
        info = x.T @ (w[:,None]*x)
        updated = beta + np.linalg.pinv(info) @ (x.T @ (y-mu))
        if np.max(np.abs(updated-beta)) < 1e-10:
            beta = updated; break
        beta = updated
    eta = off + x @ beta
    if family == "poisson": mu=np.exp(np.clip(eta,-30,30)); w=np.clip(mu,1e-10,None)
    else: mu=1/(1+np.exp(-np.clip(eta,-30,30))); w=np.clip(mu*(1-mu),1e-10,None)
    bread=np.linalg.pinv(x.T@(w[:,None]*x)); meat=np.zeros((x.shape[1],x.shape[1]))
    unique=np.unique(clusters)
    for cluster in unique:
        mask=clusters==cluster; score=x[mask].T@(y[mask]-mu[mask]); meat += np.outer(score,score)
    correction=(len(unique)/(len(unique)-1))*((len(y)-1)/(len(y)-x.shape[1]))
    cov=correction*bread@meat@bread
    return beta, np.sqrt(np.clip(np.diag(cov),0,None))


def family_effects(frame):
    clusters=frame["family_40"].to_numpy(); candidate=frame["candidate"].to_numpy(float)
    simple=np.column_stack([np.ones(len(frame)),candidate])
    adjusted=np.column_stack([np.ones(len(frame)),candidate,np.log1p(frame.sequence_length),np.log1p(frame.donor_residue_count),frame.mean_fraction_plddt_ge70])
    rows=[]
    for label,design in [("Binary unadjusted",simple),("Binary covariate-adjusted",adjusted)]:
        b,se=robust_glm_cluster(frame.recurrent_positive,design,"binomial",clusters); z=b[1]/se[1]
        rows.append({"analysis":label,"effect_candidate_vs_background":math.exp(b[1]),"ci_low":math.exp(b[1]-1.96*se[1]),"ci_high":math.exp(b[1]+1.96*se[1]),"p_value":math.erfc(abs(z)/math.sqrt(2)),"family_clusters":len(np.unique(clusters))})
    b,se=robust_glm_cluster(frame.recurrent_pair_count,simple,"poisson",clusters,np.log(frame.eligible_sequence_pairs)); z=b[1]/se[1]
    rows.append({"analysis":"Opportunity-normalized recurrent pair rate","effect_candidate_vs_background":math.exp(b[1]),"ci_low":math.exp(b[1]-1.96*se[1]),"ci_high":math.exp(b[1]+1.96*se[1]),"p_value":math.erfc(abs(z)/math.sqrt(2)),"family_clusters":len(np.unique(clusters))})
    return rows


def row(group, label, estimand, effect, low, high, p, method):
    return {
        "group": group, "label": label, "estimand": estimand,
        "effect": float(effect), "low": float(low), "high": float(high),
        "p": float(p), "method": method,
    }


def main() -> None:
    pairs = pd.read_csv(OUT.parent / "validated_endpoint_sensitivity_2026_08_25" / "donor_pair_metrics_with_freesasa.csv.gz")
    proteins = pd.read_csv(ROOT / "outputs" / "candidate_background_unified" / "protein_level_master.csv", low_memory=False)
    pass_mask = (
        pairs["sulfur_anchored"].astype(bool)
        & pairs["donor_distance_A"].between(2.5, 5.0, inclusive="both")
        & pairs["min_local_plddt"].ge(70)
        & pairs["pair_pae_A"].le(10)
        & pairs["sequence_separation"].ge(10)
        & pairs["mean_donor_sasa_A2"].ge(5)
    )
    pairs["primary_model_pass"] = pass_mask
    metrics = ops.build_protein_metrics(pairs, proteins)
    group_summary = ops.summarize_groups(metrics)
    effects = pd.DataFrame([
        ops.robust_poisson_rate_ratio(metrics, "recurrent_pair_count", "eligible_sequence_pairs", "Recurrent residue-pair rate"),
        ops.robust_poisson_rate_ratio(metrics, "recurrent_pocket_count", "eligible_sequence_pairs", "Connected recurrent-pocket rate"),
        ops.robust_poisson_rate_ratio(metrics[metrics["evaluable_model_pairs"] > 0].copy(), "qualifying_model_pair_events", "evaluable_model_pairs", "Qualifying model-pair event rate"),
        ops.adjusted_logistic_odds_ratio(metrics),
    ])
    metrics.to_csv(OUT / "protein_opportunity_metrics_freesasa.csv", index=False)
    group_summary.to_csv(OUT / "group_opportunity_summary_freesasa.csv", index=False)
    effects.to_csv(OUT / "opportunity_normalized_effects_freesasa.csv", index=False)

    families = pd.read_csv(ROOT / "outputs" / "matched_balance_family_sensitivity" / "family_assignments_all_thresholds.csv", low_memory=False)[["protein_id", "family_40"]]
    fm = metrics.merge(families, on="protein_id", validate="one_to_one")
    family_results = pd.DataFrame(family_effects(fm))
    family_results.to_csv(OUT / "family_aware_effects_freesasa.csv", index=False)

    primary = pd.read_csv(OUT / "primary_binary_effect_freesasa.csv").iloc[0]
    regressions = pd.read_csv(OUT / "logistic_effects_freesasa.csv").set_index("model")
    cmh = pd.read_csv(OUT / "cmh_effect_freesasa.csv").iloc[0]
    matched = pd.read_csv(OUT / "matched_effect_freesasa.csv").iloc[0]
    completed = pd.read_csv(OUT / "completed_scope_109_sensitivity_freesasa.csv").query("endpoint == 'Frozen'").iloc[0]
    f_unadj = family_results.query("analysis == 'Binary unadjusted'").iloc[0]
    f_adj = family_results.query("analysis == 'Binary covariate-adjusted'").iloc[0]
    f_rate = family_results.query("analysis == 'Opportunity-normalized recurrent pair rate'").iloc[0]

    rows = [
        row("Protein-level endpoint", "Primary unadjusted", "RR", primary.risk_ratio, primary.rr_ci_low, primary.rr_ci_high, primary.fisher_p, "17/93 candidate versus 10/74 background; true FreeSASA"),
        row("Protein-level endpoint", "CMH length/donor strata", "OR", cmh.common_odds_ratio, cmh.ci_low, cmh.ci_high, cmh.p_value, "7 overlapping length x donor-count strata; n=165"),
        row("Protein-level endpoint", "Length/donor adjusted", "OR", regressions.loc["length_donor_adjusted", "odds_ratio"], regressions.loc["length_donor_adjusted", "ci_low"], regressions.loc["length_donor_adjusted", "ci_high"], regressions.loc["length_donor_adjusted", "p_value"], "Logistic regression; length and Cys/Met/His burden; n=167"),
        row("Protein-level endpoint", "Covariate + structure-quality adjusted", "OR", regressions.loc["quality_adjusted", "odds_ratio"], regressions.loc["quality_adjusted", "ci_low"], regressions.loc["quality_adjusted", "ci_high"], regressions.loc["quality_adjusted", "p_value"], "Logistic regression; length, donor burden, pLDDT>=70 coverage; n=167"),
        row("Protein-level endpoint", "74-pair matched sensitivity", "OR", matched.conditional_odds_ratio, matched.ci_low, matched.ci_high, matched.mcnemar_exact_p, "7 candidate-only versus 6 background-only discordant pairs; exact CI"),
        row("Protein-level endpoint", "Post-freeze completed scope", "RR", completed.risk_ratio, completed.risk_ratio_ci_low, completed.risk_ratio_ci_high, completed.fisher_exact_p, "18/109 candidate versus 10/74 background; true FreeSASA sensitivity"),
        row("Family-aware endpoint", "Family-cluster unadjusted", "OR", f_unadj.effect_candidate_vs_background, f_unadj.ci_low, f_unadj.ci_high, f_unadj.p_value, "40% identity/80% coverage; 140 family clusters"),
        row("Family-aware endpoint", "Family-cluster covariate adjusted", "OR", f_adj.effect_candidate_vs_background, f_adj.ci_low, f_adj.ci_high, f_adj.p_value, "Cluster-robust logistic model; 140 family clusters"),
    ]
    for _, r in effects.iterrows():
        label = "Any recurrent pair, opportunity adjusted" if r.analysis.startswith("Any recurrent") else r.analysis
        rows.append(row("Opportunity-normalized geometry", label, "OR" if r.estimand == "odds_ratio" else "RR", r.effect_candidate_vs_background, r.ci_low, r.ci_high, r.p_value, r.method))
    rows.append(row("Family-aware opportunity", "Family-cluster recurrent-pair rate", "RR", f_rate.effect_candidate_vs_background, f_rate.ci_low, f_rate.ci_high, f_rate.p_value, "Poisson offset with family-cluster robust SE; 140 clusters"))

    source = FIG / "Figure2_unified_effect_forest_FreeSASA_source_data.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    draw_forest(rows)


def font(size: int, bold: bool = False):
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(path, size=size)


def draw_forest(rows):
    width, height = 5000, 2600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((140, 65), "Candidate-to-background effects after FreeSASA correction", font=font(62, True), fill="#111111")
    draw.text((140, 145), "All confidence intervals cross 1; null estimates remain compatible with moderate effects.", font=font(32), fill="#444444")
    draw.line((140, 215, width-140, 215), fill="#D9D9D9", width=4)
    label_x, plot_x0, plot_x1, num_x, p_x = 170, 2350, 3920, 4030, 4700
    for x, label in [(label_x, "Analysis"), (plot_x0, "Candidate / background effect (95% CI)"), (num_x, "Effect (95% CI)"), (p_x, "P")]:
        draw.text((x, 260), label, font=font(34, True), fill="#222222")
    xmin, xmax = 0.15, 5.0
    lx0, lx1 = math.log(xmin), math.log(xmax)
    xpos = lambda v: plot_x0 + (math.log(v)-lx0)/(lx1-lx0)*(plot_x1-plot_x0)
    for tick in [0.2, 0.5, 1, 2, 5]:
        x = xpos(tick)
        draw.line((x, 325, x, 2380), fill="#777777" if tick == 1 else "#E2E2E2", width=6 if tick == 1 else 3)
        lab = str(tick); box = draw.textbbox((0,0), lab, font=font(30))
        draw.text((x-(box[2]-box[0])/2, 2400), lab, font=font(30), fill="#333333")
    colors = {"Protein-level endpoint":"#0072B2", "Family-aware endpoint":"#7A5195", "Opportunity-normalized geometry":"#D55E00", "Family-aware opportunity":"#009E73"}
    y, last = 345, None
    for r in rows:
        if r["group"] != last:
            if last is not None: y += 18
            draw.rounded_rectangle((140,y,width-140,y+52), radius=12, fill="#F2F4F7")
            draw.text((165,y+9), r["group"], font=font(29,True), fill="#2E4053")
            y += 74; last = r["group"]
        color, center = colors[r["group"]], y+30
        draw.text((190,y+5), r["label"], font=font(30), fill="#222222")
        draw.line((xpos(r["low"]),center,xpos(r["high"]),center), fill=color, width=8)
        draw.ellipse((xpos(r["effect"])-13,center-13,xpos(r["effect"])+13,center+13), fill=color, outline="white", width=2)
        draw.text((num_x,y+4), f'{r["effect"]:.2f} ({r["low"]:.2f}–{r["high"]:.2f}) {r["estimand"]}', font=font(27), fill="#222222")
        ptxt = "1.000" if r["p"] == 1 else ("<0.001" if r["p"] < .001 else f'{r["p"]:.3f}')
        draw.text((p_x,y+4), ptxt, font=font(27), fill="#222222")
        y += 85
    draw.text((plot_x0, 2500), "Lower candidate rate", font=font(28), fill="#555555")
    rt="Higher candidate rate"; tw=draw.textbbox((0,0),rt,font=font(28))[2]
    draw.text((plot_x1-tw,2500),rt,font=font(28),fill="#555555")
    png = FIG / "Figure2_unified_effect_forest_FreeSASA.png"
    image.save(png, dpi=(400,400), optimize=True)
    image.save(FIG / "Figure2_unified_effect_forest_FreeSASA.pdf", "PDF", resolution=400)
    legend = ("Figure 2. Candidate-to-background effect estimates after true-FreeSASA correction. "
              "Risk ratios (RR) and odds ratios (OR) are distinct estimands and are not pooled. "
              "The prespecified endpoint (17/93 versus 10/74), adjusted, stratified, matched, completed-scope, "
              "opportunity-normalized and family-cluster estimates all crossed the null. The width of several "
              "intervals, including the primary RR (95% CI 0.66–2.78), precludes an equivalence claim.")
    (FIG / "Figure2_unified_effect_forest_FreeSASA_legend.txt").write_text(legend, encoding="utf-8")
    print(png)


if __name__ == "__main__":
    main()
