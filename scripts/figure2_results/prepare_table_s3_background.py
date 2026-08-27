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
MANIFEST = ROOT / "work" / "figure2" / "Figure2_candidate_manifest.json"
MAPPING = ROOT / "work" / "ncc1_mapping" / "mapping_results.tsv"
FASTA = ROOT / "medtr.R108.gnmHiC_1.ann1.Y8NH.protein.faa"
OUT = ROOT / "outputs" / "Figure2_structure_analysis" / "background"
USABLE = {"Exact unique", "High confidence", "Probable"}


def read_fasta(path):
    records = {}
    current = None
    chunks = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    records[current] = "".join(chunks).upper()
                raw = line[1:].split()[0]
                match = re.search(r"(MtrunR108HiC_\d+\.\d+)$", raw)
                current = match.group(1) if match else raw
                chunks = []
            else:
                chunks.append(line)
    if current is not None:
        records[current] = "".join(chunks).upper()
    return records


def fisher_two_sided(a, b, c, d):
    n = a + b + c + d
    row1 = a + b
    col1 = a + c

    def probability(x):
        return math.comb(col1, x) * math.comb(n - col1, row1 - x) / math.comb(n, row1)

    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed + 1e-15))


def permutation_mean_pvalue(left, right, iterations=100_000, seed=20260810):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    observed = abs(left.mean() - right.mean())
    combined = np.concatenate([left, right])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        shuffled = rng.permutation(combined)
        diff = abs(shuffled[: len(left)].mean() - shuffled[len(left):].mean())
        extreme += diff >= observed - 1e-12
    return (extreme + 1) / (iterations + 1), observed


OUT.mkdir(parents=True, exist_ok=True)
sequences = read_fasta(FASTA)
candidate_payload = json.loads(MANIFEST.read_text())
selected_records = [r for r in candidate_payload["records"] if r["cys_pair_screen_eligible"] == "Yes"]
selected_ids = {r["prediction_id"] for r in candidate_payload["records"]}

mapping = pd.read_csv(MAPPING, sep="\t")
mapping = mapping[mapping["mapping_status"].isin(USABLE)].copy()
mapping = mapping[~mapping["r108_protein_id"].isin(selected_ids)].copy()
mapping = mapping.sort_values(
    ["r108_protein_id", "identity", "coverage", "identity_margin"],
    ascending=[True, False, False, False],
).drop_duplicates("r108_protein_id")

background = []
for row in mapping.itertuples(index=False):
    sequence = sequences.get(row.r108_protein_id, "")
    if sequence.count("C") < 2:
        continue
    background.append({
        "r108_protein_id": row.r108_protein_id,
        "r108_gene_id": row.r108_gene_id,
        "uniprot_accession": row.accession,
        "mapping_status": row.mapping_status,
        "identity_pct": float(row.identity),
        "coverage_pct": float(row.coverage),
        "annotation": row.r108_annotation,
        "sequence_length": len(sequence),
        "cysteine_count": sequence.count("C"),
        "cxxc_count": len(list(re.finditer(r"C..C", sequence))),
        "sequence": sequence,
    })

selected = pd.DataFrame([{
    "r108_protein_id": r["prediction_id"],
    "sequence_length": int(r["sequence_length"]),
    "cysteine_count": int(r["cysteine_count"]),
    "cxxc_count": int(r["cxxc_count"]),
} for r in selected_records])
background_df = pd.DataFrame(background)

with (OUT / "TableS3_nonTableS4_background_manifest.csv").open("w", newline="") as handle:
    fieldnames = [k for k in background[0] if k != "sequence"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in background:
        writer.writerow({k: row[k] for k in fieldnames})

with (OUT / "TableS3_nonTableS4_background_74.fasta").open("w") as handle:
    for row in background:
        handle.write(f">{row['r108_protein_id']}\n")
        for start in range(0, len(row["sequence"]), 80):
            handle.write(row["sequence"][start:start + 80] + "\n")

selected_cxxc = int((selected["cxxc_count"] > 0).sum())
background_cxxc = int((background_df["cxxc_count"] > 0).sum())
fisher_p = fisher_two_sided(
    selected_cxxc,
    len(selected) - selected_cxxc,
    background_cxxc,
    len(background_df) - background_cxxc,
)

length_p, length_diff = permutation_mean_pvalue(selected["sequence_length"], background_df["sequence_length"])
cys_p, cys_diff = permutation_mean_pvalue(selected["cysteine_count"], background_df["cysteine_count"], seed=20260811)

comparison = {
    "selected_group": "Table S4, usable mapping, >=2 Cys",
    "background_group": "Table S3 non-Table S4, usable mapping, >=2 Cys, unique R108 IDs",
    "selected_n": int(len(selected)),
    "background_n": int(len(background_df)),
    "selected_mean_length": float(selected["sequence_length"].mean()),
    "background_mean_length": float(background_df["sequence_length"].mean()),
    "absolute_mean_length_difference": float(length_diff),
    "length_permutation_p": float(length_p),
    "selected_mean_cysteines": float(selected["cysteine_count"].mean()),
    "background_mean_cysteines": float(background_df["cysteine_count"].mean()),
    "absolute_mean_cysteine_difference": float(cys_diff),
    "cysteine_count_permutation_p": float(cys_p),
    "selected_cxxc_positive": selected_cxxc,
    "background_cxxc_positive": background_cxxc,
    "cxxc_fisher_two_sided_p": float(fisher_p),
    "important_limitation": "This sequence-level comparison does not test 3D geometry enrichment. Structural controls are not yet predicted.",
}
(OUT / "sequence_level_background_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
(OUT / "background_workbook_payload.json").write_text(json.dumps({
    "comparison": comparison,
    "records": [{k: v for k, v in row.items() if k != "sequence"} for row in background],
}, indent=2) + "\n")
pd.DataFrame([comparison]).to_csv(OUT / "sequence_level_background_comparison.csv", index=False)

print(json.dumps(comparison, indent=2))
