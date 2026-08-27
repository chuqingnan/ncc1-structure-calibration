#!/usr/bin/env python3
"""Analyze 5-seed/5-model NCC1(1-78)-candidate ColabFold multimer results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


BAIT_LEN = 78
CANDIDATE_LEN = 586
SITE_RESIDUES = (478, 501)
BAIT_CYS = (22, 25)


def median(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.median(values) if values else None


def mean(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.fmean(values) if values else None


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def parse_pdb(path: Path):
    atoms = defaultdict(list)
    bfac = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip() or "_"
        resnum = int(line[22:26])
        element = line[76:78].strip() or atom_name[0]
        if element.upper() == "H":
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        atoms[(chain, resnum)].append((atom_name, resname, xyz))
        bfac[(chain, resnum)].append(float(line[60:66]))
    return atoms, {k: mean(v) for k, v in bfac.items()}


def min_residue_distance(atoms, residue_a, residue_b):
    aa = atoms.get(residue_a, [])
    bb = atoms.get(residue_b, [])
    return min((dist(x[2], y[2]) for x in aa for y in bb), default=None)


def named_atom(atoms, key, atom_name):
    for name, _, xyz in atoms.get(key, []):
        if name == atom_name:
            return xyz
    return None


def suffix(path: Path, token: str):
    return path.name.split(token, 1)[1].rsplit(".", 1)[0]


def cross_pae(pae, rows, cols):
    return mean(pae[i][j] for i in rows for j in cols)


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def analyze_model(pdb_path: Path, score_path: Path):
    score = json.loads(score_path.read_text())
    atoms, pdb_plddt = parse_pdb(pdb_path)
    chains = sorted({k[0] for k in atoms})
    if len(chains) != 2:
        raise RuntimeError(f"Expected two chains in {pdb_path.name}; found {chains}")
    chain_a, chain_b = chains
    a_res = sorted(k for k in atoms if k[0] == chain_a)
    b_res = sorted(k for k in atoms if k[0] == chain_b)
    if len(a_res) != BAIT_LEN or len(b_res) != CANDIDATE_LEN:
        raise RuntimeError(
            f"Unexpected chain lengths in {pdb_path.name}: {len(a_res)}, {len(b_res)}"
        )

    contacts = {
        5: {"a": set(), "b": set(), "pairs": []},
        8: {"a": set(), "b": set(), "pairs": []},
    }
    # Spatial hash over candidate heavy atoms. Each NCC1 atom only tests the
    # 27 neighboring 8-A cells, preserving exact 5/8-A contact definitions.
    cell_size = 8.0
    b_grid = defaultdict(list)
    for key in b_res:
        for _, _, xyz in atoms[key]:
            cell = tuple(math.floor(v / cell_size) for v in xyz)
            b_grid[cell].append((key[1], xyz))
    pair_sets = {5: set(), 8: set()}
    for key in a_res:
        for _, _, a_xyz in atoms[key]:
            base = tuple(math.floor(v / cell_size) for v in a_xyz)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for b_num, b_xyz in b_grid.get((base[0] + dx, base[1] + dy, base[2] + dz), []):
                            d = dist(a_xyz, b_xyz)
                            if d <= 8.0:
                                pair_sets[8].add((key[1], b_num))
                                if d <= 5.0:
                                    pair_sets[5].add((key[1], b_num))
    for cutoff in (5, 8):
        contacts[cutoff]["pairs"] = sorted(pair_sets[cutoff])
        contacts[cutoff]["a"] = {p[0] for p in pair_sets[cutoff]}
        contacts[cutoff]["b"] = {p[1] for p in pair_sets[cutoff]}

    site_to_bait = {}
    site_to_bait_cys_s = {}
    for site in SITE_RESIDUES:
        site_to_bait[site] = min(
            (min_residue_distance(atoms, (chain_b, site), ak) for ak in a_res),
            default=None,
        )
        site_sd = named_atom(atoms, (chain_b, site), "SD")
        sulfur_distances = []
        if site_sd is not None:
            for cys in BAIT_CYS:
                cys_sg = named_atom(atoms, (chain_a, cys), "SG")
                if cys_sg is not None:
                    sulfur_distances.append(dist(site_sd, cys_sg))
        site_to_bait_cys_s[site] = min(sulfur_distances, default=None)

    m478 = named_atom(atoms, (chain_b, 478), "SD")
    m501 = named_atom(atoms, (chain_b, 501), "SD")
    pair_distance = dist(m478, m501) if m478 is not None and m501 is not None else None

    plddt = score.get("plddt", [])
    pae = score.get("pae", [])
    a_idx = list(range(BAIT_LEN))
    b_idx = list(range(BAIT_LEN, BAIT_LEN + CANDIDATE_LEN))
    cross_chain_pae = None
    interface_pair_pae_5A = None
    interface_pair_pae_8A = None
    site_pae = {site: None for site in SITE_RESIDUES}
    if pae and len(pae) == BAIT_LEN + CANDIDATE_LEN:
        cross_chain_pae = mean([
            cross_pae(pae, a_idx, b_idx),
            cross_pae(pae, b_idx, a_idx),
        ])
        for cutoff in (5, 8):
            pair_values = []
            for a_num, b_num in contacts[cutoff]["pairs"]:
                ai = a_num - 1
                bi = BAIT_LEN + b_num - 1
                pair_values.extend((pae[ai][bi], pae[bi][ai]))
            if cutoff == 5:
                interface_pair_pae_5A = mean(pair_values)
            else:
                interface_pair_pae_8A = mean(pair_values)
        for site in SITE_RESIDUES:
            idx = BAIT_LEN + site - 1
            site_pae[site] = mean([
                mean(pae[idx][j] for j in a_idx),
                mean(pae[j][idx] for j in a_idx),
            ])

    match = re.search(r"rank_(\d+).*model_(\d+).*seed_(\d+)", pdb_path.name)
    row = {
        "pdb_file": pdb_path.name,
        "score_file": score_path.name,
        "rank": int(match.group(1)) if match else None,
        "model": int(match.group(2)) if match else None,
        "seed": int(match.group(3)) if match else None,
        "iptm": score.get("iptm"),
        "ptm": score.get("ptm"),
        "ranking_confidence": score.get("ranking_confidence"),
        "mean_plddt_bait": mean(plddt[:BAIT_LEN]) if plddt else mean(pdb_plddt[k] for k in a_res),
        "mean_plddt_candidate": mean(plddt[BAIT_LEN:]) if plddt else mean(pdb_plddt[k] for k in b_res),
        "cross_chain_mean_pae": cross_chain_pae,
        "interface_pair_mean_pae_5A": interface_pair_pae_5A,
        "interface_pair_mean_pae_8A": interface_pair_pae_8A,
        "interface_bait_residue_count_5A": len(contacts[5]["a"]),
        "interface_candidate_residue_count_5A": len(contacts[5]["b"]),
        "interface_residue_pair_count_5A": len(contacts[5]["pairs"]),
        "candidate_interface_residues_5A": ";".join(map(str, sorted(contacts[5]["b"]))),
        "bait_interface_residues_5A": ";".join(map(str, sorted(contacts[5]["a"]))),
        "interface_bait_residue_count_8A": len(contacts[8]["a"]),
        "interface_candidate_residue_count_8A": len(contacts[8]["b"]),
        "interface_residue_pair_count_8A": len(contacts[8]["pairs"]),
        "candidate_interface_residues_8A": ";".join(map(str, sorted(contacts[8]["b"]))),
        "bait_interface_residues_8A": ";".join(map(str, sorted(contacts[8]["a"]))),
        "M478_M501_SD_distance_A": pair_distance,
    }
    for site in SITE_RESIDUES:
        row[f"M{site}_plddt"] = plddt[BAIT_LEN + site - 1] if plddt else pdb_plddt.get((chain_b, site))
        row[f"M{site}_to_bait_min_heavy_A"] = site_to_bait[site]
        row[f"M{site}_to_bait_C22_C25_S_min_A"] = site_to_bait_cys_s[site]
        row[f"M{site}_cross_chain_mean_pae"] = site_pae[site]
        row[f"M{site}_contact_5A"] = site_to_bait[site] is not None and site_to_bait[site] <= 5.0
        row[f"M{site}_near_interface_8A"] = site_to_bait[site] is not None and site_to_bait[site] <= 8.0
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/NCC1_Mtr012482_complex/multimer_analysis"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(args.results_dir.glob("*_unrelaxed_*.pdb"))
    scores = {suffix(p, "_scores_"): p for p in args.results_dir.glob("*_scores_*.json")}
    rows = []
    for pdb in pdbs:
        key = suffix(pdb, "_unrelaxed_")
        if key not in scores:
            raise RuntimeError(f"No score JSON matching {pdb.name}")
        rows.append(analyze_model(pdb, scores[key]))
    if not rows:
        raise RuntimeError("No multimer models found")

    with (args.out / "multimer_model_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    candidate_contact_frequency = Counter()
    bait_contact_frequency = Counter()
    for row in rows:
        candidate_contact_frequency.update(
            int(x) for x in row["candidate_interface_residues_5A"].split(";") if x
        )
        bait_contact_frequency.update(
            int(x) for x in row["bait_interface_residues_5A"].split(";") if x
        )
    with (args.out / "interface_contact_frequencies.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chain", "residue_number", "models_with_5A_contact", "fraction_of_models"])
        for chain, counts in [("NCC1_1-78", bait_contact_frequency), ("candidate", candidate_contact_frequency)]:
            for residue, count in sorted(counts.items()):
                writer.writerow([chain, residue, count, count / len(rows)])

    interface_sets = {}
    for chain_label, col in [
        ("NCC1_5A", "bait_interface_residues_5A"),
        ("candidate_5A", "candidate_interface_residues_5A"),
        ("NCC1_8A", "bait_interface_residues_8A"),
        ("candidate_8A", "candidate_interface_residues_8A"),
    ]:
        sets = [set(int(x) for x in row[col].split(";") if x) for row in rows]
        similarities = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
        interface_sets[chain_label] = {
            "median_pairwise_jaccard": median(similarities),
            "mean_pairwise_jaccard": mean(similarities),
        }

    with (args.out / "interface_reproducibility.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["interface_definition", "median_pairwise_jaccard", "mean_pairwise_jaccard"])
        for label, values in interface_sets.items():
            writer.writerow([label, values["median_pairwise_jaccard"], values["mean_pairwise_jaccard"]])

    med_iptm = median(r["iptm"] for r in rows)
    med_pae = median(r["cross_chain_mean_pae"] for r in rows)
    med_interface_pae_5 = median(r["interface_pair_mean_pae_5A"] for r in rows)
    med_interface_pae_8 = median(r["interface_pair_mean_pae_8A"] for r in rows)
    candidate_jaccard_5 = interface_sets["candidate_5A"]["median_pairwise_jaccard"]
    candidate_jaccard_8 = interface_sets["candidate_8A"]["median_pairwise_jaccard"]
    bait_jaccard_8 = interface_sets["NCC1_8A"]["median_pairwise_jaccard"]
    site8 = {
        site: sum(r[f"M{site}_near_interface_8A"] for r in rows) / len(rows)
        for site in SITE_RESIDUES
    }
    site5 = {
        site: sum(r[f"M{site}_contact_5A"] for r in rows) / len(rows)
        for site in SITE_RESIDUES
    }
    site_pae = {
        site: median(r[f"M{site}_cross_chain_mean_pae"] for r in rows)
        for site in SITE_RESIDUES
    }
    both8 = sum(
        r["M478_near_interface_8A"] and r["M501_near_interface_8A"] for r in rows
    ) / len(rows)

    if (med_iptm is not None and med_iptm >= 0.75 and
            med_interface_pae_8 is not None and med_interface_pae_8 <= 8 and
            candidate_jaccard_8 is not None and candidate_jaccard_8 >= 0.50):
        interface_verdict = "strong computational interface"
    elif (med_iptm is not None and med_iptm >= 0.60 and
            med_interface_pae_8 is not None and med_interface_pae_8 <= 15 and
            candidate_jaccard_8 is not None and candidate_jaccard_8 >= 0.25):
        interface_verdict = "moderate computational interface"
    else:
        interface_verdict = "weak or inconsistent computational interface"
    if both8 >= 0.60 and all((site_pae[s] or 999) <= 10 for s in SITE_RESIDUES):
        site_verdict = "the frozen M478/M501 pair repeatedly approaches the predicted NCC1 interface"
    else:
        site_verdict = "the frozen M478/M501 pair is not reproducibly positioned at a confident NCC1 interface"

    report = f"""# NCC1(1–78)–MtrunR108HiC_012482.1 multimer analysis

## Completeness

- Models analyzed: {len(rows)} (planned: 25).
- Seeds represented: {', '.join(map(str, sorted(set(r['seed'] for r in rows if r['seed'] is not None))))}.

## Interface confidence

- Median ipTM: {fmt(med_iptm)}.
- Median bidirectional cross-chain PAE: {fmt(med_pae)} Å.
- Median interface-local bidirectional PAE (5 Å contacts): {fmt(med_interface_pae_5)} Å.
- Median interface-local bidirectional PAE (8 Å contacts): {fmt(med_interface_pae_8)} Å.
- Median pairwise interface Jaccard, candidate residues: {fmt(candidate_jaccard_5)} (5 Å) and {fmt(candidate_jaccard_8)} (8 Å).
- Median pairwise interface Jaccard, NCC1 residues (8 Å): {fmt(bait_jaccard_8)}.
- Operational classification: **{interface_verdict}**.

## Frozen M478/M501 site relative to NCC1

- M478 contacts NCC1 within 5 Å in {site5[478]:.1%} of models and lies within 8 Å in {site8[478]:.1%}.
- M501 contacts NCC1 within 5 Å in {site5[501]:.1%} of models and lies within 8 Å in {site8[501]:.1%}.
- Both methionines lie within 8 Å of NCC1 in {both8:.1%} of models.
- Median M478 cross-chain PAE: {fmt(site_pae[478])} Å; median M501 cross-chain PAE: {fmt(site_pae[501])} Å.
- Site interpretation: **{site_verdict}**.

## Guardrail

AlphaFold-Multimer does not model Cu(I), oxidation state, metallation, or transfer chemistry. Even a reproducible interface would support only a testable NCC1–client interaction model; it would not demonstrate copper binding or copper transfer. Conversely, a weak multimer result does not disprove a transient, metal-dependent encounter, but it removes structural support for the current M478/M501 delivery-site hypothesis.
"""
    (args.out / "multimer_analysis_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
