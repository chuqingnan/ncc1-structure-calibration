from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(".")
CAL = ROOT / "outputs" / "Figure2_structure_analysis" / "calibration"

META = {
    "1FEE": {
        "role": "Experimental positive control: Cu(I)-ATOX1/HAH1 homodimer",
        "source_url": "https://www.rcsb.org/structure/1FEE",
    },
    "2GGP": {
        "role": "Experimental positive control: Atx1-Cu(I)-Ccc2 transfer complex",
        "source_url": "https://www.rcsb.org/structure/2GGP",
    },
    "8RNZ": {
        "role": "Experimental contrast: RAN1 MBD3 fold lacking canonical CXXC",
        "source_url": "https://www.rcsb.org/structure/8RNZ",
    },
}


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def parse_first_model(path):
    atoms = []
    in_first_model = True
    saw_model = False
    with path.open() as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if saw_model:
                    break
                saw_model = True
                in_first_model = True
                continue
            if line.startswith("ENDMDL") and saw_model:
                break
            if not in_first_model or not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                resseq = int(line[22:26])
            except ValueError:
                continue
            atoms.append({
                "record": line[:6].strip(),
                "atom": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21].strip() or "-",
                "resseq": resseq,
                "element": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
                "xyz": xyz,
            })
    return atoms


rows = []
ligand_rows = []
for pdb_id, meta in META.items():
    path = CAL / f"{pdb_id}.pdb"
    atoms = parse_first_model(path)
    cys_sg = [a for a in atoms if a["resname"] == "CYS" and a["atom"] == "SG"]
    copper = [a for a in atoms if a["element"] == "CU" or a["resname"].startswith("CU")]
    min_sg_pair = None
    min_sg_pair_distance = None
    for i, left in enumerate(cys_sg):
        for right in cys_sg[i + 1:]:
            d = distance(left["xyz"], right["xyz"])
            if min_sg_pair_distance is None or d < min_sg_pair_distance:
                min_sg_pair_distance = d
                min_sg_pair = f"{left['chain']}:C{left['resseq']}--{right['chain']}:C{right['resseq']}"

    copper_ligands = []
    for cu in copper:
        nearby = []
        for atom in atoms:
            if atom is cu or atom["element"] not in {"S", "N", "O", "SE"}:
                continue
            d = distance(cu["xyz"], atom["xyz"])
            if d <= 3.0:
                nearby.append((d, atom))
        for d, atom in sorted(nearby):
            label = f"{atom['chain']}:{atom['resname']}{atom['resseq']}:{atom['atom']}"
            copper_ligands.append(f"{label} {d:.2f} A")
            ligand_rows.append({
                "pdb_id": pdb_id,
                "cu_label": f"{cu['chain']}:{cu['resname']}{cu['resseq']}",
                "ligand": label,
                "element": atom["element"],
                "cu_ligand_distance_A": round(d, 3),
            })

    rows.append({
        "pdb_id": pdb_id,
        "role": meta["role"],
        "source_url": meta["source_url"],
        "atoms_first_model": len(atoms),
        "cysteine_sg_count": len(cys_sg),
        "copper_atom_count": len(copper),
        "nearest_cys_sg_pair": min_sg_pair or "",
        "nearest_cys_sg_distance_A": round(min_sg_pair_distance, 3) if min_sg_pair_distance is not None else None,
        "cu_ligands_within_3A": "; ".join(copper_ligands),
        "interpretation": (
            "Direct metal-bound calibration" if copper and copper_ligands
            else "Fold/functional contrast; no Cu-bound Cys geometry in deposited structure"
        ),
    })

(CAL / "calibration_summary.json").write_text(json.dumps({"structures": rows, "cu_ligands": ligand_rows}, indent=2) + "\n")
with (CAL / "calibration_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (CAL / "calibration_cu_ligands.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(ligand_rows[0]))
    writer.writeheader()
    writer.writerows(ligand_rows)

print(json.dumps(rows, indent=2))
