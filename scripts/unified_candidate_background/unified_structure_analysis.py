from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('.')
CAND_RAW = ROOT / 'Figure 2 kaggle raw data'
BG_RAW = ROOT / '74 proteins kaggle  raw data'
OUT = ROOT / 'outputs' / 'candidate_background_unified'
CAND_MANIFEST = ROOT / 'work' / 'figure2' / 'Figure2_candidate_manifest.json'
BG_MANIFEST = ROOT / 'outputs' / 'Figure2_structure_analysis' / 'background' / 'TableS3_nonTableS4_background_manifest.csv'
TABLE_S3 = ROOT / 'Table S3_R108_mapping_QC.xlsx'
TABLE_S4 = ROOT / 'NCC1 supplemental tables' / 'Table S4.xlsx'
ANNOT49 = ROOT / 'outputs' / 'Figure2_structure_analysis' / 'nonlocal_49_annotation' / 'nonlocal_49_annotation_master.csv'
ANNOT5 = OUT / 'annotation_extension_5' / 'five_new_primary_positive_annotation_master.csv'
EXPRESSION = OUT / 'mtexpress_expression_features.csv'

LOCAL_PLDDT = 70.0
PAIR_PAE = 10.0
PRIMARY_DISTANCE = 5.0
LOWER_DISTANCE = 2.5
PRIMARY_SEPARATION = 10
PRIMARY_SASA = 5.0
MIN_SUPPORT = 2
SASA_POINTS = 240

DONOR_ATOMS = {
    'CYS': ('SG',),
    'MET': ('SD',),
    'HIS': ('ND1', 'NE2'),
}
DONOR_CODE = {'CYS': 'C', 'MET': 'M', 'HIS': 'H'}
VDW = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'SE': 1.90}


def fibonacci_sphere(n=SASA_POINTS):
    i = np.arange(n, dtype=float)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    return np.column_stack((np.cos(theta) * radius, y, np.sin(theta) * radius))


SPHERE = fibonacci_sphere()


def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_rank(path):
    m = re.search(r'_rank_(\d+)_', path.name)
    return int(m.group(1)) if m else -1


def seq_id_from_path(path):
    return path.name.split('_unrelaxed_rank_')[0]


def score_path_for(pdb):
    return pdb.with_name(pdb.name.replace('_unrelaxed_', '_scores_').replace('.pdb', '.json'))


def collect_models(root, expected_ids):
    grouped = defaultdict(list)
    for path in root.rglob('*_unrelaxed_rank_*.pdb'):
        sid = seq_id_from_path(path)
        if sid in expected_ids:
            grouped[(sid, parse_rank(path))].append(path)
    chosen, duplicates = {}, []
    for key, paths in sorted(grouped.items()):
        hashes = defaultdict(list)
        for p in paths:
            hashes[file_sha256(p)].append(p)
        if len(hashes) > 1:
            raise RuntimeError(f'Non-identical duplicate PDBs for {key}: {paths}')
        selected = sorted(paths, key=lambda p: (len(str(p)), str(p)))[0]
        chosen[key] = selected
        if len(paths) > 1:
            duplicates.append({'protein_id': key[0], 'rank': key[1], 'copies': len(paths), 'selected': str(selected)})
    return chosen, duplicates


def parse_pdb(path):
    atoms, residues = [], defaultdict(list)
    seen = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith('ATOM') or line[16] not in (' ', 'A'):
                continue
            atom_name = line[12:16].strip()
            residue = line[17:20].strip()
            chain = line[21].strip() or 'A'
            try:
                resseq = int(line[22:26])
                coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                bfac = float(line[60:66])
            except ValueError:
                continue
            key = (chain, resseq, atom_name)
            if key in seen:
                continue
            seen.add(key)
            element = line[76:78].strip().upper() or re.sub('[^A-Za-z]', '', atom_name)[:1].upper()
            atom = {'chain': chain, 'resseq': resseq, 'residue': residue, 'atom': atom_name,
                    'coord': coord, 'plddt_pdb': bfac, 'element': element, 'atom_index': len(atoms)}
            atoms.append(atom)
            if residue in DONOR_ATOMS and atom_name in DONOR_ATOMS[residue]:
                residues[(chain, resseq, residue)].append(atom)
    donor_residues = []
    for (chain, resseq, residue), donor_atoms in residues.items():
        donor_residues.append({'chain': chain, 'resseq': resseq, 'residue': residue,
                               'code': DONOR_CODE[residue], 'atoms': donor_atoms})
    donor_residues.sort(key=lambda x: (x['chain'], x['resseq']))
    return atoms, donor_residues


def atom_sasa(atom_coords, atom_radii, target, probe=1.4):
    center = target['coord']
    target_radius = VDW.get(target['element'], 1.7) + probe
    points = center + SPHERE * target_radius
    distances = np.linalg.norm(atom_coords - center, axis=1)
    near = (distances <= target_radius + atom_radii)
    near[target['atom_index']] = False
    blockers = atom_coords[near]
    blocker_radii = atom_radii[near]
    if len(blockers):
        squared = np.sum((points[:, None, :] - blockers[None, :, :]) ** 2, axis=2)
        occluded = np.any(squared < blocker_radii[None, :] ** 2, axis=1)
    else:
        occluded = np.zeros(len(points), dtype=bool)
    return (1.0 - float(occluded.mean())) * 4.0 * math.pi * target_radius ** 2


def symmetric_pae(pae, i, j):
    return float((pae[i - 1][j - 1] + pae[j - 1][i - 1]) / 2.0)


def analyze_model(group, protein_id, rank, pdb, meta):
    score_path = score_path_for(pdb)
    if not score_path.exists():
        raise FileNotFoundError(f'Missing score JSON: {score_path}')
    score = json.loads(score_path.read_text())
    plddt = np.asarray(score['plddt'], dtype=float)
    pae = np.asarray(score['pae'], dtype=float)
    expected_length = int(meta['sequence_length'])
    if len(plddt) != expected_length or pae.shape != (expected_length, expected_length):
        raise ValueError(f'Length/PAE mismatch for {protein_id} rank {rank}')
    atoms, donors = parse_pdb(pdb)
    atom_coords = np.stack([a['coord'] for a in atoms])
    atom_radii = np.asarray([VDW.get(a['element'], 1.7) + 1.4 for a in atoms])
    for donor in donors:
        for atom in donor['atoms']:
            atom['sasa'] = atom_sasa(atom_coords, atom_radii, atom)

    pairs = []
    for li in range(len(donors)):
        for ri in range(li + 1, len(donors)):
            left, right = donors[li], donors[ri]
            if left['chain'] != right['chain']:
                continue
            i, j = left['resseq'], right['resseq']
            if i > j:
                left, right, i, j = right, left, j, i
            atom_choices = []
            for la in left['atoms']:
                for ra in right['atoms']:
                    atom_choices.append((float(np.linalg.norm(la['coord'] - ra['coord'])), la, ra))
            distance, la, ra = min(atom_choices, key=lambda x: x[0])
            combo = '-'.join(sorted((left['code'], right['code'])))
            min_plddt = float(min(plddt[i - 1], plddt[j - 1]))
            pair_pae = symmetric_pae(pae, i, j)
            mean_sasa = float((la['sasa'] + ra['sasa']) / 2.0)
            sulfur_anchored = left['code'] in 'CM' or right['code'] in 'CM'
            pairs.append({
                'group': group, 'protein_id': protein_id, 'rank': rank,
                'residue_i': i, 'residue_j': j, 'residue_i_type': left['code'],
                'residue_j_type': right['code'], 'atom_i': la['atom'], 'atom_j': ra['atom'],
                'pair_key': f"{left['code']}{i}-{right['code']}{j}", 'donor_combo': combo,
                'sulfur_anchored': sulfur_anchored, 'sequence_separation': j - i,
                'donor_distance_A': distance, 'min_local_plddt': min_plddt,
                'pair_pae_A': pair_pae, 'donor_sasa_i_A2': float(la['sasa']),
                'donor_sasa_j_A2': float(ra['sasa']), 'mean_donor_sasa_A2': mean_sasa,
                'base_qc': min_plddt >= LOCAL_PLDDT and pair_pae <= PAIR_PAE,
                'primary_model_pass': (sulfur_anchored and j - i >= PRIMARY_SEPARATION and
                    LOWER_DISTANCE <= distance <= PRIMARY_DISTANCE and min_plddt >= LOCAL_PLDDT and
                    pair_pae <= PAIR_PAE and mean_sasa >= PRIMARY_SASA),
                'pdb_path': str(pdb), 'score_json_path': str(score_path),
            })
    model = {
        'group': group, 'protein_id': protein_id, 'rank': rank,
        'sequence_length': expected_length, 'ptm': float(score.get('ptm', np.nan)),
        'mean_plddt': float(plddt.mean()), 'fraction_plddt_ge70': float((plddt >= 70).mean()),
        'fraction_plddt_ge80': float((plddt >= 80).mean()), 'donor_residue_count_model': len(donors),
        'pdb_path': str(pdb), 'score_json_path': str(score_path),
    }
    return model, pairs


def endpoint_pass(row, distance=5.0, plddt=70.0, separation=10, sasa=5.0, scope='sulfur'):
    if not (LOWER_DISTANCE <= row['donor_distance_A'] <= distance):
        return False
    if row['min_local_plddt'] < plddt or row['pair_pae_A'] > PAIR_PAE:
        return False
    if row['sequence_separation'] < separation or row['mean_donor_sasa_A2'] < sasa:
        return False
    if scope == 'sulfur' and not row['sulfur_anchored']:
        return False
    if scope in {'C-C', 'C-H', 'C-M', 'H-H', 'H-M', 'M-M'} and row['donor_combo'] != scope:
        return False
    return True


def supported_pairs(pair_df, **kwargs):
    if pair_df.empty:
        return pd.DataFrame()
    mask = pair_df.apply(lambda r: endpoint_pass(r, **kwargs), axis=1)
    passing = pair_df[mask]
    if passing.empty:
        return pd.DataFrame()
    return (passing.groupby(['pair_key', 'donor_combo', 'residue_i', 'residue_j'], as_index=False)
            .agg(support_models=('rank', 'nunique'), median_distance_A=('donor_distance_A', 'median'),
                 distance_range_A=('donor_distance_A', lambda s: float(s.max()-s.min())),
                 median_min_plddt=('min_local_plddt', 'median'), median_pae_A=('pair_pae_A', 'median'),
                 median_mean_sasa_A2=('mean_donor_sasa_A2', 'median'))
            .query('support_models >= @MIN_SUPPORT')
            .sort_values(['support_models', 'median_mean_sasa_A2', 'median_pae_A'], ascending=[False, False, True]))


def motifs(sequence):
    seq = sequence.upper()
    return {
        'motif_mxcxxc': bool(re.search(r'M.C..C', seq)),
        'motif_cxxc': bool(re.search(r'C..C', seq)),
        'motif_ch_window_0_4': bool(re.search(r'C.{0,4}H|H.{0,4}C', seq)),
    }


def to_number(series):
    return pd.to_numeric(series, errors='coerce')


def aggregate_table_s3():
    df = pd.read_excel(TABLE_S3, sheet_name='Proteins', header=1)
    df = df[df['Data Base'].astype(str).str.contains('Medicago', case=False, na=False)].copy()
    df = df[df['R108 protein ID'].notna()].copy()
    df['protein_id'] = df['R108 protein ID'].astype(str).str.strip()
    for c in ['# PSMs:\nControl (no bait)', '# PSMs:\nBait', 'Score Mascot:  \nControl (no bait)', 'Score Mascot: \nBait']:
        df[c] = to_number(df[c])
    rows = []
    for pid, g in df.groupby('protein_id'):
        accessions = sorted(set(g['Accession'].dropna().astype(str)))
        status = sorted(set(g['Mapping status'].dropna().astype(str)))
        bait = g['# PSMs:\nBait'].max()
        control = g['# PSMs:\nControl (no bait)'].max()
        rows.append({
            'protein_id': pid, 'table_s3_rows': int(len(g)), 'table_s3_accessions': '; '.join(accessions),
            'mapping_status_table_s3': '; '.join(status), 'bait_psm': bait, 'control_psm': control,
            'bait_mascot_score': g['Score Mascot: \nBait'].max(),
            'control_mascot_score': g['Score Mascot:  \nControl (no bait)'].max(),
            'bait_found': bool(g['Found in Sample:\nBait'].notna().any()),
            'control_found': bool(g['Found in Sample:  \nControl (no bait)'].notna().any()),
            'pull_down_log2_psm_ratio': float(np.log2(((0 if pd.isna(bait) else bait)+1)/((0 if pd.isna(control) else control)+1))),
        })
    return pd.DataFrame(rows)


def read_candidate_annotations():
    frames = []
    for path in (ANNOT49, ANNOT5):
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=['protein_id'])
    df = pd.concat(frames, ignore_index=True, sort=False)
    keep = ['sequence_id', 'deeploc2_1_localizations', 'location_compatibility_preliminary',
            'signalp6_prediction', 'signalp6_sp_probability', 'deeptmhmm_tm_count',
            'deeptmhmm_topology', 'signal_tm_evidence_status', 'domain_context_status',
            'annotation_conflict_status', 'exact_interpro_all_matches', 'working_priority_tier']
    return df[[c for c in keep if c in df.columns]].rename(columns={'sequence_id': 'protein_id'})


def mapping_points(status):
    s = str(status).lower()
    if 'exact unique' in s: return 10.0
    if 'high confidence' in s: return 9.0
    if 'probable' in s: return 6.0
    if 'ambiguous' in s: return 3.0
    if 'low confidence' in s: return 1.0
    return 0.0


def score_rows(df):
    positive_enrichment = np.clip(df['pull_down_log2_psm_ratio'].fillna(0), 0, None)
    scale = float(positive_enrichment.quantile(0.95)) or 1.0
    df['score_table_s4_15'] = np.where(df['table_s4_member'], 15.0, 0.0)
    df['score_pull_down_15'] = np.clip(positive_enrichment / scale, 0, 1) * 10 + np.where(df['bait_found'] & ~df['control_found'], 5.0, 0.0)
    df['score_mapping_10'] = df['mapping_statuses'].map(mapping_points)
    df['score_structure_30'] = (df['primary_geometry_positive'].astype(float) * 15 +
        np.clip(df['best_primary_support_models'].fillna(0)-1, 0, 2) * 4 +
        np.clip((df['best_primary_min_plddt'].fillna(70)-70)/30, 0, 1) * 4 +
        np.clip(df['best_primary_mean_sasa_A2'].fillna(0)/15, 0, 1) * 3)
    df.loc[df['models_available'] < 3, 'score_structure_30'] = np.nan
    loc = df.get('location_compatibility_preliminary', pd.Series(index=df.index, dtype=object)).fillna('')
    tm = to_number(df.get('deeptmhmm_tm_count', pd.Series(index=df.index, dtype=float))).fillna(0)
    sp = df.get('signalp6_prediction', pd.Series(index=df.index, dtype=object)).fillna('')
    df['score_cellular_20'] = np.nan
    annotated = loc.ne('')
    df.loc[annotated, 'score_cellular_20'] = np.select(
        [loc[annotated].str.contains('compatible', case=False) & ~loc[annotated].str.contains('incompatible', case=False),
         loc[annotated].str.contains('incompatible', case=False)], [15.0, 0.0], default=7.5)
    df.loc[annotated, 'score_cellular_20'] += np.where((tm[annotated] == 0) & ~sp[annotated].astype(str).str.contains('SP', case=False), 5.0, 0.0)
    if 'score_expression_10' not in df.columns:
        df['score_expression_10'] = np.nan
    df['score_expression_10'] = to_number(df['score_expression_10'])
    components = ['score_table_s4_15', 'score_pull_down_15', 'score_mapping_10', 'score_structure_30', 'score_cellular_20', 'score_expression_10']
    maxima = dict(zip(components, [15, 15, 10, 30, 20, 10]))
    df['score_raw_observed'] = df[components].fillna(0).sum(axis=1)
    df['score_max_available'] = sum(df[c].notna().astype(float) * m for c, m in maxima.items())
    df['score_normalized_0_100'] = 100 * df['score_raw_observed'] / df['score_max_available']
    df['score_completeness_fraction'] = df['score_max_available'] / 100
    df['score_interpretation'] = 'Descriptive prioritization only; not a probability of Cu binding or NCC1 client status'
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cand_payload = json.loads(CAND_MANIFEST.read_text())['records']
    cand_meta = {r['prediction_id']: r for r in cand_payload if r['cys_pair_screen_eligible'] == 'Yes'}
    bg_df = pd.read_csv(BG_MANIFEST)
    bg_meta = {r['r108_protein_id']: {
        'prediction_id': r['r108_protein_id'], 'r108_gene_id': r['r108_gene_id'],
        'uniprot_accessions': r['uniprot_accession'], 'mapping_statuses': r['mapping_status'],
        'r108_annotation': r['annotation'], 'sequence_length': int(r['sequence_length']),
        'cysteine_count': int(r['cysteine_count']), 'cxxc_count': int(r['cxxc_count']),
        'sequence': r.get('sequence', ''),
    } for _, r in bg_df.iterrows()}
    # Manifest CSV lacks sequence; load exact background FASTA.
    fasta = ROOT/'outputs'/'Figure2_structure_analysis'/'background'/'TableS3_nonTableS4_background_74.fasta'
    current, seqs = None, defaultdict(list)
    for line in fasta.read_text().splitlines():
        if line.startswith('>'):
            current = line[1:].split()[0]
        elif current:
            seqs[current].append(line.strip())
    for pid in bg_meta:
        bg_meta[pid]['sequence'] = ''.join(seqs[pid])

    cand_models, cand_dups = collect_models(CAND_RAW, set(cand_meta))
    bg_models, bg_dups = collect_models(BG_RAW, set(bg_meta))
    audit = {
        'candidate_expected': len(cand_meta), 'candidate_with_any_model': len(set(k[0] for k in cand_models)),
        'candidate_models': len(cand_models), 'background_expected': len(bg_meta),
        'background_with_any_model': len(set(k[0] for k in bg_models)), 'background_models': len(bg_models),
        'candidate_missing': sorted(set(cand_meta)-set(k[0] for k in cand_models)),
        'background_missing': sorted(set(bg_meta)-set(k[0] for k in bg_models)),
        'candidate_duplicate_copies': cand_dups, 'background_duplicate_copies': bg_dups,
    }
    (OUT/'input_audit.json').write_text(json.dumps(audit, indent=2))
    if audit['candidate_models'] != 279 or audit['background_models'] != 222 or audit['background_missing']:
        raise RuntimeError(f'Unexpected structure coverage: {audit}')

    model_rows, pair_rows = [], []
    for group, mapping, meta_map in [('Candidate', cand_models, cand_meta), ('Background', bg_models, bg_meta)]:
        for (pid, rank), path in sorted(mapping.items()):
            model, pairs = analyze_model(group, pid, rank, path, meta_map[pid])
            model_rows.append(model)
            pair_rows.extend(pairs)
    models = pd.DataFrame(model_rows).sort_values(['group', 'protein_id', 'rank'])
    pairs = pd.DataFrame(pair_rows).sort_values(['group', 'protein_id', 'rank', 'residue_i', 'residue_j'])

    protein_rows, consensus_rows = [], []
    endpoint_specs = [
        ('primary_sulfur_d5', dict(distance=5, plddt=70, separation=10, sasa=5, scope='sulfur')),
        ('all_CMH_d5', dict(distance=5, plddt=70, separation=10, sasa=5, scope='all')),
    ]
    endpoint_specs += [(f'distance_{d}A', dict(distance=d, plddt=70, separation=10, sasa=5, scope='sulfur')) for d in (4,5,6)]
    endpoint_specs += [(f'plddt_{p}', dict(distance=5, plddt=p, separation=10, sasa=5, scope='sulfur')) for p in (70,80)]
    endpoint_specs += [(f'sasa_{str(s).replace(".","p")}', dict(distance=5, plddt=70, separation=10, sasa=s, scope='sulfur')) for s in (0,2.5,5,10)]
    endpoint_specs += [(f'nonlocal_{sep}aa', dict(distance=5, plddt=70, separation=sep, sasa=5, scope='sulfur')) for sep in (5,10,20)]
    endpoint_specs += [(f'combo_{scope.replace("-","")}', dict(distance=5, plddt=70, separation=10, sasa=5, scope=scope)) for scope in ('C-C','C-H','C-M','H-H','H-M','M-M')]

    for group, meta_map in [('Candidate', cand_meta), ('Background', bg_meta)]:
        for pid, meta in sorted(meta_map.items()):
            m = models[(models.group == group) & (models.protein_id == pid)]
            p = pairs[(pairs.group == group) & (pairs.protein_id == pid)]
            sequence = meta['sequence']
            row = {
                'group': group, 'protein_id': pid, 'r108_gene_id': meta.get('r108_gene_id',''),
                'uniprot_accessions': meta.get('uniprot_accessions',''), 'mapping_statuses': meta.get('mapping_statuses',''),
                'r108_annotation': meta.get('r108_annotation',''), 'sequence_length': int(meta['sequence_length']),
                'cysteine_count': sequence.upper().count('C'), 'methionine_count': sequence.upper().count('M'),
                'histidine_count': sequence.upper().count('H'), 'donor_residue_count': sum(sequence.upper().count(x) for x in 'CMH'),
                'models_available': int(m['rank'].nunique()), 'mean_ptm': float(m.ptm.mean()),
                'mean_model_plddt': float(m.mean_plddt.mean()),
                'mean_fraction_plddt_ge70': float(m.fraction_plddt_ge70.mean()),
                'mean_fraction_plddt_ge80': float(m.fraction_plddt_ge80.mean()),
                **motifs(sequence),
            }
            row['motif_any_frozen'] = row['motif_mxcxxc'] or row['motif_cxxc'] or row['motif_ch_window_0_4']
            has_complete_structure = int(m['rank'].nunique()) == 3
            row['structure_analysis_included'] = has_complete_structure
            row['structure_exclusion_reason'] = '' if has_complete_structure else 'Technical missing: repeated GPU memory exhaustion; not counted as geometry-negative'
            primary = supported_pairs(p, distance=5, plddt=70, separation=10, sasa=5, scope='sulfur')
            row['primary_geometry_positive'] = (not primary.empty) if has_complete_structure else np.nan
            if has_complete_structure and not primary.empty:
                best = primary.iloc[0]
                row.update({
                    'best_primary_pair': best.pair_key, 'best_primary_donor_combo': best.donor_combo,
                    'best_primary_support_models': int(best.support_models),
                    'best_primary_distance_A': float(best.median_distance_A),
                    'best_primary_distance_range_A': float(best.distance_range_A),
                    'best_primary_min_plddt': float(best.median_min_plddt),
                    'best_primary_pae_A': float(best.median_pae_A),
                    'best_primary_mean_sasa_A2': float(best.median_mean_sasa_A2),
                })
                for _, site in primary.iterrows():
                    consensus_rows.append({'group': group, 'protein_id': pid, 'endpoint': 'primary', **site.to_dict()})
            else:
                row.update({'best_primary_pair':'', 'best_primary_donor_combo':'', 'best_primary_support_models':0,
                            'best_primary_distance_A':np.nan, 'best_primary_distance_range_A':np.nan,
                            'best_primary_min_plddt':np.nan, 'best_primary_pae_A':np.nan,
                            'best_primary_mean_sasa_A2':np.nan})
            for name, spec in endpoint_specs:
                row[name] = (not supported_pairs(p, **spec).empty) if has_complete_structure else np.nan
            protein_rows.append(row)
    proteins = pd.DataFrame(protein_rows)
    proteins['table_s4_member'] = proteins.group.eq('Candidate')
    proteins = proteins.merge(aggregate_table_s3(), on='protein_id', how='left')
    proteins['bait_found'] = proteins['bait_found'].fillna(False).astype(bool)
    proteins['control_found'] = proteins['control_found'].fillna(False).astype(bool)
    proteins = proteins.merge(read_candidate_annotations(), on='protein_id', how='left')
    if EXPRESSION.exists():
        expression = pd.read_csv(EXPRESSION, low_memory=False)
        expression = expression.drop(columns=['group'], errors='ignore')
        proteins = proteins.merge(expression, on='protein_id', how='left')
    proteins = score_rows(proteins)
    proteins['expression_status'] = proteins.get('expression_status', pd.Series(index=proteins.index, dtype=object)).fillna(
        'No reliable unique A17 locus mapping or locus absent from MtExpress matrices; not scored')
    proteins['analysis_note'] = np.where(proteins.group.eq('Candidate'),
        'Table S4-intersection candidate; structural geometry is compatible, not proof of Cu binding/client status',
        'Table S3 non-Table S4 comparison background; not assumed to be a true negative')

    models.to_csv(OUT/'model_level_metrics.csv', index=False)
    pairs.to_csv(OUT/'donor_pair_metrics_within_all_distances.csv.gz', index=False, compression='gzip')
    proteins.sort_values(['group','score_normalized_0_100'], ascending=[True,False]).to_csv(OUT/'protein_level_master.csv', index=False)
    pd.DataFrame(consensus_rows).to_csv(OUT/'primary_consensus_sites.csv', index=False)
    (OUT/'analysis_specification.json').write_text(json.dumps({
        'primary_endpoint': {
            'unit': 'protein', 'same_pair_support_models_min': MIN_SUPPORT,
            'local_plddt_min': LOCAL_PLDDT, 'pair_pae_max_A': PAIR_PAE,
            'sequence_separation_min_aa': PRIMARY_SEPARATION,
            'donor_distance_range_A': [LOWER_DISTANCE, PRIMARY_DISTANCE],
            'mean_selected_donor_atom_sasa_min_A2': PRIMARY_SASA,
            'donors': 'Cys/Met/His pairs with at least one sulfur donor (Cys or Met); His-His is sensitivity only',
        },
        'sensitivity_endpoints': [name for name,_ in endpoint_specs],
        'motif_dictionary': {'MXCXXC':'M.C..C','CXXC':'C..C','Cys-His window':'C.{0,4}H or H.{0,4}C'},
        'interpretation': 'geometry-compatible prioritization; not metal binding or client validation',
        'expression': {
            'source': 'MtExpress V3 (20220901), A17 5.1.9 gene quantification',
            'comparison': 'nine independent projects containing both root and nodule samples; projects are the replicate unit',
            'score': '0-5 nodule abundance plus 0-5 median project log2FC enrichment; only one reliable A17 locus is scored',
            'r108_caveat': 'Two projects use R108-source biological material, but all expression matrices are quantified against A17 5.1.9 gene models',
        },
    }, ensure_ascii=False, indent=2))
    summary = {
        'candidate_expected': int((proteins.group=='Candidate').sum()),
        'candidate_analyzed': int(((proteins.group=='Candidate') & proteins.structure_analysis_included).sum()),
        'candidate_technical_missing': proteins.loc[(proteins.group=='Candidate') & ~proteins.structure_analysis_included, 'protein_id'].tolist(),
        'background_analyzed': int((proteins.group=='Background').sum()),
        'models_analyzed': int(len(models)), 'pair_rows': int(len(pairs)),
        'candidate_primary_positive': int(proteins.loc[proteins.group=='Candidate','primary_geometry_positive'].sum()),
        'background_primary_positive': int(proteins.loc[proteins.group=='Background','primary_geometry_positive'].sum()),
    }
    (OUT/'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
