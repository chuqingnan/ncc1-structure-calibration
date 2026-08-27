from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path('.')
OUT = ROOT / 'outputs/unbiased_scope16_screen'
RAW = OUT / 'raw_results'
MANIFEST = OUT / 'scope16_unbiased_manifest.csv'
UNIFIED = ROOT / 'work/unified_candidate_background/unified_structure_analysis.py'

spec = importlib.util.spec_from_file_location('unified_structure_analysis', UNIFIED)
usa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usa)


def site_fields(sites, prefix):
    if sites.empty:
        return {
            f'{prefix}_positive': False, f'{prefix}_supported_site_count': 0,
            f'{prefix}_best_pair': '', f'{prefix}_best_combo': '',
            f'{prefix}_support_models': 0, f'{prefix}_distance_A': None,
            f'{prefix}_min_plddt': None, f'{prefix}_pair_pae_A': None,
            f'{prefix}_mean_sasa_A2': None,
        }
    best = sites.iloc[0]
    return {
        f'{prefix}_positive': True,
        f'{prefix}_supported_site_count': len(sites),
        f'{prefix}_best_pair': best.pair_key,
        f'{prefix}_best_combo': best.donor_combo,
        f'{prefix}_support_models': int(best.support_models),
        f'{prefix}_distance_A': float(best.median_distance_A),
        f'{prefix}_min_plddt': float(best.median_min_plddt),
        f'{prefix}_pair_pae_A': float(best.median_pae_A),
        f'{prefix}_mean_sasa_A2': float(best.median_mean_sasa_A2),
    }


def main():
    manifest = pd.read_csv(MANIFEST)
    meta = {r.prediction_id: {
        'sequence_length': int(r.sequence_length_verified),
        'sequence': r.sequence,
    } for r in manifest.itertuples()}
    ids = set(meta)
    models, duplicates = usa.collect_models(RAW, ids)
    coverage = {pid: len([1 for k in models if k[0] == pid]) for pid in ids}

    model_rows, pair_rows = [], []
    for (pid, rank), pdb in sorted(models.items()):
        model, pairs = usa.analyze_model('Scope16', pid, rank, pdb, meta[pid])
        model_rows.append(model); pair_rows.extend(pairs)
    model_df = pd.DataFrame(model_rows)
    pair_df = pd.DataFrame(pair_rows)

    results, sites_out = [], []
    for r in manifest.itertuples():
        pid = r.prediction_id
        complete = coverage[pid] == 3
        p = pair_df[pair_df.protein_id.eq(pid)] if not pair_df.empty else pair_df
        sulfur = usa.supported_pairs(p, distance=5, plddt=70, separation=10, sasa=5, scope='sulfur') if complete else pd.DataFrame()
        all_cmh = usa.supported_pairs(p, distance=5, plddt=70, separation=10, sasa=5, scope='all') if complete else pd.DataFrame()
        for endpoint, sites in [('sulfur_primary', sulfur), ('all_cmh_unbiased', all_cmh)]:
            for _, s in sites.iterrows():
                sites_out.append({'protein_id': pid, 'endpoint': endpoint, **s.to_dict()})
        row = r._asdict()
        row.update({
            'models_available': coverage[pid],
            'structure_complete_3models': complete,
            'mean_ptm': float(model_df.loc[model_df.protein_id.eq(pid), 'ptm'].mean()) if coverage[pid] else None,
            'mean_model_plddt': float(model_df.loc[model_df.protein_id.eq(pid), 'mean_plddt'].mean()) if coverage[pid] else None,
            **site_fields(sulfur, 'sulfur_primary'),
            **site_fields(all_cmh, 'all_cmh_unbiased'),
        })
        if not r.unbiased_pair_screen_eligible:
            row['screen_interpretation'] = 'Composition-negative: fewer than two total Cys/Met/His donor residues; no pair can exist'
        elif not complete:
            row['screen_interpretation'] = 'Technical missing; not geometry-negative'
        elif row['all_cmh_unbiased_positive']:
            row['screen_interpretation'] = 'Reproducible geometry-compatible Cys/Met/His pair; not evidence of Cu binding or NCC1 client status'
        else:
            row['screen_interpretation'] = 'No pair met the frozen consensus geometry/QC endpoint across at least 2 of 3 models'
        results.append(row)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / 'scope16_unbiased_structure_screen.csv', index=False)
    model_df.to_csv(OUT / 'scope16_model_metrics.csv', index=False)
    pair_df.to_csv(OUT / 'scope16_all_donor_pair_metrics.csv.gz', index=False, compression='gzip')
    pd.DataFrame(sites_out).to_csv(OUT / 'scope16_consensus_sites.csv', index=False)
    audit = {
        'expected_proteins': 16,
        'proteins_with_3_models': int(result_df.structure_complete_3models.sum()),
        'models_expected': 48,
        'models_found': len(model_df),
        'missing_or_incomplete': result_df.loc[~result_df.structure_complete_3models, 'prediction_id'].tolist(),
        'composition_negative_lt2_CMH': int((~result_df.unbiased_pair_screen_eligible).sum()),
        'sulfur_primary_positive': int(result_df.sulfur_primary_positive.sum()),
        'all_CMH_unbiased_positive': int(result_df.all_cmh_unbiased_positive.sum()),
        'NTF2_like': result_df.loc[result_df.ntf2_like, ['prediction_id','sulfur_primary_positive','all_cmh_unbiased_positive','all_cmh_unbiased_best_pair','screen_interpretation']].to_dict('records'),
        'duplicate_identical_model_copies': duplicates,
        'frozen_endpoint': {
            'same_pair_support_models_min': 2,
            'donors': 'Cys SG, Met SD, nearest His ND1/NE2',
            'distance_A': [2.5, 5.0],
            'sequence_separation_min_aa': 10,
            'local_plddt_min': 70,
            'pair_pae_max_A': 10,
            'mean_selected_donor_atom_sasa_min_A2': 5,
            'sulfur_primary': 'at least one Cys or Met; identical to frozen 94-protein formal endpoint',
            'all_CMH_unbiased': 'allows H-H in addition; scope-completion sensitivity endpoint',
        },
    }
    (OUT / 'scope16_screen_summary.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
