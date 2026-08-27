from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path('.')
OUT = ROOT / 'outputs' / 'robustness_5seed5model_frozen6'
RAW = ROOT / 'Figure 2 kaggle raw data'

FROZEN = [
    'MtrunR108HiC_012482.1',
    'MtrunR108HiC_005650.1',
    'MtrunR108HiC_013677.1',
    'MtrunR108HiC_001767.1',
    'MtrunR108HiC_008307.1',
    'MtrunR108HiC_031716.1',
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = OUT / 'input_a3m'
    inputs.mkdir(exist_ok=True)
    master = pd.read_csv(ROOT / 'outputs/candidate_background_unified/protein_level_master.csv', low_memory=False)
    source = {r['prediction_id']: r for r in json.loads((ROOT/'work/figure2/Figure2_candidate_manifest.json').read_text())['records']}
    rows = []
    for order, pid in enumerate(FROZEN, 1):
        hit = sorted(RAW.rglob(f'{pid}.a3m'))
        if len(hit) != 1:
            raise RuntimeError(f'Expected one A3M for {pid}, found {hit}')
        shutil.copy2(hit[0], inputs / hit[0].name)
        r = master.loc[master.protein_id.eq(pid)].iloc[0]
        rows.append({
            'frozen_order': order,
            'protein_id': pid,
            'sequence_length': int(source[pid]['sequence_length']),
            'r108_annotation': r.r108_annotation,
            'frozen_primary_pair': r.best_primary_pair,
            'donor_combo': r.best_primary_donor_combo,
            'original_support_models': int(r.best_primary_support_models),
            'original_median_distance_A': r.best_primary_distance_A,
            'original_min_plddt': r.best_primary_min_plddt,
            'original_pair_pae_A': r.best_primary_pae_A,
            'original_mean_sasa_A2': r.best_primary_mean_sasa_A2,
            'localization': r.deeploc2_1_localizations,
            'nodule_median_tmm': r.nodule_median_tmm,
            'median_project_log2fc_nodule_vs_root': r.median_project_log2fc_nodule_vs_root,
            'integrated_score_0_100': r.score_normalized_0_100,
            'selection_basis': 'Pre-frozen before 5-seed/5-model rerun: primary-geometry positive, NCC1-compartment compatible, and selected to span rank/evidence and donor-combination diversity.',
        })
    pd.DataFrame(rows).to_csv(OUT/'frozen6_manifest.csv', index=False)
    (OUT/'analysis_specification.json').write_text(json.dumps({
        'frozen_candidates': FROZEN,
        'freeze_date': '2026-08-13',
        'input_type': 'reused original ColabFold A3M to isolate model/seed robustness and avoid MSA-server variation',
        'colabfold_parameters': {
            'model_type': 'alphafold2_ptm', 'num_models': 5, 'num_seeds': 5,
            'num_recycles': 3, 'stop_at_score': 100, 'amber_relaxation': False,
        },
        'expected_predictions_per_protein': 25,
        'primary_pair_is_frozen': True,
        'robustness_readout': 'Evaluate the frozen residue pair in all 25 predictions; report qualifying fraction and seed/model consistency. Do not select a new best pair post hoc.',
        'qualification_rule': 'Both residues pLDDT >=70; symmetric pair PAE <=10 A; donor distance 2.5-5.0 A; mean selected donor-atom SASA >=5 A2.',
    }, indent=2))
    shutil.make_archive(str(OUT/'NCC1_frozen6_A3M_inputs'), 'zip', inputs)
    print(pd.DataFrame(rows)[['frozen_order','protein_id','sequence_length','frozen_primary_pair','donor_combo','integrated_score_0_100']].to_string(index=False))


if __name__ == '__main__':
    main()
