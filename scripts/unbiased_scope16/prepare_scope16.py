from pathlib import Path
import csv

ROOT = Path('.')
AUDIT = ROOT / 'outputs/final_evidence_hierarchy_matrix/table_s4_outside_frozen94_scope_16.csv'
PROTEOME = ROOT / 'medtr.R108.gnmHiC_1.ann1.Y8NH.protein.faa'
OUT = ROOT / 'outputs/unbiased_scope16_screen'


def read_fasta(path):
    records, name, chunks = {}, None, []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('>'):
            if name is not None:
                records[name] = ''.join(chunks).upper()
            token = line[1:].split()[0]
            name = token[token.index('MtrunR108HiC_'):] if 'MtrunR108HiC_' in token else token
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        records[name] = ''.join(chunks).upper()
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = read_fasta(PROTEOME)
    with AUDIT.open(newline='') as handle:
        audit = list(csv.DictReader(handle))
    ids = [r['prediction_id'] for r in audit]
    missing = [pid for pid in ids if pid not in seqs]
    if missing:
        raise RuntimeError(f'Missing R108 sequences: {missing}')

    rows = []
    for r in audit:
        pid, seq = r['prediction_id'], seqs[r['prediction_id']]
        c, m, h = seq.count('C'), seq.count('M'), seq.count('H')
        rows.append({
            **r,
            'sequence_length_verified': len(seq),
            'cysteine_count_verified': c,
            'methionine_count_verified': m,
            'histidine_count_verified': h,
            'total_CMH_donors': c + m + h,
            'sulfur_donors_C_plus_M': c + m,
            'unbiased_pair_screen_eligible': c + m + h >= 2,
            'primary_sulfur_anchored_pair_possible_by_composition': c + m >= 1 and c + m + h >= 2,
            'ntf2_like': 'NTF2' in r['r108_annotation'].upper(),
            'sequence': seq,
        })
    if len(rows) != 16 or len(set(ids)) != 16:
        raise RuntimeError('Scope-16 manifest is not exactly 16 unique proteins')
    fields = list(rows[0])
    with (OUT / 'scope16_unbiased_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (OUT / 'scope16_unbiased_CMH.fasta').open('w') as handle:
        for r in rows:
            handle.write(f">{r['prediction_id']} annotation={r['r108_annotation']} C={r['cysteine_count_verified']} M={r['methionine_count_verified']} H={r['histidine_count_verified']}\n")
            seq = r['sequence']
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i+80] + '\n')
    for start in (0, 8):
        subset = rows[start:start+8]
        with (OUT / f'scope16_batch_{start//8+1:02d}_8.fasta').open('w') as handle:
            for r in subset:
                handle.write(f">{r['prediction_id']}\n{r['sequence']}\n")
    print('protein_id\tlength\tC\tM\tH\tNTF2')
    for r in rows:
        print(f"{r['prediction_id']}\t{r['sequence_length_verified']}\t{r['cysteine_count_verified']}\t{r['methionine_count_verified']}\t{r['histidine_count_verified']}\t{r['ntf2_like']}")


if __name__ == '__main__':
    main()
