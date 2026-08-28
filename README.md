# NCC1 structure-calibration analysis

This repository contains the analysis code, frozen parameter specifications, input sequence sets, processed supplementary tables, and figure-source data supporting the manuscript **“Structural opportunity and native context constrain metal-client inference from predicted protein structures.”**

The study uses the *Medicago truncatula* NCC1 affinity-purification dataset as a worked example for evaluating whether recurrent metal-donor geometries in predicted protein structures distinguish candidate Cu clients from pull-down-derived comparison proteins.

## Main result

Using the FreeSASA-corrected primary endpoint, recurrent structural geometries occurred in 17/93 candidates and 10/74 comparison proteins (risk ratio 1.35, 95% CI 0.66–2.78; Fisher's exact P = 0.526). Opportunity-normalized, adjusted, matched, family-cluster-aware, and completed-scope analyses also crossed the null. These geometries are therefore treated as structural hypotheses rather than evidence of Cu occupancy, direct interaction, or NCC1-mediated transfer.

## Repository contents

- `scripts/` — analysis and figure-generation scripts grouped by analysis stage.
- `config/` — frozen endpoint and prediction specifications.
- `data/input_sequences/` — public R108 protein sequences submitted for structure prediction.
- `data/figure_source/` — compact source tables underlying principal quantitative figures and summaries.
- `data/processed/` — the internally consistent supplementary workbook used for submission.
- `MANIFEST.tsv` — SHA-256 checksum, file size, and relative path for every deposited file.

## Data not stored in Git

Raw ColabFold/AlphaFold coordinate models, PAE files, score JSON files, A3M alignments, and complete analysis intermediates are archived in the versioned Zenodo dataset at [https://doi.org/10.5281/zenodo.22133341](https://doi.org/10.5281/zenodo.22133341).

## Frozen structural endpoint

A protein is positive only when the same residue pair qualifies in at least two of three models. The pair must:

1. use Cys, Met, or His donor atoms and include at least one sulfur donor (Cys or Met);
2. have nearest chemically plausible donor-atom distance 2.5–5.0 Å;
3. have sequence separation at least 10 residues;
4. have pLDDT at least 70 for both residues;
5. have symmetric pair PAE at most 10 Å; and
6. have mean selected donor-atom solvent-accessible surface area at least 5 Å², calculated with FreeSASA 2.2.1 using a 1.4 Å probe.

The sensitivity analyses change only the explicitly named exposure or sequence-separation gate. See `config/freesasa_endpoint.yaml`.

## Reproducibility notes

The scripts preserve the executed analysis logic but expect the Zenodo raw-model dataset to be unpacked at the repository root using the directory names recorded in the parameter specifications. Several scripts represent sequential manuscript-development stages; they are retained for provenance and are not all required to regenerate the final endpoint. The supplementary workbook and figure-source CSV files provide the compact, submission-level outputs.

Recommended environment:

```bash
conda env create -f environment.yml
conda activate ncc1-structure-calibration
```

FreeSASA may require installation through conda-forge on some systems. R analyses require the packages listed in `environment.yml`.

## Versioning

The submission snapshot is tagged `v1.0.0-submission`. Subsequent changes that alter results will receive a new version and an updated manifest.

## Citation

For the analysis code, cite this repository and its tagged release (`v1.0.0-submission`). For the archived dataset and versioned structure-prediction outputs, cite [https://doi.org/10.5281/zenodo.22133341](https://doi.org/10.5281/zenodo.22133341). The article DOI will be added after publication.

## Licenses

Code is released under the MIT License (`LICENSE`). Deposited tables, FASTA collections, manifests, and figure-source data are released under CC BY 4.0 (`LICENSE-DATA`). Source publications and database records retain their original licenses and citation requirements.

## Contact

Qingnan Chu (GitHub: [@chuqingnan](https://github.com/chuqingnan)).
