# pMHC Guardian

Can a pMHC stability model know when it does not know?

This branch (`Dev1`) locks the scientific contract for hours 0–1. It fills in
the decisions from Developer 2's `PLAN.md` (created, then deleted on `main`).
Do not change a locked item silently. Post in chat first.

## Research question

Can predictive uncertainty identify unreliable pMHC stability predictions
under peptide and HLA distribution shift?

## Hypotheses

- **H1.** A model can predict experimental pMHC half-life from peptide and HLA
  representations.
- **H2.** Predicted uncertainty correlates with actual error, and a claimed
  90% interval covers about 90% of observations on in-distribution data.
- **H3.** Error and uncertainty both rise when the peptide is unseen or the
  HLA allele is held out. This is a test, not an assumption.

## What we predict

**Experimental peptide–MHC class I complex half-life at 37°C**, in hours.

We do **not** train on binding affinity (nM). Affinity and stability are
related but not the same quantity. Stability is the better reported correlate
of CD8 T-cell immunogenicity (Rasmussen et al., 2016).

### Transform (locked)

About 20% of the table is recorded as `0.00` hours. Those are values below
the scintillation-proximity assay floor, not biological zeros. A raw `log`
of 0 is undefined, so we train on:

```
log_half_life = log10(half_life + 0.1)
```

`0.1` is the resolution of the published table. Spearman rank is invariant
to this monotone transform. For display and for MAE in hours:

```
half_life_hat = max(10 ** log_half_life_hat - 0.1, 0)
```

A 90% interval is converted at both endpoints, so it is asymmetric on the
hour scale. That is correct.

## Data we are using, and where it comes from

**Primary table (Developer 1 owns this):** the published training set of
NetMHCstabpan.

| | |
|---|---|
| File | `Stability.txt` |
| URL | https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt |
| Paper | Rasmussen et al., *J Immunol* 2016;197:1517–1524 |
| Assay | Scintillation proximity assay, 37°C, half-life in hours |
| Download | `python scripts/download_stability.py` → `data/raw/Stability.txt` |

Inspected counts (do not re-estimate later):

- 28,166 peptide–HLA rows
- 75 HLA names, 5,633 unique peptides
- every peptide is a **9-mer** (8–11-mer NetMHCstabpan outputs are
  approximations; they stay out of training)
- no duplicate peptide–HLA pairs
- 5,679 exact zeros (20.2%); median 1.1 h; max 256.7 h
- three names are engineered mutants, not IMGT alleles:
  `HLA-B*14:01(C67S)`, `HLA-B*14:02(C67S)`, `HLA-B*39:06(C67S)`

**Why this file, not IEDB first.** This is the largest public quantitative
pMHC-I stability matrix. Most IEDB “stability” rows are the same
Buus/Harndahl experiments. An IEDB export is not an independent test unless
Developer 2 can prove sequence-level non-overlap.

**What we will not treat as a fair baseline.** NetMHCstabpan was trained on
this exact table. Its half-life and %rank on these rows are circular.
Developer 2 may still fetch them into `data/external/netmhcstabpan_preds.parquet`
as a reference. We will say so in the paper/demo.

**HLA protein sequences (Developer 2 supplies).** Until they arrive,
`hla_seq` is null. Embeddings wait. For later: mature HLA α1–α2,
residues 1–182. The three `(C67S)` alleles are the parent sequence with
cysteine 67 changed to serine.

**Novelty and structure (Developer 2, later).**
`data/features/novelty.parquet`, optional `data/features/structure.parquet`.
The ensemble does not wait on these.

## Locked splits

A random 10% row split puts the **same peptide in train and test for ~94% of
test rows**. A high score there mostly means “the model remembers the
peptide.” That is why four splits exist.

| File | Rule | What it tests |
|---|---|---|
| `data/splits/random.parquet` | row split | sanity only; report peptide leakage |
| `data/splits/peptide.parquet` | no peptide in two folds | unseen sequences |
| `data/splits/cluster.parquet` | Hamming ≤ 3 components stay together | unseen peptide families |
| `data/splits/hla.parquet` | whole alleles held out | unseen MHC molecules |

Every split has folds `train / val / test / calib`.

- **calib** is ~10% of the remainder after test/val. It is never used to
  train the stability model and never appears in test. It is reserved for
  conformal intervals.
- **HLA test allele (locked):** `HLA-B*15:02` (349 rows). Same peptides are
  mostly also measured on `HLA-B*15:01`, but the half-lives disagree
  (Spearman 0.20, median |Δ| 4.8 h). A model that ignores HLA cannot fake
  this split.
- **HLA val allele (locked):** `HLA-B*27:02`. Also fully removed from train.
  Sibling `HLA-B*27:05` stays in train.
- **`HLA-A*02:01` stays in train** so the demo allele is one the model has
  seen.

Hamming ≤ 4 collapses half the peptides into one blob. Do not use it.

## Required ablation

Every model report includes a **peptide-only** twin (same peptide features,
no HLA). If it matches the full model, we have not learned a pMHC
interaction.

## Model ladder (do not skip rungs)

1. **M1.** BLOSUM50 peptide + train-fitted HLA one-hot → MLP.
   Unseen alleles become a zero vector (training-set mean on the HLA split).
2. **M1-peptide.** BLOSUM50 only.
3. **M2.** Frozen ESM-2 `esm2_t12_35M_UR50D`, mean-pool peptide and HLA
   α1–α2, then the same MLP. Embed each unique string once. Never fine-tune
   ESM in this hackathon unless everything else is finished.
4. **M3.** Only if M2 is trained: peptide residues cross-attend HLA groove
   residues, or a cheap product/difference fallback.

Then a **5-member deep ensemble** of the best head (different seeds +
bootstrap). `y_mean` / `y_std` are in log space. 90% intervals are
split-conformal on `calib` using scores `|y - y_mean| / y_std`.

## File contract

All tables live under `data/`. Parquet. `id` is the only join key.
Predict `log_half_life`. Convert back only for display and hour-scale MAE.
Test folds are touched once, at the end of each experiment.

| File | Owner | Columns |
|---|---|---|
| `data/processed/master.parquet` | Dev 1 | `id, peptide, hla, hla_seq, log_half_life, half_life, source, assay` |
| `data/splits/{random,peptide,cluster,hla}.parquet` | Dev 1 | `id, fold` (`train` / `val` / `test` / `calib`) |
| `data/features/esm_peptide.npy`, `esm_hla.npy` + `*_index.csv` | Dev 1 | one row per unique peptide / HLA |
| `data/external/netmhcstabpan_preds.parquet` | Dev 2 | `id, nms_half_life, nms_rank` |
| `data/features/novelty.parquet` | Dev 2 | `id, hla_novelty, peptide_novelty` |
| `data/features/structure.parquet` | Dev 2 | `id, n_contacts, bsa, hbonds, ipTM, pep_plddt` |
| `results/predictions/<model>_<split>.parquet` | Dev 1 | `id, y_true, y_mean, y_std, m0..m4, fold` |

Also on the master table, documented extras: `censored` (true when
`half_life == 0`), `allele_class` (`natural` / `engineered`).

`id` format: `HLA-A*02:01|VTTEVAFGL` after `normalise_hla()`.

Allele strings keep the `HLA-` prefix and any `(C67S)` suffix. Use
`src.hla.normalise_hla` everywhere.

## Modal contract

- One shared workspace. Run `modal profile list` before every job.
- Volume `pmhc-data` mirrors `data/{processed,splits,features,external}`
  and `results/predictions` under `/vol/data/...` and `/vol/results/predictions`.
- `modal_app/common.py` defines `app`, `image`, `vol`, `VOL`. Both
  developers import it. Neither redefines it.
- Git is source of truth for code. The volume is source of truth for large
  arrays. After writing inside a Modal function, call `vol.commit()`.
- Never commit `.modal.toml` or tokens.

## Hours 0–1 checklist (this branch)

- [x] Repo cloned, branch `Dev1` (not `main`)
- [x] Question, hypotheses, target, data URL, splits locked in this README
- [x] Shared file contract written
- [x] `requirements.txt`, `.gitignore`, `modal_app/common.py`
- [x] `normalise_hla()` and the log-half-life helpers
- [x] Both people: venv + `pip install -r requirements.txt`
- [x] Both people: Modal workspace `sarah04menla`, volume `pmhc-data` visible

## Hours 1–3: master table and splits

```bash
python scripts/build_tables.py
modal volume put pmhc-data data/processed /data/processed
modal volume put pmhc-data data/splits /data/splits
```

- [x] `data/processed/master.parquet` (28,166 rows)
- [x] four splits with leakage asserts
- [x] uploaded to `pmhc-data`
- [x] M1 baseline + peptide-only ablation

Random-split peptide leakage is ~0.93 by construction. Peptide and cluster
splits are 0.00. HLA test is only `HLA-B*15:02`; val is only `HLA-B*27:02`;
`HLA-A*02:01`, `HLA-B*15:01`, and `HLA-B*27:05` stay in train.

## M1 result (locked; do not retune on test)

BLOSUM50 9-mer + train-only HLA one-hot → MLP. Peptide-only is the same MLP
without the HLA vector. Unseen alleles are a zero vector. Early stop on val
MSE only. Bootstrap 95% CI, 1,000 resamples.

| Model | Split | Spearman (95% CI) | MAE (h) | What it means |
|---|---|---|---:|---|
| M1 | random | 0.765 [0.747, 0.782] | 3.54 | sanity; peptides leak |
| M1-peptide | random | 0.411 | 4.64 | HLA identity is doing real work |
| M1 | peptide | 0.642 [0.617, 0.665] | 4.45 | H1 holds for unseen sequences |
| M1-peptide | peptide | 0.327 | 5.04 | |
| M1 | cluster | 0.629 [0.605, 0.653] | 4.07 | H1 holds for unseen families |
| M1-peptide | cluster | 0.298 | 4.59 | |
| M1 | hla | −0.210 [−0.305, −0.114] | 1.80 | one-hot cannot transfer to B*15:02 |
| M1-peptide | hla | −0.035 [−0.141, 0.064] | 1.66 | BLOSUM alone is also uninformative here |

HLA MAE looks smaller because B*15:02 complexes are short-lived (mean 2.0 h
vs 5.4 h globally). Ranking is the metric that matters.

On `hla.parquet`, M1's one-hot is an all-zero vector for val, test, and calib,
because those alleles are absent from train. That is not the same model as
M1-peptide. M1-peptide was trained without an HLA input. M1 was trained with
a real one-hot, then shown zeros, and those zeros make it worse: Spearman
−0.21 versus −0.04 for M1-peptide and +0.04 for M2. Do not describe that as
M1 beating M2 on a new allele. The peptide-split comparison (0.67 versus 0.54)
is the one where the one-hot is doing work. The ensemble is that model. Its
90% intervals on the HLA split calibrate the zero-vector network.

```bash
python scripts/train_m1.py
```

## HLA sequences and ESM-2 embeddings

`hla_seq` is the mature α1–α2 groove, 182 residues from the GSHSM motif.
Source: IPD-IMGT/HLA GitHub branch `Latest` at commit `5b915f27`
(2026-08-26), files `A_prot.fasta` and `B_prot.fasta`, fetched 2026-10-03.
Not a numbered IMGT release. `(C67S)` alleles are the parent groove with
position 67 set to Ser. Partial IMGT entries that start `SHSMR` get a
leading `G`.

The first parse treated `A*02:120` as `A*02:12` (the protein field was not
greedy) and then kept the longest protein. That made `A*02:50` identical to
`A*02:03` and put `A*24:19` one residue from `A*24:02`. Those were not
chosen proxies. Developer 2's three-way check matched IMGT and DTU
`MHC_pseudo.dat`. The corrected grooves are applied. 1,125 master rows:
`A*02:12` 372, `A*02:50` 375, `A*24:19` 378. The other 72 alleles were
unchanged. HLA embeddings for those three alleles were replaced and M2 was
retrained. M1 and the ensemble do not use `hla_seq`.

`B*15:01` vs `B*15:02` differ at mature positions 63, 94, 95, 113, 156.

```bash
python scripts/fetch_hla_sequences.py
python scripts/embed_features.py
python scripts/train_m2.py
```

Frozen `esm2_t12_35M_UR50D`, mean-pool, BOS/EOS dropped. Shapes:
peptide `(5633, 480)` and `(5633, 9, 480)`; HLA `(75, 480)` and `(75, 182, 480)`.

## M2 result (locked)

ESM peptide + ESM HLA mean-pools → the same MLP. Peptide-only is ESM peptide
alone.

| Model | Split | Spearman | vs M1 |
|---|---|---:|---|
| M2 | random | 0.646 | worse than M1 0.765 |
| M2 | peptide | 0.538 | worse than M1 0.642 |
| M2 | cluster | 0.500 | worse than M1 0.629 |
| M2 | hla | 0.041 | near zero; M1 was −0.21 |
| M2-peptide | hla | 0.056 | HLA mean-pool adds nothing here |

These are the numbers after the three groove corrections. The conclusion is
unchanged.

Mean-pooling 182 HLA residues washes out five substitutions. That is why M3
is residue-level, not another mean-pool. The uncertainty ensemble uses **M1**.
Do not retune M2 on the HLA test set.

## M1 ensemble (locked)

Five members, seeds 0–4, each trained on a bootstrap of the train fold.
`y_mean` / `y_std` are in log space. 90% intervals are split-conformal on
`calib` only: `|y - y_mean| / max(y_std, val 1st-percentile floor)`.
`q_norm` is about 5, so the members agree with each other more tightly than
they match experiment. Conformal rescales them. A constant-width interval is
the baseline.

| Split | Spearman | err–unc Spearman | 90% coverage | MAE all → most certain 50% | Constant-width 50% |
|---|---:|---:|---:|---|---:|
| random | 0.766 | 0.199 | 0.907 | 3.45 h → 2.77 h | 3.36 h |
| peptide | 0.674 | 0.122 | 0.909 | 4.11 h → 3.45 h | 3.77 h |
| cluster | 0.665 | 0.149 | 0.913 | 3.80 h → 3.20 h | 3.81 h |
| hla | −0.140 | −0.096 | 0.905 | 1.70 h → 1.69 h | 1.71 h |

H2 holds weakly where the allele was seen: higher `y_std` goes with higher
error, and keeping the more certain half lowers MAE below a random retention.
H3 does not: on `B*15:02` the ranking is still wrong and `y_std` does not
mark the worse rows. Coverage stays near 90% because the interval gets wide
(median width about 14 h on random, 23 h on `B*15:02`), not because the
model has recognised the new allele.

```bash
python scripts/train_ensemble.py
```

Predictions: `results/predictions/m1ens_*.parquet` with `m0`…`m4`, `lo`, `hi`.

## Figures

```bash
python scripts/make_figures.py
```

| File | What it shows |
|---|---|
| `results/figures/fig1_pred_vs_measured` | Ensemble hours vs measured hours, unseen peptides and B*15:02 |
| `results/figures/fig2_uncertainty_vs_error` | Ensemble std vs absolute log error |
| `results/figures/fig3_shift_bars` | Prediction Spearman and error–uncertainty Spearman across the four splits |
| `results/figures/fig4_selective_peptide` | MAE as predictions are kept, unseen peptides. Includes the error predictor |
| `results/figures/fig5_allele_novelty` | Ten calibration alleles: error rises with contact distance, ensemble std does not |
| `results/tables/comparison.csv` | M1, peptide-only, M2, ensemble, and NetMHCstabpan |
| `results/tables/error_predictor.csv` | Calib-only model of absolute error versus `y_std` |

## Frozen demo model

```bash
python scripts/freeze_ensemble.py
python scripts/predict_one.py GILGFVFTL A*02:01
```

The frozen ensemble is the peptide-split fit (test Spearman 0.67, the same
number as `m1ens`). All 75 stability-table alleles are known to it, including
`B*15:02`. The held-out HLA result came from a different fit. An allele
outside the table, such as `C*07:02`, returns `allele_known: false` and the
verdict says the interval is not calibrated. Weights are `models/member_*.pt`
on volume `pmhc-data` under `/models/` (gitignored).

Deployed endpoint, 9-mers only. `mean`, `std`, `lo`, and `hi` are
`log10(half-life + 0.1)`. Hours are also returned.

```bash
curl -X POST https://sarah04menla--pmhc-guardian-guardian-predict.modal.run \
  -H 'Content-Type: application/json' \
  -d '{"peptide":"GILGFVFTL","hla":"A*02:01"}'
```

A cold container takes about 20 seconds. The first checked reply for that
peptide was 1.92 h, interval 0–71 h, allele in training, peptide not,
predicted error in the upper half of the calibration set.

Novelty is `data/features/novelty.parquet`, built from the corrected grooves
(`seq_version=master_corrected_imgt_5b915f27`). On the peptide test, a model
fitted only on `calib` lifts error–uncertainty Spearman from 0.12 (`y_std`)
to 0.21. On the ten held-out calibration alleles, mean error rises with
groove distance (Spearman 0.55) while mean `y_std` does not (contact-distance
Spearman −0.14). `B*15:02` itself is one allele, so that correlation cannot
be computed inside the HLA test fold.

On the peptide-disjoint test the ensemble Spearman is 0.67. NetMHCstabpan on the same fold is 0.86, and that number is in-sample. On B*15:02 the row to quote together is M1 −0.21, peptide-only −0.04, M2 +0.04.

## Demo

The app is already running against the deployed ensemble:

```bash
streamlit run app/streamlit_app.py
```

Click in this order.

| Button | Query | What to say |
|---|---|---|
| Seen pair | `FVRQCFNPM` / `HLA-A*02:01` | 1.1 h, 90% interval 0.1–5.6 h. Both were in training. Not a generalisation test. |
| Unseen peptide | `GLYGNGILV` / `HLA-A*02:01` | 10.3 h, 90% interval 5.0–21.2 h. Peptide held out; 3 of 9 positions differ from the nearest training peptide. Measured half-life is 20.1 h, inside the interval. |
| Unseen HLA | `FVRQCFNPM` / `HLA-C*07:02` | 0.9 h and a narrow range, labelled uncalibrated. `C*07:02` is not in the table. The ensemble does not widen. |

`HLA-B*15:02` is in the dropdown. This frozen model trained on it. Use Unseen HLA, not `B*15:02`, for the new-allele case.

## How to reproduce

Reported numbers are the files in `results/tables/`. Retraining on Apple MPS can move the last digits. Seeds are fixed: split seed 42, ensemble members 0–4, error-predictor seed 42.

Versions used: Python 3.12, torch 2.14.1, fair-esm 2.0.0, pandas 3.0.6, pyarrow 25.0.1, numpy 2.5.3, scikit-learn 1.9.1, biopython 1.88, modal 1.6.0, streamlit 1.65.0. ESM-2 checkpoint `esm2_t12_35M_UR50D`. HLA sequences from IPD-IMGT/HLA GitHub `Latest` at `5b915f27` (2026-08-26).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_stability.py
python scripts/build_tables.py
python scripts/fetch_hla_sequences.py
python scripts/embed_features.py
python scripts/train_m1.py
python scripts/train_m2.py
python scripts/train_ensemble.py
python scripts/build_novelty.py --seq dev1
python scripts/error_predictor.py
python scripts/make_figures.py
python scripts/freeze_ensemble.py
```

Large files are not in git. `Stability.txt` is downloaded. Embeddings, novelty, and `models/*.pt` are on the Modal volume `pmhc-data` (`/data/features/`, `/data/processed/novelty_long.parquet`, `/models/`). Pull them with `modal volume get`.

## Owners

- **Dev 1 (this branch):** data, splits, embeddings, models, uncertainty, metrics, frozen endpoint
- **Dev 2:** HLA sequence check, NetMHCstabpan reference, novelty design, app shell, slides

## Handoffs

Tag `main` only when merging a finished handoff: `h1` … `h6`.
Work stays on `Dev1` until then.
