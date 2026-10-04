# Technical notes — everything, in bullets

Numbers are from `results/tables/`. If a number is not here, say "we did not measure that".

## The task

- Predict **experimental half-life of a peptide–HLA complex at 37 °C**, in hours.
- Source: Rasmussen et al., *J Immunol* 2016. Scintillation proximity assay.
- **28,166** measurements · **5,633** unique 9-mer peptides · **75** HLA alleles (36 A, 39 B, **0 C**).
- Median half-life ≈ **1.1 h**. Heavily right-skewed.
- **20.2%** of labels are exactly 0.00 — the assay floor, not a true zero.
- Train on `log10(half_life + 0.1)`. Offset avoids log(0); log tames the skew.
- Display as `10^pred − 0.1`. Converting *both* endpoints is why intervals are asymmetric in hours.
- 9-mers only. Other lengths are rejected by the API, not extrapolated.

## Features

- Peptide: BLOSUM50 row per residue → 9 × 20 = **180 numbers**. Same in every model.
- **M1** — peptide + one-hot allele label → **255** inputs (243 on the HLA split, which has 63 training alleles).
  - A new allele is all zeros. The model falls back to peptide-only behaviour.
- **M2** — frozen ESM-2 `esm2_t12_35M_UR50D`, mean-pooled → 480 peptide + 480 groove = **960**.
  - Mean-pooling discards position, which is what matters in a groove.
- **Pocket** — peptide 180 + BLOSUM50 of the **34 contact residues** 680 = **860**.
  - The 34 are the NetMHCpan pseudo-sequence positions; verified to reproduce `pseudo_best` for all 75 alleles.
  - A new allele becomes a *nearby vector*, not an unknown token. This is why it transfers.

## Network and training

- Identical everywhere, on purpose: MLP **128 → 64**, ReLU, **dropout 0.2**.
- Adam, lr **1e-3**, weight decay **1e-4**, **MSE**, batch **256**.
- Max **80** epochs, early stopping on validation loss, **patience 12**.
- Architecture held fixed so any difference is attributable to features or uncertainty method.
- Trains in minutes on laptop CPU. ESM-2 embeddings precomputed once.

## Splits (seed 42, locked before modelling)

| Split | What it tests | Test n |
|---|---|---|
| random | sanity check | 2,816 |
| peptide | unseen peptide | 2,813 |
| cluster | unseen peptide family, Hamming ≤ 3 | 2,816 |
| hla | unseen allele | 349 |

- Every split has **train / val / calibration / test** folds.
- HLA split: test **B\*15:02** (349), val **B\*27:02** (359), calibration **10 other alleles** (2,621).
- Val is for early stopping. Calibration is for conformal. Test is read once. Nothing tuned on test.

## Accuracy (Spearman)

| Model | random | peptide | cluster | unseen allele |
|---|---|---|---|---|
| M1 one-hot | 0.765 | 0.642 | 0.629 | **−0.210** |
| peptide only | 0.411 | 0.327 | 0.298 | −0.035 |
| M2 ESM-2 | 0.646 | 0.538 | 0.500 | 0.041 |
| **Pocket** | **0.784** | **0.713** | 0.683 | **0.154** |
| M1 ensemble | 0.766 | 0.674 | 0.665 | −0.140 |
| Pocket ensemble | 0.793 | 0.709 | 0.689 | 0.166 |
| Heteroscedastic | 0.754 | 0.658 | 0.643 | 0.149 |
| NetMHCstabpan | 0.880 | 0.863 | 0.848 | 0.845 |

- **NetMHCstabpan rows are in-sample.** This dataset is its training data. Ceiling, not baseline.
- One-hot on a new allele is *negative* — worse than guessing. Peptide-only is less bad (−0.03) because it at least does not have a dead input.
- Pocket beats M1 on **8 of 8** rankable held-out calibration alleles: median **0.19** vs **0.06**. B\*15:02 is typical, not cherry-picked.

## Uncertainty — how it is built

- **Ensemble**: 5 members, seeds 0–4, each on a **bootstrap resample** of training rows.
- Prediction = mean of the 5. Raw uncertainty = their **standard deviation**, in log space.
- Raw spread is *not* an error bar — members share architecture and most data, so it understates error.
- **Split conformal**, normalised:
  - On the calibration fold: score `s = |y − μ| / max(σ, floor)`.
  - `floor` = 1st percentile of validation σ, so a hairline spread cannot give a zero-width interval.
  - `q` = the `ceil((n+1)·0.90)/n` empirical quantile of those scores.
  - Interval = `μ ± q·σ`.
- **q is about 5–6** for the ensembles (5.03 M1, 6.36 pocket on the peptide split). The raw spread has to be
  multiplied by five to become honest. That is the quantitative argument against "just use the std".
- Assumption: **exchangeability** between calibration and test. True across peptides. Not true across alleles.

## Uncertainty — how it is judged

- **Coverage** — does the 90% interval contain the truth 90% of the time? Tests honesty.
- **Error–uncertainty Spearman** — rank by σ vs rank by |error|. Tests whether it flags the *right* ones.
- **Selective prediction** — keep the most-agreed half, measure MAE. Control = a *random* half.

| Peptide split | Coverage | Err–unc | Median width | MAE 100% → 50% | Random 50% |
|---|---|---|---|---|---|
| M1 ensemble (deployed) | 0.909 | 0.122 | 20.0 h | 4.11 → 3.45 | 3.77 |
| Pocket ensemble | 0.893 | 0.167 | 18.0 h | 4.23 → 3.49 | 3.90 |
| Heteroscedastic | 0.907 | 0.261 | 12.1 h | 4.43 → 3.63 | 4.15 |

- The gap between 3.49 and 3.90 is the uncertainty earning its keep. Without the random control the
  improvement could just be a smaller sample.

## Where it fails

- Unseen allele, Spearman: **0.71 → 0.17**.
- Error–uncertainty on the unseen allele: M1 ensemble **−0.096**, pocket ensemble **−0.189**.
  Negative = most confident where most wrong.
- Per-allele coverage across the 10 calibration alleles: **72%** (B\*46:01) to **98%** (A\*26:01).
  The conformal guarantee is **marginal**, not **conditional**.
- Marginal coverage on the single held-out allele still looks fine (0.885–0.905). Do not quote 85% for the
  ensembles — 0.845 is the heteroscedastic model.
- Selective prediction does not help on a new allele: keeping the most-agreed half does not reduce error.

## Things we tried to fix it

- **Heteroscedastic head** — shared trunk, mean head + log-variance head, Gaussian NLL, log-var clamped
  to [−6, 4], early stop on validation NLL.
  - Peptide split: err–unc **0.167 → 0.261**, width **18 h → 12 h**, ranking **0.709 → 0.658**.
  - Unseen allele: err–unc **0.012**, coverage **0.845**, selective MAE gets *worse* (1.51 → 1.80 at 50%).
- **Error predictor** — gradient boosting fitted on the calibration fold to predict |error|.
  - Peptide split: err–unc **0.122 → 0.211**.
  - Still does not rescue the unseen allele.
- Two mechanistically different methods failing the same way ⇒ the limit is the **data**, not the estimator.
- Groove-novelty evidence: across the 10 held-out alleles, mean error rises with groove distance
  (Spearman **0.55**) while mean ensemble σ does not (**−0.14**). The signal exists; the ensemble misses it.

## Structure panel

- AlphaFold DB, one entry per locus: **P04439** (A), **P01889** (B), **P10321** (C).
- **Not allele-specific. Not experimental. No peptide. No β2-microglobulin.**
- Mature groove starts at AlphaFold residue 25 — there is a 24-residue signal peptide first.
- Shown to make the 34 contact residues concrete. **pLDDT is AlphaFold's confidence in coordinates**,
  not the stability model's uncertainty. Never used to justify a prediction.

## Deployment

- Modal app `pmhc-guardian`, volume `pmhc-data`.
- POST endpoint returns `mean, std, lo, hi` in log units plus hours, `source: m1ens_peptide`.
- **The deployed model is the M1 five-copy ensemble**, frozen on the peptide split — *not* the pocket model.
  Deployment lagged the research by one model. Say so if asked.
- Public app: <https://sarah04menla--pepshield.modal.run>. Fallback page: `--demo.modal.run`.
- An allele outside the 75 returns `allele_known: false` and **no numeric interval**.

## Three different "confidences" — never conflate

- **Ensemble disagreement** — spread of 5 copies. Raw, uncalibrated. Not a probability.
- **Calibrated interval** — that spread × q from held-out data. A statistical claim about error.
- **pLDDT** — AlphaFold's confidence in atomic coordinates. Nothing to do with this model.

## What we do not claim

- No HLA-C accuracy, MAE, correlation or calibration. The dataset has zero HLA-C measurements.
- Not "true uncertainty" — it is a proxy for predictive uncertainty.
- Not beating NetMHCstabpan: its score here is in-sample.
- No attribution analysis, so no claim that the model learned anchor positions P2/P9.
- No claim that structure explains the uncertainty.
