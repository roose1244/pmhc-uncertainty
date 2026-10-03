# pMHC Guardian: Plan

## Question
Can predictive uncertainty identify unreliable pMHC stability
predictions under peptide and HLA distribution shift?

## Hypotheses
- H1: A model can predict pMHC stability from peptide/HLA representations.
- H2: Prediction uncertainty correlates with actual error.
- H3: This relationship is strongest under novel peptides / unseen HLAs.

## Target
log(half-life), converted back for display.

## Splits
random | peptide-disjoint | peptide-cluster | HLA-held-out
(each with train / val / test / calib folds)

## Metrics
MAE, RMSE, Spearman, error-uncertainty Spearman,
90% interval coverage, selective-prediction curve.

## Conventions
- Join key: `id`
- Allele format: HLA-A*02:01 (use normalise_hla() everywhere)
- Code in git, large data on Modal Volume `pmhc-data`

## Handoffs (clock times)
H1 __:__  H2 __:__  H3 __:__  H4 __:__  H5 __:__  H6 __:__

## Owners
Dev 1: data, splits, embeddings, models, uncertainty, metrics
Dev 2: HLA sequences, NetMHCstabpan baseline, novelty, structure,
       validation set, app, slides

## Decisions
- GPU: ____________
- NetMHCstabpan: local run / plan B stand-in
- Check-ins: every 4 hours, 10 minutes
- Hackathon deadline: ____________
