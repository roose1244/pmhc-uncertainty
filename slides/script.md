# PepShield — five minute script

Two speakers. Sarah runs the demo. Swap if you prefer, but do not swap mid-section.

**Before you are called up:** open <https://sarah04menla--pepshield.modal.run>, press enter,
and run one prediction. The container sleeps after twenty minutes and a cold start costs a
minute. Leave the tab on the landing screen, already warm. Open `slides/pitch.html` in a
second tab and press `f`.

---

## PITCH — 1:30

*Slides 1 to 4. Roughly 210 words. Do not rush the last sentence of each paragraph.*

> **[Slide 1 — title]**
>
> Peptide–HLA stability decides whether a T cell ever sees a cancer or viral target. If the
> complex falls apart in twenty minutes, the immune system never gets the message. So every
> neoantigen shortlist and every vaccine design depends on this number — and measuring it in
> the lab is slow, which is why people use models.
>
> **[Slide 2 — the problem]**
>
> Here is our problem with those models. They hand you a number. A model that says *ten hours*
> and a model that says *ten hours, and I have never seen anything like this before* should
> send you to two completely different experiments. You only ever get the first one.
>
> So we did not ask "can we predict stability". We asked whether the model can tell you when
> to ignore it.
>
> **[Slide 3 — data]**
>
> We trained on twenty-eight thousand measured half-lives: five and a half thousand peptides,
> seventy-five HLA alleles. Before modelling we locked four splits — random, unseen peptides,
> clustered peptides, and an entire allele held out.
>
> **[Slide 4 — approach]**
>
> Same network each time; what changes is how we describe the allele. The version that
> transfers best throws away the allele's name and feeds it the thirty-four groove residues
> that physically touch the peptide. And on top of any of them we train five bootstrapped
> copies, and turn their disagreement into a calibrated ninety-percent interval.

---

## DEMO — 1:30

*Live app. Say the sentence first, then click — do not narrate your own mouse.*

> **Know this before you demo.** The deployed service is the **five-copy ensemble on the
> one-hot encoding**, frozen on the peptide split — `source: m1ens_peptide`. It is not the
> contact-residue model from the results slides. On its own test fold it scores 0.67 with 91%
> coverage. If anyone asks whether the demo is the model from slide 5, the answer is no, and
> there is a Q&A entry for it below. Do not imply otherwise.

**0:00 — open the app, press enter**

> This is live. Everything you see is computed now.

**0:10 — click "Unseen peptide"** → `GLYGNGILV` / `HLA-A*02:01`, click Run prediction

> This peptide is not in the training set. The model says ten point three hours.
>
> The five bars are the five copies. They land in nearly the same place, so the model is
> confident — and the ninety-percent interval is five to twenty-one hours.

**0:35 — point at the NetMHCstabpan line**

> That is the standard tool for comparison. We show it on every prediction, and we label it,
> because it was trained on this dataset. It is a ceiling, not a fair baseline.

**0:50 — click "Show held-out measurement"**

> Here is what the lab actually measured. Twenty point one hours — inside the interval the
> model committed to before it saw this.

**1:05 — click "Unseen HLA"** → `FVRQCFNPM` / `HLA-C*07:02`

> Now the honest part. This allele does not exist in the dataset. You still get a prediction —
> and no interval, because we know our calibration does not hold here. The app refuses to
> print a number it cannot stand behind.

**1:20 — if the structure panel has loaded, one line only**

> On the right is the binding groove with those thirty-four contact residues highlighted —
> that is the AlphaFold reference protein, so it shows you what the model reads, not proof
> that the uncertainty is right.

*If anything hangs, keep talking and move to Q&A. Do not debug on stage.*

---

## Q&A — 2:00

*Short answers. Give the number, then stop.*

**"Do you beat NetMHCstabpan?"**
No, and nobody honestly can on this dataset — it scores 0.86 here because this file is its
training data. That is in-sample. We report it that way rather than quietly dropping it.

**"A rank correlation of 0.17 on a new allele is useless, isn't it?"**
Agreed, and that is the finding. The one-hot baseline is *minus* 0.21 — worse than guessing.
Describing the pocket gets you to positive territory, but nowhere near usable, and we say so
instead of only showing the peptide split.

**"Then why should I trust the uncertainty at all?"**
Where the allele is known, it is measured. For the contact-residue ensemble the ninety-percent
interval covers 89%, and discarding the half the copies most disagree about drops mean error
from 4.2 to 3.5 hours, against 3.9 for discarding a random half. Where the allele is new, it
fails, and the product withholds the interval.

**"Is the live demo the contact-residue model?"**
No — the deployed service is the five-copy ensemble on the one-hot encoding, which we froze
and shipped first. On unseen peptides it is 0.67 with 91% coverage; the contact-residue
ensemble is 0.71 with 89%. The deployment lagged the research by one model. Say that plainly
rather than letting the demo stand in for the result.

**"Why not a bigger protein language model?"**
We tried. Frozen ESM-2 scores 0.54 on unseen peptides; the thirty-four contact residues score
0.71. Mean-pooling a whole protein throws away exactly the positional detail that matters in
a groove.

**"What about HLA-C?"**
There is not a single HLA-C measurement in this dataset, so we report no HLA-C accuracy. You
can query one and get a prediction with no interval. We were not willing to show a number we
could not check.

**"How do you know the splits are clean?"**
They were written and frozen with a fixed seed before any model ran, and the hardest one holds
out an entire allele, so nothing about it is in training. Nothing was tuned on test.

**"Is the interval real, statistically?"**
Split conformal on held-out calibration alleles. The guarantee is marginal, not per-allele —
coverage across the ten calibration alleles runs from 72% to 98%, which is the honest caveat.

**"What next?"**
Two things. Conditional conformal so the ninety percent holds per allele instead of on
average. And measured half-lives for a handful of unseen alleles — that is the only thing that
turns the new-allele failure into a solved problem rather than a documented one.

**If you do not know:** "We did not test that." Do not improvise a number.

---

## TECHNICAL Q&A — if a judge goes deeper

*These are for someone who knows the field. Answer in one or two sentences and stop.*

### Modelling

**"What is the architecture?"**
Deliberately small and identical across every experiment: a two-layer MLP, 128 then 64 units,
ReLU, dropout 0.2, Adam at 1e-3 with 1e-4 weight decay, MSE, batch 256, early stopping on
validation loss with patience 12. The point of the project is the feature representation and
the uncertainty, so we held the network fixed and changed one thing at a time.

**"What exactly are the inputs?"**
860 dimensions for the pocket model: BLOSUM50 rows for the nine peptide positions, 180, plus
BLOSUM50 rows for the 34 contact residues, 680. The one-hot baseline is 180 plus a 75-way
allele indicator. The ESM-2 model is two 480-dimensional mean-pooled embeddings, peptide and
groove.

**"Why BLOSUM50 rather than learned embeddings?"**
With 5,633 unique peptides, a learned residue embedding is a lot of free parameters for the
data. BLOSUM gives you substitution structure for free, and it is what makes an unseen allele
a nearby vector rather than an unknown token.

**"Where do the 34 positions come from?"**
They are the NetMHCpan pseudo-sequence contact positions. We verified ours reproduce the
published pseudo-sequence for all 75 alleles in the dataset before using them.

**"Why predict log10 of half-life?"**
The distribution is heavily right-skewed and 20% of measurements sit exactly on the assay
floor at zero. We fit `log10(half_life + 0.1)`; the offset avoids log of zero, and we convert
both interval endpoints back to hours, which is why the displayed interval is asymmetric.

**"Why Spearman rather than R²?"**
The decision this feeds is a shortlist — which peptides go to the bench. That is a ranking
problem. We report MAE in hours alongside it.

**"Why only 9-mers?"**
The model was trained on 9-mers. Other lengths are rejected by the API rather than
extrapolated into.

### Uncertainty

**"How is the ensemble built?"**
Five members, seeds 0 to 4, each trained on a bootstrap resample of the training rows. The
prediction is their mean and the raw uncertainty is their standard deviation, both in log
space.

**"Describe the conformal procedure."**
Normalised split conformal. On a calibration fold the model never trained on, we compute
nonconformity `|y − μ| / max(σ, floor)`, take the `ceil((n+1)·0.9)/n` quantile, and the
interval is `μ ± q·σ`. The floor is the first percentile of validation σ so a confident
member set cannot produce a zero-width interval.

**"What assumption does that make, and is it satisfied?"**
Exchangeability between calibration and test. It holds across peptides, which is why coverage
is 89% there. Across alleles it does not. Be precise about the evidence: on the single
held-out allele marginal coverage still looks fine, around 89 to 91% — but per-allele coverage
across the ten calibration alleles runs from 72% to 98%, and the heteroscedastic model drops
to 85%. Average coverage hides which allele you are standing on, which is why we withhold the
interval for an allele outside the table.

**"Why normalised conformal rather than a constant-width interval?"**
We computed both. The constant-width version also covers, but it is the same width for every
prediction, so it carries no information about which prediction to distrust. Both are in the
results table.

**"How does the heteroscedastic version work?"**
Shared trunk, two heads — mean and log-variance — trained with Gaussian negative log
likelihood, log-variance clamped to [−6, 4] for stability, early stopping on validation NLL.
On unseen peptides it raises error–uncertainty from 0.167 to 0.261 and narrows the median
interval from 18 to 12 hours, costing 0.709 to 0.658 in ranking.

**"Two different uncertainty methods, same failure on a new allele. Why?"**
Because both estimate uncertainty from the training distribution, and neither has ever seen
an allele that far away. That points at the data rather than the estimator. We also fit a
separate error predictor, which lifts error–uncertainty on unseen peptides from 0.12 to 0.21,
and that one also does not rescue the new-allele case.

**"Is the ensemble disagreement epistemic or aleatoric?"**
Bootstrap spread is mostly epistemic. The variance head mixes both. We deliberately do not
label either one "true uncertainty" in the interface — we call it model disagreement and keep
it visually separate from the calibrated interval.

### Evaluation and honesty

**"Your held-out allele is a single allele — isn't 0.17 just noise on 349 rows?"**
Fair, and that is why we do not rest on it. The contact-residue encoding beats the one-hot
model on eight held-out calibration alleles, median 0.19 against 0.06. B*15:02 is typical of
that set, not the best member of it.

**"Could the groove features leak the allele identity?"**
They are the allele identity, in chemical form — that is the point. The test is whether it
transfers, and the held-out allele's 34-residue vector never appeared in training.

**"Did you tune anything on the test fold?"**
No. Each split has its own train, validation, calibration and test folds. Early stopping uses
validation, conformal uses calibration, and test is read once.

**"What about pLDDT — are you using structure confidence as uncertainty?"**
No, and that conflation is the thing we were most careful to avoid. pLDDT is AlphaFold's
confidence in coordinates of a reference protein that contains no peptide. It is displayed as
context for the features, never as evidence about the prediction.

**"How long does training take?"**
Minutes on a laptop CPU per model. ESM-2 embeddings are precomputed once. The deployed
service is the five-member ensemble behind a Modal endpoint.

---

## BIOLOGY Q&A — if the judge is an immunologist

*Say which claims are ours and which are from the literature. Do not blur the two.*

### The endpoint

**"Why stability and not binding affinity?"**
Affinity tells you whether the peptide binds; stability tells you how long the complex
survives at body temperature, and a complex that falls apart in minutes never gets displayed
long enough for a T cell to find it. The argument that stability is the better correlate of
immunogenicity is from the literature — Harndahl and colleagues, 2012 — not something we
measured here.

**"What was actually measured?"**
Rasmussen and colleagues, 2016: a scintillation proximity assay tracking dissociation of the
peptide–HLA complex at 37 °C, reported as a half-life in hours. Twenty-eight thousand
peptide–allele pairs. The median is about one hour, so most complexes are short-lived.

**"Twenty percent of your labels are zero. What does that mean biologically?"**
Those complexes dissociated faster than the assay could resolve. It is a floor, not a
measurement of zero, which is why we model on a log scale with an offset rather than
pretending the value is exact.

**"Is half-life enough to call something immunogenic?"**
No, and we are careful not to. A peptide also has to be produced by the proteasome,
transported by TAP, loaded, and then matched by a T cell receptor in that individual's
repertoire. We predict one necessary step, not the outcome.

### The molecule

**"What are the 34 residues, physically?"**
The polymorphic positions lining the groove formed by the α1 and α2 helices and the β-sheet
floor — the ones whose side chains point into the pockets that grip the peptide. They are the
standard NetMHCpan pseudo-sequence positions.

**"Why does the peptide length matter?"**
Class I grooves are closed at both ends, so they hold roughly 8 to 11 residues, with 9 the
dominant length. We trained on 9-mers only and refuse other lengths rather than extrapolate.

**"Which positions drive the binding?"**
Conventionally the anchor positions — P2 and P9 for most class I alleles — sit deepest in the
B and F pockets. We did not run a formal attribution analysis, so I would not claim our model
learned that specifically.

**"Your structure has no peptide in it. Isn't that a problem?"**
It is a reference AlphaFold model of the heavy chain, with no peptide and no β2-microglobulin.
We show it to make the 34 contact residues concrete, and we label it as exactly that. It is
not evidence about any particular prediction.

### Alleles and coverage

**"Why does performance collapse on a new allele?"**
Different alleles have chemically different pockets, so a new allele can prefer an anchor
residue nothing in training preferred. Related alleles share pocket chemistry, which is the
supertype idea, but a held-out allele with no close relative in training is genuinely out of
distribution.

**"Seventy-five alleles is a small slice of HLA diversity."**
Agreed — there are thousands of class I alleles and this dataset has 36 A and 39 B. That is
precisely the generalisation problem we are measuring rather than papering over.

**"No HLA-C at all. Does that matter clinically?"**
It does. HLA-C is expressed at lower levels and has a large role in NK recognition through
KIR, and it is under-represented in stability data generally. We will not report an HLA-C
number we cannot check, so the app gives a prediction with no interval.

**"Does this generalise across populations?"**
Allele frequencies differ substantially between populations, so a model trained on 75 mostly
well-studied alleles will be least reliable exactly where the allele is rare. That is an
equity argument for reporting uncertainty rather than a single number.

### Use and validation

**"Who would use this, and for what?"**
Anyone ranking candidate epitopes before committing bench time — neoantigen vaccine design,
TCR therapy target selection. The useful output is not just the ranking but which candidates
the model says it cannot judge.

**"What wet-lab experiment would validate it?"**
Measure half-lives for a few dozen peptides on two or three alleles outside our 75. That
single experiment tests the transfer claim and gives us the calibration data we currently
lack for unseen alleles.

**"Would you put this in a clinical pipeline today?"**
For shortlisting on well-characterised alleles, with the interval shown, yes. For a new
allele, no — and the product says so itself rather than leaving you to find out.
