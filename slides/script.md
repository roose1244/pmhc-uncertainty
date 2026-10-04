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
> Same network each time; what changes is how we describe the allele. The version that works
> throws away the allele's name and feeds it the thirty-four groove residues that physically
> touch the peptide. Then we train five bootstrapped copies, and turn their disagreement into
> a calibrated ninety-percent interval.

---

## DEMO — 1:30

*Live app. Say the sentence first, then click — do not narrate your own mouse.*

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
Where the allele is known, it is measured: the ninety-percent interval covers 89%, and
discarding the half the copies disagree about drops mean error from 4.2 to 3.5 hours, against
3.9 for discarding a random half. Where the allele is new, it fails, and the product withholds
the interval.

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
