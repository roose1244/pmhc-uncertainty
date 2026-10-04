# Demo script — PepShield live

URL: <https://sarah04menla--pepshield.modal.run> · fallback <https://sarah04menla--demo.modal.run>

## Before you present

- Open the link **five minutes early** and run one prediction. The container sleeps after 20
  minutes and a cold start costs about a minute. A warm app answers instantly.
- Leave the tab sitting on the **PEPSHIELD landing screen**. That is your opening image.
- Browser at a large zoom — the audience is reading from the back of a room.
- Know this and do not misstate it: the deployed model is the **five-copy ensemble on the
  one-hot encoding** (`source: m1ens_peptide`), not the contact-residue model from the results
  slides. If asked, say so.

## The three buttons, and what each is for

| Button | Query | The point |
|---|---|---|
| Seen pair | `FVRQCFNPM` / `HLA-A*02:01` | Both were in training. Baseline, not a test. |
| Unseen peptide | `GLYGNGILV` / `HLA-A*02:01` | The real test, with a measurement to reveal. |
| Unseen HLA | `FVRQCFNPM` / `HLA-C*07:02` | The refusal. No calibrated interval. |

**Expected values.** If what you see differs from these, something is wrong — keep talking and
do not announce it.

- Seen pair: **1.1 h**, interval **0.1 – 5.6 h**, NetMHCstabpan **0.29 h**.
- Unseen peptide: **10.3 h**, disagreement **0.013**, interval **5.0 – 21.2 h**,
  NetMHCstabpan **4.07 h**, revealed measurement **20.1 h**.
- Unseen HLA: **0.9 h**, no interval.

---

## 1:30 version — the one to rehearse

### 0:00 · Landing screen → press to enter

> This is running live. Nothing here is a recording.

*Click **PRESS TO ENTER**.*

### 0:08 · Click "Unseen peptide" → click "Run prediction"

> This peptide is not in the training data. The model says ten point three hours.

*Let the result paint. Point at the five bars.*

> Those five bars are five copies of the model, trained on different samples of the data. They
> land in nearly the same place, so the copies agree — and the ninety percent interval is five
> to twenty-one hours.

### 0:35 · Point at the NetMHCstabpan line

> That is the standard tool, shown on every prediction. We label it, because this dataset is
> its training data. It is a ceiling, not a baseline we beat.

### 0:48 · Click "Show held-out measurement"

> And here is what the laboratory actually measured. Twenty point one hours.
>
> Notice the five copies agreed on ten. If we had shown you their agreement as the error bar,
> we would have been confidently wrong. The calibrated interval contains the truth.

*This is the strongest moment in the demo. Pause for one beat before moving on.*

### 1:05 · Click "Unseen HLA"

> Now the honest part. This allele does not appear anywhere in the dataset. You still get a
> prediction — and no interval, because our calibration does not hold here. The tool refuses to
> print a number it cannot stand behind.

### 1:20 · Structure panel, one line only if it has loaded

> On the right is the binding groove, with the thirty-four residues that touch the peptide
> highlighted. That is a reference AlphaFold structure — it shows what the model reads. It is
> not evidence that the uncertainty is right.

*Stop. Go to Q&A.*

---

## 3:00 version — if you are given more time

Everything above, plus these two inserts.

### Insert after entering, before the first prediction — 20 seconds

*Point at the strip of boxes under the title.*

> Peptide and HLA go in; five copies of the predictor run; you get a prediction, a disagreement
> number, and the groove it was computed from. That is the whole product.

### Insert at the start — 25 seconds

*Click **Seen pair**, run it.*

> Start with the easy case. This peptide and this allele were both in training, and the model
> says one point one hours with a tight range. I am showing you this so the next one means
> something — this is not a test of anything.

### Insert after the HLA-C case — 30 seconds

*Click the **04 UNCERTAINTY** tab.*

> This is the evidence behind that refusal. Where the allele is known, the uncertainty tracks
> the error. On a new allele it inverts — the model is most confident where it is most wrong.
> That is why the interval is withheld rather than shown wider.

*Click **GO BACK** to return to the model.*

---

## Things that can go wrong

**Nothing happens when you press enter.** Add `?enter=1` to the URL and reload.

**The spinner sits on "Asking the frozen ensemble".** It is a cold container. Say: "it is waking
up — while it does, here is what it is about to show you" and describe the three outputs. Do not
reload; you will restart the wait.

**The 3D viewer does not appear.** Skip it. It is the one optional element; the prediction and
the interval are the demo. Never apologise for it on stage.

**The whole app is unreachable.** Open <https://sarah04menla--demo.modal.run>, the plain
fallback page. Same model, same numbers, no interface. Say: "this is the same service with the
interface stripped off" and carry on with the same three queries.

**A number differs from the table above.** Carry on with what is on screen. Do not say "that's
not what it usually shows."

---

## Lines never to say

- "The model is ninety percent sure." It is not a probability about this prediction; it is an
  interval that contains the truth about ninety percent of the time across a test set.
- "The structure proves it." The structure is context for the features, nothing more.
- "We beat NetMHCstabpan." We do not, and its score here is in-sample anyway.
- "It works on HLA-C." There is no HLA-C measurement in the dataset. It returns a prediction
  with no interval, and that is all we claim.
