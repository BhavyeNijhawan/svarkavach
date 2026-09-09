# Resume point

Last session ended here on purpose. Everything below is current as of the last
commit. Nothing is broken; three known issues are listed at the bottom and they
are the next thing to do.

## Where the project lives

`D:\PS2_VoiceClone_Fraud_Call_Detection`

It was moved off C: because C: had only 2.2 GB free. The venv is at
`.venv` on D: and was created with `--system-site-packages`, so it reuses the
torch, numpy and transformers already installed on C: instead of downloading
2.5 GB again. Always run Python as:

```
D:\PS2_VoiceClone_Fraud_Call_Detection\.venv\Scripts\python.exe
```

The old folder on the Desktop is empty apart from a pointer file. Delete it
whenever you like.

## What is finished and verified

| Area | State |
| --- | --- |
| DSP (MFCC, GFCC, LFCC, CQCC, LPCC, filterbanks, pitch, VAD) | written from scratch in numpy, 29/29 tests pass |
| Channel model (G.711, GSM, AMR, noise) | G.711 verified bit-for-bit against the stdlib C reference |
| Corpus generator (CallForge) | 31 tests pass, gold BIO / acts / language tags by construction |
| Speech synthesiser | cloned audio shows about 7x lower jitter than human, as intended |
| Linear-chain CRF from scratch | analytic gradient matches finite differences to 2e-09 |
| BiLSTM-CRF, POS, language ID, intent models, coercion, ASR adapter | all self-tests pass |
| Anti-spoofing (GMM, GBM, RawNet-lite, speaker embedding, scorer) | 19 tests pass |
| Fusion (PIM, featuriser, calibrated model, exact Shapley, streaming) | self-tests pass, kernel SHAP matches the closed form exactly |
| Evaluation engine | metric self-checks pass |
| Flask API and the console (5 views, hand-built SVG charts) | JS syntax clean, palette validated for colorblind safety in both themes |
| Colab notebooks (5) plus README | valid nbformat 4, 51 repo symbols verified to exist |
| Docs | README, CONTRACTS, ANNOTATION_GUIDELINES, ETHICS, CONSENT_FORM, REPORT |

About 20,000 lines. Two commits on `main`. No em dashes or en dashes anywhere,
checked.

## Pick up here

### Step 1: rebuild the corpus with audio

The last run was interrupted at 345 of 480 calls. It is deterministic, so just
run it again from scratch. Takes roughly 15 minutes.

```
cd D:\PS2_VoiceClone_Fraud_Call_Detection
.venv\Scripts\python.exe -m swarkavach.cli gen-corpus --n 480 --audio
```

### Step 2: fix the three known issues (details below), then

```
.venv\Scripts\python.exe -m swarkavach.cli train
.venv\Scripts\python.exe -m swarkavach.cli evaluate
.venv\Scripts\python.exe -m swarkavach.cli serve
```

### Step 3: run the whole test suite

```
.venv\Scripts\python.exe -m pytest tests -q
```

---

## The three known issues, in priority order

These were found by running the full audio path on four freshly generated
calls. That run is reproducible with the script described in issue 3.

### 1. PIM acoustic arousal saturates at 1.0, so the gap term is always zero

`fusion/pim.py` maps four acoustic quantities through fixed absolute anchors
in `ANCHORS`. Those anchors were hand-set before any corpus audio existed, and
the synthesiser's output sits above the top of every one of them. Measured on
a rendered call: `rate_std` was 0.805 against an anchor ceiling of 0.55.

Result: `aco` comes out 1.0 on every turn, `gap` is 0, and PIM falls back to
the correlation term alone (observed values 0.01 to 0.12 instead of the
0.4-plus that the controlled test in the module gives).

**Fix**: measure the four components across the generated corpus, split by
human against synthetic, and set each anchor from the pooled 10th and 90th
percentile. Load them from `data/models/pim_anchors.json` with the current
hand-set values as the fallback, and add a `calibrate_anchors(calls)` step to
`Pipeline.fit()` so it happens automatically on every retrain. The controlled
self-test in `pim.py` should keep passing, since it feeds arousal values
directly rather than through the anchors.

This is the highest priority: PIM is the project's headline novelty and it is
currently contributing almost nothing.

### 2. The anti-spoof score is flat at about 0.02 for every call

Expected, and it should resolve itself once step 1 above is done: the models
currently on disk were trained on the 120-call text-only corpus, which had no
audio at all, so the voice branch never saw a training example.

**Fix**: re-run `train` after the audio corpus exists, then confirm that
cloned calls score high and human calls score low. If they still do not
separate, check `AntiSpoofScorer.fit` is finding `call.audio_path` and is not
silently falling through to the heuristic path. `tests/test_antispoof.py`
already asserts the heuristic points the right way, so the failure would be in
the fitting, not the features.

### 3. Analysis is too slow: 18 to 51 seconds per call

Partly fixed already. `fusion/featurize.py` now caches per-turn prosody across
streaming prefixes and scores the voice branch on a 20 second trailing window
(`VOICE_WINDOW_S`) at most every 4 seconds of new audio
(`VOICE_RESCORE_GAP_S`), instead of re-running the front end over the whole
growing prefix each turn.

**Not yet verified.** Re-run the timing check and confirm a call analyses in a
couple of seconds rather than tens of seconds. The check that found this:

```python
# generate one call per cell, render audio, analyse, print risk and wall time
from swarkavach.corpus.generator import generate_one
from swarkavach.corpus.tts import render_call_audio
from swarkavach.pipeline import Pipeline
```

Target: under 3 seconds per call, because the console's Analyse button and the
time-to-detection sweep over the test split both sit on this path.

## Two smaller things already fixed, noted so they are not re-litigated

- `corpus/generator.py` and `text/langid.py` both defined the code-mixing
  index, on different scales (0 to 100 against 0 to 1). Collapsed to [0, 1],
  with `test_cmi_definitions_agree` in `tests/test_integration.py` to stop them
  drifting again.
- `generate_one` did not exist but `pipeline.py` imported it, so the console's
  "synthesise a new call" button would have failed. Added, with ids that fold
  in the resolved configuration so two calls from one seed cannot collide.

## Still to do after the three issues

- Run `evaluate` and paste the real numbers into `docs/REPORT.md` section 6,
  which currently points at the result files rather than quoting them.
- Open the console and look at all five views in a browser. It has been syntax
  checked but never rendered.
- Push to GitHub and put that URL into `REPO_URL` at the top of each notebook.
