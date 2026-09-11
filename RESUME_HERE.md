# Resume point

Current as of the last commit. Read the "open" section at the bottom first.

## Where the project lives

`D:\PS2_VoiceClone_Fraud_Call_Detection`, moved off C: because C: had 2.2 GB
free. Run Python as `.venv\Scripts\python.exe`, never plain `python`.

**The machine has 7.9 GB of RAM.** That is the binding constraint, more than
disk. Anything that loads a whole corpus of audio at once gets killed by the
OOM reaper. Two habits follow from that: run each pipeline stage as its own
process so memory is released between them, and check for orphaned
`python.exe` processes before a long run, because stopped background tasks
leave them behind holding a gigabyte each.

## What the system is

Two branches over a Hinglish call, fused with cross-modal features:

- **voice**: is this a person or a machine
- **intent**: is this conversation a scam
- **cross-modal**: prosody-intent mismatch, coercion trajectory,
  code-switching profile, dialogue-act sequence likelihood

Plus streaming: it rescores after every turn, so it reports how many seconds
of call it needed before raising an alert.

## Data, and which claim rests on what

| source | what | used for |
| --- | --- | --- |
| GramVaani (OpenSLR 118) | 2,884 real Hindi telephone utterances, 7.6 h, 2,726 speakers, gender and accent labels | bonafide side of the voice branch |
| NCSU WSPR robocalls | 1,432 genuine scam robocalls with transcripts | out-of-domain intent test, written to `robocall_ood.json`. Transcripts only, the clone is blob-filtered to 700 KB |
| CallForge (this repo) | generated Hinglish dialogues, gold BIO, acts, language tags | fusion grid and the live demo |

The anti-spoofing pairs are built here because no public Hindi anti-spoofing
set exists. Each spoof renders the SAME transcript as its bonafide, and both
sides go through the same channel with the noise floor matched.

Get the data with `swarkavach fetch-data`. Everything is direct HTTP, no
account, no approval form.

## Commands

```
swarkavach fetch-data                       # real corpora plus the pairs
swarkavach gen-corpus --n 240 --audio --backend edge
swarkavach train                            # branches on train, fusion on dev
swarkavach evaluate --codecs clean,g711u
swarkavach serve                            # console on 127.0.0.1:7860
.venv\Scripts\python.exe -m pytest tests -q
```

`--backend edge` is what makes the audio say the words on screen. The `sim`
backend produces speech-shaped noise with no words in it, which is fine for
training a detector and useless for a demo.

## Bugs fixed, so they do not get reintroduced

Most of these were silent. None raised an error, and several would have
produced better numbers than the truth.

**Evaluation validity**

- `evaluate` crashed and never wrote five of its seven result files, because a
  loop variable shadowed the list of model names.
- EER and detection cost were decided by the arbitrary order of tied scores.
  Reported 0.1339 where the best achievable was 0.0625; reordering ties moved
  it between 0.0 and 0.1339 without changing a score. Ties are collapsed now.
- A missing or failing entity tagger silently substituted the call's GOLD
  annotation, so a held-out call scored with a broken tagger produced entity
  counts identical to ground truth.
- The fusion layer was fitted on the branches' own training rows. It now fits
  on `dev`, and `dev` was resized from 15 to 25 percent to support 25 features.
- The anti-spoof calibrator was fitted on speakers the detector trained on.
- `ActHMM` was fitted on gold acts and scored on predicted ones.
- The anti-spoof EER was being computed on the generated corpus, where both
  classes are machines. It now evaluates on held-out real speakers.

**The channel confound, three layers deep**

Real phone audio has a noise floor around -41 dB; a synthesiser writes digital
silence at -88 dB. The noise floor ALONE separated the classes at AUC 0.970,
so a detector could have scored 97 percent without listening to a voice.

1. No channel matching at all. Added it, got to 0.907.
2. Wrong units: `add_noise` takes an SNR against signal RMS, `noise_floor_db`
   reads a percentile of `short_time_energy` which sums over the frame, a
   23 dB offset. Overshot by 18 dB and left the spoof noisier than the real.
3. `write_wav` peak-normalises by default, and real speech is peakier than
   synthesised speech, so writing pushed the floors 38 dB apart again after
   matching. Pairs are written with `peak=None`.

Now at 0.615, status `clean`. `confound_audit()` runs on every build and its
verdict goes into the manifest, so this cannot silently come back.

**Feature bugs**

- The prosody-intent anchors were hand-set and every real value exceeded them,
  so acoustic arousal was pinned at 1.0 on 81 percent of turns and the feature
  contributed nothing. Now calibrated from the corpus on every retrain.
- One of its four components pointed the wrong way (frame-level energy
  variance measures articulation, not emphasis, and read higher for synthetic
  speech). Replaced with the spread of syllable peak levels.
- The synthesiser's cloned path changed jitter and shimmer but left the pitch
  and loudness contour identical, so there was no macro-prosody difference for
  the feature to find. Turn arousal now comes from the dialogue act.
- Code-switching: the claim was backwards. Fraud entities are LESS English in
  scam calls (0.63 against 0.90), because threat deadlines are Hindi verb
  phrases and occur only in scam calls. See REPORT.md section 4.4.

**Infrastructure**

- The TTS service rejects concurrent bursts. At 8 in flight, 324 of 324 fresh
  requests failed while sequential probes took 1.2 s each, and every failure
  became a 0.5 second gap of silence that nothing complained about.
  `MAX_CONCURRENCY = 3`, and a call now refuses to be written if more than
  40 percent of its turns failed.
- `<inaudible>` markers in GramVaani transcripts are parsed as SSML and
  truncate the render. Stripped by `clean_for_tts`.
- The two definitions of the code-mixing index were on different scales.
- Console: the architecture diagram rendered at 0.45 scale with 4px labels,
  and light theme left it unreadable because the SVG baked in dark-theme
  colours at render time.

## Bugs found by an audit of the whole pipeline

All of these produced a number that looked fine. Listed so they are not
reintroduced.

**Results that were not this run's.** A fresh clone shipped four files in
`data/results/`, and the Drive checkpointer decided a stage was done by
counting local files, so a brand new session concluded evaluation had already
happened, skipped it, printed the checked-in numbers as its own, and pushed
them back stamped as a Colab run. Skipping is now keyed on what Drive actually
returned in this session, and results and models are no longer committed.

**The answer key inside the highest weighted feature.** `TfidfIntent`
`_entity_counts` read `Turn.bio`, which is the gold annotation, and its output
overwrites `intent_score`. `CRFTagger.tag_call` was the only thing that would
ever have replaced that with predictions, and it is called from nowhere. Mean
scam score on the test split was 0.960 with the annotation present, 0.741
without. It now takes the predicted spans as an argument, and blanking the
gold BIO changes the score by exactly zero.

**An "ASR" row that was the gold row.** `_eval_ner` gated on whisper being
importable and then ran the tagger over the gold tokens, so on Colab it
emitted an asr column identical to the gold one and the console read it as
"entity F1 survives ASR unchanged". Nothing was ever transcribed. It now
transcribes for real, or there is no row and the log says why.

**Ablation arms fitted on in-sample branch outputs.** `Pipeline.fit` fits
fusion on dev and explains why; `run_full_evaluation` then fitted its own arms
on train, where the tagger scores 1.000 because it memorised those rows. The
arms now fit on dev, and the file records which.

**Two headline numbers from two different classifiers.** Fused AUC came from
the ablation arms, time to detection from the shipped fusion model, presented
as one system. Now stated in the file.

**A threshold that was ignored.** `_eval_intent` took a threshold argument and
used 0.5, while everything else used 0.65 and the file recorded no threshold,
so its recall and the robocall recall were not comparable.

**A failure that scored 0.0.** The codec sweep gave any call that failed to
read, degrade or featurise the most benign possible score. A condition that
failed entirely produced AUC exactly 0.5, which reads as a hard channel rather
than a broken one. Failures are now counted, excluded, and the number is
withheld once too many are missing.

**Colour, not level.** See REPORT section 4.7. Level-matched white noise
against real line hiss separated the classes at AUC 1.000 on spectral tilt,
which is what nine EERs of exactly 0.0 were measuring.

**Stages that failed and exited 0.** `train` returned normally with
`failed: ...` in its report, `fetch-data` swallowed download and pair-build
errors, and `gen-corpus` fell back to speech-shaped noise per call when the
synthesis service was unreachable. All three now exit non-zero, and gen-corpus
checks how many calls really came from the neural backend.

**Tests writing into the real data directory.** `test_checkpoints` set
`SWARKAVACH_DATA` at module import, which is too late under pytest, and left
100 byte stub archives in `data/raw` that `fetch-data` accepted as real
downloads. There is a `conftest.py` now, and the archive check is by size.

## Running it

**Training and evaluation happen on Colab, not on this machine.** That was the
constraint from the start and it has not changed. Open
`notebooks/00_RUN_EVERYTHING.ipynb` in Colab, point it at the repo, run it top
to bottom. Locally, keep to the fast checks: `py_compile`, the notebook syntax
sweep, and `tests/test_checkpoints.py`, which needs neither a corpus nor a
network. Anything that generates audio or fits a model belongs on Colab.

Everything expensive is checkpointed to Google Drive as it is produced, under
`MyDrive/swarkavach_checkpoints/`, and restored at the start of the next run.
A disconnect costs the stage that was in flight and nothing before it. The
speech cache is saved even when generation fails partway, since that is the
hour you would otherwise repeat. Set `FORCE_REBUILD = ["corpus"]` in the Drive
cell to redo a stage anyway. Total footprint is about 1 GB:

| artifact | size | what a loss costs |
| --- | --- | --- |
| tts_cache | 440 MB | about an hour of rate-limited synthesis |
| corpus | 410 MB | 10 minutes, given the cache |
| raw | 160 MB | 5 minutes of download |
| pairs | 45 MB | 15 minutes |
| models | 5 MB | 15 minutes of training |
| robocall, results | 1 MB | seconds |

## Open

**The numbers in `data/results/` are from before these fixes. Treat them as
void.** A full run was in progress at the last commit: pairs, corpus, train,
evaluate, verify. Re-run it with the commands above if it did not finish.

Still to do after that:

1. Paste the real numbers into `docs/REPORT.md` section 6, which currently
   points at the result files rather than quoting them.
2. Confirm the three issues from the earlier resume point are actually closed,
   using the verify script: PIM separating on real audio, the voice branch
   separating human from machine, and analysis under 3 seconds per call.
3. Watch the training log for the anchor warning. If it says a component reads
   higher for synthetic speech than human, the headline feature is partly
   cancelling itself and the component needs replacing, as `energy_std` did.
4. Open the console and watch a call whose audio matches its subtitles.
5. Read `robocall_ood.json`. Before the fixes, the TF-IDF intent model flagged
   74.7 percent of 1,413 real FTC robocall scripts at threshold 0.5 while the
   Hinglish rule lexicon managed 13.4 percent, which is the expected shape:
   the lexicon is language-specific and the learned model is not. Both numbers
   need re-measuring after a clean training run.
