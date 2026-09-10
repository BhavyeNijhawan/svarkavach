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
| NCSU WSPR robocalls | 1,432 genuine scam robocalls with transcripts | out-of-domain intent test |
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
4. Push to GitHub and run `python scripts/set_repo_url.py <your repo>` so the
   five Colab notebooks clone the right place.
5. Open the console and watch a call whose audio matches its subtitles.
