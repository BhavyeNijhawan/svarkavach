# Colab notebooks

**Start with `00_RUN_EVERYTHING.ipynb`.** It is the whole pipeline in one
notebook: fetch the real corpora, build the Hindi anti-spoofing set, generate
the Hinglish corpus with neural voices, train, evaluate, and hand the results
back. Roughly 45 to 70 minutes. The other four go deeper on one piece each and
are optional.

Run it on Colab rather than a laptop because the corpus build holds a lot of
audio at once. On an 8 GB machine the same run gets killed part way through.

## The token

The repo can be private. Put a GitHub Personal Access Token in Colab's
**Secrets** panel (key icon, left sidebar) under the name `GH_PAT` with
"Notebook access" on. A fine-grained token needs **Contents: read and write**
on this one repository and nothing else.

Use Secrets rather than pasting the token into a cell: a pasted token is saved
inside the notebook file and travels with every copy of it. The notebook never
prints the token and strips it from the git remote right after cloning, and it
can push the finished results back to a `colab-results` branch so you do not
have to shuffle zip files around.

| notebook | needs | time | writes back |
| --- | --- | --- | --- |
| `00_RUN_EVERYTHING.ipynb` | GPU optional, a GH_PAT for a private repo | 45 to 70 min | everything below, plus trained models |
| `01_setup_and_data.ipynb` | no GPU, 20 to 45 GB free disk | 3 min plus 1 to 2 h of downloading | `dataset_manifest.json` |
| `02_antispoof_train.ipynb` | T4 GPU, notebook 01 first | 35 to 60 min | `antispoof_results.json`, `antispoof_branch_scores.json`, `data/models/antispoof_rawnet.pt` |
| `03_asr_ner_intent.ipynb` | T4 GPU | 25 to 45 min | `ner_results.json`, `intent_results.json`, `asr_results.json`, `intent_branch_scores.json`, attention PNGs |
| `04_fusion_and_eval.ipynb` | GPU optional, notebooks 02 and 03 first | 10 to 20 min | `ablation_results.json`, `calibration.json`, `robustness_results.json`, `ttd_results.json` |
| `05_voice_cloning.ipynb` | T4 GPU, signed `docs/CONSENT_FORM.md` | 10 min | `cloning_manifest.json`, cloned Call JSON |

Every notebook degrades rather than crashing when a GPU or a dataset is
missing, prints what it did instead, and tags the export with `reduced: true`
so a reduced run cannot be quoted as a full one.

## Getting the results back to the laptop

1. Run `swarkavach evaluate` locally first, then unzip each Colab download
   into the project root so `data/results/` gains the Colab files. The Colab
   `provenance.json` merges with whatever it finds, so unzipping second keeps
   both sets of entries.
2. Restart `swarkavach serve` (or refresh the Results lab) and the provenance
   table will show the Colab rows next to the local ones.

## Order

01 first, because it verifies the clone and pulls the data. Then 02 and 03 in
either order, they do not depend on each other. Then 04, which reads the
branch scores both of them export. 05 is independent and can run any time,
subject to its consent gate.

## What each notebook is for, in one line

- **01** proves the environment works and gets ASVspoof 2019 LA and In-the-Wild
  onto the machine, with a generated-corpus fallback if a download fails.
- **02** trains `RawNetLite` on real spoofing data at 8 kHz, fits the LFCC-GMM
  and GFCC-GMM baselines beside it, and measures the generalisation gap between
  the benchmark and In-the-Wild.
- **03** measures what a Whisper transcript costs the entity taggers, which is
  the honest version of the text branch's accuracy, and fine-tunes MuRIL for
  scam intent against the TF-IDF and rule baselines.
- **04** rebuilds the 25-feature fusion vectors with the real-data branch
  scores in place and runs the four-arm ablation the whole project is built to
  test, plus calibration, codecs and time to detection.
- **05** produces consented cloned-voice corpus cells with XTTS-v2, with
  edge-tts and Piper as the fallbacks Plan B section 4 names, and deletes the
  voice model and reference recording at the end.
