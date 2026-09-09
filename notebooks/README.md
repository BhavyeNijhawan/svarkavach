# Colab notebooks

Everything the laptop cannot do. Each notebook clones this repository, runs on
Colab's hardware, and downloads a small zip of JSON results that drops into
`data/results/` on the laptop. The console's provenance table then shows which
numbers came from which machine.

Change `REPO_URL` at the top of the first code cell of every notebook to your
own fork before running anything.

| notebook | needs | time | writes back |
| --- | --- | --- | --- |
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
