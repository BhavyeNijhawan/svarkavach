# Problem Statement 2 - Plan B (Fallback Plan)

Trigger-based fallbacks. All free, no hardware. Sections map to risks in `02_Implementation_Plan.md`.

---

## §1. If neural anti-spoofing (RawNet2) can't be trained on free Colab

1. **Feature-based only:** LFCC/GFCC/MFCC + GMM and + Gradient Boosting run entirely on CPU and still reproduce respectable published-baseline EERs on ASVspoof LA - a complete, defensible anti-spoof branch by itself (and it maps to Module 6 even more directly).
2. **Use pretrained detectors instead of training:** free pretrained deepfake-audio detectors exist on HuggingFace (e.g., AASIST/wav2vec2-based anti-spoofing checkpoints). Inference-only on Colab/CPU. Report them as "off-the-shelf branch" and focus your contribution on the fusion + Hinglish intent side (which no pretrained model provides).
3. **Dataset too big to download:** use the ASVspoof 2019 LA *dev* partition only (~2 GB) or the 2-second-clip subsets mirrored on Kaggle; document the subset precisely.

**What survives:** the entire fusion story and evaluation grid are unchanged - only the audio branch's internals swap.

---

## §2. If Whisper's Hinglish transcription is too poor

1. **Gold-script mode:** the corpus is scripted, so perfect transcripts already exist for free. Run the NER/intent branch on gold text and report the ASR-degradation experiment as an analysis section instead of a dependency. The system claim becomes "transcript-in" (which is honest: production systems use telecom-grade ASR anyway).
2. **Try AI4Bharat IndicWhisper / IndicConformer** (free) - usually better on Hinglish phone speech.
3. **Romanization normalization:** add a preprocessing step mapping common Hinglish spellings to canonical forms (small dictionary, free) - often recovers several F1 points for NER on noisy transcripts.

---

## §3. If the scam corpus ends up too small for transformer fine-tuning

1. CRF + TF-IDF/SVM are designed for small data - keep them as the main models (they are ALSO the syllabus-listed methods, so this is not a downgrade for grading).
2. **Data augmentation (free):** paraphrase scam utterances with a local LLM / synonym substitution; entity-swap augmentation (swap bank names, amounts, deadlines) multiplies NER data 5–10×.
3. Reduce entity tagset to the 4 highest-frequency tags (`OTP`, `THREAT_DEADLINE`, `AUTHORITY_CLAIM`, `PAYMENT_HANDLE`).

---

## §4. If Coqui XTTS voice cloning fails to install/run

1. Alternative free TTS for the "synthetic" audio cells: **Piper TTS** (very light, CPU), **edge-tts** (free Microsoft neural voices), F5-TTS / OpenVoice (Colab notebooks available). Cloning of a *specific* person's voice is nice-to-have, not essential - the anti-spoof branch only needs *synthetic vs human* audio, which any neural TTS provides.
2. Also include ASVspoof's own TTS attack samples as the synthetic class for the grid (already downloaded).

---

## §5. If the fusion result is not clearly better than single branches

1. Check calibration first (uncalibrated scores ruin late fusion) - apply Platt scaling before fusing.
2. Enrich the test grid with more hard negatives (benign robocalls = synthetic + benign) - fusion advantages appear exactly on these cells; if the grid is too easy, single branches look falsely sufficient.
3. Worst case, reframe headline as **complementarity analysis**: show the confusion matrices where each single branch fails and fusion covers - a per-cell analysis table is publishable/defensible even if aggregate AUC gains are modest.

---

## §6. Absolute minimum viable submission

Guaranteed deliverable with zero external risk (everything local/self-created):
- LFCC/GFCC + GMM anti-spoof classifier evaluated on ASVspoof dev subset (or Kaggle mirror),
- CRF scam-entity NER + SVM intent classifier on the self-built text corpus (no audio needed for this branch),
- Rule-based + logistic fusion on the self-recorded/edge-tts audio grid,
- Streamlit demo + full report.

**Decision rule:** if by end of Week 4 the neural anti-spoof branch is not training reliably, switch permanently to §1.2 (pretrained detector + feature-based baseline) and continue the main timeline unchanged.
