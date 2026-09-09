# Problem Statement 2 - Detailed Implementation Plan

## Voice-Clone Fraud Call Detection (Acoustic Anti-Spoofing + Scam-Intent NLP Fusion)

**Guiding rule:** only free software, direct-download datasets (no approval gates at all for this PS), free Colab compute, a normal laptop. Written to not need changes later - every phase has a hard output, a checkpoint, and a fallback pointer to `03_Plan_B.md`.

---

## Phase 0 - Setup (Week 1)

**Tasks**
1. GitHub repo (private):
   ```
   ps2-fraudcall-detection/
   ├── data/                (gitignored)
   │   ├── asvspoof19_la/
   │   ├── in_the_wild/
   │   └── scam_corpus/     (text + recorded + cloned audio)
   ├── notebooks/
   ├── src/
   │   ├── audio/           (features.py, antispoof.py, embeddings.py)
   │   ├── text/            (ner.py, intent.py, asr.py)
   │   ├── fusion/
   │   └── utils/
   ├── reports/
   └── requirements.txt
   ```
2. Download **ASVspoof 2019 LA** (Edinburgh DataShare, ~7 GB - start early, direct link, no signup barrier) and the **In-the-Wild** dataset.
3. Colab setup: `pip install librosa spafe speechbrain openai-whisper sklearn-crfsuite transformers shap streamlit webrtcvad TTS` (Coqui).
4. Verify `ffmpeg` codecs for GSM/AMR simulation work: `ffmpeg -i in.wav -ar 8000 -acodec libgsm out.gsm`.

**Checkpoint:** both datasets load; one ASVspoof file → LFCC features extracted; Coqui XTTS generates a test sentence locally/Colab.

---

## Phase 1 - Scam-Script Corpus Creation (Week 2) - do this early, it's the only "manual" work

**Tasks**
1. Write **150–300 short dialogues** (5–15 turns each) in natural Hinglish across classes:
   - Scam: fake KYC ("aapka account block ho jayega"), OTP theft, digital-arrest (CBI/police impersonation), lottery/prize, fake-relative emergency, electricity-bill threat.
   - Benign-but-similar (hard negatives): real bank reminders, delivery OTP calls (legit context), family calls, telemarketing.
   Use LLM assistance for drafting variety, then manually edit for realism (free).
2. Annotate entities in BIO format (Module 3): `OTP`, `BANK_ENTITY`, `AUTHORITY_CLAIM`, `THREAT_DEADLINE`, `PAYMENT_HANDLE`, `PERSONAL_INFO_REQ`. Use **doccano** (free, local) or plain text markup. Write a 1-page annotation guideline (needed for report + patent record).
3. Record ~60–100 of the dialogues: me + 3–5 consenting friends reading scam/benign scripts on phone mics. Save consent forms.
4. Generate **cloned versions**: clone each consenting speaker with Coqui XTTS-v2 (needs ~6 s reference audio) reading the same scripts → the {cloned × scam}, {cloned × benign} cells of the evaluation grid.
5. Split: speaker-disjoint train/dev/test; metadata CSV.

**Checkpoint:** corpus stats table (dialogues, tokens, entities per class); 4-cell audio grid populated {human, cloned} × {scam, benign}.

---

## Phase 2 - Anti-Spoofing Branch (Weeks 3–4)

**Tasks**
1. **Classical baseline (CPU, guaranteed to work):** LFCC (20 coeffs + Δ + ΔΔ) → 2×GMM (bonafide vs spoof, 512 components, `sklearn.mixture`) on ASVspoof 2019 LA. Target: EER in the published ~8–11% ballpark. Also run GFCC and MFCC variants for the syllabus-feature comparison table (Module 6).
2. **Neural model:** train **RawNet2** (public reference code) on LA train subset in Colab (reduce epochs/subset if GPU time is tight - document exactly what subset). Target: EER well under the GMM baseline.
3. **Speaker embeddings:** ECAPA-TDNN via `speechbrain` (pretrained, free) → cosine-consistency score between call audio and (optional) enrolled reference of the "claimed" speaker; also use embedding statistics as spoof features (cloned voices show characteristic embedding artifacts).
4. Evaluate EER + min t-DCF on LA eval; generalization EER on In-the-Wild; codec-degraded EER (GSM/AMR, 8 kHz).
5. Run branch on the self-built audio grid: report scores per cell - cloned cells must score high.

**Checkpoint:** anti-spoof results table (3 feature sets × 2 models × 3 conditions). **If RawNet2 training infeasible on free GPU →** Plan B §1.

---

## Phase 3 - ASR + Intent/NER Branch (Weeks 4–5)

**Tasks**
1. Whisper-small/medium transcription of the recorded corpus; measure WER vs scripts (scripts ARE the gold transcripts - clever advantage of scripted data: free perfect references).
2. **NER:** train `sklearn-crfsuite` CRF with classic features (word shape, POS, gazetteers of bank/authority terms, ±2 context) - Module 3 exact match. Then BiLSTM (PyTorch, optionally + CRF layer) comparison. Report entity-level F1 for both.
3. **Intent classifier (utterance + call level):** TF-IDF + SVM baseline; DistilBERT-multilingual or MuRIL fine-tune (Colab, ~minutes) main model. Also an interpretable regex/pattern rule baseline (Module 1 tie-in) - urgency lexicon ("turant", "immediately", "block ho jayega", "warrant").
4. Aggregate to call-level intent score: max + mean pooling of utterance scores + entity-count features.
5. Robustness test: run NER/intent on Whisper (noisy) transcripts vs gold scripts → quantify degradation (this WER-vs-F1 analysis is a strong report section).

**Checkpoint:** NER F1 table (CRF vs BiLSTM; gold vs ASR transcripts); intent accuracy table. **If Hinglish ASR is too poor →** Plan B §2.

---

## Phase 4 - Fusion + Explainability (Week 6)

**Tasks**
1. Feature vector per call: [authenticity score, embedding-consistency, LFCC-GMM LLR] + [intent score, entity counts per type, urgency density].
2. Late fusion: logistic regression AND gradient boosting (`sklearn`), calibrated (Platt/isotonic); 5-fold CV on train, single final run on held-out test grid.
3. **Headline ablation:** audio-only vs text-only vs fused on the 4-cell grid. Expected story: audio-only misses {human × scam}; text-only misses {cloned × benign-script pretext}; fusion catches both.
4. Explainability: SHAP for fusion features + transcript-span highlighting (NER spans + top attention/TF-IDF terms). Design the evidence-panel JSON schema (also a patent artifact).

**Checkpoint:** fused AUC/accuracy beats both single branches; evidence panel renders for 3 demo calls.

---

## Phase 5 - Demo App + Robustness Hardening (Week 7)

**Tasks**
1. **Streamlit app:** upload WAV (or record via mic component) → pipeline → risk gauge + evidence panel (synthetic-voice meter, highlighted transcript, entity chips). All local, free.
2. Codec/noise stress test on the full pipeline (GSM, AMR-NB, added babble noise from free MUSAN dataset).
3. Latency measurement (CPU-only inference path documented - shows deployability without GPU).

**Checkpoint:** end-to-end demo on a fresh cloned-scam recording works live; robustness table complete.

---

## Phase 6 - Report, Slides, Freeze (Week 8)

**Tasks**
1. IEEE-format report: motivation (I4C/RBI stats), gap table, corpus description + annotation guidelines, both branches, fusion ablation, robustness, ethics/misuse section, limitations. Doubles as invention-disclosure draft.
2. 10-slide deck: threat landscape → gap → architecture diagram → 4-cell ablation table (headline) → demo video.
3. Demo video (OBS screen recording, free). Tag repo `v1.0`.

**Deliverables:** report PDF, slides, repo, demo video, corpus (text public, audio private/consented), results CSVs.

---

## Risk Register

| Risk | Likelihood | Mitigation | Fallback |
|---|---|---|---|
| ASVspoof download slow (7 GB) | Medium | Start Week 1; use LA subset if needed | Plan B §1 (feature-based only) |
| RawNet2 training exceeds free GPU quota | Medium | Train on subset, fewer epochs; checkpoint often | Plan B §1 (GMM + pretrained models) |
| Whisper Hinglish WER high | Medium | Scripts = free gold transcripts; report ASR-vs-gold gap | Plan B §2 |
| Corpus too small for DistilBERT | Low | TF-IDF+SVM and CRF work fine on small data | Plan B §3 |
| Coqui XTTS install issues | Low | Use Colab (known-working env) | Plan B §4 (alternative free TTS) |

## Timeline Summary

| Week | Phase | Hard output |
|---|---|---|
| 1 | Setup + downloads | Repo, datasets on disk, tools verified |
| 2 | Scam corpus | Annotated Hinglish corpus + 4-cell audio grid |
| 3–4 | Anti-spoof branch | EER/t-DCF results tables |
| 4–5 | ASR + NER/intent branch | F1 + accuracy tables |
| 6 | Fusion + explainability | Headline ablation + evidence panel |
| 7 | Demo + robustness | Streamlit app + stress tables |
| 8 | Report + slides | Final deliverables, v1.0 tag |
