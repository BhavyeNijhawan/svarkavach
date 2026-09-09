# Problem Statement 2 - Research Document

## Voice-Clone Fraud Call Detection: Fusing Acoustic Anti-Spoofing with Scam-Intent NLP for Hindi–English Calls

**Course:** BCSE419L - Introduction to Natural Language Processing (Speech and Language Processing Lab)
**Student:** Bhavye Nijhawan (23BAI0100)
**Assessment:** Lab Assessment 2 - Individual Problem Statement
**Cost constraint:** 100% free - open datasets (direct download, no approval), open-source tools, free Colab compute. No telephony hardware - everything runs on recorded audio files.

---

## 1. Formal Problem Statement

> Develop a software system that flags fraudulent phone calls by simultaneously (a) detecting **AI-cloned / synthetic voices** using anti-spoofing acoustic features (GFCC, LFCC, CQCC-style spectral features, i-vector/embedding-based speaker representations, prosodic dynamics) and (b) detecting **scam intent** in the transcribed conversation (urgency language, coercion patterns, and entities such as OTP / account number / KYC / "digital arrest") using NER + sequence models, over Hindi–English code-mixed audio, and fusing both signals into a single explainable fraud-risk score.

Input: an audio file of a call (or live microphone recording). Output: fraud probability + evidence panel, e.g., *"Synthetic-voice score 0.87; detected entities: OTP request, urgency phrase 'account will be blocked in 10 minutes'."*

---

## 2. Background and Motivation

- Voice-cloning scams are exploding in India: "digital arrest" scams, fake-relative emergency calls, and bank-KYC frauds. RBI, TRAI, and I4C have issued repeated public advisories (2023–2025); reported cyber-fraud losses in India crossed **₹11,000+ crore in FY2024** (I4C data).
- Modern TTS (voice cloning from 3 seconds of audio) makes the "grandparent scam" trivially scalable - a genuinely current threat.
- **Existing defenses are one-eyed:**
  - Anti-spoofing research (ASVspoof community) analyzes **only the audio** - it can say "this voice is synthetic" but not "this is a scam" (a legitimate robocall is synthetic too, and a human scammer defeats audio-only checks).
  - Text spam/scam classifiers analyze **only words** - defeated by novel phrasings, and blind to cloned voices reading innocent-sounding scripts.
- **The gap = the fusion**: a call is dangerous when *voice authenticity signals* and *conversational intent signals* are combined - and no public system does this for **code-mixed Hindi–English telephone speech**. That is the research contribution and the patent angle.

---

## 3. Literature Survey (key prior work)

| # | Work | What it did | What it lacks (our opportunity) |
|---|------|-------------|-------------------------------|
| 1 | **ASVspoof 2019 (LA)** - Todisco et al., INTERSPEECH 2019 | Standard benchmark for logical-access spoofing (TTS/VC attacks); baselines LFCC-GMM, CQCC-GMM | Clean studio audio; English; no intent analysis |
| 2 | **ASVspoof 2021** - Yamagishi et al. | Added **telephony codecs & compression** (LA/PA/DF tracks) - directly relevant to phone-call conditions | Still audio-only |
| 3 | **RawNet2** (Tak et al., 2021) | End-to-end raw-waveform anti-spoofing CNN, strong free baseline with public code | No explainability, no text |
| 4 | **AASIST** (Jung et al., 2022) | Graph-attention anti-spoofing, SOTA-class, public PyTorch code | Same |
| 5 | **"In-the-Wild" audio deepfake dataset** (Müller et al., 2022) | ~38 h real + deepfake audio of celebrities collected from the wild; shows lab models degrade badly in the wild - free direct download | English; no conversation/intent dimension |
| 6 | Scam/vishing text detection literature (phishing NLP, spam SMS datasets, fraud-call transcript studies) | Intent/urgency classification from text works well with classical + neural models | English/monolingual; not fused with audio authenticity |
| 7 | i-vector / x-vector speaker verification (Dehak et al. 2011; Snyder et al. 2018) | Speaker embeddings usable for caller-identity consistency checks | Not applied to scam-call screening pipelines |

**Summary of the research gap:** (a) no public **audio-authenticity + scam-intent fusion** system; (b) nothing for **Hinglish** telephone speech; (c) explainable evidence output (needed for real-world trust and for a patent claim) is absent in prior systems.

---

## 4. Novelty Claims

1. **Two-channel fusion architecture:** anti-spoofing branch (is the voice synthetic?) + intent branch (is the conversation a scam?) fused into one calibrated risk score - each branch covers the other's blind spot (human scammer → intent branch catches; innocent synthetic voice → intent branch clears).
2. **Code-mixed scam-entity NER:** a purpose-built entity tagset for Indian fraud calls - `OTP`, `BANK_ENTITY`, `THREAT_DEADLINE`, `AUTHORITY_CLAIM` (police/CBI/RBI impersonation), `PAYMENT_HANDLE` (UPI ID) - trained with CRF and BiLSTM on a self-built Hinglish scam-script corpus (Module 3 syllabus: NER + CRF + LSTM, exactly).
3. **Telephone-channel robustness by design:** all training/eval at 8 kHz with codec simulation (free `ffmpeg` AMR/GSM codecs), aligned with ASVspoof-2021 findings.
4. **Explainable evidence panel** combining SHAP feature attributions (audio) with highlighted transcript spans (text) - the human-reviewable output format is itself a claimable system element.

---

## 5. Patent Angle (for possible later filing)

- **Type:** utility patent - "System and method for detecting fraudulent voice calls using joint voice-authenticity and conversational-intent analysis."
- **Draft independent claim (method):** A method comprising: receiving call audio; extracting spoofing-discriminative spectral features and a speaker embedding; computing a voice-authenticity score; transcribing the audio with per-token language identification; extracting scam-intent features including named fraud entities and urgency indicators; fusing the authenticity score and intent features in a trained model to produce a fraud-risk score; and presenting evidence comprising attributed audio features and highlighted transcript spans.
- **Dependent claims:** codec-robust 8 kHz operation; code-mixed NER tagset; caller-embedding consistency check across a user's call history; on-device inference variant; escalation API to a blocklist service.
- **Prior-art caution:** patents exist for audio-only spoof detection (e.g., Pindrop's phoneprinting family) and for text-only spam filtering. The **joint fusion + code-mixed entity layer + evidence panel** is the differentiation; keep dated records (this document = invention-disclosure draft).

---

## 6. Datasets (all free, direct download - no approval process)

| Dataset | Content | Access | Use |
|---------|---------|--------|-----|
| **ASVspoof 2019 LA** | ~120k utterances, bonafide vs 19 TTS/VC attack types | Direct download, Edinburgh DataShare | Train/benchmark anti-spoof branch |
| **ASVspoof 2021 LA/DF eval** | Codec-degraded + deepfake eval sets | Direct download, Zenodo | Telephone-robustness evaluation |
| **In-the-Wild** (Müller et al.) | 38 h real vs deepfake celebrity audio | Direct download (deepfake-total.com / Fraunhofer page) | Generalization test |
| **Self-built Hinglish scam-script corpus** | 150–300 short scripted dialogues: scam (KYC, OTP, digital-arrest, lottery, fake-relative) vs benign (bank reminders, delivery calls, family calls). Written by me + friends acting; text first, then voice-recorded readings | Self-created, consent from participants | Train/eval intent + NER branch |
| **Self-generated cloned audio** | Clone consenting participants' voices with **Coqui XTTS-v2** (free, local/Colab) reading both scam and benign scripts | Self-created | End-to-end demo: cloned-voice scam call caught by both branches |
| **Kaggle spam/fraud text sets** (SMS Spam Collection; fraud-call transcript sets) | Extra text training data | Free Kaggle download | Augment intent classifier |

Ethics note: voice cloning done ONLY on consenting participants' own voices, used only inside this project; scripts fictional; no real bank/UPI data anywhere.

---

## 7. Tools and Software Stack (all free / open source)

- **Anti-spoof branch:** `librosa` + `spafe` (LFCC, GFCC, CQCC-approx, MFCC), `speechbrain` (free) for x-vector/ECAPA speaker embeddings (modern i-vector successor; classical i-vector via `kaldi`-free implementations if syllabus alignment demanded), public **RawNet2 / AASIST reference code** (GitHub, MIT-style licenses) as strong baselines.
- **ASR:** Whisper small/medium (free) with language hints for Hinglish; `whisper-timestamped` for spans.
- **Intent/NER branch:** `sklearn-crfsuite` (CRF - Module 3), PyTorch BiLSTM tagger, TF-IDF + SVM and DistilBERT/MuRIL fine-tune (free Colab) for utterance-level intent.
- **Fusion + explainability:** logistic/gradient-boosting late fusion (`scikit-learn`), SHAP, transcript-span highlighting in a **Streamlit** demo app.
- **Audio degradation:** `ffmpeg` (GSM/AMR codec simulation, 8 kHz resampling).
- **Compute:** Google Colab free T4 - RawNet2-scale training on ASVspoof LA subset is feasible; feature-based (LFCC-GMM) baselines run on CPU.

---

## 8. System Architecture (detailed)

```
                        ┌────────────────────────────────────────┐
 call audio (wav) ─────►│ Pre-processing: 8 kHz mono, VAD (webrtcvad, free)
                        └───────────────┬────────────────────────┘
            ┌───────────────────────────┴───────────────────────────┐
            ▼ (audio branch)                                        ▼ (text branch)
 ┌─────────────────────────┐                          ┌─────────────────────────────┐
 │ Anti-spoofing            │                         │ Whisper ASR + lang-ID        │
 │ • LFCC/GFCC/MFCC stats   │                         │ → Hinglish transcript        │
 │ • ECAPA speaker embedding│                         │ ┌─────────────────────────┐ │
 │ • GMM baseline +         │                         │ │ Scam-entity NER (CRF /  │ │
 │   RawNet2 CNN            │                         │ │ BiLSTM): OTP, THREAT,   │ │
 │ → authenticity score     │                         │ │ AUTHORITY, PAYMENT      │ │
 └───────────┬─────────────┘                          │ └─────────────────────────┘ │
             │                                        │ • urgency/coercion intent   │
             │                                        │   classifier (TF-IDF+SVM /  │
             │                                        │   DistilBERT)               │
             │                                        │ → intent score + entities   │
             │                                        └──────────────┬──────────────┘
             └──────────────────┬─────────────────────────────────── ┘
                                ▼
                  ┌───────────────────────────────┐
                  │ Calibrated late fusion (LR/GBM)│
                  │ → fraud-risk score [0,1]       │
                  │ + evidence panel (SHAP + spans)│
                  └───────────────────────────────┘
```

---

## 9. Evaluation Plan

- **Anti-spoof branch:** Equal Error Rate (EER) and min t-DCF on ASVspoof 2019 LA eval (published baselines: LFCC-GMM ≈ 8–9% EER; RawNet2 ≈ 1–5%); generalization EER on In-the-Wild (expect large degradation - analyzing WHY is a results section, not a failure).
- **Intent/NER branch:** entity-level Precision/Recall/F1 (BIO tagging, Module 3); intent accuracy/F1 with 5-fold CV on the scam-script corpus; confusion analysis benign-urgent (bank reminder) vs scam-urgent - the hard class.
- **Fusion:** end-to-end detection accuracy/AUC on a held-out mixed test set of 4 cell types: {human, cloned} × {benign, scam}; ablation: audio-only vs text-only vs fused - the fused-wins table is the headline result.
- **Robustness:** all metrics repeated after GSM/AMR codec simulation.
- **Rigor:** speaker-disjoint splits, fixed seeds, confidence intervals.

---

## 10. Ethics and Safety

- Voice cloning only with written consent, only participants' own voices, models/audio deleted after the project; scripts fictional; the tool is **defensive** (detection), and no scam automation is built.
- Dataset scripts avoid real institution names in generated audio where possible ("your bank" instead of a specific bank) to prevent misuse.
- Report includes a misuse-and-limitations section (false-positive harm: legitimate synthetic-voice services).

---

## 11. Syllabus Mapping (viva defense table)

| Syllabus module | Where it is used |
|---|---|
| M1 - POS, parsing, regex | Regex + POS patterns for urgency phrases; ELIZA-style pattern rules as an interpretable baseline |
| M2 - N-grams, TF-IDF/BoW, embeddings | Intent classifier features; MuRIL embeddings for Hinglish |
| M3 - **NER, CRF, LSTMs**, sentiment | Core of the intent branch: scam-entity CRF + BiLSTM tagger |
| M4 - Attention, encoder-decoder | DistilBERT/attention intent model; attention visualization of suspicious spans |
| M5 - Short-time analysis, energy, ZCR, STFT | VAD, framing, spectral front-end |
| M6 - **MFCC, GFCC, LPCC, i-vector** | Anti-spoofing feature set + speaker embeddings (i-vector concept → ECAPA) |
| M7 - ASR, speaker recognition, DNN/CNN speech models, evaluation metrics | Whisper ASR, ECAPA speaker recognition, RawNet2 CNN, EER/t-DCF metrics |

---

## 12. References (starting reading list)

1. Todisco M. et al. (2019). ASVspoof 2019: Future horizons in spoofed and fake audio detection. *INTERSPEECH 2019*.
2. Yamagishi J. et al. (2021). ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection. *ASVspoof Workshop*.
3. Tak H. et al. (2021). End-to-end anti-spoofing with RawNet2. *ICASSP 2021*.
4. Jung J. et al. (2022). AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks. *ICASSP 2022*.
5. Müller N. et al. (2022). Does audio deepfake detection generalize? ("In-the-Wild" dataset). *INTERSPEECH 2022*.
6. Dehak N. et al. (2011). Front-end factor analysis (i-vectors) for speaker verification. *IEEE TASLP*.
7. Desplanques B. et al. (2020). ECAPA-TDNN: Emphasized channel attention for speaker verification. *INTERSPEECH 2020*.
8. Snyder D. et al. (2018). X-vectors: Robust DNN embeddings for speaker recognition. *ICASSP 2018*.
9. Lafferty J. et al. (2001). Conditional Random Fields. *ICML 2001*.
10. Radford A. et al. (2022). Whisper: Robust speech recognition via large-scale weak supervision. *arXiv:2212.04356*.
11. Khanuja S. et al. (2021). MuRIL: Multilingual representations for Indian languages. *arXiv:2103.10730*.
12. I4C / RBI public advisories on digital-arrest and vishing frauds (2023–2025) - cite for motivation statistics.
