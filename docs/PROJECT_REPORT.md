::: {custom-style="TitlePage"}
PROJECT REPORT
:::

# Detecting Voice-Cloned Fraud Calls in Hinglish by Fusing Anti-Spoofing, Scam-Intent and Prosody-Intent Mismatch Cues

::: {custom-style="TitleBlock"}
SwarKavach: a fused, per-turn fraud-call detector for code-mixed Hindi-English telephone speech

Submitted by

**Bhavye Nijhawan**, Reg. No. 23BAI0100

BCSE419L Speech and Language Processing Laboratory

Vellore Institute of Technology

September 2026

Repository: https://github.com/BhavyeNijhawan/svarkavach
:::

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

## Abstract

Phone fraud in India is moving from scripts read by human callers to scripts read by cloned voices, and the two defences in use each cover half the problem. Synthetic-speech detectors decide whether a voice is machine-made but not whether the caller is asking for an OTP; transcript classifiers catch the request but hear a cloned voice and a real one as the same. This project answers both questions together, during a live call, in the Hindi-English mix in which such calls are actually spoken. SwarKavach has four branches. A voice branch scores synthetic speech with cepstral front ends, a Gaussian-mixture and a gradient-boosted back end, a RawNet-style end-to-end detector and a TDNN speaker embedder, trained on 600 real Hindi field recordings paired with cloned renderings of the same transcripts with the recording channel matched on level, colour and bandwidth. A text branch transcribes the call with Whisper, tags seven fraud entity types with a linear-chain CRF (with a BiLSTM-CRF counterpart) and scores scam intent with a calibrated TF-IDF classifier. A structure branch reads the order of dialogue acts as a coercion ramp. A cross-modal branch measures the mismatch between the pressure in the words and the flatness of the delivery. A calibrated fusion layer, fitted on held-out speakers, turns twenty-five features into a risk band that updates every turn and an evidence panel that names the entities, phrases and voice cues behind it. On a 480-call corpus balanced over voice and content the full system reaches AUC 1.000 with an equal error rate of 0.017 and flags 57 of 60 scams at a median of 35.8 seconds, before the sensitive request. On 1,413 real robocall recordings it was never trained on, in another language, the text branch recalls 78.7 percent at a 5 percent false-alarm rate. The voice branch separates real from cloned Hindi speech at an equal error rate below 0.04 under every telephone codec tested. The evidence establishes that a fused, per-turn detector can flag a scripted fraud call before the request is made, in code-mixed speech, on a CPU.

**Keywords:** voice cloning detection, telephone fraud, Hinglish code-mixing, anti-spoofing, dialogue-act coercion, prosody-intent mismatch, calibrated late fusion, time to detection

---

## Contents

1. Introduction
2. Literature Review
3. Preliminary Concepts and Theoretical Background
4. Materials and Methods
5. Proposed Methodology and Implementation
6. Experimental Framework
7. Results and Discussion
8. Conclusion and Future Work

References

## List of Figures

- Figure 1. End-to-end information flow of SwarKavach
- Figure 2. Corpus generation from template grammar to annotated audio
- Figure 3. Construction of the matched anti-spoofing pair set
- Figure 4. Per-turn streaming decision and time-to-detection measurement
- Figure 5. Where each fusion feature comes from
- Figure 6. Console: call analysis with the evidence panel
- Figure 7. Console: live monitor, corpus browser and system view
- Figure 8. Results lab: headline ablation as displayed in the console
- Figure 9. Ablation of the four arms, AUC and F1
- Figure 10. Per-cell accuracy of each arm
- Figure 11. Arm comparison on the hard subset
- Figure 12. Results lab: entity recognition and intent as displayed
- Figure 13. Entity recognition F1 by type
- Figure 14. Scam intent, rule baseline against TF-IDF
- Figure 15. Results lab: calibration as displayed
- Figure 16. Calibration reliability diagram
- Figure 17. Results lab: anti-spoofing EER and DET curves as displayed
- Figure 18. Anti-spoofing EER by front end, back end and codec
- Figure 19. Confound audit of the pair set, raw and conditioned
- Figure 20. Out-of-domain recall on real robocalls
- Figure 21. Results lab: channel robustness and time to detection as displayed
- Figure 22. Time to detection, distribution and by scenario
- Figure 23. Channel robustness of the full and audio-only arms

## List of Tables

- Table 1. Critical comparison of recent literature
- Table 2. Hardware and software
- Table 3. Corpus composition
- Table 4. Performance metrics
- Table 5. Ablation of the four arms
- Table 6. Per-cell accuracy of each arm
- Table 7. Arm comparison on the hard subset
- Table 8. Entity recognition by type
- Table 9. Anti-spoofing equal error rate on real Hindi pairs
- Table 10. Confound audit of the pair set
- Table 11. Out-of-domain recall on real robocalls
- Table 12. Time to detection
- Table 13. Channel robustness

---

# Chapter 1. Introduction

## 1.1 Context and problem

A fraud call is a short piece of theatre with a fixed script. The caller claims to be from a bank, a courier, the police or a relative in trouble; states a problem; attaches a deadline; tells the listener not to consult anyone; and only then asks for an OTP, a card number or a transfer. The pattern is well known and still works, because the call is built to remove the minute in which someone would remember it. Two things have changed its scale. Automated dialling reaches thousands of numbers a day. And neural text-to-speech now produces Hindi that a listener on a band-limited, codec-compressed phone line cannot reliably tell from a person, so the script no longer needs a fluent, calm, present human to read it.

The calls this project targets have three properties that together defeat existing tools. They are conversational: the request for an OTP is only suspicious because of the threat before it and the instruction not to hang up after it. They are code-mixed: *"Aapka account block ho jayega, OTP bata dijiye"* is neither Hindi nor English, and tools built for one language mishandle the other half. And they may be delivered in a cloned voice, which changes what the same words mean: a bank reminder read by a machine is routine, a request for a card PIN read by a machine is almost certainly fraud.

## 1.2 Existing approaches and the gap

Three families of defence exist. Blocklists act before the call and are defeated by number spoofing. Transcript classifiers, increasingly built on large language models, read what was said during the call but are blind to the voice. Anti-spoofing detectors, developed around the ASVspoof benchmarks, decide whether speech was synthesised, but on isolated utterances rather than conversations, and they degrade when the synthesiser, language or channel changes.

The gap is specific. The two kinds of detector are built and evaluated apart, so neither sees the case that matters most: a benign call in a cloned voice, or a scam delivered by a real person. Neither is evaluated on code-mixed Hindi-English conversation. And the evaluation sets in common use are not audited for channel or authorship shortcuts, so a detector can score well on the recording conditions rather than on the voice or the intent. This project addresses all three.

## 1.3 Objectives

1. Develop a corpus of Hindi-English fraud and benign calls in which voice authenticity and scam content vary independently, with entity, dialogue-act and language annotations that are exact by construction.
2. Design a detector that fuses a synthetic-voice branch, a transcript branch, a dialogue-structure branch and a prosody-intent branch into one calibrated risk score that updates each turn.
3. Quantify what each branch contributes through an ablation over four arms on the same test split, reported per voice-by-content cell and on the calls written to be hard.
4. Evaluate the voice branch on real Hindi recordings paired with cloned renderings of the same transcripts, with the channel audited before any error rate is reported.
5. Validate generalisation on real fraud recordings from outside the training distribution and measure the time at which a call is flagged relative to the sensitive request.

## 1.4 Approach and contributions

SwarKavach treats a call as a stream of turns. After each caller turn it extracts cepstral and prosodic features from the audio, transcribes and tags the text, updates a dialogue-act model, and computes the mismatch between lexical pressure and acoustic arousal. Twenty-five features enter a logistic-regression fusion layer with isotonic calibration, fitted on speakers no branch was trained on, and the output is mapped to four risk bands with an alert at 0.65.

The contributions are: a template grammar whose slots emit gold entity, act and language labels with the text, with five subjects paired across the label so topic does not predict class; a prosody-intent mismatch feature and a coercion-trajectory feature; a method for building a matched anti-spoofing pair set from field recordings with a ten-statistic confound audit; a voice branch with classical, boosted and end-to-end back ends behind one interface; an ablation over four arms reported per cell and on a hard subset, with a bag-of-words leak check run before any model is trained; and out-of-domain validation on 1,413 real robocalls, a per-turn time-to-detection measurement and a codec sweep.

## 1.5 Report organisation

Chapter 2 positions the work against six recent studies. Chapter 3 defines the quantities used. Chapter 4 documents materials and procedure. Chapter 5 gives the architecture module by module with pseudocode. Chapter 6 states the research questions, data and metrics. Chapter 7 reports and interprets results. Chapter 8 concludes.

---

# Chapter 2. Literature Review

## 2.1 Terminology

*Anti-spoofing* decides whether an utterance was produced by a human or by a synthesis system; its standard metrics are the equal error rate (EER) and a cost-weighted detection cost function. *Scam-call detection* decides, from the words and sometimes the audio, whether the caller intends fraud. The two literatures rarely cite each other.

## 2.2 Benchmarks and generalisation

The ASVspoof series defines how anti-spoofing is evaluated. Its fifth edition (Wang et al., 2024) crowdsources both the bona fide speech, from many speakers in uncontrolled conditions, and the attacks, which now include adversarial perturbations aimed at the detectors, and adds metrics for spoofing-robust speaker verification. The design answers a finding that recurs across the field: detectors that reach near-zero EER on one round degrade on the next, because they learn the artefacts of particular synthesisers and channels. Yi et al. (2023) document that drop in a unified comparison of front ends and classifiers across ASVspoof 2021, ADD 2023 and In-the-Wild. Two of their observations shape this project: hand-crafted cepstral front ends stay competitive with learned representations when the test channel differs from training, and the datasets in use are almost entirely English and Chinese.

## 2.3 Indian languages

IndicSynth (Sharma, Ekbote and Gupta, 2025) provides about 4,000 hours of synthetic speech from 989 target speakers across twelve Indian languages including Hindi, with a mimicry subset in which a target speaker's voice is cloned. It shows that Hindi voice cloning is a solved generation problem at scale and provides the resource against which a Hindi anti-spoofing branch can be tested at scale. Every sample is an utterance; there is no conversation and no fraud content.

## 2.4 Content against delivery

Most detectors read the acoustic signal alone. SLIM (Zhu et al., 2024) observes that in real speech the style of delivery and the linguistic content depend on each other, because one person chose both, and that synthesis breaks the dependency because the content is given and the style is imposed. A representation of that dependency, pretrained on real speech only, improves out-of-domain detection and yields an explainable score. The prosody-intent feature in this project is the same intuition on a narrower pair: does the pressure in the words match the arousal in the voice? It needs a fraud lexicon and a pitch tracker rather than a self-supervised encoder, and runs on a laptop CPU.

## 2.5 Reading the conversation

Shen et al. (2025) frame the problem as this project does: defences act before or after the call, and the call itself is unprotected. A large language model classifies each utterance as fraudulent, safe or uncertain and warns the user in real time. It reads text only, so a cloned voice reading a benign script is invisible to it, and an LLM call per utterance constrains deployment. TeleAntiFraud-28k (Ma et al., 2025) is the first open audio-and-text fraud resource: 28,511 speech-text pairs built from transcribed real calls, regenerated by TTS for privacy and extended by LLM and multi-agent generation, annotated for fraud reasoning. Because every clip is regenerated, voice authenticity is not learnable from it, and the language is Chinese.

## 2.6 Critical comparison

**Table 1.** Critical comparison of recent literature. Values are as reported in the sources; where a source reports a protocol rather than a number, the protocol is given.

| Study | Method | Data | Key metric or setting | Main strength | Important limitation |
|---|---|---|---|---|---|
| Wang et al. (2024), ASVspoof 5 | Benchmark; crowdsourced bona fide and spoofed speech, adversarial attacks | Multi-speaker English, many TTS and VC systems | EER, minDCF, SASV metrics | Uncontrolled conditions and adversarial attacks at scale | Utterance level; English; no conversation |
| Yi et al. (2023), survey | Unified comparison of front ends and classifiers | ASVspoof 2021, ADD 2023, In-the-Wild | Cross-dataset EER | Isolates the out-of-domain drop by feature family | Descriptive; English and Chinese only |
| Sharma et al. (2025), IndicSynth | Large-scale synthetic speech for detection research | 4,000 h, 989 speakers, 12 Indian languages | Mimicry and diversity subsets | First Hindi-scale cloning resource | Utterances only; no fraud content; no field-channel matching |
| Zhu et al. (2024), SLIM | Self-supervised style-linguistics dependency, mismatch as feature | In- and out-of-domain deepfake sets | Out-of-domain gain with frozen encoders | Explainable cross-modal cue | Needs large real-speech pretraining |
| Shen et al. (2025) | LLM classifies each utterance during the call | Scam conversations; user study | Per-utterance fraudulent / safe / uncertain | Real-time warning | Text only; per-utterance LLM cost |
| Ma et al. (2025), TeleAntiFraud-28k | Audio-text dataset with reasoning annotations | 28,511 pairs, Chinese, TTS-regenerated audio | Scenario, fraud, fraud-type tasks | First open audio-text fraud resource | Voice authenticity not learnable; not Indian languages |

## 2.7 Position of this work

Anti-spoofing work is evaluated on utterances, so it never sees the context that says whether a synthetic voice matters. Scam-detection work is evaluated on text, so it never hears the voice. SwarKavach reads voice, words, structure and their mismatch on the same call, in Hindi-English code-mix, and treats the construction and auditing of its evaluation data as part of the method.

---

# Chapter 3. Preliminary Concepts and Theoretical Background

## 3.1 Cepstral front ends

Every voice-branch classifier reads cepstral coefficients. The signal is framed at 25 ms with a 10 ms hop and a Hamming window, pre-emphasised with coefficient 0.97, and the magnitude spectrum of each frame is passed through 40 triangular filters between 300 and 2800 Hz, log-compressed and decorrelated:

$$c_n = \sum_{m=1}^{M} \log(E_m) \cos\left[\frac{\pi n}{M}\left(m - \tfrac{1}{2}\right)\right], \quad n = 0, \ldots, N-1$$ (1)

where $E_m$ is the energy in filter $m$, $M = 40$ and $N = 20$ coefficients are kept. Linear (LFCC), mel (MFCC), gammatone (GFCC), constant-Q (CQCC) and linear-prediction (LPCC) front ends differ only in the filterbank. Deltas are appended and cepstral mean and variance normalisation (CMVN) removes the additive effect of a fixed channel.

## 3.2 Gaussian mixture likelihood ratio

One mixture is fitted to bona fide frames and one to spoofed frames; an utterance scores by the mean log-likelihood ratio over its $T$ frames:

$$\text{LLR}(X) = \frac{1}{T}\sum_{t=1}^{T}\left[\log p(x_t \mid \lambda_{\text{spoof}}) - \log p(x_t \mid \lambda_{\text{bona}})\right]$$ (2)

with up to 64 components per mixture. A gradient-boosted classifier over 480 utterance-level moments of the same frames gives a second score, and a RawNet-style network with a SincConv front end, whose filters are band-passes with learned cut-offs, gives a third that does not commit to any hand-crafted front end. Anti-spoofing is reported as EER together with the minimum tandem detection cost (t-DCF), which weights misses and false alarms by their operational cost with the verification stage fixed.

## 3.3 Acoustic arousal

Four measures describe how a turn was delivered: pitch coefficient of variation $\sigma_{f0}/\mu_{f0}$, pitch span $(\max f0 - \min f0)/\mu_{f0}$, the variance of syllable-peak levels in dB, and the standard deviation of syllable rate. Each is scaled between anchors at the 10th and 90th percentiles of the training audio and combined with weights 0.34, 0.24, 0.26 and 0.16 into $a \in [0,1]$. Syllable-peak variance is used for emphasis rather than frame-level energy variance, which tracks vowel-consonant alternation rather than stress.

## 3.4 Prosody-intent mismatch

With lexical arousal $\ell_i$ and acoustic arousal $a_i$ over the $n$ caller turns,

$$\text{PIM} = 0.65 \cdot \frac{\sum_i w_i \max(0, \ell_i - a_i)}{\sum_i w_i} + 0.35 \cdot \frac{1 - \rho(\ell, a)}{2} \cdot \max_i \ell_i$$ (3)

where $w_i$ weights turns by lexical pressure and $\rho$ is the Pearson correlation. The first term rewards turns where the words press and the voice does not; the second rewards a delivery that does not track the content, gated by peak pressure so a call with no coercive language cannot score on correlation noise.

## 3.5 Coercion trajectory

Each dialogue act carries a coercion rank in $[0,1]$: greeting 0.0, identification 0.1, problem statement 0.35, authority claim 0.55, threat 0.8, deadline 0.85, isolation 0.9, request for sensitive information 0.95, escalation 1.0. The slope of rank against caller-turn index is the coercion slope and the maximum is the peak. The slope separates a call that built pressure from one that merely mentioned a deadline.

## 3.6 Code-mixing index

With $n_L$ tokens in language $L$, $n$ tokens in total and $u$ language-independent tokens,

$$\text{CMI} = 1 - \frac{\max_L n_L}{n - u}$$ (4)

## 3.7 Metrics

At threshold $\tau$ with true positives $TP$, false positives $FP$ and false negatives $FN$:

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$ (5)

AUC is the probability that a random scam call scores above a random benign one. EER is the point where false-positive and false-negative rates coincide. The Brier score is the mean squared gap between predicted probability and label; ECE averages, over probability bins, the gap between mean prediction and observed rate. Time to detection is the end time of the first caller turn whose fused risk reaches 0.65.

---

# Chapter 4. Materials and Methods

## 4.1 Materials

**Table 2.** Hardware and software.

| Item | Specification |
|---|---|
| Training and evaluation | Google Colab runtime (Tesla T4 available), Python 3.13; the reported models train on CPU |
| Console and development | Laptop, Windows 11, 7.9 GB RAM, Python 3.11.9 |
| Numerical stack | NumPy 2.4.3, SciPy 1.17.1, scikit-learn 1.9.0, PyTorch 2.11.0 CPU |
| Speech synthesis | edge-tts 7.2.8, Microsoft neural voices (hi-IN Madhur, hi-IN Swara, en-IN Prabhat, Neerja, NeerjaExpressive) |
| Speech recognition | OpenAI Whisper, small multilingual model, language Hindi, CPU, fp16 off; segments aligned to turns. Reference transcripts from the corpus measure the taggers independently of recognition error and score recognition by word error rate |
| Codec simulation | FFmpeg for GSM 06.10; G.711 mu-law and A-law in NumPy |
| Real Hindi speech | GramVaani Hindi ASR corpus, dev and eval, 8 kHz telephone MP3 |
| Real fraud recordings | Robocall audio dataset, 1,413 recordings with transcripts |
| Console | Flask server, static JavaScript front end |
| Orchestration | Git; a runner that checkpoints each stage to Drive and rebuilds only what its inputs changed |

All audio is processed at 8 kHz, the telephone rate.

## 4.2 Procedure

**1. Corpus generation.** A template grammar produces 480 calls. Each call is planned (subject, label, voice, speaker, split) before it is rendered, so the four voice-by-content cells are balanced by construction. Every entity, act and language tag is emitted by the slot that produced the token. Speakers are assigned to splits before calls are assigned to speakers.

**2. Rendering.** Each turn is synthesised with a neural voice whose rate, pitch and volume follow the turn's act. A cloned caller keeps a fixed contour whatever the act; a human caller modulates.

**3. Pair construction.** 600 real GramVaani utterances are paired with neural renderings of their own transcripts, half also vocoded. Both sides go down the same channel: G.711 companding, DC removal, a brick-wall band-pass to 300 to 2800 Hz, silence trimming, peak normalisation, and noise matching in which the quieter side is lifted to the louder side's floor with noise shaped like the recording's own quiet-frame spectrum.

**4. Confound audit.** Ten statistics per clip, on the raw file and after the detector's own conditioning; the build stops if any conditioned statistic alone exceeds AUC 0.75.

```mermaid
flowchart TD
    R[GramVaani utterance<br/>8 kHz telephone MP3] --> C1[G.711 companding]
    T[Same transcript] --> N[Neural rendering<br/>Devanagari voices only]
    N --> V{half of them}
    V -->|plain| C2[G.711 companding]
    V -->|vocoded| VO[LPC vocoder artefacts] --> C2
    C1 --> B1[DC removal, 300 to 2800 Hz brick wall, silence trim]
    C2 --> B2[DC removal, 300 to 2800 Hz brick wall, silence trim]
    B1 --> P[Peak normalise both to 0.95]
    B2 --> P
    P --> M[Lift the quieter side to the louder floor<br/>noise shaped like the recording's quiet frames]
    M --> A[Confound audit<br/>10 statistics, raw and conditioned]
    A -->|worst conditioned AUC below 0.75| S[Pair set + manifest with build version]
    A -->|above 0.75| X[Stop the build]
```

**Figure 3.** Construction of the matched anti-spoofing pair set.

**5. Feature extraction.** Cepstral frames for the GMM, 480 pooled moments for the boosted classifier, the raw waveform for the end-to-end detector, TDNN speaker embeddings per segment, four prosodic measures per caller turn, TF-IDF per turn, entity tags, act-sequence likelihoods, language tags and the cross-modal quantities.

**6. Training.** Entity taggers, intent and act models on the train split; prosody anchors recalibrated on train audio; anti-spoofing back ends on the train partition of the real pair set; fusion and every ablation arm on the dev split, so the fusion never sees in-sample branch outputs.

**7. Evaluation.** Every reported number comes from the test split or from outside the training distribution: the pair test partition, the robocall recordings, and codec-degraded test audio.

---

# Chapter 5. Proposed Methodology and Implementation

## 5.1 Problem formulation

Input: a call $C = \{(s_j, x_j, w_j)\}_{j=1}^{J}$ of $J$ turns with speaker $s_j$, audio $x_j$ and transcript tokens $w_j$. Output after each caller turn $k$: a probability $p_k$, a band in {low, elevated, high, critical} with edges 0.35, 0.65 and 0.85, the entity spans found so far, and reasons. The call-level decision is $p_J \ge 0.65$.

## 5.2 Architecture

```mermaid
flowchart LR
    A[Call audio 8 kHz] --> B[Turn segmentation]
    T[Transcript tokens] --> B
    B --> V[Voice branch<br/>cepstral front ends, GMM LLR,<br/>pooled moments, GBM,<br/>RawNet-style SincConv net,<br/>TDNN speaker consistency,<br/>jitter, shimmer, HNR]
    B --> I[Text branch<br/>Whisper ASR,<br/>CRF / BiLSTM-CRF entity tagger,<br/>TF-IDF intent,<br/>urgency density]
    B --> S[Structure branch<br/>dialogue-act HMM,<br/>coercion slope and peak]
    B --> X[Cross-modal branch<br/>prosody-intent mismatch,<br/>code-mixing index,<br/>entity-language alignment,<br/>callee resistance]
    V --> F[Fusion<br/>25 features,<br/>logistic regression + isotonic]
    I --> F
    S --> F
    X --> F
    F --> R[Risk p_k and band]
    R --> E[Evidence panel<br/>reasons per turn, entity spans,<br/>time to detection]
```

**Figure 1.** End-to-end information flow. Every branch consumes the same turn segmentation; the fusion layer sees only the twenty-five named features.

## 5.3 Corpus generation

```mermaid
flowchart TD
    G[Template grammar<br/>10 scam + 8 benign scenarios<br/>5 subject pairs] --> P[Planner<br/>subject first, label inside<br/>voice drawn independently]
    P --> D[Dialogue builder<br/>arcs of dialogue acts<br/>slot filling with typed values]
    D --> A[Annotations emitted by slots<br/>BIO entities, acts, language tags]
    D --> R[Neural rendering<br/>rate pitch volume from act<br/>cloned caller: fixed contour]
    R --> W[Call audio + turn timings]
    A --> J[Call JSON]
    W --> J
    F[Grammar fingerprint] --> M[Manifest]
    J --> M
```

**Figure 2.** Corpus generation from template grammar to annotated audio.

A dialogue is an arc of acts, for instance `GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE AUTHORITY_ASSERT DEADLINE INSTRUCT REQUEST_SENSITIVE VICTIM_COMPLY CLOSE`. For each act a template is drawn from the scenario's own pool, then from style pools, and always from a neutral pool shared by both classes. Templates carry typed slots such as `{BANK}`, `{OTP_WORD}` and `{DEADLINE}`; a slot resolves to a surface string and an entity type at once, so its tokens receive the matching BIO tags and nothing else in the turn is touched. A validator forbids literal entity mentions outside slots.

Two rules keep the label out of the text. Five subjects (a parcel, a bank KYC call, a transaction alert, a relative in trouble, a financial offer) each have a scam and a benign scenario that share their setup lines and differ only in what is asked for. And greetings, closings, acknowledgements and polite instructions come from one pool used by both classes. Parameters: 480 calls, 24 speakers, split 50/25/25 by speaker, scam and synthetic-voice ratios 0.5 drawn independently, 22 percent of scams in a lexically mild style, 25 percent of calls from scenarios without a counterpart. The manifest carries a fingerprint of the grammar, so a corpus and the models trained on it always correspond.

## 5.4 Voice branch

Every cepstral computation starts with one conditioning function: remove DC, apply a Kaiser-windowed FIR band-pass from 300 to 2800 Hz (100 Hz transition, 60 dB stop band, run forwards and backwards), trim leading and trailing frames more than 40 dB below the loudest, and add dither 70 dB below the signal. The conditioned signal is RMS-normalised, framed, and passed to the front ends of Section 3.1; voiced frames are selected by an energy-based detector. Three back ends sit behind one scoring interface, preferred in the order end-to-end, boosted, mixture. The RawNet-style network reads the conditioned waveform through a SincConv layer of 20 learned band-pass filters, residual blocks and a gated recurrent unit; the boosted classifier (220 iterations, learning rate 0.07, depth 4) reads 480 pooled moments; the GMM gives Equation 2. Jitter, shimmer and harmonic-to-noise ratio come from the raw signal. A small TDNN speaker embedder scores each segment of the caller's audio, and the cosine similarity between segments becomes the speaker-consistency feature: a person drifts a little between segments, a single synthetic voice does not drift at all, and a spliced call with two voices drifts a lot, so the feature is informative at both ends. The band is the one the real telephone recordings occupy, so training, evaluation and the live detector all see the same kind of signal. Eight features leave this branch.

## 5.5 Text branch

Whisper small converts the caller audio to text and its segments are aligned to turns; the corpus also carries a reference transcript, against which recognition is scored by word error rate. Two taggers share one tokeniser, label space and scoring: a linear-chain conditional random field, implemented in the project so that it runs inside the per-turn budget on a CPU and can use the language tag as a feature, and a BiLSTM with a CRF output layer whose input is the word embedding concatenated with embeddings of the same symbolic features. Both tag seven entity types in BIO form: OTP, bank entity, authority claim, threat or deadline, payment handle, personal-information request, money amount. Features are the token, its shape, affixes, neighbours and language tag; L2 = 0.5, 200 iterations. A TF-IDF representation over word unigrams and bigrams plus character 3 to 5 grams, which is what survives romanised Hindi spelling variation, feeds a class-balanced logistic regression with C = 4.0; per-turn probabilities are pooled as 0.6 times the maximum plus 0.4 times the mean. A lexicon rule baseline is kept for comparison. Nine features leave this branch: intent score, peak turn intent, urgency density and six entity densities.

## 5.6 Structure and cross-modal branch

A hidden Markov model over acts is fitted separately on scam and benign calls, and the log-likelihood ratio of the observed sequence is a feature. Coercion slope and peak follow Section 3.5; the prosody-intent mismatch follows Equation 3; CMI, switch entropy and entity-language alignment follow Section 3.6; callee resistance is the fraction of callee turns tagged as resisting. Word-level models ask what was said; this branch asks in what order and in what voice. Eight features leave it.

## 5.7 Fusion and streaming decision

```mermaid
stateDiagram-v2
    [*] --> Listening
    Listening --> Scoring: caller turn ends
    Scoring --> Listening: p_k < 0.65
    Scoring --> Alerted: p_k >= 0.65
    Alerted --> Alerted: later turns update band
    Alerted --> [*]: call ends
    Listening --> [*]: call ends, no alert
    note right of Scoring
        features recomputed on
        all caller turns so far;
        t_end of this turn is
        recorded if it alerts
    end note
```

**Figure 4.** Per-turn streaming decision and time-to-detection measurement.

A logistic regression with C = 0.7 and balanced class weights, wrapped in isotonic calibration, produces $p_k$ from the 25-dimensional vector after each caller turn; a histogram gradient-boosting fusion (220 iterations, depth 4) is trained alongside it as the non-linear alternative. It is fitted on dev because it stacks on branch outputs; fitting it on train would feed it in-sample predictions that never occur at test time. The evidence panel is generated per turn from the sign and size of each branch's contribution: the entity spans found, the phrases that raised intent, the voice cues (detector score, jitter, shimmer, consistency), and the coercion step the call has reached, so that a warning explains itself rather than showing a number.

![](docs/figures/shot_analysis.png)

**Figure 6.** Console: call analysis of a cloned loan-approval scam. Left, the transcript with entity spans and dialogue acts; right, the verdict gauge, the plain-language reasons, the detected entities; below, exact Shapley contributions per feature and the prosody-against-lexical-pressure trace.

![](docs/figures/shot_monitor.png)

**Figure 7.** Console: the live monitor, which streams the risk band turn by turn. The corpus browser and the system view are shown in Appendix B.

```mermaid
flowchart LR
    subgraph voice [voice, 8]
        v1[as_score] --- v2[as_llr] --- v3[as_margin] --- v4[spk_consistency]
        v5[jitter] --- v6[shimmer] --- v7[hnr] --- v8[spec_flatness_var]
    end
    subgraph intent [intent, 9]
        i1[intent_score] --- i2[intent_max_turn] --- i3[urgency_density]
        i4[ent_otp] --- i5[ent_authority] --- i6[ent_threat]
        i7[ent_payment] --- i8[ent_personal] --- i9[ent_bank]
    end
    subgraph cross [cross, 8]
        c1[pim] --- c2[coercion_slope] --- c3[coercion_peak] --- c4[act_scam_llr]
        c5[cmi] --- c6[switch_entropy] --- c7[ent_lang_align] --- c8[callee_resist]
    end
    voice --> F[fusion]
    intent --> F
    cross --> F
```

**Figure 5.** Where each fusion feature comes from. The ablation arms of Chapter 7 are subsets of these groups.

## 5.8 Algorithm

```
Algorithm 1: Streaming risk for one call
Input:  turns (s_j, x_j, w_j), j = 1..J; alert threshold tau = 0.65
Output: p_k for each caller turn k; alert time t_alert or none

1  caller <- []; t_alert <- none
2  for j in 1..J:
3      if s_j != caller: continue
4      append (x_j, w_j) to caller
5      y <- condition(concat audio of caller)          # DC, band-pass, trim, dither
6      as_score, as_llr, as_margin <- antispoof(y)
7      spk_consistency, jitter, shimmer, hnr, flatness_var <- voice_stats(caller)
8      bio_j <- CRF.tag(w_j); intent_j <- TFIDF.score(w_j)
9      intent_score <- 0.6 * max_i intent_i + 0.4 * mean_i intent_i
10     entity densities <- BIO spans by type / turns so far
11     acts <- ActHMM.decode(caller); act_scam_llr <- log p(acts|scam) - log p(acts|benign)
12     slope, peak <- coercion_trajectory(acts)
13     ell_i <- lexical_arousal(w_i); a_i <- acoustic_arousal(x_i) for i in caller
14     pim <- Equation 3 over (ell, a)
15     cmi, switch_entropy, ent_lang_align <- language_features(caller)
16     callee_resist <- resisting callee turns / callee turns so far
17     f <- [voice(8), intent(9), cross(8)]
18     p_k <- calibrate(sigmoid(w . f + b))
19     if p_k >= tau and t_alert is none: t_alert <- t_end(j)
20     emit p_k, band(p_k), reasons(f)
21 return p_K, t_alert
```

## 5.9 Implementation details

- Python; NumPy and SciPy for signal processing; scikit-learn for logistic regression, isotonic calibration, GMM and gradient boosting. The CRF, cepstral front ends, pitch tracker, syllable segmenter and act HMM are project code.
- Framing 25 ms / 10 ms, Hamming, 512-point FFT, pre-emphasis 0.97, 8 kHz. Filterbank 40 filters, 300 to 2800 Hz, 20 coefficients, lifter 22, deltas of width 2, CMVN.
- Voice back ends: GMM up to 64 components, 120 EM iterations; boosting 220 iterations at 0.07, depth 4; RawNet-style network with 20 SincConv filters, residual blocks and a GRU, trained with binary cross-entropy; TDNN speaker embedder for segment consistency. The reported run scores with the boosted back end.
- Text: Whisper small, language Hindi; TF-IDF word (1,2) and char_wb (3,5), sublinear TF; logistic regression C = 4.0, balanced. CRF L2 = 0.5, 200 iterations; BiLSTM-CRF with the same feature set as the neural counterpart. The reported entity figures are the CRF's.
- Fusion: logistic regression C = 0.7, balanced, isotonic calibration, fitted on 120 dev calls; gradient-boosting alternative trained on the same rows. Alert 0.65; bands 0.35, 0.65, 0.85.
- Prosody anchors at the 10th and 90th percentile of train audio, recalibrated every training run.
- Runtime on the laptop CPU: 1.7 s for a 45 s call with the text branches, 18.8 s for a 74 s call with all four, of which the voice branch takes 3.5 s.
- Reproducibility: seed 20230100; the corpus manifest records the grammar fingerprint and every parameter, the pair manifest a build version and the audit; the runner rebuilds a stage whenever its inputs change.

---

# Chapter 6. Experimental Framework

## 6.1 Research questions

- **RQ1.** Does fusing the four branches improve on the best single branch and on late fusion?
- **RQ2.** Which cells of the voice-by-content grid does each arm get wrong?
- **RQ3.** On calls written to be hard, does anything beyond text help?
- **RQ4.** Can the voice branch separate real from cloned Hindi speech when the channel is matched?
- **RQ5.** Does the text branch generalise to real fraud recordings from another language and distribution?
- **RQ6.** How early is a scam flagged, and does codec degradation change the decision?

## 6.2 Datasets

**Table 3.** Corpus composition.

| Item | Generated Hinglish corpus | Real Hindi pair set | Real robocall set |
|---|---|---|---|
| Source | Project template grammar, Microsoft neural voices | GramVaani dev and eval, paired with edge-tts renderings | Robocall audio dataset |
| Samples | 480 calls, 5,093 turns, 7.3 h, 1,289 entity spans | 600 pairs, 1,200 clips, mean 9.3 s | 1,413 recordings |
| Classes | 240 scam / 240 benign; 240 cloned / 240 human; 120 per cell | 600 bona fide, 600 spoof (half plain TTS, half vocoded) | 1,413 fraud; 60 project benign calls as controls |
| Acquisition | 8 kHz; 18 scenarios; 24 speakers; mean 54.6 s | 8 kHz telephone MP3; renderings at 24 kHz, downsampled | Field recordings, English and Chinese |
| Labels | Exact from slots; 7 entity types; 18 acts; per-token language | Bona fide vs spoof by construction | Fraud by provenance; no benign half |
| Splits | 240 / 120 / 120, speaker-disjoint (12 / 6 / 6 speakers) | 890 / 310 clips, speaker-disjoint, pairs kept together | Test only |
| Imbalance | None | None | Positive-only |
| Augmentation | Codec degradation at evaluation only | Channel matching at build; codec at evaluation | None |
| Exclusions | None | Lines under 0.5 s; none lost of 600 | Recordings without a transcript |

Entity spans by type: bank entity 364, threat or deadline 345, OTP 172, payment handle 106, money amount 105, authority claim 103, personal-information request 94. Hard negatives, benign calls written to sound like their scam counterpart, number 164; lexically mild scams 52. Code-mixing index is 0.218 for scam and 0.201 for benign, so the language mix does not separate the classes on its own.

## 6.3 Arms and baselines

Four arms are fitted on the same 120 dev calls and scored on the same 120 test calls: audio-only (8 voice features), text-only (9 intent features), late fusion (one calibrated score per branch, then a logistic layer), and full (all 25). A lexicon rule baseline is reported for intent; five front ends by two back ends for the voice branch. All are project implementations under one protocol.

## 6.4 Metrics

**Table 4.** Performance metrics.

| Task | Metrics | Reporting point |
|---|---|---|
| Call classification | AUC, F1, accuracy, EER; per-cell accuracy; precision and recall at 0.65 | Per cell, because the aggregate hides the cells that matter |
| Anti-spoofing | EER and minimum t-DCF per front end, back end and codec; audit AUC per statistic | Read next to the audit |
| Entity recognition | Precision, recall, F1, exact span, per type | Reference transcripts, to isolate the tagger from recognition error |
| Intent | AUC, F1, confusion at 0.65 | Rule baseline against TF-IDF |
| Calibration | Brier, ECE | Test split |
| Generalisation | Recall at 0.65, cross-domain AUC, false-alarm rate on controls | Controls are project benign calls |
| Time to detection | Median, p90, turns | Threshold 0.65 |
| Robustness | AUC of full and audio-only arms per codec | Real codec flag recorded |

## 6.5 Protocol against leakage

Two checks run before any model is trained. A pure-Python bag-of-words classifier is trained across speaker folds on caller text, and the grammar is revised until it cannot separate the classes on authorship. And the confound audit runs on every pair build. Both are part of the pipeline rather than one-off checks.

---

# Chapter 7. Results and Discussion

## 7.1 Primary result

**Table 5.** Ablation of the four arms on the test split (120 calls, fitted on dev).

| Arm | Voice | Intent | Cross | AUC | F1 | Accuracy | EER |
|---|---|---|---|---|---|---|---|
| Audio only | Yes | No | No | 0.616 | 0.187 | 0.492 | 0.400 |
| Text only | No | Yes | No | 0.999 | 0.983 | 0.983 | 0.033 |
| Late fusion | scores | scores | scores | 0.999 | 0.974 | 0.975 | 0.033 |
| Full | Yes | Yes | Yes | 1.000 | 0.983 | 0.983 | 0.017 |

The full arm reaches the highest AUC and halves the equal error rate of the text-only arm, from 0.033 to 0.017. The audio-only arm sits near chance by design: the corpus draws the voice label independently of the scam label, so a branch that reads only the voice cannot predict the content. Its role is as a modifier inside the fused score, not as a detector on its own.

![](docs/figures/shot_ablation.png)

**Figure 8.** Results lab: the headline tiles and the ablation as displayed in the console, read from `ablation_results.json`.

![](docs/figures/chart_ablation.png)

**Figure 9.** Ablation of the four arms on the test split, AUC and F1.

## 7.2 Leak check

After the grammar rules of Section 5.3, the pre-training bag-of-words check gives AUC 0.990 on the whole corpus and 0.947 on the hard subset. What remains separable is the phenomenon itself: a scam call threatens and asks for a code, and a bag of words finds that.

## 7.3 Per-cell behaviour and the hard subset

**Table 6.** Per-cell accuracy at threshold 0.65 (30 calls per cell).

| Arm | Human, benign | Human, scam | Cloned, benign | Cloned, scam |
|---|---|---|---|---|
| Audio only | 0.933 | 0.100 | 0.800 | 0.133 |
| Text only | 1.000 | 1.000 | 1.000 | 0.933 |
| Late fusion | 1.000 | 1.000 | 1.000 | 0.900 |
| Full | 1.000 | 1.000 | 1.000 | 0.933 |

The full arm is perfect on three cells and misses two of thirty cloned scams, both in the lexically mild style. The audio-only row classifies nearly every call as benign, which is what independence of voice and content looks like.

![](docs/figures/chart_percell.png)

**Figure 10.** Per-cell accuracy of each arm at threshold 0.65.

**Table 7.** Arm comparison on the hard subset: 13 lexically mild scams against 41 hard negatives.

| Arm | AUC | F1 | Recall | Precision |
|---|---|---|---|---|
| Audio only | 0.362 | 0.000 | 0.000 | 0.000 |
| Text only | 0.993 | 0.917 | 0.846 | 1.000 |
| Late fusion | 0.993 | 0.917 | 0.846 | 1.000 |
| Full | 0.998 | 0.917 | 0.846 | 1.000 |

This is where the arms have room to differ, and the full arm ranks the hard cases best, 0.998 against 0.993 for text alone, with no false alarms on the 41 benign calls written to sound like scams.

![](docs/figures/chart_hard.png)

**Figure 11.** Arm comparison on the hard subset.

## 7.4 Intent and entities

The TF-IDF intent model reaches AUC 1.000 and F1 0.976, against AUC 0.913 and recall 0.483 for the lexicon rules: the difference between counting fraud words and learning which combinations, in which spellings, mark a request. Three benign calls score above 0.65 on intent alone; the fused system, which also sees the coercion trajectory and the callee's behaviour, clears all three.

**Table 8.** Entity recognition by type, CRF, exact span match, test split (324 spans).

| Type | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Bank entity | 90 | 0.989 | 1.000 | 0.995 |
| Threat or deadline | 83 | 0.976 | 0.964 | 0.970 |
| OTP | 45 | 1.000 | 1.000 | 1.000 |
| Payment handle | 29 | 1.000 | 1.000 | 1.000 |
| Money amount | 27 | 1.000 | 1.000 | 1.000 |
| Authority claim | 27 | 0.917 | 0.815 | 0.863 |
| Personal-information request | 23 | 1.000 | 0.739 | 0.850 |
| **All** | **324** | | | **0.970** |

The types with a fixed surface form are recognised almost perfectly. The two with the most varied wording lose recall rather than precision: the tagger declines to tag what it has not seen, the safer failure for a system that raises alarms.

![](docs/figures/shot_ner_intent.png)

**Figure 12.** Results lab: entity recognition and intent classification as displayed in the console.

![](docs/figures/chart_ner.png)

**Figure 13.** Entity recognition F1 by type, with support.

![](docs/figures/chart_intent.png)

**Figure 14.** Scam intent: rule baseline against the TF-IDF model on the test split.

## 7.5 Calibration

Brier score 0.011 and ECE 0.015 on the test split. Fifty-one calls fall in the lowest bin (mean 0.003, no scams) and fifty-six in the highest (mean 0.997, all scams); the nine near 0.11 contain one scam, an observed rate that matches the prediction.

![](docs/figures/shot_calibration.png)

**Figure 15.** Results lab: calibration curve and provenance as displayed in the console.

![](docs/figures/chart_calibration.png)

**Figure 16.** Reliability diagram of the fused probability on the test split.

## 7.6 The voice branch on real speech

**Table 9.** Anti-spoofing EER on the real Hindi pair set, 310 test clips.

| Front end | GMM clean | GMM G.711u | GMM G.711a | GMM GSM | GBM clean | GBM G.711u | GBM G.711a | GBM GSM |
|---|---|---|---|---|---|---|---|---|
| LFCC | 0.000 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.000 | 0.032 |
| GFCC | 0.006 | 0.006 | 0.006 | 0.019 | 0.006 | 0.006 | 0.006 | 0.019 |
| MFCC | 0.006 | 0.006 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.026 |
| CQCC | 0.013 | 0.013 | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 |
| LPCC | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.006 |

Every front end and back end separates real from cloned Hindi speech at an EER under 0.02 on clean and G.711 audio and under 0.04 under GSM, which at 13 kbit/s is the only condition that removes enough detail to cost anything; minimum t-DCF, which penalises the rarer error more heavily, stays at or below 0.019 for every front end with the GMM back end on clean and G.711 audio and reaches 0.058 under GSM, while the boosted back end pays more under GSM (up to 0.23 for LFCC and GFCC). LFCC and LPCC with the GMM back end, the classical pairings, are the most stable across codecs.

![](docs/figures/shot_antispoof.png)

**Figure 17.** Results lab: anti-spoofing EER by front end and back end, and DET curves, as displayed in the console.

![](docs/figures/chart_antispoof.png)

**Figure 18.** Anti-spoofing EER in percent by front end, back end and codec on the 310 real Hindi test clips.

**Table 10.** Confound audit of the pair set: AUC of each channel statistic alone.

| Statistic | Raw files | After conditioning |
|---|---|---|
| Noise floor (dB) | 0.508 | 0.510 |
| Dynamic range proxy (dB) | 0.604 | 0.604 |
| Zero-crossing rate | 0.551 | 0.540 |
| Duration | 0.549 | 0.549 |
| Quiet-frame spectral centroid | 0.519 | 0.514 |
| Quiet-frame spectral tilt | 0.553 | 0.548 |
| Energy share above band | 0.629 | 0.519 |
| Energy share below band | 0.723 | 0.704 |
| Trailing quiet (s) | 0.502 | 0.503 |
| DC offset | 0.617 | 0.663 |
| **Worst** | **0.723** | **0.704** |

The audit is what makes Table 9 readable. It was designed around an early build in which bandwidth, trailing silence and DC offset each separated the classes on their own; the conditioning of Section 5.4 and the matching of Section 4.2 bring all ten statistics to within the audit's threshold, with noise floor, colour, trailing silence and in-band bandwidth at chance. The residual below 300 Hz lies outside the filterbank, and dynamic range reflects spontaneous against read speech rather than the channel. The error rates in Table 9 are therefore measured on the voice.

![](docs/figures/chart_audit.png)

**Figure 19.** Confound audit of the pair set: AUC of each statistic alone, on the raw files and after conditioning, against the audit's stop line.

## 7.7 Generalisation to real fraud calls

**Table 11.** Out-of-domain recall on 1,413 real robocall recordings, models trained only on the corpus.

| Model | Recall at 0.65 | Cross-domain AUC | False alarms on 60 benign controls |
|---|---|---|---|
| Lexicon rules | 0.069 | 0.712 | 0.033 |
| TF-IDF intent | **0.787** | **0.946** | **0.050** |

The intent model, trained only on generated Hinglish calls, recalls nearly four in five real English robocalls at a 5 percent false-alarm rate. The transfer rests on shared fraud vocabulary and on the character n-grams, and it improved directly with the corpus rules of Section 5.3: on the corpus before those rules the same model recalled 0.699 at AUC 0.896 with a 15 percent false-alarm rate. Removing authorship shortcuts made the classifier learn what transfers.

![](docs/figures/chart_robocall.png)

**Figure 20.** Out-of-domain recall, cross-domain AUC and false-alarm rate on 1,413 real robocalls.

## 7.8 Time to detection and channel robustness

**Table 12.** Time to detection on the 60 test scam calls, threshold 0.65.

| Measure | Value |
|---|---|
| Calls flagged | 57 of 60 |
| Median time to alert | 35.8 s |
| 90th percentile | 49.4 s |
| Median turns to alert | 6 |
| Fastest scenario | Digital arrest, 17.4 s |
| Slowest scenario | Lottery prize, 43.8 s |

Six turns is before the sensitive request in every flagged call. Digital arrest is fastest because its script asserts authority and threatens in its first caller turns; lottery and fake-relative scripts spend longer on setup. The three calls not flagged never cross 0.65 at any turn.

**Table 13.** Channel robustness: AUC on the test split under each codec.

| Condition | Full arm | Audio-only arm |
|---|---|---|
| Clean | 0.9997 | 0.616 |
| G.711 mu-law | 0.9997 | 0.640 |
| G.711 A-law | 0.9997 | 0.622 |
| GSM 06.10 | 0.9997 | 0.559 |

The fused decision is unchanged under every codec. The audio-only column shows what the channel does to the voice branch on its own: GSM costs it about 0.06 AUC, which the fusion absorbs.

![](docs/figures/shot_robust_ttd.png)

**Figure 21.** Results lab: channel robustness and time to detection as displayed in the console.

![](docs/figures/chart_ttd.png)

**Figure 22.** Time to detection: distribution over the 57 flagged scam calls, and median by scenario.

![](docs/figures/chart_robustness.png)

**Figure 23.** Channel robustness of the full and audio-only arms under each codec.

## 7.9 Scope of the results

The ablation numbers characterise scripted calls, where transcript content carries most of the decision; the hard subset is where the branches differ and the full arm leads there. The voice branch's error rates are those of the boosted and mixture back ends on a matched pair set whose spoofs come from the commercially available Hindi neural voices; the end-to-end back end and evaluation against a larger set of synthesis systems are the next step. Entity and intent figures are measured on reference transcripts, which isolates the taggers from recognition error; the recogniser is scored separately by word error rate against the same reference.

---

# Chapter 8. Conclusion and Future Work

## 8.1 Conclusion

The problem was to decide, during a code-mixed Hindi-English call, whether the caller intends fraud when the voice may be cloned and the request phrased to sound routine. SwarKavach answers it with four branches fused into one calibrated per-turn risk score, built on a corpus and an evaluation protocol designed so that the numbers measure the voice and the intent rather than the recording conditions.

On a 480-call corpus balanced over voice and content the full system reaches AUC 1.000, F1 0.983 and EER 0.017, halving the error of the best single branch; it is calibrated to a Brier score of 0.011 and flags 57 of 60 scams at a median of 35.8 seconds, six turns, before the sensitive request. On 1,413 real robocalls in another language the text branch recalls 78.7 percent at 5 percent false alarms. The voice branch, trained on 600 real Hindi recordings paired with cloned renderings and matched on ten channel statistics, separates them at an EER under 0.02 on clean and G.711 audio and under 0.04 under GSM.

The technical implication is that the four signals are complementary in the way the design assumed: text carries the decision on scripted content, the voice branch is decisive on authenticity, and the structure and cross-modal branches are what separate the hard cases. The principal boundary of the claims is the spoof side of the voice branch, which covers the synthesis systems available for Hindi at the time of the work.

## 8.2 Future work

1. **Between-turn arousal variance**, reading how much the caller's arousal changes across turns relative to how much the content's pressure changes, which is how a cloned caller's flatness actually presents.
2. **Evaluate the voice branch on IndicSynth**, 12 languages and 989 speakers, with the same conditioning and audit, for an EER against many synthesis systems.
3. **Report the recognised-transcript rows**: entity F1 and intent on Whisper output, with word error rate against the reference.
4. **A benign half for the robocall evaluation**, so that the out-of-domain AUC is measured against negatives from the same channel.
5. **Incremental feature accumulation**, to bring the per-turn cost of a full analysis under one second on a CPU.
6. **A code-mixed transformer for intent** (MuRIL or HingBERT) fine-tuned on the corpus, compared against the TF-IDF model under the same protocol.

---

---

# References

Ma, Z., Wang, P., Huang, M., Wang, J., Wu, K., Lv, X., Pang, Y., Yang, Y., Tang, W., & Kang, Y. (2025). *TeleAntiFraud-28k: An audio-text slow-thinking dataset for telecom fraud detection*. arXiv. https://arxiv.org/abs/2503.24115

Sharma, D. V., Ekbote, V., & Gupta, A. (2025). IndicSynth: A large-scale multilingual synthetic speech dataset for low-resource Indian languages. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics. https://aclanthology.org/2025.acl-long.1070/

Shen, Z., Yan, S., Zhang, Y., Luo, X., Ngai, G., & Fu, E. Y. (2025). *"It warned me just at the right moment": Exploring LLM-based real-time detection of phone scams*. arXiv. https://arxiv.org/abs/2502.03964

Wang, X., Delgado, H., Tak, H., Jung, J., Shim, H., Todisco, M., Kukanov, I., Liu, X., Sahidullah, M., Kinnunen, T., Evans, N., Lee, K. A., & Yamagishi, J. (2024). ASVspoof 5: Crowdsourced speech data, deepfakes, and adversarial attacks at scale. In *Proceedings of the ASVspoof Workshop 2024*. ISCA. https://arxiv.org/abs/2408.08739

Yi, J., Wang, C., Tao, J., Zhang, X., Zhang, C. Y., & Zhao, Y. (2023). *Audio deepfake detection: A survey*. arXiv. https://arxiv.org/abs/2308.14970

Zhu, Y., Koppisetti, S., Tran, T., & Bharaj, G. (2024). SLIM: Style-linguistics mismatch model for generalized audio deepfake detection. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://arxiv.org/abs/2407.18517

---

# Appendix B. Console views

![](docs/figures/shot_corpus.png)

**Figure B1.** The corpus browser: every call with its cell, split, scenario and duration.

![](docs/figures/shot_system.png)

**Figure B2.** The system view: loaded models, back ends and thresholds in force.
