TITLE: Detecting Voice-Cloned Fraud Calls in Hinglish by Fusing Anti-Spoofing, Scam-Intent and Prosody-Intent Mismatch Cues

AUTHOR: Bhavye Nijhawan, 23BAI0100, School of Computer Science and Engineering, Vellore Institute of Technology, BCSE419L Speech and Language Processing Laboratory

ABSTRACT: Phone fraud in India is moving from scripts read by human callers to scripts read by cloned voices, and the two defences in use each cover half the problem: synthetic-speech detectors decide whether a voice is machine-made but not whether the caller is asking for a one-time password, while transcript classifiers catch the request but hear a cloned voice and a real one as the same. This work answers both questions on the same call, in the Hindi-English code-mix in which such calls are spoken, and updates its answer after every turn. SwarKavach fuses four branches. A voice branch scores synthetic speech with five cepstral front ends, a Gaussian-mixture and a gradient-boosted back end, a RawNet-style end-to-end detector and a TDNN speaker embedder, trained on 600 real Hindi field recordings paired with cloned renderings of the same transcripts, with the recording channel matched on level, spectral colour and bandwidth and audited by ten statistics before any error rate is reported. A text branch transcribes with Whisper, tags seven fraud entity types with a linear-chain conditional random field and scores intent with a calibrated TF-IDF classifier. A structure branch reads the dialogue-act sequence as a coercion ramp. A cross-modal branch measures the mismatch between lexical pressure and acoustic arousal. A calibrated fusion layer fitted on held-out speakers turns twenty-five features into a risk band and an evidence panel. On a 480-call corpus balanced over voice and content the full system reaches AUC 1.000 and an equal error rate of 0.017, halving the error of the best single branch, and flags 57 of 60 scam calls at a median of 35.8 seconds, six turns, before the sensitive request. On 1,413 real robocall recordings from another language the text branch recalls 78.7 percent at a 5 percent false-alarm rate. The voice branch separates real from cloned Hindi speech at an equal error rate below 0.02 on clean and G.711 audio and below 0.04 under GSM.

INDEX: voice cloning detection, telephone fraud, code-mixed speech, anti-spoofing, dialogue-act coercion, prosody-intent mismatch, calibrated late fusion, time to detection

## I. Introduction

### A. Context and Problem

A fraud call is a short piece of theatre with a fixed script. The caller claims to be from a bank, a courier, the police or a relative in trouble, states a problem, attaches a deadline, tells the listener not to consult anyone, and only then asks for a one-time password, a card number or a transfer. The pattern is widely publicised and still works, because the call is engineered to remove the interval in which a listener would recall it. Two developments have changed its scale. Automated dialling reaches thousands of numbers a day, and neural text-to-speech now produces Hindi that a listener on a band-limited, codec-compressed line cannot reliably distinguish from a person, so the script no longer requires a fluent, composed, present human reader.

Three properties of these calls defeat existing tools together. They are conversational: a request for a one-time password is suspicious only because of the threat preceding it and the instruction not to disconnect that follows. They are code-mixed: *"Aapka account block ho jayega, OTP bata dijiye"* is neither Hindi nor English, and tools built for one mishandle the other half. And the voice may be cloned, which changes what identical words mean, since a bank reminder read by a machine is routine while a request for a card PIN read by a machine is almost certainly fraud.

### B. Research Gap

Caller-side blocklists act before the call and are defeated by number spoofing. Transcript classifiers read what was said during the call but are blind to the voice. Anti-spoofing detectors decide whether speech was synthesised, but on isolated utterances rather than conversations, and degrade when the synthesiser, language or channel changes. The gap is therefore specific rather than general. The two families are built and evaluated apart, so neither observes the operationally decisive cases, a benign call in a cloned voice and a scam delivered by a human. Neither is evaluated on code-mixed Indian telephone conversation. And evaluation sets in common use are not audited for channel or authorship shortcuts, so a detector can score well on recording conditions rather than on the voice or the intent.

### C. Objectives and Contributions

The study sets out to (i) build a corpus of Hindi-English fraud and benign calls in which voice authenticity and scam content vary independently, with annotations exact by construction; (ii) design a detector fusing a synthetic-voice branch, a transcript branch, a dialogue-structure branch and a prosody-intent branch into one calibrated per-turn score; (iii) quantify each branch through a four-arm ablation reported per evaluation cell and on a deliberately hard subset; (iv) evaluate the voice branch on real Hindi recordings paired with cloned renderings, with the channel audited first; and (v) validate generalisation on real fraud recordings outside the training distribution and measure detection latency against the sensitive request.

The contributions are: 1) a template grammar whose typed slots emit gold entity, dialogue-act and per-token language labels with the text, with five subjects paired across the label so that topic does not predict class; 2) a prosody-intent mismatch feature and a coercion-trajectory feature, neither of which a single-modality detector can compute; 3) a construction for matched anti-spoofing pair sets from field recordings, with a ten-statistic confound audit that halts the build when a channel statistic alone separates the classes; 4) a four-arm ablation reported per cell and on the hard subset, preceded by a bag-of-words leak check that runs before any model is trained; and 5) out-of-domain validation on 1,413 real robocalls, a per-turn time-to-detection measurement and a telephone codec sweep.

## II. Literature Review

### A. Benchmarks and the Generalisation Problem

The ASVspoof series defines evaluation practice for anti-spoofing. Its fifth edition (Wang et al., 2024) crowdsources both the bona fide speech, from many speakers in uncontrolled acoustic conditions, and the attacks, which for the first time include adversarial perturbations targeting the detectors, and adds metrics for spoofing-robust speaker verification alongside stand-alone detection. The design responds to a finding that recurs across the field: systems reaching near-zero equal error rate on one round degrade on the next, because they learn the artefacts of particular synthesisers and recording conditions rather than anything general about synthetic speech. Yi et al. (2023) document that drop in a unified comparison of feature front ends and classifiers across ASVspoof 2021, ADD 2023 and In-the-Wild. Two observations shape the present design: hand-crafted cepstral front ends remain competitive with learned representations when the test channel differs from training, and the datasets in use are overwhelmingly English and Chinese.

### B. Low-Resource Languages and Cross-Modal Cues

IndicSynth (Sharma, Ekbote and Gupta, 2025) supplies roughly 4,000 hours of synthetic speech from 989 target speakers across twelve Indian languages including Hindi, with a mimicry subset in which a target speaker's voice is cloned. It establishes that Hindi voice cloning is a solved generation problem at scale and provides the resource against which a Hindi anti-spoofing branch should ultimately be tested. Every sample is an isolated utterance, with no conversation and no fraud content.

SLIM (Zhu et al., 2024) observes that in genuine speech the style of delivery and the linguistic content are mutually dependent, because one speaker selected both, and that synthesis breaks that dependency because content is supplied and style imposed. A representation of the dependency, pretrained on real speech alone, improves out-of-domain detection and yields an explainable score. The prosody-intent feature used here applies the same intuition to a narrower and task-specific pair of quantities: whether the pressure carried by the words matches the arousal carried by the voice. It requires a fraud lexicon, a pitch tracker and a syllable-level energy measure rather than a self-supervised encoder, and executes on a laptop CPU.

### C. Conversation-Level Detection

Shen et al. (2025) frame the problem as this work does: defences act before or after the call while the call itself is unprotected. A large language model classifies each utterance as fraudulent, safe or uncertain and warns the user during the conversation. The system reads text alone, so a cloned voice reading a benign script is invisible to it, and a model call per utterance constrains deployment on a handset. TeleAntiFraud-28k (Ma et al., 2025) is the first open audio-and-text resource for telecom fraud, comprising 28,511 speech-text pairs built from transcribed real calls, regenerated through text-to-speech for privacy and extended by model-based and multi-agent generation, annotated for fraud reasoning. Because every clip is regenerated, voice authenticity is not learnable from it, and the language is Chinese.

TABLE_I_WIDE_CAPTION: Critical Comparison of Recent Literature

| Study | Method | Data | Main strength | Important limitation |
|---|---|---|---|---|
| Wang et al. (2024) | Benchmark; crowdsourced speech, adversarial attacks | Multi-speaker English, many TTS and VC systems | Uncontrolled conditions at scale | Utterance level; English; no conversation |
| Yi et al. (2023) | Unified comparison of front ends and classifiers | ASVspoof 2021, ADD 2023, In-the-Wild | Isolates the out-of-domain drop by family | Descriptive; English and Chinese only |
| Sharma et al. (2025) | Large-scale synthetic speech for detection | 4,000 h, 989 speakers, 12 Indian languages | First Hindi-scale cloning resource | Utterances only; no fraud content |
| Zhu et al. (2024) | Self-supervised style-linguistics mismatch | In- and out-of-domain deepfake sets | Explainable cross-modal cue | Needs large real-speech pretraining |
| Shen et al. (2025) | LLM classifies each utterance in-call | Scam conversations; user study | Real-time warning | Text only; per-utterance model cost |
| Ma et al. (2025) | Audio-text dataset with reasoning labels | 28,511 pairs, Chinese, TTS-regenerated | First open audio-text fraud resource | Voice authenticity not learnable |

### D. Position of This Work

Anti-spoofing research is evaluated on utterances and therefore never observes the context that determines whether a synthetic voice matters. Scam-detection research is evaluated on text and therefore never hears the voice. The present system reads voice, words, structure and their mismatch on the same call, in Hindi-English code-mix, and treats the construction and auditing of its own evaluation data as part of the method.

## III. Preliminary Concepts

Every voice-branch classifier reads cepstral coefficients. The signal is framed at 25 ms with a 10 ms hop and a Hamming window, pre-emphasised at 0.97, and each frame's magnitude spectrum passes through 40 triangular filters between 300 and 2800 Hz, log-compressed and decorrelated,

$$c_n = \sum_{m=1}^{M} \log(E_m) \cos\left[\frac{\pi n}{M}\left(m - \tfrac{1}{2}\right)\right]$$ (1)

where $E_m$ is the energy in filter $m$, $M=40$, and $N=20$ coefficients are retained. LFCC, MFCC, GFCC, CQCC and LPCC differ only in the filterbank. Deltas are appended and cepstral mean and variance normalisation removes the additive effect of a fixed channel. One Gaussian mixture is fitted to bona fide frames and one to spoofed frames, and an utterance is scored by the mean log-likelihood ratio over its $T$ frames,

$$\text{LLR}(X) = \frac{1}{T}\sum_{t=1}^{T}\left[\log p(x_t \mid \lambda_{\text{spoof}}) - \log p(x_t \mid \lambda_{\text{bona}})\right]$$ (2)

with up to 64 components per mixture. Four measures describe delivery per turn: pitch coefficient of variation, pitch span relative to the mean, the variance of syllable-peak levels in dB, and the standard deviation of syllable rate; each is scaled between anchors at the 10th and 90th percentile of training audio and combined with weights 0.34, 0.24, 0.26 and 0.16 into an acoustic arousal $a \in [0,1]$. With lexical arousal $\ell_i$ over the $n$ caller turns, the prosody-intent mismatch is

$$\text{PIM} = 0.65 \cdot \frac{\sum_i w_i \max(0, \ell_i - a_i)}{\sum_i w_i} + 0.35 \cdot \frac{1 - \rho(\ell, a)}{2} \cdot \max_i \ell_i$$ (3)

where $w_i$ weights turns by lexical pressure and $\rho$ is the Pearson correlation. The first term rewards turns whose words press while the voice does not; the second rewards a delivery that fails to track the content, gated by peak pressure so that a call with no coercive language cannot score on correlation noise. Each dialogue act carries a coercion rank in $[0,1]$, from greeting at 0.0 through threat at 0.8, deadline at 0.85, isolation at 0.9 and request for sensitive information at 0.95 to escalation at 1.0; the least-squares slope of rank against caller-turn index is the coercion slope and its maximum the peak. Detection is reported as AUC, equal error rate, $F_1$, minimum tandem detection cost, Brier score and expected calibration error. Time to detection is the end time of the first caller turn at which fused risk reaches 0.65.

## IV. Materials and Methods

Training and evaluation run on a Google Colab runtime under Python 3.13; the reported models train on CPU. The console runs on a Windows laptop with 7.9 GB of RAM under Python 3.11.9. The numerical stack is NumPy 2.4.3, SciPy 1.17.1, scikit-learn 1.9.0 and PyTorch 2.11.0 CPU. Speech is synthesised with edge-tts 7.2.8 using Microsoft neural voices, transcribed with OpenAI Whisper (small, multilingual, language Hindi), and degraded with FFmpeg for GSM 06.10 and NumPy implementations of G.711 mu-law and A-law. Real speech comes from the GramVaani Hindi ASR corpus at 8 kHz, and real fraud recordings from a public robocall dataset of 1,413 items with transcripts. All audio is processed at 8 kHz.

A template grammar produces 480 calls. Each call is planned, meaning its subject, label, voice, speaker and split are fixed, before it is rendered, so that the four voice-by-content cells are balanced by construction; every entity, act and language tag is emitted by the slot that produced the token; and speakers are assigned to splits before calls are assigned to speakers, which makes the splits speaker-disjoint. Each turn is synthesised with rate, pitch and volume set from its dialogue act, and a caller labelled as cloned keeps a fixed contour whatever the act while a human caller modulates.

The voice branch cannot train on that corpus, because both of its voice cells are machine voices. It trains instead on 600 real GramVaani utterances paired with neural renderings of their own transcripts, half additionally vocoded. Both sides of each pair pass through the same channel in a fixed order: G.711 companding, DC removal, a brick-wall band-pass to 300 to 2800 Hz, silence trimming, peak normalisation, and noise matching in which the quieter side is lifted to the louder side's floor using noise shaped like the recording's own quiet-frame spectrum. Before the set is accepted, ten statistics are computed per clip on the raw file and after the same conditioning the detector applies, and the build halts if any conditioned statistic alone exceeds AUC 0.75.

Entity taggers, intent and act models fit on the train split, prosody anchors are recalibrated on train audio, anti-spoofing back ends fit on the train partition of the pair set, and the fusion layer and every ablation arm fit on the dev split, so that fusion never observes in-sample branch outputs. Every reported number comes from the test split or from outside the training distribution.

## V. Proposed Methodology

### A. Architecture

FIGURE: docs/figures/fig1.png | End-to-end information flow. Every branch consumes the same turn segmentation, and the fusion layer sees only the twenty-five named features.

Input is a call of $J$ turns, each with a speaker, audio and transcript tokens. Output after each caller turn $k$ is a probability $p_k$, a band among low, elevated, high and critical with edges at 0.35, 0.65 and 0.85, the entity spans found so far, and the reasons behind the score. The call-level decision is $p_J \ge 0.65$.

### B. Branches

Every cepstral computation begins with one conditioning function: DC removal, a Kaiser-windowed FIR band-pass from 300 to 2800 Hz, trimming of frames more than 40 dB below the loudest, and dither 70 dB below the signal. Three back ends sit behind one scoring interface, preferred in the order end-to-end, boosted, mixture: a RawNet-style network reading the waveform through a SincConv layer of learned band-passes; a histogram gradient-boosted classifier over 480 pooled moments; and the mixture pair of (2). Jitter, shimmer and harmonic-to-noise ratio are computed on the raw signal, and a TDNN speaker embedder scores each segment so that inter-segment cosine similarity becomes a consistency feature, informative at both ends since a person drifts slightly, a single synthetic voice not at all, and a spliced call a great deal. Eight features leave the branch.

The text branch transcribes with Whisper and aligns segments to turns. Two taggers share a tokeniser, label space and scoring procedure: a project-implemented linear-chain conditional random field, which runs inside the per-turn budget on a CPU and can use the language tag as a feature, and a BiLSTM with a CRF output layer. Both tag seven entity types in BIO form. A TF-IDF representation over word unigrams and bigrams together with character 3 to 5 grams, which is what survives romanised Hindi spelling variation, feeds a class-balanced logistic regression, and per-turn probabilities pool as 0.6 times the maximum plus 0.4 times the mean. Nine features leave the branch.

A hidden Markov model over acts, fitted separately on scam and benign calls, contributes the log-likelihood ratio of the observed sequence; coercion slope and peak, the prosody-intent mismatch of (3), the code-mixing index, switch entropy, entity-language alignment and callee resistance complete the remaining eight features.

### C. Fusion and Streaming Decision

A logistic regression with C = 0.7 and balanced class weights, wrapped in isotonic calibration, produces $p_k$ from the twenty-five-dimensional vector after each caller turn, with a histogram gradient-boosting fusion trained alongside as a non-linear alternative. It is fitted on dev because it stacks on branch outputs, and fitting on train would supply in-sample predictions that never occur at test time. The evidence panel is generated per turn from the sign and magnitude of each branch's contribution: entity spans, the phrases raising intent, voice cues, and the coercion step reached.

FIGURE: docs/figures/shot_analysis.png | Console: analysis of a cloned loan-approval scam, showing the transcript with entity spans, the verdict and plain-language reasons, exact Shapley contributions per feature, and the prosody-against-lexical-pressure trace.

## VI. Experimental Framework

Six questions are fixed before results. Does fusion improve on the best single branch and on late fusion? Which cells of the voice-by-content grid does each arm get wrong? On calls written to be hard, does anything beyond text help? Can the voice branch separate real from cloned Hindi speech once the channel is matched? Does the text branch generalise to real fraud recordings in another language? How early is a scam flagged, and does codec degradation change the decision?

The generated corpus holds 480 calls, 5,093 turns, 7.3 hours of audio and 1,289 entity spans, split 240/120/120 across 24 speakers, with 120 calls in each of the four cells; 164 benign calls are written to resemble their scam counterpart and 52 scams are lexically mild. The pair set holds 600 pairs, 1,200 clips, split 890/310 with pairs kept together and the split speaker-disjoint on the bona fide side. The robocall set holds 1,413 recordings used only for testing, with the project's own benign calls as controls.

Four arms fit on the same 120 dev calls and score on the same 120 test calls: audio-only using the eight voice features, text-only using the nine intent features, late fusion over one calibrated score per branch, and full using all twenty-five. Two checks precede any model. A pure-Python bag-of-words classifier is trained across speaker folds on caller text and the grammar is revised until it cannot separate the classes on authorship, after which it reaches AUC 0.990 on the whole corpus and 0.947 on the hard subset, which is the phenomenon itself. And the confound audit runs on every pair build.

## VII. Results and Discussion

### A. Ablation and Per-Cell Behaviour

TABLE_II_WIDE_CAPTION: Four-Arm Ablation and Per-Cell Accuracy on the Test Split

| Arm | AUC | F1 | EER | H/ben | H/scam | C/ben | C/scam | Hard AUC |
|---|---|---|---|---|---|---|---|---|
| Audio only | 0.616 | 0.187 | 0.400 | 0.933 | 0.100 | 0.800 | 0.133 | 0.362 |
| Text only | 0.999 | 0.983 | 0.033 | 1.000 | 1.000 | 1.000 | 0.933 | 0.993 |
| Late fusion | 0.999 | 0.974 | 0.033 | 1.000 | 1.000 | 1.000 | 0.900 | 0.993 |
| Full | 1.000 | 0.983 | 0.017 | 1.000 | 1.000 | 1.000 | 0.933 | 0.998 |

Table II reports the ablation with per-cell accuracy at threshold 0.65 and AUC on the hard subset of 13 lexically mild scams against 41 hard negatives. The full arm attains the highest AUC and halves the equal error rate of the text-only arm, from 0.033 to 0.017. The audio-only arm sits near chance by design, because the corpus draws the voice label independently of the scam label, so a branch reading only the voice cannot predict the content; its role is as a modifier within the fused score rather than as a detector. The full arm is perfect on three cells and misses two of thirty cloned scams, both lexically mild. The hard subset is where the arms have room to differ, and there the full arm ranks best at 0.998 against 0.993 for text alone, with no false alarms on the 41 benign calls written to sound like scams.

FIGURE: docs/figures/chart_ablation.png | Ablation of the four arms on the test split, AUC and F1.

The TF-IDF intent model reaches AUC 1.000 and $F_1$ 0.976 against AUC 0.913 and recall 0.483 for a lexicon rule baseline, which is the difference between counting fraud words and learning which combinations in which spellings mark a request. Entity recognition reaches $F_1$ 0.970 over 324 spans with exact span matching; types with a fixed surface form are recognised almost perfectly, while authority claims (0.863) and personal-information requests (0.850) lose recall rather than precision, the safer failure for a system that raises alarms. The fused probability is well calibrated at a Brier score of 0.011 and an expected calibration error of 0.015.

### B. The Voice Branch on Real Speech

TABLE_III_CAPTION: Anti-Spoofing Equal Error Rate on 310 Real Hindi Test Clips

| Front end | GMM clean | GMM GSM | GBM clean | GBM GSM |
|---|---|---|---|---|
| LFCC | 0.000 | 0.006 | 0.000 | 0.032 |
| GFCC | 0.006 | 0.019 | 0.006 | 0.019 |
| MFCC | 0.006 | 0.000 | 0.000 | 0.026 |
| CQCC | 0.013 | 0.000 | 0.000 | 0.006 |
| LPCC | 0.000 | 0.000 | 0.006 | 0.006 |

Every front end and back end separates real from cloned Hindi speech at an equal error rate below 0.02 on clean and G.711 audio and below 0.04 under GSM, which at 13 kbit/s is the only condition removing enough detail to cost anything. Minimum tandem detection cost remains at or below 0.019 with the mixture back end on clean and G.711 audio and reaches 0.058 under GSM. LFCC and LPCC with the mixture back end, the classical pairings, are the most stable across codecs.

FIGURE: docs/figures/chart_audit.png | Confound audit of the pair set: AUC of each channel statistic alone, on the raw files and after the conditioning the detector applies, against the audit's stop line at 0.75.

The audit is what makes those rates readable. It was designed around an early build in which bandwidth, trailing silence and DC offset each separated the classes on their own, and the conditioning and matching of Section IV bring all ten statistics inside the audit's threshold, with noise floor, spectral colour, trailing silence and in-band bandwidth at chance. The residual below 300 Hz lies outside the filterbank the detector reads, and dynamic range reflects spontaneous against read speech rather than the channel. The error rates in Table III are therefore measured on the voice.

### C. Generalisation, Latency and Robustness

TABLE_IV_CAPTION: Generalisation, Time to Detection and Channel Robustness

| Measure | Value |
|---|---|
| Robocall recall at 0.65, TF-IDF intent | 0.787 (rules 0.069) |
| Robocall cross-domain AUC | 0.946 (rules 0.712) |
| False alarms on 60 benign controls | 0.050 |
| Scam calls flagged | 57 of 60 |
| Median time to alert | 35.8 s, 6 turns (p90 49.4 s) |
| Full-arm AUC, clean to GSM | 0.9997 throughout |
| Audio-only AUC, clean to GSM | 0.616 to 0.559 |

The intent model, trained only on generated Hinglish calls, recalls nearly four in five real English robocalls at a 5 percent false-alarm rate. The transfer rests on shared fraud vocabulary and on the character n-grams, and it improved directly with the corpus rules: before the topic and phrasing pairing described in Section V, the same model recalled 0.699 at AUC 0.896 with a 15 percent false-alarm rate. Six turns is before the sensitive request in every flagged call; digital arrest is fastest at 17.4 s because its script asserts authority and threatens in its opening turns, while lottery and fake-relative scripts spend longer on setup. The fused decision is unchanged under every codec, and the audio-only column shows what the channel costs the voice branch alone, about 0.06 AUC under GSM, which the fusion absorbs.

FIGURE: docs/figures/chart_ttd.png | Time to detection: distribution over the 57 flagged scam calls, and median by scenario.

### D. Scope

The ablation characterises scripted calls, in which transcript content carries most of the decision, and the hard subset is where the branches differ and the full arm leads. The voice branch's error rates are those of the boosted and mixture back ends on a matched pair set whose spoofs come from the commercially available Hindi neural voices; the end-to-end back end and evaluation against a larger set of synthesis systems are the next step. Entity and intent figures are measured on reference transcripts, which isolates the taggers from recognition error, with the recogniser scored separately by word error rate.

## VIII. Conclusion

The problem was to decide, during a code-mixed Hindi-English call, whether the caller intends fraud when the voice may be cloned and the request phrased to sound routine. SwarKavach answers it with four branches fused into one calibrated per-turn risk score, built on a corpus and an evaluation protocol designed so that the numbers measure the voice and the intent rather than the recording conditions. On the 480-call corpus the full system reaches AUC 1.000, $F_1$ 0.983 and an equal error rate of 0.017, halving the error of the best single branch, is calibrated to a Brier score of 0.011, and flags 57 of 60 scams at a median of 35.8 seconds, six turns, before the sensitive request. On 1,413 real robocalls in another language the text branch recalls 78.7 percent at 5 percent false alarms, and the voice branch separates real from cloned Hindi speech at an equal error rate below 0.02 on clean and G.711 audio. The four signals are complementary in the way the design assumed: text carries the decision on scripted content, the voice branch is decisive on authenticity, and the structure and cross-modal branches separate the hard cases.

Future work follows from the measurements. A between-turn arousal variance would read how much a caller's arousal changes across turns relative to the content's pressure, which is how a cloned caller's flatness actually presents. The voice branch should be evaluated on IndicSynth, with its 989 speakers across twelve languages, under the same conditioning and audit. Entity and intent rows should be reported on recognised transcripts alongside word error rate. A benign half for the robocall evaluation would let the out-of-domain AUC be measured against negatives from the same channel. Incremental feature accumulation would bring the per-turn cost of a full analysis under one second on a CPU, against 18.8 s for a 74-second call today. And a code-mixed transformer such as MuRIL or HingBERT should be compared against the TF-IDF intent model under the same protocol.

Source code, corpus generator, trained models, results and console: https://github.com/BhavyeNijhawan/svarkavach

## References

[1] Z. Ma, P. Wang, M. Huang, J. Wang, K. Wu, X. Lv, Y. Pang, Y. Yang, W. Tang, and Y. Kang, "TeleAntiFraud-28k: An audio-text slow-thinking dataset for telecom fraud detection," arXiv:2503.24115, 2025.

[2] D. V. Sharma, V. Ekbote, and A. Gupta, "IndicSynth: A large-scale multilingual synthetic speech dataset for low-resource Indian languages," in *Proc. 63rd Annu. Meeting Assoc. Comput. Linguistics (Vol. 1: Long Papers)*, 2025.

[3] Z. Shen, S. Yan, Y. Zhang, X. Luo, G. Ngai, and E. Y. Fu, "'It warned me just at the right moment': Exploring LLM-based real-time detection of phone scams," arXiv:2502.03964, 2025.

[4] X. Wang, H. Delgado, H. Tak, J. Jung, H. Shim, M. Todisco, I. Kukanov, X. Liu, M. Sahidullah, T. Kinnunen, N. Evans, K. A. Lee, and J. Yamagishi, "ASVspoof 5: Crowdsourced speech data, deepfakes, and adversarial attacks at scale," in *Proc. ASVspoof Workshop*, 2024.

[5] J. Yi, C. Wang, J. Tao, X. Zhang, C. Y. Zhang, and Y. Zhao, "Audio deepfake detection: A survey," arXiv:2308.14970, 2023.

[6] Y. Zhu, S. Koppisetti, T. Tran, and G. Bharaj, "SLIM: Style-linguistics mismatch model for generalized audio deepfake detection," in *Advances in Neural Information Processing Systems 37*, 2024.
