# SwarKavach module contracts

Read this before writing any module. It fixes the interfaces so five people
(or five agents) can build in parallel and have the pieces snap together.

Project root: `D:\PS2_VoiceClone_Fraud_Call_Detection`
Package root: `src/swarkavach/`
Python: 3.11, run through `D:\PS2_VoiceClone_Fraud_Call_Detection\.venv\Scripts\python.exe`

## Hard rules

1. **Available libraries only**: numpy, scipy, scikit-learn, pandas, matplotlib,
   torch (CPU), Flask, PyYAML, typer, rich, tqdm, joblib, soundfile.
   Nothing else. No librosa, no whisper, no transformers at import time, no
   seaborn, no shap, no sklearn-crfsuite. If a module can optionally use one of
   those, guard it with try/except and provide a working fallback.
2. **Import the shared contracts**, never redefine them:
   `from ..schema import Call, Turn, EntitySpan, BranchScore, ENTITY_TYPES, ...`
   `from ..config import SETTINGS, TARGET_SR, MODELS_DIR, ...`
3. **One tokeniser**: `swarkavach.schema.tokenize`. BIO lists must always be the
   same length as the token list.
4. **Determinism**: seed everything from `SETTINGS.pipeline.seed`. Same input,
   same output, every run.
5. **No network calls, ever.** Everything must work offline.
6. **Windows paths**: use `pathlib`, never hard-code a separator.
7. **Prose style** in docstrings and comments: plain, direct, no em dashes and
   no en dashes anywhere. Use commas, colons or parentheses instead. Do not
   write marketing copy. Do not use the words "comprehensive", "seamless",
   "leverage", "robust" as filler, "delve", or "it's not just X, it's Y".
8. Every module you write must have a `if __name__ == "__main__":` self-test
   block that runs in under 20 seconds and prints something a human can check.

## Already written (do not modify)

- `swarkavach/schema.py` : `Call`, `Turn`, `EntitySpan`, `Evidence`,
  `BranchScore`, `Verdict`, `tokenize`, `decode_bio`, `ENTITY_TYPES`,
  `BIO_LABELS`, `DIALOGUE_ACTS`, `COERCION_RANK`, `SCAM_SCENARIOS`,
  `BENIGN_SCENARIOS`, `FUSION_FEATURES`, `FUSION_FEATURE_NAMES`,
  `FEATURE_GROUPS`, `FEATURE_LABELS`, `ABLATION_ARMS`, `risk_band`,
  `empty_feature_dict`, `feature_vector`.
- `swarkavach/config.py` : `SETTINGS` (with `.frame`, `.cepstral`, `.prosody`,
  `.vad`, `.pipeline`), `TARGET_SR=8000`, `WIDEBAND_SR=16000`, `ROOT`,
  `DATA_DIR`, `CORPUS_DIR`, `CORPUS_AUDIO_DIR`, `CORPUS_CALLS_DIR`,
  `MODELS_DIR`, `RESULTS_DIR`, `ensure_dirs()`, `artifact_path(name)`,
  `result_path(name)`, `find_ffmpeg()`, `N_SPEAKERS`, `DEFAULT_SPLIT`.
- `swarkavach/audioio.py` : `read_audio(path, sr) -> (np.float32[n], sr)`,
  `read_audio_bytes(raw, sr)`, `write_wav(path, x, sr)`, `wav_bytes(x, sr)`,
  `resample(x, sr_in, sr_out)`, `normalize`, `rms_normalize`, `pad_or_trim`,
  `duration_s`, `concat_with_gaps(segments, sr, gap_s) -> (signal, spans)`.

## The 25 fusion features

`schema.FUSION_FEATURE_NAMES` is the contract between the branches and the
fusion layer. Each branch fills in its own group and leaves the rest at 0.0.

| group | features | owner |
|---|---|---|
| `voice` | `as_score as_llr as_margin spk_consistency jitter shimmer hnr spec_flatness_var` | anti-spoof branch |
| `intent` | `intent_score intent_max_turn urgency_density ent_otp ent_authority ent_threat ent_payment ent_personal ent_bank` | text branch |
| `cross` | `pim coercion_slope coercion_peak act_scam_llr cmi switch_entropy ent_lang_align callee_resist` | fusion layer |

Values should be roughly on a comparable scale. Probabilities stay in [0, 1],
counts are normalised per 100 tokens, log-ratios are clipped to [-10, 10].

## Module interfaces to implement

### A. `swarkavach/dsp/` (module 5 and 6 of the syllabus)

```python
# framing.py
preemphasis(x, coef=0.97) -> np.ndarray
frame_signal(x, frame_len, hop_len, window="hamming") -> np.ndarray  # (n_frames, frame_len)
get_window(name, n) -> np.ndarray                 # hamming, hann, rect, blackman
short_time_energy(frames) -> np.ndarray           # (n_frames,)
log_energy(frames) -> np.ndarray
zero_crossing_rate(frames) -> np.ndarray

# spectral.py
stft(x, cfg: FrameConfig) -> np.ndarray           # complex (n_frames, n_fft//2+1)
power_spectrum(x, cfg) -> np.ndarray              # magnitude squared
spectrogram_db(x, cfg, top_db=80) -> np.ndarray   # for the dashboard
spectral_centroid / spectral_bandwidth / spectral_rolloff / spectral_flatness / spectral_flux
  each -> (n_frames,)

# filterbanks.py
mel_filterbank(sr, n_fft, n_filters, fmin, fmax) -> (n_filters, n_fft//2+1)
gammatone_filterbank(...)   -> same shape, ERB spaced, used for GFCC
linear_filterbank(...)      -> same shape, linear spaced, used for LFCC
cqt_like_filterbank(...)    -> constant-Q geometric spacing, used for CQCC-approx
hz_to_mel / mel_to_hz / hz_to_erb / erb_to_hz

# cepstral.py
dct2(x, n_out) -> np.ndarray                      # DCT-II, orthonormal, implemented here
mfcc(x, cfg, cep) -> (n_frames, n_ceps)
gfcc(x, cfg, cep) -> (n_frames, n_ceps)
lfcc(x, cfg, cep) -> (n_frames, n_ceps)
cqcc(x, cfg, cep) -> (n_frames, n_ceps)           # constant-Q cepstral, approximation
lpcc(x, cfg, order=16, n_ceps=20) -> (n_frames, n_ceps)   # via Levinson-Durbin
FEATURE_EXTRACTORS: dict[str, callable]           # "mfcc"|"gfcc"|"lfcc"|"cqcc"|"lpcc"

# deltas.py
delta(feat, width=2) -> same shape
add_deltas(feat, width=2, order=2) -> (n_frames, n_ceps*3)
cmvn(feat) -> same shape                          # per-utterance mean/var norm

# prosody.py
f0_track(x, sr, cfg: ProsodyConfig) -> (f0: np.ndarray, voiced: np.ndarray[bool])
  # autocorrelation or YIN, per frame, 0.0 where unvoiced
jitter(f0, voiced) -> float                       # local jitter, relative
shimmer(x, sr, f0, voiced) -> float
hnr(x, sr, f0, voiced) -> float                   # harmonic to noise ratio, dB
speech_rate(x, sr) -> float                       # syllable-ish rate from energy peaks
prosody_summary(x, sr) -> dict[str, float]
  # keys: f0_mean f0_std f0_range f0_slope voiced_ratio jitter shimmer hnr
  #       energy_std energy_range rate rate_std pause_ratio
  # This dict is the ACOUSTIC AROUSAL input to the PIM feature. It must always
  # return every key, using 0.0 when a value cannot be computed.

# vad.py
vad_mask(x, sr, cfg: VADConfig) -> np.ndarray[bool]   # per frame
speech_segments(x, sr, cfg) -> list[tuple[float,float]]   # seconds
trim_silence(x, sr, cfg) -> np.ndarray
```

### B. `swarkavach/channel.py`

```python
apply_codec(x, sr, codec, **kw) -> (np.ndarray, sr)
  # codec in {"clean","g711u","g711a","gsm","amrnb","narrowband","packet_loss"}
  # g711u / g711a are exact mu-law and A-law companding, implemented in numpy
  # gsm / amrnb shell out to ffmpeg if config.find_ffmpeg() finds it, otherwise
  #   they fall back to a documented numpy approximation and set a flag
add_noise(x, snr_db, kind="babble"|"white"|"hum") -> np.ndarray
telephone_band(x, sr, low=300, high=3400) -> np.ndarray
CODECS: dict[str, dict]      # name -> {"label":..., "real": bool, "needs_ffmpeg": bool}
degrade(x, sr, codec="g711u", snr_db=None, seed=0) -> (np.ndarray, dict)
```

### C. `swarkavach/corpus/`

```python
# lexicon.py
HINDI_WORDS: set[str]        # romanised Hindi function and content words, >=600
ENGLISH_WORDS: set[str]      # >=600
URGENCY_TERMS: dict[str, float]      # term -> weight in [0,1]
THREAT_TERMS / AUTHORITY_TERMS / BANK_TERMS / PAYMENT_TERMS / PERSONAL_TERMS: dict
GAZETTEERS: dict[str, set[str]]      # entity type -> surface forms
BENIGN_MARKERS: dict[str, float]     # phrases that argue AGAINST fraud

# generator.py
generate_corpus(n_calls, seed, out_dir, audio=False, audio_backend="sim") -> list[Call]
  # writes data/corpus/calls/<call_id>.json and data/corpus/manifest.json
  # gold BIO, gold dialogue acts, gold per-token language tags, all automatic
load_corpus(split=None) -> list[Call]
corpus_stats(calls) -> dict

# tts.py
render_call_audio(call, backend="sim", out_path=None, sr=8000) -> (np.ndarray, dict)
  # backend "sim": built-in formant synthesiser in synth.py
  # backend "sapi"/"edge"/"xtts": optional, guarded, never required
# synth.py
synthesize_turn(text, sr, voice, synthetic: bool, seed) -> np.ndarray
  # A source-filter synthesiser. `synthetic=False` adds human micro-variation
  # (jitter, shimmer, breath noise, F0 drift). `synthetic=True` produces the
  # over-smoothed, low-jitter, phase-regular signal that vocoder output has.
  # This is what gives the anti-spoof branch real signal to learn offline.
```

### D. `swarkavach/text/`

```python
# normalize.py
normalize_text(s) -> str          # lowercase, unify spellings, strip diacritics
HINGLISH_VARIANTS: dict[str,str]  # "kripya"->"kripaya" style canonicalisation
# langid.py
tag_languages(tokens) -> list[str]                  # "hi"|"en"|"univ"
code_mixing_features(tokens, langs, bio=None) -> dict
  # returns cmi, switch_points, switch_entropy, hi_ratio, en_ratio,
  #         ent_lang_align  (share of fraud entities carried by English)
# pos.py
pos_tag(tokens) -> list[str]      # coarse tagset, lexicon plus suffix rules
# crf.py         <- implement a linear-chain CRF from scratch (numpy + scipy L-BFGS)
class LinearChainCRF:  fit(X, y), predict(X), score(X, y), save(path), load(path)
# ner_crf.py
class CRFTagger: fit(calls), predict_turn(tokens) -> list[str], evaluate(calls) -> dict
# ner_bilstm.py
class BiLSTMTagger: same interface, torch, CPU trainable in under 2 minutes
# intent_rules.py
class RuleIntent: score_turn(tokens) -> float, explain(tokens) -> list[dict]
# intent_tfidf.py
class TfidfIntent: fit(calls), score_turn(tokens) -> float, score_call(call) -> dict
# coercion.py
act_classifier(turn) -> str                # rule based, predicts a DIALOGUE_ACT
coercion_features(call, acts=None) -> dict # coercion_slope, coercion_peak,
                                           # act_scam_llr, callee_resist
class ActHMM: fit(calls), llr(acts) -> float
# asr.py
class ASR:  transcribe(path_or_array, sr) -> {"text":..., "segments":[...], "backend":...}
  # backends: "gold" (sidecar JSON), "whisper" (if installed), "manual"
```

### E. `swarkavach/antispoof/`

```python
# features.py
utterance_features(x, sr, feature_set="lfcc") -> np.ndarray   # pooled vector
frame_features(x, sr, feature_set) -> np.ndarray              # (n_frames, d)
POOLING = mean, std, skew, kurtosis, plus delta stats
antispoof_feature_dict(x, sr) -> dict[str,float]   # jitter, shimmer, hnr,
                                                   # spec_flatness_var, etc.
# gmm.py
class GMMScorer: fit(X_bona, X_spoof), llr(X) -> float, save/load
# gbm.py
class GBMScorer: fit(X, y), score(X) -> float, save/load
# rawnet.py
class RawNetLite(torch.nn.Module)          # SincConv front end, residual blocks
train_rawnet(...) / load_rawnet(path)      # CPU inference must work
# embeddings.py
class SpeakerEmbedder: embed(x, sr) -> np.ndarray(192,)   # small TDNN, torch
cosine_consistency(embs) -> float          # within-call embedding stability
# scorer.py
class AntiSpoofScorer:
    def __init__(self, backend="auto"): ...
    def score(self, x, sr) -> BranchScore      # name="antispoof", score=P(synthetic)
    def fit(self, calls): ...
    def save(self, path) / load(path)
```

### F. `swarkavach/fusion/`

```python
# pim.py
lexical_arousal(call, intent_scores) -> np.ndarray     # per caller turn, [0,1]
acoustic_arousal(x, sr, spans) -> np.ndarray           # per caller turn, [0,1]
prosody_intent_mismatch(lex, aco) -> dict
  # {"pim": float, "per_turn": [...], "corr": float}
  # PIM is high when lexical arousal is high and acoustic arousal is flat.
# featurize.py
build_features(call, audio=None, sr=None, models=...) -> (dict[str,float], detail)
# model.py
class FusionModel: fit(X, y, arm="full"), predict_proba(X), calibrate(...),
                   explain(x) -> list[dict], save/load
# explain.py
linear_shap(model, x, background_mean) -> list[dict]   # exact for logistic
kernel_shap(predict_fn, x, background, n_samples) -> list[dict]  # for the GBM
build_evidence(call, features, contributions, spans) -> Evidence
# streaming.py
stream_call(call, audio, sr, models, threshold) -> list[dict]
  # incremental verdict after each turn; each dict has turn_index, t_end, risk,
  # intent, authenticity, pim, new_entities, reasons
time_to_detection(timeline, threshold) -> (seconds, turn_index) | (None, None)
```

### G. `swarkavach/evaluate.py`

```python
eer(scores, labels) -> (eer, threshold)
min_tdcf(scores, labels, ...) -> float        # ASVspoof style, documented priors
det_curve(scores, labels) -> (fpr, fnr, thresholds)
roc_auc / accuracy / precision_recall_f1
entity_prf(gold_calls, pred_bio) -> dict      # entity-level, exact span match
ablation_table(calls, models) -> pandas.DataFrame
robustness_table(calls, models, codecs) -> pandas.DataFrame
ttd_stats(verdicts) -> dict
run_full_evaluation(...) -> dict              # writes data/results/*.json
```

## Output files the dashboard reads

Everything the dashboard shows comes from JSON in `data/results/`:

- `corpus_stats.json`
- `antispoof_results.json`  (feature set x model x condition, EER and t-DCF)
- `ner_results.json`        (CRF vs BiLSTM, gold vs ASR transcripts)
- `intent_results.json`
- `ablation_results.json`   (the headline four-cell table)
- `robustness_results.json`
- `ttd_results.json`
- `calibration.json`
- `provenance.json`         (which runs are local, which came from Colab)

Write them with `json.dump(obj, fh, indent=2)`. Numbers must be plain floats,
not numpy scalars (call `float()` on them).
