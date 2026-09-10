"""Real neural speech for the corpus, using Microsoft Edge's free TTS voices.

Why this exists: the built-in `synth.py` produces a signal with the right
acoustic statistics but no intelligible words, which is fine for training a
spoof detector and useless for a demo where the transcript is on screen next
to the audio. This module renders the actual Hinglish text with real neural
voices, so what you hear is what the subtitle says.

Three things it has to get right.

**Script.** The Hindi voices expect Devanagari. Our corpus is romanised
Hinglish. Feeding "aapka account block ho jayega" to a Hindi voice in Latin
script gives a mangled reading, so Hindi words are transliterated and English
words are deliberately left in Latin, which is how a Hinglish speaker actually
says them.

**Speaker variety.** Edge exposes only five Indian voices. Varying rate and
pitch per speaker turns those five into something closer to twenty distinct
sounding people, which the speaker-embedding consistency feature needs.

**The human against cloned axis.** Every Edge voice is synthetic, so rendering
both classes with it would leave the anti-spoofing branch nothing real to
find. The cloned cells therefore go through `vocoder_artifacts` below, a
magnitude-preserving phase vocoder re-synthesis that imposes the specific
artefacts real voice cloning leaves: over-smoothed spectral envelope, regular
phase, near-zero jitter and shimmer. Intelligibility survives because the
magnitude spectrum is preserved.

Be clear about what that buys and what it does not. Offline, both classes are
machine generated, so the local anti-spoofing numbers measure "can it find
vocoder artefacts", not "can it tell a person from a machine". The real
human-against-spoof evaluation is ASVspoof 2019 LA and In-the-Wild, in
notebook 02. The report says this rather than implying otherwise.

Needs a network connection. Everything is cached on disk by content hash, so
a second run over the same corpus costs nothing, and the template grammar
repeats many lines verbatim, which cuts the request count by roughly half.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audioio import read_audio, rms_normalize, write_wav
from ..config import CORPUS_DIR, SETTINGS, TARGET_SR

# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------

#: (short name, gender, script the voice expects)
VOICE_POOL: Tuple[Tuple[str, str, str], ...] = (
    ("hi-IN-MadhurNeural", "male", "deva"),
    ("hi-IN-SwaraNeural", "female", "deva"),
    ("en-IN-PrabhatNeural", "male", "latin"),
    ("en-IN-NeerjaNeural", "female", "latin"),
    ("en-IN-NeerjaExpressiveNeural", "female", "latin"),
)

MAX_CONCURRENCY = 8
CACHE_DIR = CORPUS_DIR / "tts_cache"


def _stable_hash(*parts: Any) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def voice_for_speaker(speaker_id: str, seed: int = 0) -> Dict[str, Any]:
    """A stable voice, rate and pitch for one speaker id.

    Five voices is not many. Rate and pitch offsets multiply that out so two
    calls from different speakers do not sound like the same person, which
    matters because the speaker-embedding consistency feature is measuring
    exactly that.
    """
    h = _stable_hash("voice", speaker_id, seed)
    name, gender, script = VOICE_POOL[h % len(VOICE_POOL)]
    rate = -12 + (h >> 8) % 27           # -12% to +14%
    pitch = -9 + (h >> 16) % 19          # -9Hz to +9Hz
    return {
        "voice": name,
        "gender": gender,
        "script": script,
        "rate": f"{rate:+d}%",
        "pitch": f"{pitch:+d}Hz",
        "speaker_id": speaker_id,
    }


# --------------------------------------------------------------------------
# Romanised Hinglish to Devanagari
# --------------------------------------------------------------------------

#: English words that must stay in Latin script. A Hindi voice reading
#: "अकाउन्ट" sounds worse than one reading "account", and these are exactly the
#: tokens a Hinglish speaker says in English anyway.
_KEEP_LATIN = {
    "otp", "kyc", "upi", "atm", "pin", "cvv", "sms", "id", "qr", "apk",
    "account", "bank", "card", "credit", "debit", "netbanking", "branch",
    "block", "blocked", "verify", "verification", "transaction", "transfer",
    "payment", "balance", "statement", "cyber", "crime", "police", "case",
    "fir", "warrant", "court", "legal", "notice", "officer", "inspector",
    "department", "customer", "care", "service", "support", "helpline",
    "link", "click", "download", "install", "app", "screen", "share",
    "anydesk", "teamviewer", "delivery", "parcel", "courier", "order",
    "package", "address", "pincode", "offer", "scheme", "prize", "lottery",
    "loan", "emi", "interest", "policy", "insurance", "premium", "claim",
    "job", "interview", "salary", "company", "electricity", "bill", "meter",
    "connection", "disconnect", "reading", "appointment", "reminder",
    "confirm", "booking", "survey", "feedback", "rating", "school", "class",
    "exam", "result", "fee", "hospital", "doctor", "medical", "emergency",
    "process", "system", "server", "portal", "website", "online", "offline",
    "update", "record", "file", "document", "signature", "message", "number",
    "mobile", "phone", "call", "sir", "madam", "ok", "okay", "hello", "please",
    "thank", "thanks", "sorry", "minute", "minutes", "hour", "second",
    "rbi", "trai", "cbi", "ncb", "sbi", "pan", "aadhaar", "aadhar",
    "processing", "registration", "security", "deposit", "fraud", "risk",
    "team", "management", "wallet", "gpay", "phonepe", "paytm", "bhim",
}

#: Romanised Hinglish is not ITRANS. ITRANS is case sensitive and writes long
#: vowels as capitals, so "aapka" has to become "Apka" before transliteration
#: or it comes out as a broken vowel cluster. These rules cover the spellings
#: the corpus generator actually produces.
_ROMAN_TO_ITRANS: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p), r) for p, r in [
        (r"aa", "A"), (r"ee", "I"), (r"ii", "I"), (r"oo", "U"), (r"uu", "U"),
        (r"ck", "k"), (r"x", "ks"), (r"z", "j"), (r"w", "v"),
        (r"chh", "Ch"), (r"ch", "ch"), (r"sh", "sh"), (r"ss", "s"),
        (r"th", "th"), (r"dh", "dh"), (r"ph", "ph"), (r"bh", "bh"),
        (r"gh", "gh"), (r"kh", "kh"), (r"jh", "jh"), (r"tt", "T"), (r"dd", "D"),
        (r"ain$", "ain"), (r"ein$", "ein"),
    ]
)

_DIGIT_HI = {
    "0": "shunya", "1": "ek", "2": "do", "3": "teen", "4": "chaar",
    "5": "paanch", "6": "chhe", "7": "saat", "8": "aath", "9": "nau",
}

_transliterate = None


def _load_transliterator():
    global _transliterate
    if _transliterate is None:
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate

            def fn(s: str) -> str:
                return transliterate(s, sanscript.ITRANS, sanscript.DEVANAGARI)

            _transliterate = fn
        except Exception:
            _transliterate = False        # tried and unavailable
    return _transliterate or None


#: Words the rules below get wrong often enough to be worth pinning. Values
#: are ITRANS, not Devanagari, so they still go through the transliterator.
_ITRANS_OVERRIDES: Dict[str, str] = {
    "main": "maiM", "mein": "meM", "me": "meM", "hai": "hai", "hain": "haiM",
    "hoon": "hUM", "hun": "hUM", "hu": "hUM", "ho": "ho", "hoga": "hogA",
    "hogi": "hogI", "raha": "rahA", "rahi": "rahI", "rahe": "rahe",
    "jayega": "jAyegA", "jayegi": "jAyegI", "jaayega": "jAyegA",
    "aapka": "ApkA", "aapki": "ApkI", "aapke": "Apke", "aap": "Ap",
    "aapko": "ApkO", "aapse": "Apse", "apna": "apnA", "apne": "apne",
    "nahi": "nahIM", "nahin": "nahIM", "haan": "hAM", "han": "hAM",
    "kya": "kyA", "kyun": "kyoM", "kyon": "kyoM", "kaise": "kaise",
    "abhi": "abhI", "turant": "turant", "jaldi": "jaldI",
    "namaste": "namaste", "namaskar": "namaskAr", "dhanyavaad": "dhanyavAd",
    "kijiye": "kIjie", "dijiye": "dIjie", "bataiye": "batAie",
    "batao": "batAo", "boliye": "bolie", "suniye": "sunie",
    "sakta": "saktA", "sakti": "saktI", "sakte": "sakte",
    "karo": "karo", "karna": "karnA", "kar": "kar", "kiya": "kiyA",
    "gaya": "gayA", "gayi": "gaI", "diya": "diyA", "liya": "liyA",
    "band": "band", "bandh": "band", "das": "das", "paisa": "paisA",
    "paise": "paise", "rupaye": "rupaye", "lakh": "lAkh", "crore": "karoR",
    "mat": "mat", "koi": "koI", "kuch": "kuch", "sab": "sab",
    "thik": "ThIk", "theek": "ThIk", "acha": "acchA", "achha": "acchA",
    "ji": "jI", "bhai": "bhAI", "beta": "beTA", "ghante": "ghanTe",
    "minute": "minute", "andar": "andar", "bahar": "bAhar",
    "warna": "varnA", "varna": "varnA", "agar": "agar", "phir": "phir",
    "bol": "bol", "sun": "sun", "dekh": "dekh", "samajh": "samajh",
}


def _to_itrans(token: str) -> str:
    """Romanised Hinglish to ITRANS, well enough for a Hindi voice to read.

    Two systematic corrections beyond the character rules, because Hinglish
    spelling leaves out things Devanagari needs:

    A word-final "a" in Hinglish is nearly always a long vowel ("raha" is
    rahaa, not rah), so it becomes ITRANS "A". And a word ending in a
    consonant needs an inherent vowel appended, otherwise ITRANS puts a halant
    on it and the voice reads a clipped consonant ("bol" would come out as
    बोल् rather than बोल).
    """
    t = token.lower()
    if t in _ITRANS_OVERRIDES:
        return _ITRANS_OVERRIDES[t]

    for pat, rep in _ROMAN_TO_ITRANS:
        t = pat.sub(rep, t)

    if t.endswith("a"):
        t = t[:-1] + "A"
    elif t and t[-1] not in "aAiIuUeoEO":
        t = t + "a"
    return t


def to_devanagari(text: str) -> str:
    """Romanised Hinglish to mixed Devanagari, English words left in Latin.

    Falls back to the original string when the transliteration package is not
    installed, which just means the Hindi voice reads Latin script.
    """
    fn = _load_transliterator()
    if fn is None:
        return text

    out: List[str] = []
    for tok in text.split():
        core = tok.strip(".,!?;:").lower()
        punct = "".join(c for c in tok if c in ".,!?;:")
        if not core:
            out.append(tok)
        elif core in _KEEP_LATIN or not core.isalpha():
            # digits read better spelled out for a Hindi voice, but an OTP
            # should stay as digits so it is read one by one
            out.append(tok)
        else:
            try:
                out.append(fn(_to_itrans(core)) + punct)
            except Exception:
                out.append(tok)
    return " ".join(out)


#: Anything in angle brackets is stripped before synthesis. The service parses
#: its input as SSML, so a transcription marker like `<inaudible>` is read as a
#: tag and the render stops there. Measured on the GramVaani pairs: 95 percent
#: of the truncated renders contained one of these, against 8 percent of the
#: intact ones, and a 200 character line came back as 1.6 seconds of audio.
_SSML_UNSAFE = re.compile(r"<[^>]*>")


def clean_for_tts(text: str) -> str:
    """Make a transcript safe to hand to the TTS service."""
    t = _SSML_UNSAFE.sub(" ", text)
    t = t.replace("&", " and ")
    return re.sub(r"\s+", " ", t).strip()


def prepare_text(text: str, script: str) -> str:
    text = clean_for_tts(text)
    return to_devanagari(text) if script == "deva" else text


# --------------------------------------------------------------------------
# Vocoder artefacts, for the cloned cells
# --------------------------------------------------------------------------


def vocoder_artifacts(
    x: np.ndarray,
    sr: int = TARGET_SR,
    order: int = 16,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    envelope_smooth: int = 5,
) -> np.ndarray:
    """Impose the artefacts real voice cloning leaves, keeping the words.

    A classic LPC vocoder: estimate the vocal-tract filter frame by frame,
    throw the original excitation away, and drive the filter with a perfectly
    periodic pulse train on voiced frames and white noise on unvoiced ones.

    The point is what the discarded excitation was carrying. Cycle-to-cycle
    pitch variation (jitter) and amplitude variation (shimmer) live in the
    excitation, so replacing it with a regular pulse train drives both toward
    zero. That is exactly the signature the anti-spoofing features look for,
    and it is what real vocoder-based cloning leaves behind. Smoothing the
    filter coefficients across time adds the over-smoothed spectral envelope
    that goes with it. The filter is preserved, so the words survive.

    The first version of this used a phase vocoder with a fixed per-bin phase
    advance. That was wrong: partials sitting between bin centres beat against
    the imposed phase, which produced amplitude modulation and sent shimmer UP
    by a factor of seven instead of down.
    """
    from ..dsp.cepstral import autocorrelation, levinson_durbin
    from ..dsp.framing import frame_signal
    from ..dsp.prosody import f0_track

    x = np.asarray(x, dtype=np.float64).ravel()
    frame_len = max(16, int(round(sr * frame_ms / 1000.0)))
    hop = max(4, int(round(sr * hop_ms / 1000.0)))
    if x.size < frame_len * 3:
        return np.asarray(x, dtype=np.float32)

    frames = frame_signal(x, frame_len, hop, window="hamming")
    n_frames = frames.shape[0]
    if n_frames < 3:
        return np.asarray(x, dtype=np.float32)

    r = autocorrelation(frames, order)
    a, err = levinson_durbin(r, order)          # a[:, 0] == 1

    # Over-smooth the filter across time. A vocoder updates its envelope
    # parameters slowly, so fast spectral detail is lost.
    if envelope_smooth > 1:
        k = np.ones(envelope_smooth) / envelope_smooth
        pad = envelope_smooth // 2
        padded = np.vstack([np.repeat(a[:1], pad, axis=0), a,
                            np.repeat(a[-1:], pad, axis=0)])
        a = np.stack([np.convolve(padded[:, j], k, mode="same")[pad:pad + n_frames]
                      for j in range(a.shape[1])], axis=1)
        a[:, 0] = 1.0

    f0, voiced = f0_track(x, sr)
    m = min(n_frames, f0.size)
    f0 = np.concatenate([f0[:m], np.full(max(0, n_frames - m), 0.0)])
    voiced = np.concatenate([voiced[:m], np.zeros(max(0, n_frames - m), bool)])

    # A single median pitch per utterance rather than the real contour: the
    # excitation becomes strictly periodic, which is the whole point.
    vf = f0[voiced & (f0 > 0)]
    base = float(np.median(vf)) if vf.size else 140.0
    base = float(np.clip(base, 60.0, 400.0))

    gain = np.sqrt(np.maximum(err[:n_frames], 1e-12))
    # flatten the gain contour too: constant loudness per voiced stretch is
    # what removes shimmer
    if gain.size > 3:
        gk = np.ones(9) / 9.0
        gain = np.convolve(gain, gk, mode="same")

    try:
        from scipy.signal import lfilter
    except Exception:                            # pragma: no cover
        return np.asarray(x, dtype=np.float32)

    rng = np.random.default_rng(1234)
    period = max(2, int(round(sr / base)))
    total = x.size + frame_len

    # ONE excitation for the whole utterance, not one per frame. Building it
    # per frame lets overlapping frames disagree about where the pulses go,
    # and the overlap-add then smears them into something less regular than
    # the original rather than more.
    exc = rng.normal(0.0, 0.35, total)
    voiced_mask = np.zeros(total, dtype=bool)
    for i in range(n_frames):
        if voiced[i]:
            voiced_mask[i * hop: i * hop + hop] = True
    pulses = np.zeros(total)
    pulses[::period] = 1.0
    exc = np.where(voiced_mask, pulses, exc)

    # per-sample gain, from the flattened per-frame gain
    g = np.interp(np.arange(total), np.arange(n_frames) * hop + frame_len / 2.0,
                  gain[:n_frames])
    exc = exc * g

    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros_like(out)
    win = np.hanning(frame_len + 1)[:-1]

    for i in range(n_frames):
        s = i * hop
        try:
            seg = lfilter([1.0], a[i], exc[s:s + frame_len])
        except Exception:
            seg = exc[s:s + frame_len]
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.max(np.abs(seg)) if seg.size else 0.0
        if peak > 1e3:                          # unstable filter for this frame
            seg = seg / peak
        out[s:s + frame_len] += seg * win
        norm[s:s + frame_len] += win

    out = out[: x.size] / np.maximum(norm[: x.size], 1e-6)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return rms_normalize(out.astype(np.float32), target_dbfs=-20.0)


# --------------------------------------------------------------------------
# Synthesis with a disk cache
# --------------------------------------------------------------------------


#: Last few synthesis errors, so a failed run says why instead of just
#: reporting a count of zero.
FAILURES: List[str] = []


def _record_failure(text: str, voice: str, exc: BaseException) -> None:
    msg = f"{voice}: {type(exc).__name__}: {exc}"
    if len(FAILURES) < 8 and msg not in FAILURES:
        FAILURES.append(msg)


def _cache_path(text: str, voice: str, rate: str, pitch: str) -> Path:
    key = hashlib.sha1(
        "|".join([text, voice, rate, pitch]).encode("utf-8")
    ).hexdigest()[:20]
    return CACHE_DIR / f"{key}.wav"


async def _synth_one(
    text: str, voice: str, rate: str, pitch: str, sr: int, sem: asyncio.Semaphore
) -> Optional[np.ndarray]:
    """One line of speech, from the cache when possible."""
    import edge_tts  # type: ignore

    path = _cache_path(text, voice, rate, pitch)
    if path.exists():
        try:
            x, _ = read_audio(path, sr=sr)
            return x
        except Exception:
            pass

    # the directory has to exist before edge-tts writes into it, not after
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".mp3.part")
    last_error: Optional[BaseException] = None
    async with sem:
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                await comm.save(str(tmp))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    _record_failure(text, voice, exc)
                    return None
                await asyncio.sleep(1.5 * (attempt + 1))
    if last_error is not None:
        return None

    try:
        x, _ = read_audio(tmp, sr=sr)
        x = rms_normalize(x, target_dbfs=-20.0)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        write_wav(path, x, sr)
        return x
    except Exception:
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


async def synth_many(
    items: Sequence[Tuple[str, str, str, str]], sr: int = TARGET_SR
) -> List[Optional[np.ndarray]]:
    """Synthesise (text, voice, rate, pitch) tuples concurrently."""
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [_synth_one(t, v, r, p, sr, sem) for (t, v, r, p) in items]
    return list(await asyncio.gather(*tasks))


def synth_batch(
    items: Sequence[Tuple[str, str, str, str]], sr: int = TARGET_SR
) -> List[Optional[np.ndarray]]:
    """Blocking wrapper around `synth_many`, safe to call from normal code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(synth_many(items, sr))
    # already inside a loop (a notebook, say): run in a private one
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(synth_many(items, sr))).result()


def cache_stats() -> Dict[str, Any]:
    if not CACHE_DIR.exists():
        return {"files": 0, "mb": 0.0, "dir": str(CACHE_DIR)}
    files = list(CACHE_DIR.glob("*.wav"))
    return {
        "files": len(files),
        "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        "dir": str(CACHE_DIR),
    }


if __name__ == "__main__":
    import sys
    import time

    sys.stdout.reconfigure(encoding="utf-8")

    line = "Namaste, main cyber crime branch se bol raha hoon, aapka account das minute mein block ho jayega"
    print("latin     :", line)
    print("devanagari:", to_devanagari(line))

    for spk in ("spk_00", "spk_07", "spk_13"):
        print(f"  {spk}: {voice_for_speaker(spk)}")

    print("\nsynthesising three lines")
    v = voice_for_speaker("spk_00")
    items = [
        (prepare_text(line, v["script"]), v["voice"], v["rate"], v["pitch"]),
        (prepare_text("Ji boliye, kya baat hai", v["script"]), v["voice"], v["rate"], v["pitch"]),
        (prepare_text("Main branch jaunga, phone pe kuch nahi bataunga", v["script"]),
         v["voice"], v["rate"], v["pitch"]),
    ]
    t0 = time.time()
    got = synth_batch(items)
    ok = [g for g in got if g is not None]
    print(f"  {len(ok)}/{len(items)} rendered in {time.time() - t0:.1f}s")
    for i, g in enumerate(ok):
        print(f"    line {i}: {g.size / TARGET_SR:.2f}s")

    if ok:
        from ..dsp.prosody import prosody_summary

        clean = ok[0]
        cloned = vocoder_artifacts(clean)
        pc, pv = prosody_summary(clean, TARGET_SR), prosody_summary(cloned, TARGET_SR)
        print("\nvocoder artefacts, clean against cloned:")
        for k in ("jitter", "shimmer", "hnr", "f0_std"):
            print(f"    {k:10s} {pc[k]:8.4f}  ->  {pv[k]:8.4f}")
        print(f"    duration   {clean.size / TARGET_SR:8.2f}  ->  {cloned.size / TARGET_SR:8.2f}")

    print("\ncache:", cache_stats())
