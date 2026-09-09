"""Transcription front end, with three interchangeable backends.

The rest of the text branch only ever sees a list of `Turn` objects, so where
those turns came from is this module's problem alone.

* `gold` reads the sidecar Call JSON that sits next to the audio file. The
  corpus ships a perfect transcript with every call, so the whole NLP stack can
  be developed and measured without an ASR in the loop, and the ASR can then be
  measured against that same transcript.
* `whisper` runs OpenAI Whisper if it happens to be installed. The import is
  guarded and never happens at module import time, because the offline
  environment this project runs in does not have it and must not break.
* `manual` takes text the caller supplies, which is what the dashboard's paste
  box uses.

`auto` picks whisper if it can be imported, then gold if a sidecar exists, then
manual. That ordering is what makes the same code path work in Colab (where
whisper is installed) and on the offline machine (where it is not).

`word_error_rate` closes the loop: gold transcripts are free here, so the WER of
a real ASR against them is a number worth putting in the results table, and the
whole pipeline can be re-run on ASR output to show how much entity F1 and intent
accuracy a real transcription costs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..config import CORPUS_CALLS_DIR, TARGET_SR, WIDEBAND_SR
from ..schema import Call, Turn, tokenize
from .langid import tag_languages

__all__ = ["ASR", "word_error_rate", "wer_details", "aggregate_wer", "find_sidecar"]

_BACKENDS = ("auto", "gold", "whisper", "manual")


# --------------------------------------------------------------------------
# Word error rate
# --------------------------------------------------------------------------


def _wer_tokens(s: str, hinglish: bool = False) -> List[str]:
    """Words for scoring: lowercased, punctuation dropped, digits kept whole."""
    from .normalize import normalize_token

    toks = [t for t in tokenize(s.lower()) if t.isalnum()]
    if hinglish:
        toks = [normalize_token(t) for t in toks]
    return toks


def wer_details(ref: str, hyp: str, hinglish: bool = False) -> Dict[str, Any]:
    """Levenshtein alignment counts between two transcripts.

    Standard word-level edit distance with unit costs. `hinglish=True` first
    maps both sides to the canonical spelling, which separates real recognition
    errors from romanisation variants (kripaya against kripya is not an ASR
    mistake worth counting, but it costs a full substitution otherwise).
    """
    r = _wer_tokens(ref, hinglish)
    h = _wer_tokens(hyp, hinglish)
    n, m = len(r), len(h)
    if n == 0:
        return {"wer": 0.0 if m == 0 else 1.0, "sub": 0, "del": 0,
                "ins": m, "n_ref": 0, "n_hyp": m, "distance": m}

    # Full DP table, so the backtrace can report substitutions separately from
    # insertions and deletions. Transcripts here are short enough for O(n*m).
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)

    i, j = n, m
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            sub += int(r[i - 1] != h[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {
        "wer": float(int(d[n, m]) / n),
        "sub": int(sub), "del": int(dele), "ins": int(ins),
        "n_ref": int(n), "n_hyp": int(m), "distance": int(d[n, m]),
    }


def word_error_rate(ref: str, hyp: str, hinglish: bool = False) -> float:
    """Word error rate: edit distance over reference length.

    Zero for identical strings. One when nothing matches and both sides are the
    same length; it can exceed one when the hypothesis is longer than the
    reference, which is the standard definition and not a bug.
    """
    return float(wer_details(ref, hyp, hinglish)["wer"])


def aggregate_wer(refs: Sequence[str], hyps: Sequence[str], hinglish: bool = False) -> Dict[str, float]:
    """Corpus WER: total edits over total reference words, not a mean of WERs.

    Averaging per-utterance rates would let a three-word turn outweigh a
    forty-word one, which is why no ASR paper reports it that way.
    """
    tot_err = tot_ref = 0
    per_call: List[float] = []
    for r, h in zip(refs, hyps):
        d = wer_details(r, h, hinglish)
        tot_err += d["distance"]
        tot_ref += d["n_ref"]
        per_call.append(d["wer"])
    return {
        "wer": float(tot_err / max(1, tot_ref)),
        "mean_utterance_wer": float(np.mean(per_call)) if per_call else 0.0,
        "n_utterances": len(per_call),
        "n_ref_words": int(tot_ref),
    }


# --------------------------------------------------------------------------
# Sidecar lookup
# --------------------------------------------------------------------------

#: Suffixes the corpus adds to an audio file name for degraded copies. They are
#: stripped one at a time when looking for the matching call JSON.
_CHANNEL_SUFFIXES = ("_clean", "_g711u", "_g711a", "_gsm", "_amrnb",
                     "_narrowband", "_packet_loss", "_noisy")


def find_sidecar(path: Union[str, Path]) -> Optional[Path]:
    """Locate the Call JSON that belongs to an audio file, if there is one."""
    p = Path(path)
    stems = [p.stem]
    for suf in _CHANNEL_SUFFIXES:
        if p.stem.endswith(suf):
            stems.append(p.stem[: -len(suf)])
    candidates: List[Path] = []
    for stem in stems:
        candidates.append(p.with_name(stem + ".json"))
        candidates.append(p.parent / "calls" / (stem + ".json"))
        candidates.append(p.parent.parent / "calls" / (stem + ".json"))
        candidates.append(Path(CORPUS_CALLS_DIR) / (stem + ".json"))
    for c in candidates:
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------


class ASR:
    """Uniform transcription interface over gold, Whisper and manual text."""

    def __init__(
        self,
        backend: str = "auto",
        model_size: str = "small",
        language: Optional[str] = None,
        text: Optional[str] = None,
        sr: int = TARGET_SR,
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        self.requested = backend
        self.model_size = model_size
        self.language = language
        self.manual_text = text
        self.sr = sr
        self._whisper_model = None

    # -- backend resolution ----------------------------------------------

    @staticmethod
    def whisper_available() -> bool:
        """True when Whisper can be imported. Never imports it as a side effect."""
        return importlib.util.find_spec("whisper") is not None

    def resolve(self, source: Any = None, text: Optional[str] = None) -> str:
        """Which backend an actual call would use for this input."""
        if self.requested != "auto":
            return self.requested
        if self.whisper_available():
            return "whisper"
        if isinstance(source, (str, Path)) and find_sidecar(source) is not None:
            return "gold"
        return "manual" if (text or self.manual_text) else "gold"

    def set_manual_text(self, text: str) -> None:
        self.manual_text = text

    # -- transcription ----------------------------------------------------

    def transcribe(
        self,
        path_or_array: Union[str, Path, np.ndarray, None] = None,
        sr: Optional[int] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio or accept text, and return turns plus segments.

        The returned dict always has `text`, `segments`, `backend` and `turns`.
        `turns` are `schema.Turn` objects with language tags filled in, ready
        for the NER tagger to write BIO onto.
        """
        backend = self.resolve(path_or_array, text)
        if backend == "gold":
            return self._gold(path_or_array)
        if backend == "whisper":
            return self._whisper(path_or_array, sr or self.sr)
        return self._manual(text or self.manual_text)

    # -- backends ---------------------------------------------------------

    def _gold(self, path_or_array) -> Dict[str, Any]:
        if not isinstance(path_or_array, (str, Path)):
            raise ValueError("the gold backend needs a file path so it can find the sidecar JSON")
        side = find_sidecar(path_or_array)
        if side is None:
            raise FileNotFoundError(f"no sidecar Call JSON next to {path_or_array}")
        with open(side, "r", encoding="utf-8") as fh:
            call = Call.from_dict(json.load(fh))
        segments = [
            {"start": float(t.t_start), "end": float(t.t_end), "text": t.text,
             "speaker": t.speaker, "index": int(t.index)}
            for t in call.turns
        ]
        return {
            "text": call.full_text(),
            "segments": segments,
            "backend": "gold",
            "turns": call.turns,
            "call_id": call.call_id,
            "source": str(side),
        }

    def _load_whisper(self):
        if self._whisper_model is None:
            import whisper  # imported here on purpose, see the module docstring

            self._whisper_model = whisper.load_model(self.model_size)
        return self._whisper_model

    def _whisper(self, path_or_array, sr: int) -> Dict[str, Any]:
        model = self._load_whisper()
        audio: Any
        if isinstance(path_or_array, (str, Path)):
            audio = str(path_or_array)
        else:
            x = np.asarray(path_or_array, dtype=np.float32)
            if sr != WIDEBAND_SR:
                from ..audioio import resample

                x = resample(x, sr, WIDEBAND_SR).astype(np.float32)
            audio = x
        result = model.transcribe(audio, language=self.language, fp16=False)
        segments = [
            {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)),
             "text": str(s.get("text", "")).strip(), "speaker": "caller", "index": i}
            for i, s in enumerate(result.get("segments", []))
        ]
        # Whisper gives no diarisation, so every segment is attributed to the
        # caller and the callee side stays empty. Anything that depends on
        # speaker roles has to be told the transcript came from here.
        return {
            "text": str(result.get("text", "")).strip(),
            "segments": segments,
            "backend": "whisper",
            "turns": self._to_turns(segments),
            "language": result.get("language"),
        }

    def _manual(self, text: Optional[str]) -> Dict[str, Any]:
        if not text:
            return {"text": "", "segments": [], "backend": "manual", "turns": []}
        segments: List[Dict[str, Any]] = []
        t = 0.0
        for i, raw in enumerate([l.strip() for l in text.splitlines() if l.strip()]):
            speaker = "caller"
            body = raw
            if ":" in raw[:12]:
                head, rest = raw.split(":", 1)
                if head.strip().lower() in ("caller", "callee", "agent", "victim", "user"):
                    speaker = "callee" if head.strip().lower() in ("callee", "victim", "user") else "caller"
                    body = rest.strip()
            dur = 1.0 + 0.3 * len(tokenize(body))
            segments.append({"start": t, "end": t + dur, "text": body,
                             "speaker": speaker, "index": i})
            t += dur + 0.3
        return {
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
            "backend": "manual",
            "turns": self._to_turns(segments),
        }

    @staticmethod
    def _to_turns(segments: Sequence[Dict[str, Any]]) -> List[Turn]:
        turns: List[Turn] = []
        for i, s in enumerate(segments):
            toks = tokenize(s["text"])
            turns.append(Turn(
                index=i, speaker=s.get("speaker", "caller"), text=s["text"],
                tokens=toks, bio=["O"] * len(toks), lang=tag_languages(toks),
                t_start=float(s.get("start", 0.0)), t_end=float(s.get("end", 0.0)),
            ))
        return turns

    def to_call(self, result: Dict[str, Any], call_id: str = "uploaded") -> Call:
        """Wrap a transcription result in a Call so the pipeline can consume it."""
        return Call(
            call_id=result.get("call_id", call_id),
            turns=result["turns"],
            meta={"transcript_source": result["backend"]},
        )


if __name__ == "__main__":
    import tempfile

    from .ner_crf import demo_calls

    call = demo_calls(n_repeat=1)[0]
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio" / f"{call.call_id}_g711u.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"not really a wav, the gold backend never opens it")
        (Path(tmp) / "calls").mkdir(parents=True, exist_ok=True)
        call.save(str(Path(tmp) / "calls" / f"{call.call_id}.json"))

        asr = ASR(backend="gold")
        res = asr.transcribe(str(audio))
        print(f"backend={res['backend']}  call={res['call_id']}  "
              f"{len(res['turns'])} turns, {len(res['text'].split())} words")
        print("  first segment:", res["segments"][0])

    manual = ASR(backend="manual").transcribe(
        text="caller: sir aapka account band ho jayega\ncallee: kya baat hai"
    )
    print(f"manual: {len(manual['turns'])} turns, speakers="
          f"{[t.speaker for t in manual['turns']]}")

    print("whisper importable:", ASR.whisper_available())
    print("auto resolves to  :", ASR(backend='auto').resolve(None, text="hello"))

    ref = "aapka account 10 minute me band ho jayega"
    print("\nWER checks")
    print("  identical      :", word_error_rate(ref, ref))
    print("  total mismatch :", word_error_rate("ek do teen", "seven eight nine"))
    print("  one sub        :", round(word_error_rate(ref, ref.replace("band", "blocked")), 3))
    noisy = "apka akaunt 10 minute me bandh ho jayega"
    print("  noisy asr      :", round(word_error_rate(ref, noisy), 3),
          " hinglish-normalised:", round(word_error_rate(ref, noisy, hinglish=True), 3))
    print("  aggregate      :", aggregate_wer([ref, "otp bataiye"], [noisy, "otp bataiye"]))
    assert word_error_rate(ref, ref) == 0.0
    assert word_error_rate("ek do teen", "seven eight nine") == 1.0
