"""Text normalisation for romanised Hinglish transcripts.

Romanised Hindi has no spelling standard. The same word arrives as kripya,
kripaya or kripayaa depending on who typed it or which ASR produced it, so a
dictionary look-up on the raw surface form misses roughly as often as it hits.
Everything downstream (lexicon matching, TF-IDF, CRF features) goes through the
canonical form defined in `corpus.lexicon` so that the variation is collapsed
once, in one place.

Two entry points, and the difference between them matters:

* `normalize_text(s)` works on a whole string and may change the number of
  tokens (it glues dotted acronyms such as "o.t.p." back into "otp"). Use it
  for bag-of-words style work where no per-token labels are attached.
* `normalize_tokens(tokens)` maps token to token and never changes the length,
  so BIO tags and language tags stay aligned. Use it anywhere a sequence label
  exists.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

from ..corpus.lexicon import HINGLISH_VARIANTS, canonical
from ..schema import tokenize

__all__ = [
    "HINGLISH_VARIANTS",
    "normalize_text",
    "normalize_tokens",
    "normalize_token",
    "strip_diacritics",
    "collapse_repeats",
    "normalize_for_match",
]

# Curly quotes, long dashes and non-breaking spaces come in from copy-pasted
# transcripts. Built with chr() so this source file stays plain ASCII.
_PUNCT_MAP: Dict[str, str] = {
    chr(0x2018): "'", chr(0x2019): "'", chr(0x201B): "'",   # curly single quotes
    chr(0x201C): '"', chr(0x201D): '"',                       # curly double quotes
    chr(0x2013): "-", chr(0x2014): "-", chr(0x2212): "-",    # long dashes, minus sign
    chr(0x2026): "...", chr(0x00A0): " ", chr(0x200B): "",   # ellipsis, nbsp, zero width
    chr(0x0964): ".",                                         # devanagari danda
}
_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT_MAP))

_COMBINING_RE = re.compile("[" + chr(0x0300) + "-" + chr(0x036F) + "]")
_WS_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")

# "o.t.p." / "k.y.c" / "u p i" written out letter by letter. ASR and hand
# transcripts both do this to acronyms, and the gazetteers only hold the glued
# form, so the letters get put back together before any look-up.
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\s*\.\s*){2,}[a-z]\b")
_SPACED_ACRONYM_RE = re.compile(r"\b(?:o\s+t\s+p|u\s+p\s+i|k\s+y\s+c|c\s+v\s+v|p\s+a\s+n|a\s+t\s+m)\b")

#: Digit separators that split a single OTP or account number into pieces.
#: "4 4 5 5 6 6" is one entity, not six, and the tokeniser would otherwise hand
#: the CRF six unrelated single-digit tokens.
_SPACED_DIGITS_RE = re.compile(r"\b\d(?:[ -]\d){3,}\b")


def strip_diacritics(s: str) -> str:
    """Drop combining accents, keeping the base letters.

    Transcribers sometimes mark long vowels with a macron on romanised Hindi.
    The dictionaries are written without them, so the marks are dropped rather
    than treated as different characters.
    """
    decomposed = unicodedata.normalize("NFKD", s)
    return _COMBINING_RE.sub("", decomposed)


def collapse_repeats(s: str) -> str:
    """turaaaant -> turaant. Elongation is emphasis, not a different word."""
    return _REPEAT_RE.sub(r"\1\1", s)


def normalize_token(token: str) -> str:
    """Canonical form of one token. Length of the token list never changes."""
    t = strip_diacritics(token).lower().strip()
    if not t:
        return t
    if t.isdigit():
        return t
    return canonical(collapse_repeats(t))


def normalize_tokens(tokens: Iterable[str]) -> List[str]:
    """Alignment-safe normalisation: one output token per input token."""
    return [normalize_token(t) for t in tokens]


def normalize_text(s: str) -> str:
    """Lowercase, unify spellings, strip diacritics, tidy punctuation.

    May merge tokens (dotted acronyms, spaced-out digit runs), so it is not
    safe on text that already carries per-token labels. See `normalize_tokens`
    for that case.
    """
    if not s:
        return ""
    s = strip_diacritics(s)
    s = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group(0)], s)
    s = s.lower()
    s = _DOTTED_ACRONYM_RE.sub(lambda m: re.sub(r"[\s.]", "", m.group(0)), s)
    s = _SPACED_ACRONYM_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), s)
    s = _SPACED_DIGITS_RE.sub(lambda m: re.sub(r"[\s-]", "", m.group(0)), s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    # Per-word canonicalisation last, so the acronym and digit repairs above
    # have already run and the dictionary sees whole words.
    parts = s.split(" ")
    out = []
    for p in parts:
        head = p.rstrip(".,!?;:")
        tail = p[len(head):]
        out.append(canonical(collapse_repeats(head)) + tail if head else p)
    return " ".join(out)


def normalize_for_match(s: str) -> str:
    """Normalised text with punctuation removed, for phrase look-ups.

    `lexicon.lexicon_hits` matches multi-word terms against a plain string, and
    a comma sitting inside "account, block ho jayega" would otherwise hide the
    phrase.
    """
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return _WS_RE.sub(" ", s).strip()


if __name__ == "__main__":
    samples = [
        "Namaste! Aapka KYC pending hai, kripayaa turaaaant O.T.P. batayiye.",
        "Sir main C.B.I. se bol raha hoon, OTP 4 4 5 5 6 6 abhi bataiye",
        "Hum kabhi OTP nahin maangte, aap chaahe to branch aakar mil sakte hain.",
    ]
    for s in samples:
        print("raw :", s)
        print("norm:", normalize_text(s))
        toks = tokenize(s)
        print("tok :", len(toks), "->", len(normalize_tokens(toks)), normalize_tokens(toks)[:12])
        print()
    assert len(normalize_tokens(tokenize(samples[0]))) == len(tokenize(samples[0]))
    print("token-count preservation of normalize_tokens: ok")
