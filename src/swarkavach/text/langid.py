"""Per-token language identification and code-mixing statistics.

Hindi and English share the Latin alphabet in a romanised transcript, so the
script gives nothing away and language ID has to come from vocabulary plus
orthography. `corpus.lexicon.token_language` does that part. This module turns
the resulting tag sequence into the numbers the fusion layer consumes.

Why code-mixing measurements belong in a fraud detector at all:

* Scam scripts are written, rehearsed and read out. Read speech mixes
  differently from spontaneous speech: fewer, larger monolingual blocks and a
  more regular switching rhythm. `switch_entropy` measures that rhythm.
* Indian fraud scripts put the coercive material in Hindi (band ho jayega,
  48 ghante ke andar) while the financial and technical vocabulary stays in
  English (OTP, KYC, UPI, account). `ent_lang_align` measures that split as
  the share of entity tokens carried by English.

  Read the direction carefully, because the measured result is the reverse of
  the obvious guess. On this corpus, bank, OTP, payment, authority and
  personal-information entities run 89 to 96 percent English in scam and
  benign calls alike. Only THREAT_DEADLINE breaks the pattern, at 23 percent
  English, and threat deadlines appear only in scam calls. So a scam call's
  fraud entities average LESS English (0.63) than a benign call's (0.90).
  Lower alignment means more fraudulent, not higher.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

from ..corpus.lexicon import token_language
from ..schema import tokenize

__all__ = ["tag_languages", "code_mixing_features", "LANG_FEATURE_KEYS"]

#: Keys `code_mixing_features` always returns. Callers can rely on all of them
#: existing, with 0.0 where a value is undefined (empty turn, no entities).
LANG_FEATURE_KEYS = (
    "cmi",
    "switch_points",
    "switch_entropy",
    "hi_ratio",
    "en_ratio",
    "ent_lang_align",
)


def tag_languages(tokens: Sequence[str]) -> List[str]:
    """Tag every token 'hi', 'en' or 'univ'.

    'univ' covers digits, punctuation and words that genuinely belong to both
    languages (ok, sir, ji). Keeping them separate rather than forcing a guess
    is what stops the switch-point count from turning into noise: "ok" between
    two Hindi words is not a code switch.
    """
    return [token_language(t) for t in tokens]


def _transition_entropy(seq: Sequence[str]) -> float:
    """Normalised Shannon entropy of the language-transition distribution.

    Computed over the four ordered pairs (hi hi, hi en, en hi, en en) taken
    from the sequence with universal tokens removed. Divided by log2(4) so the
    result lands in [0, 1]: 0 means one single transition type (monolingual, or
    strict alternation), 1 means all four are equally likely.
    """
    if len(seq) < 2:
        return 0.0
    counts = Counter(zip(seq[:-1], seq[1:]))
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h / 2.0          # log2(4) = 2 possible transition types


def code_mixing_features(
    tokens: Sequence[str],
    langs: Optional[Sequence[str]] = None,
    bio: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Code-mixing summary for one turn or one whole call.

    `cmi` is the Code-Mixing Index of Gamback and Das (2014):

        CMI = 1 - max_L(w_L) / (n - u)

    with n tokens, u language-independent tokens and w_L the count of tokens in
    language L. It is reported here on the [0, 1] scale rather than the usual
    percentage, so it sits on the same scale as the other fusion features. A
    monolingual turn gives exactly 0.

    `ent_lang_align` needs `bio`; without it the key is still present and 0.0.
    """
    tokens = list(tokens)
    langs = list(langs) if langs is not None else tag_languages(tokens)
    if len(langs) != len(tokens):
        raise ValueError(f"{len(langs)} language tags for {len(tokens)} tokens")

    n = len(tokens)
    out = {k: 0.0 for k in LANG_FEATURE_KEYS}
    out["n_tokens"] = float(n)
    out["univ_ratio"] = 0.0
    out["switch_rate"] = 0.0
    out["ent_tokens"] = 0.0
    if n == 0:
        return out

    counts = Counter(langs)
    n_hi, n_en, n_univ = counts.get("hi", 0), counts.get("en", 0), counts.get("univ", 0)
    out["hi_ratio"] = n_hi / n
    out["en_ratio"] = n_en / n
    out["univ_ratio"] = n_univ / n

    mixed = [l for l in langs if l in ("hi", "en")]
    denom = len(mixed)
    if denom > 0:
        out["cmi"] = 1.0 - max(n_hi, n_en) / denom

    # Switch points are counted on the sequence with universal tokens removed,
    # so a digit run inside a Hindi clause does not fake two switches.
    switches = sum(1 for a, b in zip(mixed[:-1], mixed[1:]) if a != b)
    out["switch_points"] = float(switches)
    out["switch_rate"] = switches / max(1, denom - 1)
    out["switch_entropy"] = _transition_entropy(mixed)

    if bio is not None:
        if len(bio) != n:
            raise ValueError(f"{len(bio)} BIO tags for {n} tokens")
        # Denominator counts only entity tokens that carry a language at all.
        # Digits inside an OTP span are 'univ' and say nothing about which
        # language the fraud vocabulary is riding on.
        ent_langs = [l for l, t in zip(langs, bio) if t != "O" and l in ("hi", "en")]
        out["ent_tokens"] = float(len(ent_langs))
        if ent_langs:
            out["ent_lang_align"] = sum(1 for l in ent_langs if l == "en") / len(ent_langs)
    return out


def call_code_mixing(call, speaker: Optional[str] = "caller") -> Dict[str, float]:
    """Code-mixing features pooled over a whole call.

    Concatenating the turns rather than averaging per-turn values is deliberate:
    switch statistics on a five-token turn are almost pure noise, and the
    quantity of interest is how the speaker mixes across the call.
    """
    turns = call.turns if speaker is None else [t for t in call.turns if t.speaker == speaker]
    toks: List[str] = []
    langs: List[str] = []
    bio: List[str] = []
    for t in turns:
        toks.extend(t.tokens)
        langs.extend(t.lang if len(t.lang) == len(t.tokens) else tag_languages(t.tokens))
        bio.extend(t.bio if len(t.bio) == len(t.tokens) else ["O"] * len(t.tokens))
    return code_mixing_features(toks, langs, bio)


if __name__ == "__main__":
    demo = [
        ("monolingual english", "please confirm your account number for the delivery"),
        ("monolingual hindi", "aap apna naam aur pata bataiye phir hum aage badhenge"),
        ("scam hinglish", "sir main cyber crime branch se bol raha hoon aapka account "
                          "turant block ho jayega OTP 445566 abhi batao"),
    ]
    for name, text in demo:
        toks = tokenize(text)
        langs = tag_languages(toks)
        feats = code_mixing_features(toks, langs)
        print(f"{name}")
        print("  " + " ".join(f"{t}/{l}" for t, l in zip(toks, langs)))
        print("  " + "  ".join(f"{k}={feats[k]:.3f}" for k in LANG_FEATURE_KEYS))

    # Entity-language alignment on a hand-tagged span: the English word
    # 'account' is the entity, the pressure around it is Hindi.
    toks = tokenize("aapka account turant band ho jayega")
    bio = ["O", "B-BANK_ENTITY", "O", "B-THREAT_DEADLINE", "I-THREAT_DEADLINE", "I-THREAT_DEADLINE"]
    f = code_mixing_features(toks, None, bio)
    print("\nent_lang_align on mixed span:", round(f["ent_lang_align"], 3),
          "over", int(f["ent_tokens"]), "entity tokens")

    mono = code_mixing_features(tokenize("please confirm your account number today"))
    assert mono["cmi"] == 0.0, mono
    print("CMI on monolingual input:", mono["cmi"])
