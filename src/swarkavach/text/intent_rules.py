"""Interpretable rule baseline for scam intent.

Two things a bag of words cannot do, and both matter here.

First, evidence. A statistical classifier gives a number; a reviewer looking at
a flagged call needs to see which words caused it. `explain` returns the matched
token spans with their weights, and the dashboard highlights exactly those.

Second, construction. Fraud speech is recognisable by how words combine, not
only by which words appear. Three constructions carry most of the signal and
are invisible to unigram counts:

* a threat verb inside a short window of a time expression
  ("10 minute mein block ho jayega"). "block" and "minute" are both ordinary
  words in a genuine bank call. The pairing is not.
* an imperative aimed at the callee together with a sensitive-information noun
  ("OTP batao", "cvv bata dijiye"). Any real institution says the opposite
  sentence, and a unigram model sees the same words in both.
* an authority noun with a first-person copula ("main CBI se bol raha hoon").
  Talking about the police and claiming to be the police are different acts.

The lexical part of the score comes from the weighted tables in
`corpus.lexicon`, combined with `PRESSURE_WEIGHTS`, which includes a negative
weight for the benign markers. Without those, every genuine bank call scores as
fraud, which is the hard-negative problem this baseline exists to expose.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from ..corpus.lexicon import LEXICON_TABLES, PRESSURE_WEIGHTS, canonical
from ..schema import Call, tokenize

__all__ = ["RuleIntent", "PATTERN_WEIGHTS"]

#: How much each construction adds on top of its component words. These sit
#: above the single-word weights on purpose: the construction is the evidence.
PATTERN_WEIGHTS: Dict[str, float] = {
    "threat_near_time": 1.60,
    "imperative_sensitive": 1.90,
    "authority_copula": 1.50,
    "secrecy_command": 1.60,
    "conditional_threat": 1.20,
    # The mirror image of imperative_sensitive, and the reason a real bank
    # call does not trip this baseline: "hum kabhi OTP nahi mangte". The same
    # nouns appear, under negation, from the first person.
    "never_ask_disclaimer": -1.70,
}

_TIME_UNITS = {
    "minute", "minutes", "min", "mint", "ghanta", "ghante", "second", "seconds",
    "hour", "hours", "din", "days", "day", "baje", "baar", "hafte", "week",
}
#: A deadline needs a clock rather than mere impatience. "10 minute mein" and "aaj
#: raat" bind a consequence to a time; "jaldi kijiye" only asks for speed. The
#: two are different acts (DEADLINE against PRESSURE_ESCALATE) and mixing them
#: makes the coercion ladder meaningless.
_CLOCK_WORDS = _TIME_UNITS | {
    "aaj", "kal", "raat", "subah", "shaam", "today", "tonight", "tomorrow",
    "midnight", "deadline", "expire", "expiring", "within", "before",
}
_TIME_WORDS = _CLOCK_WORDS | {
    "abhi", "turant", "foran", "fauran", "jaldi", "immediately", "now",
    "urgently", "last", "aakhri",
}

_THREAT_WORDS = {
    "block", "blocked", "band", "bandh", "suspend", "suspended", "freeze",
    "frozen", "seize", "seal", "deactivate", "disconnect", "cancel", "terminate",
    "arrest", "arrested", "giraftar", "giraftari", "warrant", "jail", "fir",
    "raid", "penalty", "khatam", "expire", "blacklist", "nikal",
}

_IMPERATIVES = {
    "batao", "bataiye", "bataye", "batana", "bata", "bolo", "boliye",
    "bhejo", "bhejiye", "bhej", "do", "dijiye", "dijiyega", "de", "dedo",
    "daaliye", "daalo", "dalo", "likho", "likhiye", "padho", "padhiye",
    "karo", "kijiye", "kro", "share", "tell", "give", "send", "provide",
    "type", "enter", "read", "confirm", "forward", "verify", "submit",
}

#: Single tokens that name information a genuine caller never asks for.
#: "code" is here because the fraud scripts rarely say "OTP" out loud: they say
#: verification code, security code, six digit code, SMS code. The gazetteer
#: already treats those as OTP mentions and the constructions are identical.
_SENSITIVE = {
    "otp", "cvv", "pin", "mpin", "password", "aadhaar", "aadhar", "pan",
    "dob", "expiry", "passcode", "cpin", "code",
}
#: Multi-token variants of the same.
_SENSITIVE_PHRASES = (
    ("card", "number"), ("account", "number"), ("date", "of", "birth"),
    ("atm", "pin"), ("expiry", "date"), ("one", "time", "password"),
    ("full", "number"), ("card", "details"),
)

_AUTHORITY = {
    "police", "cbi", "rbi", "trai", "ncb", "customs", "interpol", "adalat",
    "court", "magistrate", "inspector", "commissioner", "thana", "sarkar",
    "vibhag", "officer", "adhikari", "cyber", "crime", "narcotics",
    "enforcement", "income", "ed",
}
_FIRST_PERSON = {"main", "mai", "mein", "hum", "ham", "i", "we", "myself"}
_SPEAK_MARKERS = {
    "bol", "bolta", "bolti", "bol_raha", "raha", "rahi", "rahe", "hoon", "hun",
    "calling", "speaking", "call", "from", "se", "hain", "am", "baat",
}
_NEGATIVE_IMPERATIVE = {"mat", "nahi", "na", "never", "dont", "not"}
_SECRECY_TARGETS = {
    "kisi", "koi", "family", "ghar", "walo", "walon", "bank", "police",
    "beta", "beti", "anyone", "anybody", "husband", "wife", "line", "phone",
    "call", "disconnect",
}


def _entries(table: Dict[str, float]) -> List[Tuple[Tuple[str, ...], Tuple[str, ...], str, float]]:
    """Table terms as (raw tokens, canonical tokens, term, weight), longest first."""
    out = []
    for term, w in table.items():
        raw = tuple(t.lower() for t in tokenize(term))
        if not raw:
            continue
        out.append((raw, tuple(canonical(t) for t in raw), term, float(w)))
    out.sort(key=lambda e: -len(e[0]))
    return out


_TABLE_ENTRIES = {name: _entries(tbl) for name, tbl in LEXICON_TABLES.items()}


def _window(spans: Sequence[int], n: int, pad: int = 0) -> Tuple[int, int]:
    lo = max(0, min(spans) - pad)
    hi = min(n, max(spans) + 1 + pad)
    return lo, hi


class RuleIntent:
    """Weighted lexicon plus construction patterns. No training, no fitting.

    `gain` and `bias` shape the logistic squashing of the raw weighted sum. The
    defaults put a single strong cue near 0.6 and a full coercion sentence
    (threat, deadline and an OTP request) above 0.95, with benign markers able
    to pull a turn back down.
    """

    name = "rules"

    def __init__(self, gain: float = 1.10, bias: float = 1.40) -> None:
        self.gain = float(gain)
        self.bias = float(bias)

    # -- matching --------------------------------------------------------

    def _lexicon_hits(self, low: Sequence[str], can: Sequence[str]) -> List[Dict[str, Any]]:
        n = len(low)
        hits: List[Dict[str, Any]] = []
        for table, entries in _TABLE_ENTRIES.items():
            table_w = PRESSURE_WEIGHTS.get(table, 0.0)
            if table_w == 0.0:
                continue
            used = [False] * n
            for raw, canon_toks, term, w in entries:
                k = len(raw)
                for i in range(n - k + 1):
                    if any(used[i:i + k]):
                        continue
                    if all(
                        low[i + j] == raw[j] or can[i + j] == canon_toks[j]
                        for j in range(k)
                    ):
                        for j in range(i, i + k):
                            used[j] = True
                        hits.append({
                            "source": "lexicon",
                            "type": table,
                            "term": term,
                            "tok_start": i,
                            "tok_end": i + k,
                            "weight": float(table_w * w),
                        })
        return hits

    def _pattern_hits(self, low: Sequence[str], can: Sequence[str]) -> List[Dict[str, Any]]:
        n = len(low)
        hits: List[Dict[str, Any]] = []
        idx = lambda pred: [i for i in range(n) if pred(i)]

        threats = idx(lambda i: low[i] in _THREAT_WORDS or can[i] in _THREAT_WORDS)
        times = idx(lambda i: low[i] in _CLOCK_WORDS or can[i] in _CLOCK_WORDS
                    or (low[i].isdigit() and i + 1 < n and low[i + 1] in _TIME_UNITS))
        imperatives = idx(lambda i: low[i] in _IMPERATIVES or can[i] in _IMPERATIVES)
        sensitive = idx(lambda i: low[i] in _SENSITIVE or can[i] in _SENSITIVE)
        for phrase in _SENSITIVE_PHRASES:
            k = len(phrase)
            for i in range(n - k + 1):
                if tuple(low[i:i + k]) == phrase:
                    sensitive.append(i)
        authority = idx(lambda i: low[i] in _AUTHORITY or can[i] in _AUTHORITY)

        # 1. threat verb near a time expression: the deadline construction
        for t in threats:
            near = [u for u in times if abs(u - t) <= 8]
            if near:
                u = min(near, key=lambda x: abs(x - t))
                lo, hi = _window([t, u], n)
                hits.append({
                    "source": "pattern", "type": "threat_near_time",
                    "term": " ".join(low[lo:hi]), "tok_start": lo, "tok_end": hi,
                    "weight": PATTERN_WEIGHTS["threat_near_time"],
                })
                break

        # 2. imperative addressed at the callee plus a sensitive noun
        for s in sensitive:
            near = [v for v in imperatives if abs(v - s) <= 5]
            if near:
                v = min(near, key=lambda x: abs(x - s))
                lo, hi = _window([s, v], n)
                hits.append({
                    "source": "pattern", "type": "imperative_sensitive",
                    "term": " ".join(low[lo:hi]), "tok_start": lo, "tok_end": hi,
                    "weight": PATTERN_WEIGHTS["imperative_sensitive"],
                })
                break

        # 3. authority noun with a first-person copula: claiming to BE the
        #    authority, as opposed to talking about one
        if authority:
            fp = idx(lambda i: low[i] in _FIRST_PERSON)
            sp = idx(lambda i: low[i] in _SPEAK_MARKERS or can[i] in _SPEAK_MARKERS)
            if fp and sp:
                a = authority[0]
                if any(f < a for f in fp) and any(s > min(fp) for s in sp):
                    lo, hi = _window([min(fp), a] + [s for s in sp if s > a][:1], n)
                    hits.append({
                        "source": "pattern", "type": "authority_copula",
                        "term": " ".join(low[lo:hi]), "tok_start": lo, "tok_end": hi,
                        "weight": PATTERN_WEIGHTS["authority_copula"],
                    })

        # 4. negative imperative aimed at the victim's support network
        for i in range(n):
            if low[i] in _NEGATIVE_IMPERATIVE:
                near = [j for j in range(max(0, i - 4), min(n, i + 5))
                        if low[j] in _SECRECY_TARGETS]
                verb = [j for j in range(i, min(n, i + 4))
                        if low[j] in _IMPERATIVES or can[j] in _IMPERATIVES]
                if near and verb:
                    lo, hi = _window([min(near), i, max(verb)], n)
                    hits.append({
                        "source": "pattern", "type": "secrecy_command",
                        "term": " ".join(low[lo:hi]), "tok_start": lo, "tok_end": hi,
                        "weight": PATTERN_WEIGHTS["secrecy_command"],
                    })
                    break

        # 5. the disclaimer: first person, negated, over a sensitive noun.
        #    "hum kabhi otp nahi mangte" contains every word an OTP request
        #    contains, so without this the strongest hard negatives in the
        #    corpus score like fraud.
        negations = idx(lambda i: low[i] in _NEGATIVE_IMPERATIVE or low[i] in
                        ("kabhi", "never", "nahin"))
        if sensitive and negations:
            fp_all = idx(lambda i: low[i] in _FIRST_PERSON)
            for sidx in sensitive:
                near_neg = [g for g in negations if abs(g - sidx) <= 5]
                if near_neg and fp_all and min(fp_all) < sidx + 4:
                    lo, hi = _window([min(fp_all), sidx, near_neg[0]], n)
                    hits.append({
                        "source": "pattern", "type": "never_ask_disclaimer",
                        "term": " ".join(low[lo:hi]), "tok_start": lo, "tok_end": hi,
                        "weight": PATTERN_WEIGHTS["never_ask_disclaimer"],
                    })
                    break

        # 6. explicit conditional: comply or else
        for i in range(n):
            if low[i] in ("warna", "varna", "otherwise", "nahi") and i + 1 < n:
                after = [j for j in range(i, min(n, i + 8)) if j in threats]
                if after:
                    lo, hi = _window([i, after[0]], n)
                    hits.append({
                        "source": "pattern", "type": "conditional_threat",
                        "term": " ".join(low[lo:hi]), "tok_start": lo, "tok_end": hi,
                        "weight": PATTERN_WEIGHTS["conditional_threat"],
                    })
                    break
        return hits

    # -- api -------------------------------------------------------------

    def explain(self, tokens: Sequence[str], turn_index: int = 0) -> List[Dict[str, Any]]:
        """Every matched span with its signed weight, strongest first.

        The dicts have the keys `schema.Evidence.spans` expects, so they can go
        straight into a verdict.
        """
        toks = list(tokens)
        low = [t.lower() for t in toks]
        can = [canonical(t) for t in low]
        hits = self._lexicon_hits(low, can) + self._pattern_hits(low, can)
        for h in hits:
            h["turn_index"] = turn_index
            h["text"] = " ".join(toks[h["tok_start"]:h["tok_end"]])
        hits.sort(key=lambda h: -abs(h["weight"]))
        return hits

    def raw_score(self, tokens: Sequence[str]) -> float:
        """Signed sum of matched weights, before squashing."""
        return float(sum(h["weight"] for h in self.explain(tokens)))

    def score_turn(self, tokens: Sequence[str]) -> float:
        """Probability-like scam score for one turn, in [0, 1]."""
        toks = list(tokens)
        if not toks:
            return 0.0
        raw = self.raw_score(toks)
        return 1.0 / (1.0 + math.exp(-self.gain * (raw - self.bias)))

    def score_text(self, text: str) -> float:
        return self.score_turn(tokenize(text))

    def score_call(self, call: Call) -> Dict[str, Any]:
        """Call-level rollup, same shape as the TF-IDF model returns.

        Only caller turns count toward the score. A victim reading an OTP back
        is evidence about the caller's pressure, not about the victim's intent,
        and pooling it in would reward the model for the wrong reason.
        """
        per_turn: List[float] = []
        spans: List[Dict[str, Any]] = []
        for t in call.turns:
            s = self.score_turn(t.tokens)
            if t.speaker == "caller":
                per_turn.append(s)
                spans.extend(self.explain(t.tokens, turn_index=t.index))
        if not per_turn:
            per_turn = [0.0]
        mean = sum(per_turn) / len(per_turn)
        peak = max(per_turn)
        return {
            "intent_score": float(0.5 * mean + 0.5 * peak),
            "intent_max_turn": float(peak),
            "intent_mean_turn": float(mean),
            "per_turn": [float(x) for x in per_turn],
            "spans": sorted(spans, key=lambda h: -abs(h["weight"]))[:25],
            "backend": self.name,
        }


if __name__ == "__main__":
    ri = RuleIntent()
    scam = [
        "sir aapka account 10 minute mein block ho jayega",
        "OTP batao warna FIR darj ho jayegi",
        "main cyber crime branch se bol raha hoon",
        "kisi ko mat bataiye aur line mat kaatiye",
        "verification ke liye apna cvv aur atm pin bataiye",
    ]
    benign = [
        "namaste ye sbi bank ka reminder call hai",
        "hum kabhi otp nahi mangte aap chahe to branch aakar mil sakte hain",
        "aapka parcel kal subah pahunch jayega",
        "kal doctor ke saath appointment hai koi jaldi nahi hai",
        "survey ke liye do minute chahiye koi jankari nahi chahiye",
    ]
    print("scam turns")
    for s in scam:
        toks = tokenize(s)
        print(f"  {ri.score_turn(toks):.3f}  {s}")
        for h in ri.explain(toks)[:3]:
            print(f"        {h['type']:22s} {h['weight']:+.2f}  {h['text']!r}")
    print("benign turns")
    for s in benign:
        toks = tokenize(s)
        print(f"  {ri.score_turn(toks):.3f}  {s}")
        for h in ri.explain(toks)[:2]:
            print(f"        {h['type']:22s} {h['weight']:+.2f}  {h['text']!r}")

    ms = sum(ri.score_text(s) for s in scam) / len(scam)
    mb = sum(ri.score_text(s) for s in benign) / len(benign)
    print(f"\nmean scam {ms:.3f} vs mean benign {mb:.3f}")
    assert ms > mb + 0.2, (ms, mb)
