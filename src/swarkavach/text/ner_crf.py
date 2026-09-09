"""Fraud-entity NER with the hand-written linear-chain CRF.

The tagset is not general-purpose NER. There is no PERSON or LOC here, because
knowing that "Rahul" is a name tells a fraud analyst nothing. The seven types in
`schema.ENTITY_TYPES` each answer a question an analyst actually asks: was an
OTP requested, was an authority impersonated, was a consequence tied to a clock,
where would the money have gone.

Feature design follows the standard CRF NER recipe (word, shape, affixes,
gazetteer, context window), with three additions for code-mixed Hinglish:

* the canonical spelling from `corpus.lexicon`, so bataiye / bataye / batana are
  one feature instead of three,
* the per-token language tag, because the language a token is written in is
  itself evidence here (the financial nouns stay English while the coercion
  around them is Hindi),
* a coarse POS tag, which separates "block ho jayega" (verb phrase, a threat)
  from "block number" (noun phrase, not a threat) without needing a separate
  lexical feature for every context.

Scoring is entity level with exact span match, never token level. A tagger that
gets two of the three tokens in "band ho jayega" right has produced a span that
cannot be highlighted, counted or explained, so it should score zero for that
entity, and token-level accuracy would instead report a comfortable 0.9 because
most tokens are O.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS
from ..corpus.lexicon import canonical, gazetteer_type, is_digit_run, token_language, word_shape
from ..schema import BIO_LABELS, Call, ENTITY_TYPES, Turn, decode_bio, tokenize
from .crf import LinearChainCRF
from .langid import tag_languages
from .pos import pos_tag

__all__ = [
    "CRFTagger",
    "token_features",
    "sequence_features",
    "entity_prf",
    "bio_spans",
    "bio_constraints",
    "demo_calls",
]

_BOS = "__BOS__"
_EOS = "__EOS__"


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _pad(seq: Sequence[str], i: int, offset: int, empty: str) -> str:
    j = i + offset
    if j < 0:
        return _BOS
    if j >= len(seq):
        return _EOS
    return seq[j] or empty


def token_features(
    tokens: Sequence[str],
    i: int,
    pos: Sequence[str],
    langs: Sequence[str],
    canon: Sequence[str],
    gaz: Sequence[str],
) -> List[str]:
    """Active feature names for one token. All binary, all strings."""
    w = tokens[i]
    low = w.lower()
    feats = [
        "bias",
        f"w={low}",
        f"canon={canon[i]}",
        f"shape={word_shape(w)}",
        f"pre2={low[:2]}",
        f"pre3={low[:3]}",
        f"suf2={low[-2:]}",
        f"suf3={low[-3:]}",
        f"pos={pos[i]}",
        f"lang={langs[i]}",
        f"gaz={gaz[i]}",
        f"len={min(len(low), 12)}",
    ]
    if low.isdigit():
        feats.append("isdigit")
        feats.append(f"ndigits={min(len(low), 16)}")
    if is_digit_run(w):
        feats.append("isdigitrun")
    if w.isupper() and len(w) > 1:
        feats.append("allcaps")
    elif w[:1].isupper():
        feats.append("istitle")
    if not low.isalnum():
        feats.append("ispunct")
    if i == 0:
        feats.append("first")
    if i == len(tokens) - 1:
        feats.append("last")

    # Context window of plus and minus two tokens. Fraud entities are short
    # phrases whose left context does most of the work ("aapka account",
    # "otp 445566"), so the window is small and the conjunctions below carry
    # the phrase-internal evidence.
    for d in (-2, -1, 1, 2):
        feats.append(f"w[{d}]=" + _pad([t.lower() for t in tokens], i, d, ""))
        feats.append(f"pos[{d}]=" + _pad(pos, i, d, "X"))
        feats.append(f"gaz[{d}]=" + _pad(gaz, i, d, "NONE"))
        if abs(d) == 1:
            feats.append(f"canon[{d}]=" + _pad(canon, i, d, ""))
            feats.append(f"lang[{d}]=" + _pad(langs, i, d, "univ"))

    prev_w = _pad([t.lower() for t in tokens], i, -1, "")
    next_w = _pad([t.lower() for t in tokens], i, 1, "")
    feats.append(f"w[-1]|w[0]={prev_w}|{low}")
    feats.append(f"w[0]|w[1]={low}|{next_w}")
    feats.append(f"pos[-1]|pos[0]={_pad(pos, i, -1, 'X')}|{pos[i]}")
    feats.append(f"gaz[-1]|gaz[0]={_pad(gaz, i, -1, 'NONE')}|{gaz[i]}")
    return feats


def sequence_features(tokens: Sequence[str], langs: Optional[Sequence[str]] = None) -> List[List[str]]:
    """Feature lists for a whole token sequence."""
    tokens = list(tokens)
    if not tokens:
        return []
    pos = pos_tag(tokens)
    if langs is None or len(langs) != len(tokens):
        langs = tag_languages(tokens)
    canon = [canonical(t.lower()) for t in tokens]
    gaz = [gazetteer_type(t) or "NONE" for t in tokens]
    return [token_features(tokens, i, pos, langs, canon, gaz) for i in range(len(tokens))]


def bio_constraints(labels: Sequence[str]):
    """Legal BIO transitions, as boolean (trans, start, stop) arrays.

    I-X may only follow B-X or I-X, and no sequence may open on an I- tag.
    Feeding this to the CRF means the partition function sums over valid
    taggings only, so the model never spends probability mass on sequences that
    would be thrown away at decode time.
    """
    L = len(labels)
    trans = np.ones((L, L), dtype=bool)
    start = np.ones(L, dtype=bool)
    stop = np.ones(L, dtype=bool)
    for j, lj in enumerate(labels):
        if not lj.startswith("I-"):
            continue
        start[j] = False
        etype = lj[2:]
        for i, li in enumerate(labels):
            trans[i, j] = li in (f"B-{etype}", f"I-{etype}")
    return trans, start, stop


# --------------------------------------------------------------------------
# Entity-level evaluation
# --------------------------------------------------------------------------


def bio_spans(bio: Sequence[str]) -> List[Tuple[int, int, str]]:
    """(start, end, type) spans from a BIO sequence, end exclusive."""
    spans: List[Tuple[int, int, str]] = []
    i, n = 0, len(bio)
    while i < n:
        tag = bio[i]
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        etype = tag.split("-", 1)[1]
        j = i + 1
        while j < n and bio[j] == f"I-{etype}":
            j += 1
        spans.append((i, j, etype))
        i = j
    return spans


def _prf(tp: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def entity_prf(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """Micro precision, recall and F1 over exactly matched entity spans.

    A predicted span counts only when its start, end and type all match a gold
    span in the same sequence. Per-type numbers use the same rule.
    """
    tp = n_pred = n_gold = 0
    per_type: Dict[str, List[int]] = {t: [0, 0, 0] for t in ENTITY_TYPES}
    tok_correct = tok_total = 0
    for g_seq, p_seq in zip(gold, pred):
        g_spans = set((s, e, t) for s, e, t in bio_spans(g_seq))
        p_spans = set((s, e, t) for s, e, t in bio_spans(p_seq))
        tp += len(g_spans & p_spans)
        n_pred += len(p_spans)
        n_gold += len(g_spans)
        for s, e, t in g_spans:
            per_type.setdefault(t, [0, 0, 0])[2] += 1
            if (s, e, t) in p_spans:
                per_type[t][0] += 1
        for s, e, t in p_spans:
            per_type.setdefault(t, [0, 0, 0])[1] += 1
        for a, b in zip(g_seq, p_seq):
            tok_correct += int(a == b)
            tok_total += 1

    p, r, f = _prf(tp, n_pred, n_gold)
    out: Dict[str, Any] = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "support": int(n_gold),
        "n_pred": int(n_pred),
        "n_correct": int(tp),
        "token_accuracy": float(tok_correct / max(1, tok_total)),
        "n_sequences": int(len(gold)),
        "per_type": {},
    }
    for t, (c, np_, ng) in per_type.items():
        tp_, tr_, tf_ = _prf(c, np_, ng)
        out["per_type"][t] = {
            "precision": float(tp_), "recall": float(tr_), "f1": float(tf_),
            "support": int(ng), "n_pred": int(np_),
        }
    return out


# --------------------------------------------------------------------------
# Tagger
# --------------------------------------------------------------------------


class CRFTagger:
    """Fraud-entity tagger over the shared tokeniser.

    Same `fit` / `predict_turn` / `evaluate` surface as `BiLSTMTagger`, so the
    two can be dropped into the same results table without adapters.
    """

    name = "crf"

    def __init__(
        self,
        l2: float = 0.5,
        max_iter: int = 150,
        min_freq: int = 1,
        constrained: bool = True,
        verbose: bool = False,
    ) -> None:
        self.model = LinearChainCRF(
            l2=l2,
            max_iter=max_iter,
            min_freq=min_freq,
            labels=list(BIO_LABELS),
            constraints=bio_constraints if constrained else None,
            verbose=verbose,
        )
        self.fitted = False

    # -- data ------------------------------------------------------------

    @staticmethod
    def _turns(calls: Iterable[Call]) -> List[Turn]:
        out: List[Turn] = []
        for c in calls:
            for t in c.turns:
                if t.tokens:
                    out.append(t)
        return out

    def _xy(self, calls: Iterable[Call]) -> Tuple[List[List[List[str]]], List[List[str]]]:
        X, y = [], []
        for t in self._turns(calls):
            X.append(sequence_features(t.tokens, t.lang))
            y.append(list(t.bio))
        return X, y

    # -- api -------------------------------------------------------------

    def fit(self, calls: Sequence[Call]) -> "CRFTagger":
        X, y = self._xy(calls)
        if not X:
            raise ValueError("no non-empty turns to train on")
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_turn(self, tokens: Sequence[str], langs: Optional[Sequence[str]] = None) -> List[str]:
        tokens = list(tokens)
        if not tokens:
            return []
        return self.model.predict_one(sequence_features(tokens, langs))

    def predict_call(self, call: Call) -> List[List[str]]:
        return [self.predict_turn(t.tokens, t.lang) for t in call.turns]

    def tag_call(self, call: Call) -> Call:
        """Write predicted BIO onto a copy of the call, leaving gold alone."""
        import copy

        out = copy.deepcopy(call)
        for turn, bio in zip(out.turns, self.predict_call(out)):
            turn.bio = bio if bio else ["O"] * len(turn.tokens)
        return out

    def entities(self, tokens: Sequence[str], turn_index: int = 0):
        bio = self.predict_turn(tokens)
        return decode_bio(list(tokens), bio, turn_index=turn_index)

    def evaluate(self, calls: Sequence[Call]) -> Dict[str, Any]:
        turns = self._turns(calls)
        gold = [list(t.bio) for t in turns]
        pred = [self.predict_turn(t.tokens, t.lang) for t in turns]
        out = entity_prf(gold, pred)
        out["model"] = self.name
        out["n_turns"] = len(turns)
        return out

    def save(self, path) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path) -> "CRFTagger":
        tagger = cls()
        tagger.model = LinearChainCRF.load(path)
        tagger.fitted = True
        return tagger


# --------------------------------------------------------------------------
# A small hand-built corpus, used by every text module's self-test
# --------------------------------------------------------------------------

_SLOTS: Dict[str, List[str]] = {
    "name": ["rahul", "priya", "amit", "sunita", "vikram", "meena"],
    "bank": ["sbi", "hdfc", "icici", "axis", "kotak"],
    "agency": ["cyber crime branch", "cbi", "narcotics control bureau", "delhi police"],
    "otp": ["445566", "883021", "710294", "560118"],
    "amount": ["45000", "12500", "98000", "23400"],
    "minutes": ["10", "15", "30", "20"],
    "upi": ["helpdesk@ybl", "verify@okaxis", "support@paytm"],
    "brand": ["amazon", "flipkart", "myntra"],
    "city": ["delhi", "mumbai", "pune"],
}

#: (speaker, act, text template, [(entity phrase template, type), ...])
_TurnSpec = Tuple[str, str, str, List[Tuple[str, str]]]

_SCAM_SCRIPTS: List[Tuple[str, List[_TurnSpec]]] = [
    ("kyc_freeze", [
        ("caller", "GREET", "namaste {name} ji main {bank} bank se bol raha hoon", [("{bank} bank", "BANK_ENTITY")]),
        ("callee", "VICTIM_QUESTION", "ji boliye kya baat hai", []),
        ("caller", "PROBLEM_STATE", "aapka kyc pending hai isliye account me dikkat aa gayi hai",
         [("kyc", "BANK_ENTITY"), ("account", "BANK_ENTITY")]),
        ("caller", "DEADLINE", "agar {minutes} minute me update nahi hua to account band ho jayega",
         [("band ho jayega", "THREAT_DEADLINE"), ("account", "BANK_ENTITY")]),
        ("callee", "VICTIM_QUESTION", "mujhe kya karna hoga sir", []),
        ("caller", "REQUEST_SENSITIVE", "aapke mobile par otp aaya hai wo otp turant bataiye",
         [("otp", "OTP")]),
        ("callee", "VICTIM_COMPLY", "otp {otp} hai", [("otp {otp}", "OTP")]),
        ("caller", "CLOSE", "dhanyavaad line pe rahiye", []),
    ]),
    ("digital_arrest", [
        ("caller", "IDENTIFY_SELF", "main {agency} se inspector sharma bol raha hoon", [("{agency}", "AUTHORITY_CLAIM")]),
        ("caller", "PROBLEM_STATE", "aapke naam par ek illegal parcel {city} me pakda gaya hai", []),
        ("caller", "THREAT", "aapke khilaf warrant jaari ho chuka hai", [("warrant", "THREAT_DEADLINE")]),
        ("callee", "VICTIM_RESIST", "ye galat hai main thane jaunga", []),
        ("caller", "ISOLATE", "ye case confidential hai kisi ko mat bataiye aur line mat kaatiye", []),
        ("caller", "REQUEST_SENSITIVE", "verification ke liye apna aadhaar number bataiye",
         [("aadhaar number", "PERSONAL_INFO_REQ")]),
        ("caller", "INSTRUCT", "ab {amount} rupees {upi} par transfer kijiye",
         [("{amount} rupees", "MONEY_AMOUNT"), ("{upi}", "PAYMENT_HANDLE")]),
        ("caller", "PRESSURE_ESCALATE", "jaldi kijiye warna abhi giraftari ho jayegi", []),
    ]),
    ("otp_theft", [
        ("caller", "GREET", "hello main {brand} delivery support se bol raha hoon", []),
        ("caller", "PROBLEM_STATE", "aapka parcel {city} warehouse me atka hua hai", []),
        ("caller", "REQUEST_SENSITIVE", "confirm karne ke liye ek link bhej raha hoon usme card number daaliye",
         [("link", "PAYMENT_HANDLE"), ("card number", "PERSONAL_INFO_REQ")]),
        ("callee", "VICTIM_QUESTION", "kya ye safe hai", []),
        ("caller", "REASSURE", "bilkul safe hai chinta mat kijiye", []),
        ("caller", "REQUEST_SENSITIVE", "abhi jo otp aaya hai aur cvv dono bataiye",
         [("otp", "OTP"), ("cvv", "PERSONAL_INFO_REQ")]),
        ("callee", "VICTIM_COMPLY", "thik hai otp {otp} likh lijiye", [("otp {otp}", "OTP")]),
    ]),
    ("electricity_disconnect", [
        ("caller", "IDENTIFY_SELF", "main bijli vibhag ke office se bol raha hoon", []),
        ("caller", "PROBLEM_STATE", "aapka pichla bill {amount} rupees pending hai", [("{amount} rupees", "MONEY_AMOUNT")]),
        ("caller", "DEADLINE", "aaj raat {minutes} baje connection band ho jayega", [("band ho jayega", "THREAT_DEADLINE")]),
        ("callee", "VICTIM_RESIST", "maine to pura bill bhar diya tha", []),
        ("caller", "INSTRUCT", "abhi is anydesk app ko download kijiye aur screen share kijiye",
         [("anydesk", "PERSONAL_INFO_REQ"), ("screen share", "PERSONAL_INFO_REQ")]),
        ("caller", "REQUEST_SENSITIVE", "phir apna atm pin daaliye", [("atm pin", "PERSONAL_INFO_REQ")]),
        ("caller", "PRESSURE_ESCALATE", "time bahut kam hai turant kijiye", []),
    ]),
    ("lottery_prize", [
        ("caller", "GREET", "badhai ho {name} ji aapka number lucky draw me nikla hai", []),
        ("caller", "INFORM", "aapko {amount} rupees ka inaam mila hai", [("{amount} rupees", "MONEY_AMOUNT")]),
        ("callee", "VICTIM_QUESTION", "sach me kaise mila mujhe", []),
        ("caller", "INSTRUCT", "inaam lene ke liye processing fee {upi} par bhejiye",
         [("processing fee", "PAYMENT_HANDLE"), ("{upi}", "PAYMENT_HANDLE")]),
        ("caller", "REQUEST_SENSITIVE", "apna account number aur pan number bhi bataiye",
         [("account number", "PAYMENT_HANDLE"), ("pan number", "PERSONAL_INFO_REQ")]),
        ("caller", "DEADLINE", "ye offer sirf {minutes} minute ke liye hai warna cancel ho jayega", []),
    ]),
    ("sim_block", [
        ("caller", "IDENTIFY_SELF", "main trai ke customer verification department se bol raha hoon",
         [("trai", "AUTHORITY_CLAIM")]),
        ("caller", "INFORM", "aapke aadhaar par teen sim chal rahe hain", [("aadhaar", "PERSONAL_INFO_REQ")]),
        ("caller", "DEADLINE", "isliye aapka number {minutes} minute me band ho jayega",
         [("band ho jayega", "THREAT_DEADLINE")]),
        ("callee", "VICTIM_QUESTION", "phir main kya karun", []),
        ("caller", "REQUEST_SENSITIVE", "sms me jo otp aaya hai wo otp bataiye", [("otp", "OTP")]),
        ("callee", "VICTIM_COMPLY", "otp {otp} aaya hai", [("otp {otp}", "OTP")]),
        ("caller", "CLOSE", "shukriya aapka sim safe ho gaya hai", []),
    ]),
]

_BENIGN_SCRIPTS: List[Tuple[str, List[_TurnSpec]]] = [
    ("bank_reminder", [
        ("caller", "GREET", "namaste ye {bank} bank ka reminder call hai", [("{bank} bank", "BANK_ENTITY")]),
        ("caller", "INFORM", "aapke account ka statement email par bhej diya gaya hai", [("account", "BANK_ENTITY")]),
        ("caller", "REASSURE", "hum kabhi otp ya pin nahi mangte",
         [("otp", "OTP"), ("pin", "PERSONAL_INFO_REQ")]),
        ("callee", "VICTIM_QUESTION", "thik hai koi aur kaam hai kya", []),
        ("caller", "INFORM", "aap chahe to branch aakar bhi mil sakte hain koi jaldi nahi hai", []),
        ("caller", "CLOSE", "dhanyavaad aapka din shubh ho", []),
    ]),
    ("delivery_otp", [
        ("caller", "GREET", "hello main {brand} ka delivery partner bol raha hoon", []),
        ("caller", "INFORM", "aapka parcel gate par pahunch gaya hai", []),
        ("caller", "REQUEST_SENSITIVE", "delivery confirm karne ke liye otp bataiye", [("otp", "OTP")]),
        ("callee", "VICTIM_COMPLY", "haan otp {otp} hai", [("otp {otp}", "OTP")]),
        ("caller", "CLOSE", "dhanyavaad parcel de diya gaya hai", []),
    ]),
    ("family_call", [
        ("caller", "GREET", "hello {name} kaise ho", []),
        ("callee", "INFORM", "main thik hoon aap sunao", []),
        ("caller", "SMALLTALK", "ghar par sab thik hai bacche school gaye hain", []),
        ("callee", "VICTIM_COMPLY", "haan sab badhiya hai shaam ko baat karte hain", []),
        ("caller", "CLOSE", "thik hai apna khyal rakhna", []),
    ]),
    ("appointment_reminder", [
        ("caller", "GREET", "namaste main city hospital se bol rahi hoon", []),
        ("caller", "INFORM", "kal subah aapka doctor ke saath appointment hai", []),
        ("caller", "CONFIRM", "kya ye time aapke liye thik rahega", []),
        ("callee", "VICTIM_COMPLY", "haan bilkul thik hai", []),
        ("caller", "CLOSE", "dhanyavaad milte hain kal", []),
    ]),
    ("customer_support", [
        ("caller", "GREET", "namaste main {bank} bank customer care se bol raha hoon", [("{bank} bank", "BANK_ENTITY")]),
        ("callee", "INFORM", "mera netbanking password reset nahi ho raha", [("netbanking", "BANK_ENTITY")]),
        ("caller", "INFORM", "aap official website par jaakar reset kar sakte hain", []),
        ("caller", "REASSURE", "hum aapse password kabhi nahi puchenge", [("password", "PERSONAL_INFO_REQ")]),
        ("callee", "VICTIM_COMPLY", "thik hai main website par try karta hoon", []),
        ("caller", "CLOSE", "dhanyavaad aur koi madad chahiye to batayiye", []),
    ]),
    ("survey_call", [
        ("caller", "GREET", "namaste main ek survey ke liye call kar raha hoon", []),
        ("caller", "INFORM", "sirf do minute lagenge aur koi jankari nahi chahiye", []),
        ("callee", "VICTIM_QUESTION", "kis baare me survey hai", []),
        ("caller", "INFORM", "hamare service ke feedback ke baare me", []),
        ("callee", "VICTIM_COMPLY", "thik hai puchiye", []),
        ("caller", "CLOSE", "aapke samay ke liye dhanyavaad", []),
    ]),
]


def _fill(text: str, slots: Dict[str, str]) -> str:
    return text.format(**slots)


def _build_turn(index: int, speaker: str, act: str, text: str,
                ents: Sequence[Tuple[str, str]], t0: float) -> Turn:
    tokens = tokenize(text)
    low = [t.lower() for t in tokens]
    bio = ["O"] * len(tokens)
    for phrase, etype in ents:
        ptoks = [t.lower() for t in tokenize(phrase)]
        n = len(ptoks)
        placed = False
        for i in range(len(tokens) - n + 1):
            if low[i:i + n] == ptoks and all(b == "O" for b in bio[i:i + n]):
                bio[i] = f"B-{etype}"
                for j in range(i + 1, i + n):
                    bio[j] = f"I-{etype}"
                placed = True
                break
        if not placed:
            raise ValueError(f"entity {phrase!r} not found in {text!r}")
    dur = 1.2 + 0.28 * len(tokens)
    return Turn(
        index=index, speaker=speaker, text=text, act=act, tokens=tokens, bio=bio,
        lang=[token_language(t) for t in tokens], t_start=t0, t_end=t0 + dur,
    )


def demo_calls(n_repeat: int = 4, seed: Optional[int] = None) -> List[Call]:
    """A hand-written corpus for the module self-tests.

    The real corpus comes from `corpus.generator.load_corpus`. This exists so
    every text module can be run and checked on a clean checkout, before the
    generator has produced anything, and so the self-tests never depend on
    another module's output.
    """
    rng = random.Random(SETTINGS.pipeline.seed if seed is None else seed)
    calls: List[Call] = []
    scripts = [(1, s) for s in _SCAM_SCRIPTS] + [(0, s) for s in _BENIGN_SCRIPTS]
    for rep in range(n_repeat):
        for label, (scenario, spec) in scripts:
            slots = {k: rng.choice(v) for k, v in _SLOTS.items()}
            turns: List[Turn] = []
            t = 0.0
            for i, (speaker, act, text, ents) in enumerate(spec):
                filled_ents = [(_fill(p, slots), et) for p, et in ents]
                turn = _build_turn(i, speaker, act, _fill(text, slots), filled_ents, t)
                t = turn.t_end + 0.35
                turns.append(turn)
            idx = len(calls)
            calls.append(Call(
                call_id=f"demo_{idx:03d}",
                turns=turns,
                label_scam=label,
                label_voice="synthetic" if (label and rep % 2 == 0) else "human",
                scenario=scenario,
                speaker_id=f"spk_{idx % 8:02d}",
                split="train" if rep < max(1, n_repeat - 1) else "test",
            ))
    return calls


if __name__ == "__main__":
    import time
    from collections import Counter

    calls = demo_calls(n_repeat=4)
    train = [c for c in calls if c.split == "train"]
    test = [c for c in calls if c.split == "test"]
    n_ent = Counter(e.type for c in calls for e in c.entities())
    print(f"corpus: {len(calls)} calls ({len(train)} train, {len(test)} test), "
          f"{sum(len(c.turns) for c in calls)} turns")
    print("entities:", dict(n_ent))

    tagger = CRFTagger(l2=0.5, max_iter=100)
    t0 = time.time()
    tagger.fit(train)
    print(f"trained in {time.time() - t0:.1f}s   {tagger.model.history_['n_features']} features, "
          f"{tagger.model.history_['n_params']} params, {tagger.model.history_['n_iter']} iters")

    res = tagger.evaluate(test)
    print(f"entity-level  P={res['precision']:.3f}  R={res['recall']:.3f}  "
          f"F1={res['f1']:.3f}  (support {res['support']}, token acc {res['token_accuracy']:.3f})")
    for t, v in sorted(res["per_type"].items()):
        if v["support"]:
            print(f"   {t:18s} P={v['precision']:.2f} R={v['recall']:.2f} "
                  f"F1={v['f1']:.2f}  n={v['support']}")

    demo = "sir aapka account 10 minute me band ho jayega otp 445566 abhi bataiye"
    toks = tokenize(demo)
    print("\n" + " ".join(f"{w}/{b}" for w, b in zip(toks, tagger.predict_turn(toks))))
    print("spans:", [(e.type, e.text) for e in tagger.entities(toks)])
