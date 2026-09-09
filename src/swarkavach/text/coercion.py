"""Dialogue-act structure and the coercion trajectory.

This is the structural half of the intent branch. Word-level models ask what
was said; this module asks in what order, which is where a scripted fraud call
gives itself away.

A social-engineering script is a ladder. It opens indistinguishably from a real
call (GREET, IDENTIFY_SELF), states a problem, asserts authority, attaches a
consequence to a clock, isolates the victim, and only then asks for the thing it
came for. A genuine bank reminder or delivery call stays flat: it informs,
confirms and closes. `schema.COERCION_RANK` puts a number on each act, and the
slope of that number across the caller's turns is the ramp. Peak pressure alone
would fire on any call that mentions a deadline; the slope only fires when the
pressure was built.

Four features come out of here, and all four are in the `cross` group of
`schema.FUSION_FEATURES`:

* `coercion_slope`, the fitted rise in coercive rank per turn,
* `coercion_peak`, the highest rank reached,
* `act_scam_llr`, how much more likely the act sequence is under a
  scam-trained first-order Markov model than under a benign-trained one,
* `callee_resist`, the share of the callee's turns spent questioning or
  pushing back. A pressured victim argues, and that argument is evidence about
  the caller even though the victim produced it.

The act classifier is rules over the same lexicon and construction patterns the
rule-based intent baseline uses, so an act label and its intent evidence never
disagree about what a turn contained.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import Call, COERCION_RANK, DIALOGUE_ACTS, Turn, tokenize
from .intent_rules import RuleIntent

__all__ = [
    "act_classifier",
    "predict_acts",
    "coercion_features",
    "ActHMM",
    "evaluate_act_classifier",
    "set_act_hmm",
    "COERCION_FEATURE_KEYS",
]

COERCION_FEATURE_KEYS = ("coercion_slope", "coercion_peak", "act_scam_llr", "callee_resist")

_RULES = RuleIntent()

# -- surface cues -----------------------------------------------------------

#: "hi" is deliberately absent: in Hinglish it is the emphatic particle in
#: "aaj hi", not the English greeting, and including it turned every sentence
#: with emphasis into a GREET.
_GREET = {"namaste", "namaskar", "hello", "hey", "good", "morning",
          "evening", "afternoon", "badhai", "salaam"}
_CLOSE = {"dhanyavaad", "dhanyawad", "shukriya", "thanks", "thank", "bye",
          "alvida", "khyal", "khayal", "milte", "shubh", "chaliye", "chalo",
          "rakhta", "rakhti", "rakhte"}
_CLOSE_PHRASES = (("ho", "gaya", "ji"), ("kaam", "ho", "gaya"), ("bas", "itna"),
                  ("good", "day"), ("din", "achha"), ("rakhta", "hoon"),
                  ("rakhti", "hoon"), ("phone", "rakh"), ("baat", "karte", "hain"))
_SELF_ID = {"bol", "bolta", "bolti", "calling", "speaking", "department",
            "vibhag", "care", "support", "executive", "partner", "team",
            "desk", "office", "wala", "wali", "officer", "inspector",
            "engineer", "manager", "agent", "representative"}
_SELF_ID_PHRASES = (("bol", "raha", "hoon"), ("bol", "rahi", "hoon"),
                    ("baat", "kar", "raha"), ("baat", "kar", "rahi"),
                    ("call", "kar", "raha"), ("call", "kar", "rahi"),
                    ("call", "kiya", "hai"), ("bol", "rahe", "hain"))
_FIRST_PERSON = {"main", "mai", "hum", "ham", "i", "we", "mera", "meri", "hamara"}
#: Institutions a caller can claim or invoke. Naming one while introducing
#: yourself is IDENTIFY_SELF; naming one to lend weight to an instruction
#: ("ye RBI ka order hai") is AUTHORITY_ASSERT.
_AUTHORITY_WORDS = {"police", "cbi", "rbi", "trai", "ncb", "customs", "interpol",
                    "adalat", "magistrate", "inspector", "commissioner", "thana",
                    "sarkar", "sarkari", "narcotics", "enforcement", "cyber",
                    "crime", "vibhag", "court", "station"}
_PROBLEM = {"problem", "dikkat", "pareshani", "samasya", "issue", "error", "pending",
            "galat", "suspicious", "atka", "atki", "fail", "failed", "hold",
            "complaint", "mismatch", "expired", "overdue", "illegal", "unpaid",
            "defaulter", "bakaya", "notice", "reject", "kharab", "toot",
            "accident", "jhagde", "trafficking", "charge", "formality"}
_PROBLEM_PHRASES = (("rok", "liya"), ("try", "hua"), ("valid", "nahi"),
                    ("update", "nahi"), ("allowed", "nahi"), ("mila", "hai"),
                    ("naam", "aa", "gaya"), ("use", "ho", "rahi"))
#: Reassurance is a small set of explicit comfort moves plus the benign
#: markers from the lexicon ("aap chahe to", "aapki marzi", "koi jaldi nahi").
#: A statement that merely happens to be harmless is INFORM, not REASSURE.
_REASSURE = {"chinta", "tension", "fikar", "worry", "ghabraiye", "ghabrao",
             "ghabra", "aaram", "marzi", "refundable", "bharosa", "surakshit"}
_REASSURE_PHRASES = (("aap", "chahe", "to"), ("koi", "jaldi", "nahi"),
                     ("jaldi", "ki", "baat", "nahi"), ("koi", "zaroorat", "nahi"),
                     ("main", "hoon", "na"), ("nahi", "karna", "hai"),
                     ("nahi", "bhejna", "hai"), ("jaana", "nahi", "hai"),
                     ("bilkul", "safe"), ("paisa", "safe"))
_CONFIRM = {"confirm", "confirmation", "sahi", "correct", "verify", "check",
            "sure", "right", "thik"}
_SMALLTALK = {"kaise", "kaisa", "kaisi", "ghar", "bacche", "bacchon", "khana",
              "tabiyat", "family", "chai", "weekend", "garmi", "barish",
              "result", "padhai", "homework"}
_INSTRUCT = {"karo", "kijiye", "kariye", "click", "install", "download", "open",
             "dabao", "dabaiye", "press", "jao", "jaiye", "aao", "aaiye", "scan",
             "transfer", "bhejo", "bhejiye", "pay", "follow", "type", "enter",
             "lijiye", "lijiyega", "dijiye", "dijiyega", "rakhiye", "rakhiyega",
             "dikhaiye", "bhariye", "kholiye", "khol", "daaliye", "dial",
             "likhwa", "delete", "padhkar", "note", "jaakar", "sunaiye"}
_URGENT = {"turant", "abhi", "jaldi", "foran", "fauran", "immediately", "urgent",
           "urgently", "quickly", "hurry", "asap", "last", "aakhri", "khatam",
           "nikal", "waqt", "time"}
#: Escalation is pressure with no new content: repeating the ask, counting down
#: the speaker's own patience, or dangling the loss of the offer. The urgency
#: table supplies the lexical half through its high-weight terms.
_ESCALATE_PHRASES = (("last", "warning"), ("last", "chance"), ("decide", "nahi"),
                     ("decide", "kijiye"), ("soch", "lijiye"), ("soch", "rahe"),
                     ("soch", "kya"), ("wait", "kar"), ("kitna", "wait"),
                     ("dobara", "nahi"), ("samajh", "nahi", "rahe"),
                     ("mat", "badalna"), ("reject", "kar"), ("close", "kar"),
                     ("shift", "khatam"), ("roz", "nahi"), ("chakkar", "kaatne"),
                     ("teesri", "baar"), ("mujhe", "mat"), ("jaldi", "kijiye"),
                     ("jaldi", "karo"), ("jaldi", "kariye"), ("jaldi", "kar"))
#: Terms in the threat table that name an institution rather than a
#: consequence. "main police se bol raha hoon" is an authority claim, not a
#: threat, and treating it as one puts the opening turn of a digital-arrest
#: script two rungs too high on the coercion ladder.
_WEAK_THREAT_TERMS = {"police", "court", "adalat", "case", "action", "fine"}
#: Units that turn a number into a deadline. A DEADLINE act needs one of these
#: ("10 minute ke andar", "aaj raat 12 baje tak"); impatience on its own is
#: PRESSURE_ESCALATE and a consequence on its own is THREAT.
_CLOCK_UNITS = {"minute", "minutes", "ghante", "ghanta", "din", "hafte", "hour",
                "hours", "baje", "second", "seconds", "days"}
_CLOCK_WORDS = {"aaj", "kal", "raat", "subah", "shaam", "today", "tonight",
                "tomorrow", "deadline", "midnight"}

_WH = {"kya", "kyun", "kyu", "kyon", "kaise", "kaun", "kahan", "kitna", "kitne",
       "kab", "kis", "kisme", "kisko", "kaunsa", "sach", "matlab",
       "what", "why", "how", "who", "where", "when", "which"}
#: Resistance cues are deliberately narrow on single words. A bare "nahi" is
#: the most common word in the corpus and appears in agreement, in disclaimers
#: and in ordinary statements, so most pushback is recognised by phrase. Real
#: victims rarely refuse outright: they stall, defer to a relative, or promise
#: to call back, and those hedges are the resistance that matters.
_RESIST = {"galat", "jhoot", "jhut", "fraud", "thagi", "thane", "police",
           "complaint", "bharosa", "bakwas", "nakli", "fake", "katunga",
           "kaatunga", "shikayat", "cybercrime"}
_RESIST_PHRASES = (("bhar", "diya"), ("kar", "diya"), ("de", "diya"),
                   ("pehle", "hi"), ("kyu", "batau"), ("nahi", "dunga"),
                   ("nahi", "batauga"), ("nahi", "bataunga"), ("nahi", "manta"),
                   ("nahi", "karunga"), ("yakeen", "nahi"), ("believe", "you"),
                   ("rehne", "dijiye"), ("wapas", "call"), ("baad", "me", "call"),
                   ("pooch", "leta"), ("pooch", "lunga"), ("pooch", "leti"),
                   ("interest", "nahi"), ("nahi", "chahiye"), ("phir", "kabhi"),
                   ("nahi", "lag"), ("khud", "jaake"), ("khud", "pata"),
                   ("abhi", "nahi"))
#: Backchannels: an acknowledgement particle plus a stock continuation, with no
#: new information. The corpus labels these CONFIRM, and they are the single
#: most frequent callee act, so separating them from real compliance and from
#: real questions matters more than any other callee distinction.
_ACK_OPENERS = {"thik", "theek", "haan", "han", "ok", "okay", "yes", "acha",
                "accha", "achha", "bilkul", "sure", "ji", "hmm", "hm"}
_BACKCHANNEL = {"boliye", "bolo", "bol", "sunai", "samajh", "dekh", "note",
                "wahi", "suniye", "rahe", "leta", "lunga", "leti"}
_COMPLY_PHRASES = (("kar", "deta", "hoon"), ("kar", "deti", "hoon"),
                   ("karta", "hoon"), ("karti", "hoon"), ("likh", "lijiye"),
                   ("likha", "hai"), ("aaya", "hai"), ("aa", "gaya"),
                   ("correct", "hai"), ("wahi", "hai", "jo"), ("bhej", "raha"),
                   ("bhej", "diya"), ("kar", "diya", "hai"))


def _has(tokens: Sequence[str], vocab: set) -> bool:
    return any(t in vocab for t in tokens)


def _has_phrase(tokens: Sequence[str], phrases: Sequence[Tuple[str, ...]]) -> bool:
    n = len(tokens)
    for ph in phrases:
        k = len(ph)
        for i in range(n - k + 1):
            if tuple(tokens[i:i + k]) == ph:
                return True
    return False


# -- act classifier ---------------------------------------------------------


def _has_clock(low: Sequence[str]) -> bool:
    """True when the turn names a time bound, not merely a hurry.

    "10 minute ke andar" and "aaj raat 12 baje tak" bind a consequence to a
    clock. "jaldi kijiye" does not, and the difference is what separates
    DEADLINE from PRESSURE_ESCALATE on the coercion ladder.
    """
    for i, t in enumerate(low):
        if t in _CLOCK_UNITS:
            return True
        if t.isdigit() and i + 1 < len(low) and low[i + 1] in _CLOCK_UNITS:
            return True
    return _has(low, _CLOCK_WORDS) and _has(low, {"tak", "andar", "pehle", "baad", "ko"})


def act_classifier(turn: Turn) -> str:
    """Predict one dialogue act from `schema.DIALOGUE_ACTS`.

    Rules, not a model, for two reasons: there is no act-annotated Hinglish
    telephony corpus to train on, and the act label goes into the evidence
    panel, where "why did you call this a THREAT" has to have an answer a person
    can read.

    Caller rules are ordered by pressure, because the strongest construction in
    a turn is the one the victim experiences. A sentence that both threatens and
    asks for an OTP is a REQUEST_SENSITIVE turn wrapped in a threat, and the
    request is the part that costs money.

    Four distinctions do most of the work:

    * DEADLINE needs a clock; a threat surrounded only by impatience is
      PRESSURE_ESCALATE, and a bare consequence is THREAT.
    * Introducing yourself is IDENTIFY_SELF whoever you claim to be. Invoking
      an institution without introducing yourself ("ye RBI ka order hai") is
      AUTHORITY_ASSERT: the first is how every call opens, the second is a move.
    * A callee backchannel ("achha achha, boliye") is CONFIRM. It looks like
      compliance and often ends in a question mark, but it hands over nothing,
      and it is the most frequent callee turn in any call.
    * Victims mostly resist by stalling, not refusing: "ek minute, bete se
      pooch leta hoon" is VICTIM_RESIST even though every word is polite.
    """
    tokens = list(turn.tokens) if turn.tokens else tokenize(turn.text or "")
    low = [t.lower() for t in tokens]
    if not low:
        return "INFORM"

    if turn.speaker == "callee":
        if _has(low, _RESIST) or _has_phrase(low, _RESIST_PHRASES):
            return "VICTIM_RESIST"
        if any(t.isdigit() for t in low) or _has_phrase(low, _COMPLY_PHRASES):
            return "VICTIM_COMPLY"
        # An acknowledgement anywhere in the opening two tokens, since "ji" and
        # "hmm" routinely precede the real particle ("ji, thik hai").
        ack = bool(set(low[:2]) & _ACK_OPENERS)
        if ack and len(low) <= 11 and (_has(low, _BACKCHANNEL) or len(low) <= 7):
            return "CONFIRM"
        if "?" in low or _has(low, _WH):
            return "VICTIM_QUESTION"
        if ack:
            return "VICTIM_COMPLY"
        if _has(low[:2], _GREET):
            return "GREET"
        return "INFORM"

    hits = _RULES.explain(tokens)
    kinds = {h["type"] for h in hits}
    threatening = any(
        h["type"] in ("threat", "conditional_threat") and h["term"] not in _WEAK_THREAT_TERMS
        for h in hits
    )
    # High-weight urgency terms only: "abhi" is ordinary speech, "last chance"
    # and "turant" are the escalation itself.
    hard_urgency = any(h["type"] == "urgency" and h["weight"] >= 0.85 for h in hits)
    clock = _has_clock(low)

    # High-pressure constructions first.
    if "secrecy_command" in kinds or "isolation" in kinds:
        return "ISOLATE"
    if "imperative_sensitive" in kinds:
        return "REQUEST_SENSITIVE"
    if threatening and clock:
        return "DEADLINE"
    if clock and _has(low, {"warna", "varna", "otherwise"}):
        return "DEADLINE"
    if _has_phrase(low, _ESCALATE_PHRASES):
        return "PRESSURE_ESCALATE"
    if threatening and hard_urgency:
        return "PRESSURE_ESCALATE"
    if threatening:
        return "THREAT"
    if hard_urgency and _has(low, _INSTRUCT):
        return "PRESSURE_ESCALATE"

    # Then the ordinary business of a phone call.
    # Hindi drops the subject, so "Meena bol rahi hoon, Metro Bank se" is a
    # self-introduction with no pronoun in it at all. The speaking phrase alone
    # is enough; a pronoun plus a role word is the other way in.
    named_self = any(
        t in ("main", "mai") and i + 1 < len(tokens) and tokens[i + 1][:1].isupper()
        for i, t in enumerate(low[:4])
    )
    self_intro = (
        _has_phrase(low, _SELF_ID_PHRASES)
        or named_self
        or (_has(low, _FIRST_PERSON) and _has(low, _SELF_ID))
    )
    if "authority_copula" in kinds or _has(low, _AUTHORITY_WORDS):
        return "IDENTIFY_SELF" if self_intro else "AUTHORITY_ASSERT"
    if _has(low[:3], _CLOSE) or _has_phrase(low, _CLOSE_PHRASES) or (
        _has(low, _CLOSE) and turn.index > 2
    ):
        return "CLOSE"
    if _has(low[:2], _GREET) and turn.index <= 1:
        return "GREET"
    if self_intro and turn.index <= 4:
        return "IDENTIFY_SELF"
    if _has(low, _REASSURE) or _has_phrase(low, _REASSURE_PHRASES):
        return "REASSURE"
    if _has(low, _INSTRUCT):
        return "INSTRUCT"
    if _has(low, _PROBLEM) or _has_phrase(low, _PROBLEM_PHRASES):
        return "PROBLEM_STATE"
    if _has(low, _SMALLTALK):
        return "SMALLTALK"
    if _has(low, _CONFIRM):
        return "CONFIRM"
    if low[0] in _GREET:
        return "GREET"
    return "INFORM"


def predict_acts(call: Call) -> List[str]:
    """Predicted act for every turn in a call, in order."""
    return [act_classifier(t) for t in call.turns]


def evaluate_act_classifier(calls: Sequence[Call]) -> Dict[str, Any]:
    """Accuracy of the rules against the gold acts, overall and per act."""
    correct = total = 0
    per_act: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    confusions: Counter = Counter()
    for c in calls:
        for t in c.turns:
            gold = t.act
            pred = act_classifier(t)
            per_act[gold][1] += 1
            total += 1
            if pred == gold:
                per_act[gold][0] += 1
                correct += 1
            else:
                confusions[(gold, pred)] += 1
    return {
        "accuracy": float(correct / max(1, total)),
        "n_turns": int(total),
        "per_act": {
            a: {"accuracy": float(c / max(1, n)), "support": int(n)}
            for a, (c, n) in sorted(per_act.items())
        },
        "top_confusions": [
            {"gold": g, "pred": p, "n": int(n)} for (g, p), n in confusions.most_common(8)
        ],
    }


# -- act sequence model -----------------------------------------------------


class ActHMM:
    """First-order Markov model over dialogue acts, one per class.

    Called an HMM by the project contract; strictly it is a visible-state
    Markov chain, because the acts are observed (predicted by the rules) rather
    than latent. The quantity of interest is the same either way: how much more
    probable this act sequence is under the scam-trained chain than under the
    benign-trained one.

    Add-one smoothing throughout. With a few hundred calls most of the 18 by 18
    transition table is empty, and an unsmoothed zero would send the ratio to
    infinity on the first unseen transition.
    """

    def __init__(self, alpha: float = 1.0, use_gold: bool = True) -> None:
        self.alpha = float(alpha)
        self.use_gold = use_gold
        self.acts_: List[str] = list(DIALOGUE_ACTS)
        self.index_ = {a: i for i, a in enumerate(self.acts_)}
        self.log_pi_: Optional[np.ndarray] = None      # (2, A)
        self.log_A_: Optional[np.ndarray] = None       # (2, A, A)
        self.counts_: Dict[str, int] = {}

    def _seq(self, call: Call) -> List[str]:
        if self.use_gold and all(t.act for t in call.turns):
            return [t.act for t in call.turns]
        return predict_acts(call)

    def fit(self, calls: Sequence[Call]) -> "ActHMM":
        A = len(self.acts_)
        pi = np.full((2, A), self.alpha)
        trans = np.full((2, A, A), self.alpha)
        n_calls = [0, 0]
        for c in calls:
            k = 1 if int(c.label_scam) else 0
            seq = [a for a in self._seq(c) if a in self.index_]
            if not seq:
                continue
            n_calls[k] += 1
            pi[k, self.index_[seq[0]]] += 1.0
            for a, b in zip(seq[:-1], seq[1:]):
                trans[k, self.index_[a], self.index_[b]] += 1.0
        self.log_pi_ = np.log(pi / pi.sum(axis=1, keepdims=True))
        self.log_A_ = np.log(trans / trans.sum(axis=2, keepdims=True))
        self.counts_ = {"scam_calls": int(n_calls[1]), "benign_calls": int(n_calls[0])}
        return self

    def log_prob(self, acts: Sequence[str], scam: bool) -> float:
        if self.log_pi_ is None:
            return 0.0
        k = 1 if scam else 0
        seq = [self.index_[a] for a in acts if a in self.index_]
        if not seq:
            return 0.0
        lp = float(self.log_pi_[k, seq[0]])
        for a, b in zip(seq[:-1], seq[1:]):
            lp += float(self.log_A_[k, a, b])
        return lp

    def llr(self, acts: Sequence[str]) -> float:
        """Per-act log-likelihood ratio, scam over benign, clipped to [-10, 10].

        Dividing by the sequence length keeps a long call from outscoring a
        short one purely on length, which matters because the fusion layer sees
        calls of very different sizes.
        """
        seq = [a for a in acts if a in self.index_]
        if not seq or self.log_pi_ is None:
            return 0.0
        raw = (self.log_prob(seq, True) - self.log_prob(seq, False)) / len(seq)
        return float(np.clip(raw, -10.0, 10.0))

    def transition_table(self, scam: bool = True) -> np.ndarray:
        k = 1 if scam else 0
        return np.exp(self.log_A_[k]) if self.log_A_ is not None else np.zeros((0, 0))

    def save(self, path) -> None:
        import joblib

        joblib.dump({"alpha": self.alpha, "use_gold": self.use_gold,
                     "acts": self.acts_, "log_pi": self.log_pi_, "log_A": self.log_A_,
                     "counts": self.counts_}, path)

    @classmethod
    def load(cls, path) -> "ActHMM":
        import joblib

        d = joblib.load(path)
        m = cls(alpha=d["alpha"], use_gold=d["use_gold"])
        m.acts_ = list(d["acts"])
        m.index_ = {a: i for i, a in enumerate(m.acts_)}
        m.log_pi_ = d["log_pi"]
        m.log_A_ = d["log_A"]
        m.counts_ = d.get("counts", {})
        return m


#: Process-wide model used by `coercion_features` when no model is passed in.
_DEFAULT_HMM: Optional[ActHMM] = None


def set_act_hmm(hmm: Optional[ActHMM]) -> None:
    """Install the fitted act model that `coercion_features` should use."""
    global _DEFAULT_HMM
    _DEFAULT_HMM = hmm


# -- trajectory features ----------------------------------------------------


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of y on x. Zero when x has no spread."""
    if len(x) < 2:
        return 0.0
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    vx = float(((ax - ax.mean()) ** 2).sum())
    if vx <= 1e-12:
        return 0.0
    return float(((ax - ax.mean()) * (ay - ay.mean())).sum() / vx)


def coercion_features(
    call: Call,
    acts: Optional[Sequence[str]] = None,
    hmm: Optional[ActHMM] = None,
) -> Dict[str, float]:
    """Escalation-trajectory features for one call.

    `acts` defaults to the rule classifier's own predictions rather than the
    gold labels on the turns. That is deliberate: at inference time there are no
    gold acts, so defaulting to them would report a number the deployed system
    can never reproduce. Pass gold acts explicitly to measure the ceiling.
    """
    turns = call.turns
    if acts is None:
        acts = predict_acts(call)
    if len(acts) != len(turns):
        raise ValueError(f"{len(acts)} acts for {len(turns)} turns")

    caller_x: List[float] = []
    caller_rank: List[float] = []
    callee_acts: List[str] = []
    for t, a in zip(turns, acts):
        if t.speaker == "caller":
            caller_x.append(float(t.index))
            caller_rank.append(float(COERCION_RANK.get(a, 0.0)))
        else:
            callee_acts.append(a)

    slope = _slope(caller_x, caller_rank)
    peak = max(caller_rank) if caller_rank else 0.0
    mean = float(np.mean(caller_rank)) if caller_rank else 0.0
    resisted = sum(1 for a in callee_acts if a in ("VICTIM_RESIST", "VICTIM_QUESTION"))

    model = hmm if hmm is not None else _DEFAULT_HMM
    llr = model.llr(list(acts)) if model is not None else 0.0

    first_pressure = -1
    for t, a in zip(turns, acts):
        if t.speaker == "caller" and COERCION_RANK.get(a, 0.0) >= 0.5:
            first_pressure = int(t.index)
            break

    return {
        "coercion_slope": float(np.clip(slope, -1.0, 1.0)),
        "coercion_peak": float(peak),
        "act_scam_llr": float(llr),
        "callee_resist": float(resisted / len(callee_acts)) if callee_acts else 0.0,
        "coercion_mean": mean,
        "n_caller_turns": float(len(caller_rank)),
        "first_pressure_turn": float(first_pressure),
        "acts": list(acts),
    }


if __name__ == "__main__":
    from .ner_crf import demo_calls

    calls = demo_calls(n_repeat=4)
    train = [c for c in calls if c.split == "train"]
    test = [c for c in calls if c.split == "test"]

    ev = evaluate_act_classifier(calls)
    print(f"act classifier accuracy: {ev['accuracy']:.3f} over {ev['n_turns']} turns")
    for a, v in ev["per_act"].items():
        print(f"   {a:20s} {v['accuracy']:.2f}  n={v['support']}")
    print("top confusions:", [(c["gold"], c["pred"], c["n"]) for c in ev["top_confusions"]])

    hmm = ActHMM().fit(train)
    set_act_hmm(hmm)
    print("\nact model fitted on", hmm.counts_)

    rows = []
    for c in test:
        f = coercion_features(c)
        rows.append((c.label_scam, f))
    for label in (1, 0):
        sel = [f for l, f in rows if l == label]
        if not sel:
            continue
        name = "scam  " if label else "benign"
        print(f"{name}  slope={np.mean([f['coercion_slope'] for f in sel]):+.3f}  "
              f"peak={np.mean([f['coercion_peak'] for f in sel]):.3f}  "
              f"llr={np.mean([f['act_scam_llr'] for f in sel]):+.3f}  "
              f"resist={np.mean([f['callee_resist'] for f in sel]):.3f}   n={len(sel)}")

    scam_slope = np.mean([f["coercion_slope"] for l, f in rows if l == 1])
    benign_slope = np.mean([f["coercion_slope"] for l, f in rows if l == 0])
    print(f"\nslope separation: scam {scam_slope:+.3f} vs benign {benign_slope:+.3f}")
    assert scam_slope > benign_slope

    c = test[0]
    print(f"\nexample trajectory for {c.call_id} "
          f"({'scam' if c.label_scam else 'benign'}):")
    for t, a in zip(c.turns, predict_acts(c)):
        print(f"  {t.index} {t.speaker:6s} gold={t.act:18s} pred={a:18s} "
              f"rank={COERCION_RANK.get(a, 0.0):.2f}  {t.text[:52]}")
