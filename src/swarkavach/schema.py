"""Core data contracts for SwarKavach.

Every module in this project reads and writes these objects. They serialise to
plain JSON so the corpus on disk, the Flask API and the Colab notebooks all
speak the same language.

Nothing here imports numpy, torch or sklearn on purpose: the schema has to be
importable from a notebook before any of the heavy stack is installed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------
# Label spaces
# --------------------------------------------------------------------------

#: Fraud entity tagset. Designed for Indian voice-fraud scripts, not for
#: general-purpose NER (no PERSON / LOC / ORG). Each type answers a question a
#: fraud analyst actually asks about a call.
ENTITY_TYPES: Tuple[str, ...] = (
    "OTP",                # one-time password: mention, request or dictation
    "BANK_ENTITY",        # account, card, KYC, netbanking, cheque
    "AUTHORITY_CLAIM",    # police / CBI / RBI / TRAI / customs / court claim
    "THREAT_DEADLINE",    # a consequence bound to a clock
    "PAYMENT_HANDLE",     # UPI id, account number, payment link, wallet
    "PERSONAL_INFO_REQ",  # aadhaar, PAN, DOB, CVV, date of birth
    "MONEY_AMOUNT",       # rupee amounts
)

#: BIO label space derived from ENTITY_TYPES.
BIO_LABELS: Tuple[str, ...] = ("O",) + tuple(
    f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")
)

#: Dialogue acts. The ordering matters: COERCION_RANK below turns the act
#: sequence into the escalation trajectory that the fusion layer consumes.
DIALOGUE_ACTS: Tuple[str, ...] = (
    "GREET",
    "IDENTIFY_SELF",
    "SMALLTALK",
    "INFORM",
    "CONFIRM",
    "AUTHORITY_ASSERT",
    "PROBLEM_STATE",
    "THREAT",
    "DEADLINE",
    "ISOLATE",             # stay on the line, do not tell anyone
    "INSTRUCT",
    "REQUEST_SENSITIVE",
    "REASSURE",
    "PRESSURE_ESCALATE",
    "CLOSE",
    "VICTIM_QUESTION",
    "VICTIM_RESIST",
    "VICTIM_COMPLY",
)

#: How much coercive pressure each act carries, 0.0 to 1.0. This is the ladder
#: that the coercion-trajectory feature climbs. Hand-set from the script
#: structure described in I4C and RBI advisories, not learned, so it stays
#: interpretable in the evidence panel.
COERCION_RANK: Dict[str, float] = {
    "GREET": 0.00,
    "SMALLTALK": 0.00,
    "IDENTIFY_SELF": 0.10,
    "INFORM": 0.10,
    "CONFIRM": 0.15,
    "VICTIM_QUESTION": 0.00,
    "VICTIM_RESIST": 0.00,
    "VICTIM_COMPLY": 0.00,
    "PROBLEM_STATE": 0.35,
    "REASSURE": 0.30,
    "INSTRUCT": 0.50,
    "AUTHORITY_ASSERT": 0.55,
    "THREAT": 0.80,
    "DEADLINE": 0.85,
    "ISOLATE": 0.90,
    "REQUEST_SENSITIVE": 0.95,
    "PRESSURE_ESCALATE": 1.00,
    "CLOSE": 0.20,
}

SCAM_SCENARIOS: Tuple[str, ...] = (
    "kyc_freeze",
    "otp_theft",
    "digital_arrest",
    "lottery_prize",
    "fake_relative",
    "electricity_disconnect",
    "courier_customs",
    "job_offer",
    "loan_approval",
    "sim_block",
)

BENIGN_SCENARIOS: Tuple[str, ...] = (
    "bank_reminder",
    "delivery_otp",
    "family_call",
    "telemarketing",
    "appointment_reminder",
    "customer_support",
    "survey_call",
    "school_notice",
)

SPEAKER_ROLES: Tuple[str, ...] = ("caller", "callee")
VOICE_LABELS: Tuple[str, ...] = ("human", "synthetic")
LANG_TAGS: Tuple[str, ...] = ("hi", "en", "univ")  # univ = digits, names, punctuation

RISK_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.00, "low"),
    (0.35, "elevated"),
    (0.65, "high"),
    (0.85, "critical"),
)


def risk_band(p: float) -> str:
    """Map a fused risk probability onto the four operational bands."""
    band = "low"
    for threshold, name in RISK_BANDS:
        if p >= threshold:
            band = name
    return band


# --------------------------------------------------------------------------
# Tokenisation (shared, so BIO offsets always line up)
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+(?:'[A-Za-z]+)?|[^\sA-Za-z\d]")


def tokenize(text: str) -> List[str]:
    """The one tokeniser used across the whole project.

    Keeps numbers whole (OTP digits matter), splits punctuation off, and leaves
    romanised Hindi words alone. Every module must call this, otherwise BIO
    tags stop lining up with tokens.
    """
    return _TOKEN_RE.findall(text)


def tokenize_spans(text: str) -> List[Tuple[str, int, int]]:
    """Same tokenisation but with character offsets, for span highlighting."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


# --------------------------------------------------------------------------
# Core records
# --------------------------------------------------------------------------


@dataclass
class EntitySpan:
    """A decoded entity: which tokens, which type, which turn."""

    type: str
    text: str
    turn_index: int
    tok_start: int
    tok_end: int          # exclusive
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decode_bio(tokens: List[str], bio: List[str], turn_index: int = 0) -> List[EntitySpan]:
    """Standard BIO decoding, tolerant of a stray I- with no B- before it."""
    spans: List[EntitySpan] = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = bio[i] if i < len(bio) else "O"
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        _, etype = tag.split("-", 1)
        j = i + 1
        while j < n and j < len(bio) and bio[j] == f"I-{etype}":
            j += 1
        spans.append(
            EntitySpan(
                type=etype,
                text=" ".join(tokens[i:j]),
                turn_index=turn_index,
                tok_start=i,
                tok_end=j,
            )
        )
        i = j
    return spans


@dataclass
class Turn:
    """One utterance by one speaker."""

    index: int
    speaker: str                       # 'caller' | 'callee'
    text: str
    act: str = "INFORM"
    tokens: List[str] = field(default_factory=list)
    bio: List[str] = field(default_factory=list)      # gold or predicted BIO
    lang: List[str] = field(default_factory=list)     # per-token language tag
    t_start: float = 0.0                              # seconds into the call
    t_end: float = 0.0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(self.text)
        if not self.bio:
            self.bio = ["O"] * len(self.tokens)
        if not self.lang:
            self.lang = ["univ"] * len(self.tokens)
        self.validate()

    def validate(self) -> None:
        n = len(self.tokens)
        if len(self.bio) != n:
            raise ValueError(
                f"turn {self.index}: {len(self.bio)} BIO tags for {n} tokens"
            )
        if len(self.lang) != n:
            raise ValueError(
                f"turn {self.index}: {len(self.lang)} lang tags for {n} tokens"
            )

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    def entities(self) -> List[EntitySpan]:
        """Decode this turn's BIO sequence into entity spans."""
        return decode_bio(self.tokens, self.bio, turn_index=self.index)


@dataclass
class Call:
    """A whole call: the transcript, the labels and (optionally) the audio."""

    call_id: str
    turns: List[Turn]
    label_scam: int = 0                 # 1 = fraudulent
    label_voice: str = "human"          # 'human' | 'synthetic'
    scenario: str = "unknown"
    speaker_id: str = "spk_unknown"
    split: str = "train"                # train | dev | test
    audio_path: Optional[str] = None
    sample_rate: int = 8000
    channel: str = "clean"              # clean | g711u | g711a | gsm | amrnb | noisy
    audio_source: str = "none"          # none | sim | sapi | edge | xtts | recorded
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # -- convenience ------------------------------------------------------

    @property
    def cell(self) -> str:
        """Which of the four evaluation cells this call belongs to."""
        v = "cloned" if self.label_voice == "synthetic" else "human"
        s = "scam" if self.label_scam else "benign"
        return f"{v}x{s}"

    @property
    def duration(self) -> float:
        return self.turns[-1].t_end if self.turns else 0.0

    def caller_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.speaker == "caller"]

    def full_text(self, speaker: Optional[str] = None) -> str:
        turns = self.turns if speaker is None else [t for t in self.turns if t.speaker == speaker]
        return " ".join(t.text for t in turns)

    def entities(self) -> List[EntitySpan]:
        out: List[EntitySpan] = []
        for t in self.turns:
            out.extend(t.entities())
        return out

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Call":
        d = dict(d)
        turns = []
        for t in d.get("turns", []):
            t = dict(t)
            t = {k: v for k, v in t.items() if k in Turn.__dataclass_fields__}
            turns.append(Turn(**t))
        d["turns"] = turns
        known = set(cls.__dataclass_fields__)
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        if extra:
            d.setdefault("meta", {})
            d["meta"] = {**d["meta"], **extra}
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Call":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------
# Analysis outputs
# --------------------------------------------------------------------------


@dataclass
class Evidence:
    """Everything a human reviewer needs in order to agree or disagree."""

    reasons: List[str] = field(default_factory=list)
    contributions: List[Dict[str, Any]] = field(default_factory=list)
    # each: {feature, value, contribution, group, label}
    spans: List[Dict[str, Any]] = field(default_factory=list)
    # each: {turn_index, tok_start, tok_end, type, text, weight, source}
    audio_markers: List[Dict[str, Any]] = field(default_factory=list)
    # each: {t_start, t_end, kind, note}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BranchScore:
    """Output of one analysis branch."""

    name: str
    score: float                                  # calibrated probability
    backend: str = "unknown"
    raw: float = 0.0                              # pre-calibration score / LLR
    features: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """The final, explainable answer for one call."""

    call_id: str
    risk: float                                   # fused calibrated probability
    band: str = ""
    authenticity: float = 0.0                     # P(voice is synthetic)
    intent: float = 0.0                           # P(conversation is a scam)
    pim: float = 0.0                              # prosody-intent mismatch
    branches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    evidence: Evidence = field(default_factory=Evidence)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    # each: {turn_index, t_end, risk, intent, authenticity, pim, new_entities}
    ttd: Optional[float] = None                   # seconds until risk crossed alert
    ttd_turns: Optional[int] = None
    latency_ms: Dict[str, float] = field(default_factory=dict)
    transcript_source: str = "gold"               # gold | whisper | manual
    label_scam: Optional[int] = None              # carried through for evaluation
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.band:
            self.band = risk_band(self.risk)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Verdict":
        d = dict(d)
        ev = d.get("evidence") or {}
        if isinstance(ev, dict):
            ev = {k: v for k, v in ev.items() if k in Evidence.__dataclass_fields__}
            d["evidence"] = Evidence(**ev)
        else:
            d["evidence"] = Evidence()
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


# --------------------------------------------------------------------------
# Feature registry: the single source of truth for fusion feature names
# --------------------------------------------------------------------------

#: (name, group, human-readable label, direction hint)
#: direction is +1 when a larger value should mean "more fraudulent".
FUSION_FEATURES: Tuple[Tuple[str, str, str, int], ...] = (
    # -- voice authenticity branch ---------------------------------------
    ("as_score",          "voice",  "Synthetic-voice score",            +1),
    ("as_llr",            "voice",  "LFCC-GMM log-likelihood ratio",    +1),
    ("as_margin",         "voice",  "Detector margin",                  +1),
    ("spk_consistency",   "voice",  "Speaker-embedding consistency",    -1),
    ("jitter",            "voice",  "Pitch jitter",                     -1),
    ("shimmer",           "voice",  "Amplitude shimmer",                -1),
    ("hnr",               "voice",  "Harmonic-to-noise ratio",          +1),
    ("spec_flatness_var", "voice",  "Spectral-flatness variance",       -1),
    # -- scam intent branch ----------------------------------------------
    ("intent_score",      "intent", "Scam-intent score",                +1),
    ("intent_max_turn",   "intent", "Peak turn intent",                 +1),
    ("urgency_density",   "intent", "Urgency-word density",             +1),
    ("ent_otp",           "intent", "OTP mentions",                     +1),
    ("ent_authority",     "intent", "Authority claims",                 +1),
    ("ent_threat",        "intent", "Threat deadlines",                 +1),
    ("ent_payment",       "intent", "Payment handles",                  +1),
    ("ent_personal",      "intent", "Personal-info requests",           +1),
    ("ent_bank",          "intent", "Bank references",                  +1),
    # -- cross-modal and structural (the novel block) --------------------
    ("pim",               "cross",  "Prosody-intent mismatch",          +1),
    ("coercion_slope",    "cross",  "Coercion escalation slope",        +1),
    ("coercion_peak",     "cross",  "Peak coercive pressure",           +1),
    ("act_scam_llr",      "cross",  "Dialogue-act sequence LLR",        +1),
    ("cmi",               "cross",  "Code-mixing index",                +1),
    ("switch_entropy",    "cross",  "Code-switch entropy",              +1),
    ("ent_lang_align",    "cross",  "Entity-language alignment",        +1),
    ("callee_resist",     "cross",  "Callee resistance ratio",          +1),
)

FUSION_FEATURE_NAMES: Tuple[str, ...] = tuple(f[0] for f in FUSION_FEATURES)
FEATURE_GROUPS: Dict[str, str] = {f[0]: f[1] for f in FUSION_FEATURES}
FEATURE_LABELS: Dict[str, str] = {f[0]: f[2] for f in FUSION_FEATURES}
FEATURE_DIRECTION: Dict[str, int] = {f[0]: f[3] for f in FUSION_FEATURES}

#: Which fusion features each ablation arm is allowed to see. The fused arm
#: gets everything; the single-branch arms get only their own group. This is
#: what makes the headline ablation table an apples-to-apples comparison.
ABLATION_ARMS: Dict[str, Tuple[str, ...]] = {
    "audio_only": tuple(n for n in FUSION_FEATURE_NAMES if FEATURE_GROUPS[n] == "voice"),
    "text_only": tuple(n for n in FUSION_FEATURE_NAMES if FEATURE_GROUPS[n] == "intent"),
    "late_fusion": tuple(
        n for n in FUSION_FEATURE_NAMES if FEATURE_GROUPS[n] in ("voice", "intent")
    ),
    "full": FUSION_FEATURE_NAMES,
}


def empty_feature_dict() -> Dict[str, float]:
    return {name: 0.0 for name in FUSION_FEATURE_NAMES}


def feature_vector(d: Dict[str, float], names: Optional[Tuple[str, ...]] = None) -> List[float]:
    """Dict to ordered vector. Missing keys become 0.0, never an exception."""
    names = names or FUSION_FEATURE_NAMES
    return [float(d.get(name, 0.0)) for name in names]
