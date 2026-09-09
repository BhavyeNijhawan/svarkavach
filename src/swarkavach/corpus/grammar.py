"""Template grammar for the Hinglish fraud-call corpus.

A template is a turn skeleton with typed slots:

    ("AUTHORITY_ASSERT", "caller", "Main {AUTH} se {NAME} bol raha hoon")

A slot resolves to a surface string AND to an entity type at the same time, so
the BIO tags fall out of generation instead of being annotated by hand. When a
slot of type OTP contributes three tokens, those three tokens get
B-OTP I-OTP I-OTP and nothing else in the turn is touched. That automatic-gold
property is the point of the whole module: gold entities, gold dialogue acts
and gold token language tags come for free and can never drift out of sync
with the text.

The annotation guideline follows from that in one line: every entity mention
comes from a slot, so the literal part of a pattern never names an entity. A
stray literal "OTP" or "account" would be a silent false negative in the gold,
and it would teach a tagger that the same word is an entity in one turn and
not in the next. `literal_entity_leaks` checks for exactly that and
`validate_grammar` reports it, so the rule cannot quietly rot.

Three kinds of call shape live here:

1. Scam, hard. The pressure ramp that Indian fraud advisories describe:
   greet, identify, authority or problem, threat, deadline, isolate, instruct,
   request sensitive, escalate.
2. Scam, mild. A soft "verification call" that asks one innocent-sounding
   question and never threatens anybody. Lexically these look benign, so the
   text branch misses them and the voice branch has to carry the call.
3. Benign, flat. Greet, identify, inform, confirm, close. Several of these are
   deliberate hard negatives: a real bank reminder that talks about accounts
   and KYC, a real delivery agent who asks the customer to read out an OTP, a
   real electricity bill notice with a due date. Those carry explicit
   anti-fraud markers from lexicon.BENIGN_MARKERS where a real caller would
   say them.

Ethics: no real bank names, no real phone numbers, no real UPI handles. Every
institution, company, hospital and school in the filler tables is invented.
Regulator and police names (RBI, TRAI, CBI, customs) do appear, because
impersonating them is the fraud pattern being modelled, and they are already
part of the shared lexicon.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..schema import (
    BENIGN_SCENARIOS,
    DIALOGUE_ACTS,
    ENTITY_TYPES,
    SCAM_SCENARIOS,
    tokenize,
)

# --------------------------------------------------------------------------
# Who says what
# --------------------------------------------------------------------------

#: Default speaker for each dialogue act. An arc can override it with a
#: "ACT:callee" suffix when a scenario needs the other side to say it.
ACT_SPEAKER: Dict[str, str] = {
    "GREET": "caller",
    "IDENTIFY_SELF": "caller",
    "SMALLTALK": "callee",
    "INFORM": "caller",
    "CONFIRM": "callee",
    "AUTHORITY_ASSERT": "caller",
    "PROBLEM_STATE": "caller",
    "THREAT": "caller",
    "DEADLINE": "caller",
    "ISOLATE": "caller",
    "INSTRUCT": "caller",
    "REQUEST_SENSITIVE": "caller",
    "REASSURE": "caller",
    "PRESSURE_ESCALATE": "caller",
    "CLOSE": "caller",
    "VICTIM_QUESTION": "callee",
    "VICTIM_RESIST": "callee",
    "VICTIM_COMPLY": "callee",
}

#: Benign scenarios written to look like their scam counterparts. These are the
#: calls a text-only classifier gets wrong, and the ablation table needs them.
HARD_NEGATIVE_SCENARIOS: Tuple[str, ...] = (
    "bank_reminder",
    "delivery_otp",
    "customer_support",
    "telemarketing",
    "school_notice",
)


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotDef:
    """One fillable slot.

    `entity` is None for slots that carry no gold entity (names, fillers,
    cities) and an ENTITY_TYPES member otherwise. `sticky` slots are drawn once
    per call and reused, so the bank a caller names at turn 2 is still the bank
    they name at turn 9.
    """

    name: str
    entity: Optional[str]
    sticky: bool
    options: Tuple[str, ...]


def _slot(name: str, entity: Optional[str], sticky: bool, options: Sequence[str]) -> SlotDef:
    if entity is not None and entity not in ENTITY_TYPES:
        raise ValueError(f"slot {name}: unknown entity type {entity}")
    return SlotDef(name=name, entity=entity, sticky=sticky, options=tuple(options))


#: The caller keeps one gender for the whole call. The name and every gendered
#: verb form the caller uses come from the same draw, otherwise a call ends up
#: saying "Priya bol {RAHA} hoon", which no Hindi speaker would say. SlotBinder
#: ties the two together.
MALE_NAMES: Tuple[str, ...] = (
    "Rahul", "Amit", "Vikas", "Suresh", "Manish", "Deepak", "Rohit",
    "Sandeep", "Ashish", "Nitin",
)
FEMALE_NAMES: Tuple[str, ...] = (
    "Priya", "Sneha", "Neha", "Anjali", "Pooja", "Kavita", "Meena",
    "Shweta", "Ritu", "Divya",
)

#: Slots whose surface form is deliberately fixed, so the eight-option floor
#: in validate_grammar does not apply to them.
FIXED_FORM_SLOTS: Tuple[str, ...] = ("OTP_LIT",)

#: slot -> (masculine options, feminine options), drawn from the caller gender.
GENDERED_SLOTS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "NAME": (MALE_NAMES, FEMALE_NAMES),
    "RAHA": (("raha",), ("rahi",)),
    "DETA": (("deta",), ("deti",)),
    "KARTA": (("karta",), ("karti",)),
    "LETA": (("leta",), ("leti",)),
    "BOLTA": (("bolta",), ("bolti",)),
    "SAKTA": (("sakta",), ("sakti",)),
    "LAYA": (("laya",), ("layi",)),
}

_SLOT_LIST: Tuple[SlotDef, ...] = (
    # -- people and places, no entity ------------------------------------
    _slot("NAME", None, True, MALE_NAMES + FEMALE_NAMES),
    _slot("RAHA", None, True, ("raha", "rahi")),
    _slot("DETA", None, True, ("deta", "deti")),
    _slot("KARTA", None, True, ("karta", "karti")),
    _slot("LETA", None, True, ("leta", "leti")),
    _slot("BOLTA", None, True, ("bolta", "bolti")),
    _slot("SAKTA", None, True, ("sakta", "sakti")),
    _slot("LAYA", None, True, ("laya", "layi")),
    _slot("CUST", None, True, [
        "Sharma ji", "Verma ji", "Gupta ji", "Mehta ji", "Malhotra ji",
        "Reddy ji", "Kapoor ji", "Joshi ji", "Bansal ji", "Nair ji",
        "Chauhan ji", "Saxena ji",
    ]),
    _slot("RELATIVE", None, True, [
        "aapka beta", "aapka bhatija", "aapki beti", "aapka bhai",
        "aapka bhanja", "aapki bahan", "aapka pota", "aapka damaad",
    ]),
    _slot("CITY", None, True, [
        "Delhi", "Mumbai", "Pune", "Jaipur", "Lucknow", "Indore",
        "Nagpur", "Kanpur", "Bhopal", "Surat",
    ]),
    _slot("GREET_WORD", None, False, [
        "Hello", "Namaste", "Ji namaste", "Haan hello", "Hello ji",
        "Namaskar", "Good morning", "Good evening",
    ]),
    _slot("FILLER", None, False, [
        "haan", "achha", "dekhiye", "suniye", "ji", "arre", "matlab",
        "ek minute", "suno", "hmm",
    ]),
    _slot("ACK", None, False, [
        "haan ji", "achha", "achha ji", "hmm", "haan bolo", "ji",
        "haan haan", "achha achha",
    ]),
    _slot("EXCUSE", None, False, [
        "main abhi office me hoon", "main gaadi chala raha hoon",
        "ghar pe koi nahi hai", "network thoda kharab hai",
        "main abhi bahar hoon", "mere paas abhi phone nahi tha",
        "main thoda busy hoon", "abhi khana kha raha tha",
    ]),
    _slot("DEPT", None, False, [
        "verification department", "customer care team",
        "fraud prevention team", "risk management team", "back office",
        "head office", "operations team", "helpdesk",
    ]),
    # -- organisations, all invented -------------------------------------
    _slot("BANK", "BANK_ENTITY", True, [
        "XYZ Bank", "Metro Bank", "Sunrise Bank", "Nova Bank", "Apex Bank",
        "Bluestone Bank", "Riverline Bank", "Northwind Bank",
        "Silverline Bank", "Greenleaf Bank",
    ]),
    _slot("COMPANY", None, True, [
        "FastTrack Courier", "Skyline Logistics", "QuickShip",
        "Metro Logistics", "Speedway Couriers", "Cityline Delivery",
        "Nova Logistics", "Prime Parcels",
    ]),
    _slot("JOB_COMPANY", None, True, [
        "Nexa Solutions", "Skyline Infotech", "Orbit Technologies",
        "Bluewave Services", "Greenfield Consultancy", "Vertex Systems",
        "Pinnacle BPO", "Zenith Enterprises",
    ]),
    _slot("UTIL", None, True, [
        "City Power Board", "Metro Electricity Board", "Sunrise Power Company",
        "Northline Electricity", "Greenfield Power", "Cityline Power",
        "Nova Power Board", "Riverside Electricity",
    ]),
    _slot("HOSPITAL", None, True, [
        "City Care Hospital", "Sunrise Hospital", "Metro Nursing Home",
        "Green Valley Hospital", "Apex Medical Centre", "Riverside Clinic",
        "Lifeline Hospital", "Sanjeevani Hospital",
    ]),
    _slot("SCHOOL", None, True, [
        "Sunrise Public School", "Green Valley School", "Riverdale Academy",
        "Vidya Bhawan School", "Nova Public School", "Silver Oak School",
        "Cityline School", "Bright Future Academy",
    ]),
    _slot("DOCTOR", None, True, [
        "Dr Mehta", "Dr Rao", "Dr Sinha", "Dr Kulkarni", "Dr Bose",
        "Dr Patel", "Dr Khanna", "Dr Iyer",
    ]),
    _slot("TELCO", None, True, [
        "Nova Telecom", "Skyline Mobile", "Metro Telecom", "Bluewave Mobile",
        "Cityline Telecom", "Northwind Mobile", "Prime Cellular",
        "Riverline Telecom",
    ]),
    # -- entity-bearing slots --------------------------------------------
    _slot("BANK_ITEM", "BANK_ENTITY", False, [
        "account", "savings account", "debit card", "credit card",
        "net banking", "KYC", "KYC verification", "ATM card", "passbook",
        "cheque book",
    ]),
    _slot("AUTH", "AUTHORITY_CLAIM", True, [
        "cyber crime branch", "cyber cell", "Delhi police", "CBI",
        "narcotics control bureau", "income tax department",
        "customs department", "RBI", "TRAI", "crime branch",
        "police station", "enforcement directorate",
    ]),
    _slot("THREAT", "THREAT_DEADLINE", False, [
        "block ho jayega", "band ho jayega", "suspend ho jayega",
        "freeze ho jayega", "permanently deactivate ho jayega",
        "cancel ho jayega", "legal action liya jayega",
        "FIR darj ho jayegi", "arrest warrant nikal jayega",
        "case darj ho jayega",
    ]),
    _slot("DEADLINE", "THREAT_DEADLINE", False, [
        "aaj raat 12 baje tak", "agle 30 minute ke andar",
        "sirf 2 ghante ke andar", "aaj shaam 5 baje tak", "24 ghante ke andar",
        "kal subah tak", "agle 15 minute ke andar", "48 ghante ke andar",
        "aaj din khatam hone se pehle", "agle 10 minute ke andar",
    ]),
    _slot("WINDOW", "THREAT_DEADLINE", False, [
        "2 ghante", "30 minute", "24 ghante", "15 minute", "48 ghante",
        "aaj raat 12 baje", "kal subah", "aaj shaam 5 baje",
    ]),
    _slot("PAY", "PAYMENT_HANDLE", False, [
        "UPI id", "QR code", "payment link", "account number",
        "beneficiary account", "wallet", "UPI app", "verification link",
    ]),
    _slot("PAY_ID", "PAYMENT_HANDLE", True, [
        "help@xyzupi", "verify@metropay", "care@novapay", "support@apexupi",
        "refund@sunrisepay", "kyc@primeupi", "claim@riverpay",
        "clear@northpay",
    ]),
    _slot("PERS", "PERSONAL_INFO_REQ", False, [
        "CVV", "CVV number", "MPIN", "ATM pin", "card number",
        "expiry date", "aadhaar number", "PAN number", "date of birth",
        "mother maiden name", "AnyDesk app", "screen share",
    ]),
    _slot("ID_DOC", "PERSONAL_INFO_REQ", False, [
        "aadhaar", "aadhaar card", "aadhaar number", "PAN card", "PAN number",
        "voter id", "passport", "driving licence",
    ]),
    _slot("CARD", "BANK_ENTITY", False, [
        "card", "debit card", "credit card", "ATM card", "bank card",
        "debit wala card", "credit wala card", "ATM wala card",
    ]),
    _slot("SMALL_MONEY", "MONEY_AMOUNT", False, [
        "10 rupees", "5 rupees", "1 rupee", "2 rupees", "20 rupees",
        "10 rupaye", "5 rupaye", "15 rupees",
    ]),
    #: Fixed surface form on purpose. The benign anti-fraud markers in
    #: lexicon.BENIGN_MARKERS are exact phrases ("kabhi otp nahi"), so this one
    #: slot may not vary its wording or those phrases stop matching.
    _slot("OTP_LIT", "OTP", False, ["OTP"]),
    #: Used only by the mild scam arm. Every option has to sound like ordinary
    #: paperwork: a PIN or a CVV would give the call away immediately, which is
    #: the opposite of what the mild arm is for.
    _slot("SOFT_PERS", "PERSONAL_INFO_REQ", False, [
        "date of birth", "PAN number", "aadhaar number",
        "registered mobile number", "mother maiden name", "PAN card number",
        "aadhaar card number", "voter id number",
    ]),
    _slot("OTP_WORD", "OTP", False, [
        "OTP", "one time password", "verification code", "six digit code",
        "OTP code", "SMS code", "security code", "6 digit OTP",
    ]),
    _slot("OTP_CODE", "OTP", True, [
        "445566", "882134", "701298", "639014", "512877", "348190",
        "926745", "170362", "584129", "233908",
    ]),
    _slot("MONEY", "MONEY_AMOUNT", True, [
        "12,500 rupees", "50,000 rupees", "2 lakh rupees", "9,999 rupees",
        "85,000 rupees", "45,600 rupees", "25 lakh rupees", "7,500 rupees",
        "1 lakh 20 hazaar", "3 lakh rupees",
    ]),
    _slot("FEE", "MONEY_AMOUNT", True, [
        "999 rupees", "1,500 rupees", "2,499 rupees", "4,999 rupees",
        "750 rupees", "3,200 rupees", "1,100 rupees", "5,500 rupees",
    ]),
    # -- small neutral fillers -------------------------------------------
    _slot("TIME", None, False, [
        "kal shaam", "parso subah", "agle hafte", "is mahine ke end tak",
        "agle mahine ki 5 tarikh", "somvar ko", "budhwar subah 11 baje",
        "aaj shaam 6 baje", "agle 3 din me", "is hafte ke andar",
    ]),
    _slot("LAST4", None, False, [
        "4417", "8092", "3365", "7710", "2284", "9536", "6148", "5023",
    ]),
    _slot("ORDER_ID", None, True, [
        "771204", "339087", "824156", "556703", "190845", "443219",
        "667590", "201938",
    ]),
    _slot("TICKET", None, True, [
        "884512", "230917", "615403", "479028", "158364", "902471",
        "336859", "740126",
    ]),
    _slot("PRODUCT", None, True, [
        "broadband plan", "mobile postpaid plan", "health insurance plan",
        "term insurance plan", "gym membership", "DTH package",
        "annual maintenance plan", "travel package",
    ]),
    _slot("SURVEY_TOPIC", None, True, [
        "branch service", "mobile app", "customer care experience",
        "ATM service", "loan process", "insurance service",
        "online banking experience", "delivery experience",
    ]),
    _slot("SUBJECT", None, True, [
        "maths", "science", "hindi", "english", "social science",
        "computer", "sanskrit", "drawing",
    ]),
)

SLOTS: Dict[str, SlotDef] = {s.name: s for s in _SLOT_LIST}

#: Scenario-specific overrides. A courier scam claims customs, a SIM scam
#: claims TRAI, and an electricity scam threatens the supply, not the account.
SCENARIO_SLOTS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "digital_arrest": {
        "PERS": (
            "aadhaar number", "PAN number", "date of birth",
            "mother maiden name", "aadhaar card number", "voter id number",
            "passport number", "registered mobile number",
        ),
        "BANK_ITEM": (
            "account", "savings account", "bank account", "current account",
            "salary account", "joint account", "net banking", "debit card",
        ),
        "AUTH": (
            "cyber crime branch", "Delhi police cyber cell", "CBI",
            "crime branch", "narcotics control bureau", "cyber police",
            "Mumbai police cyber cell", "enforcement directorate",
        ),
        "THREAT": (
            "arrest warrant nikal jayega", "giraftari ho jayegi",
            "FIR darj ho jayegi", "case darj ho jayega",
            "legal action liya jayega", "non bailable warrant nikal jayega",
            "account seize kar liya jayega", "police ghar pe bhej di jayegi",
        ),
    },
    "courier_customs": {
        "PERS": (
            "aadhaar number", "PAN number", "date of birth",
            "aadhaar card number", "passport number", "voter id number",
            "registered mobile number", "mother maiden name",
        ),
        "AUTH": (
            "customs department", "customs office", "narcotics control bureau",
            "cyber crime branch", "airport customs", "crime branch",
            "customs clearance department", "Delhi police",
        ),
        "THREAT": (
            "FIR darj ho jayegi", "case darj ho jayega",
            "parcel seize kar liya jayega", "legal action liya jayega",
            "arrest warrant nikal jayega", "customs case ban jayega",
            "giraftari ho sakti hai", "aapka naam case me aa jayega",
        ),
    },
    "sim_block": {
        "PERS": (
            "aadhaar number", "PAN number", "date of birth",
            "aadhaar card number", "voter id number", "passport number",
            "registered mobile number", "mother maiden name",
        ),
        "AUTH": (
            "TRAI", "TRAI verification cell", "telecom department",
            "TRAI head office", "department of telecom", "cyber crime branch",
            "telecom regulatory authority", "TRAI compliance team",
        ),
        "THREAT": (
            "block ho jayega", "band ho jayega",
            "permanently deactivate ho jayega", "suspend ho jayega",
            "disconnect ho jayega", "sim band ho jayegi",
            "number band ho jayega", "service band ho jayegi",
        ),
    },
    "electricity_disconnect": {
        "PERS": (
            "AnyDesk app", "screen share", "remote access", "AnyDesk",
            "TeamViewer app", "screen sharing app", "remote access app",
            "AnyDesk code",
        ),
        "AUTH": (
            "electricity department", "bijli vibhag",
            "state electricity board", "power department",
            "meter inspection team", "vigilance team",
            "electricity board head office", "bijli board",
        ),
        "THREAT": (
            "connection kat diya jayega", "connection disconnect kar diya jayega",
            "connection band kar diya jayega", "supply cut ho jayegi",
            "connection kat jayega", "meter uthwa liya jayega",
            "line cut ho jayegi", "connection permanently disconnect ho jayega",
        ),
    },
    "fake_relative": {
        "AUTH": (
            "police station", "thana", "local police station",
            "police chowki", "city police station", "police thana",
            "nearby police station", "area police station",
        ),
    },
    "lottery_prize": {
        "PERS": (
            "aadhaar number", "PAN number", "date of birth", "card number",
            "PAN card number", "aadhaar card number", "mother maiden name",
            "registered mobile number",
        ),
    },
    "job_offer": {
        "PERS": (
            "aadhaar number", "PAN number", "date of birth",
            "PAN card number", "aadhaar card number", "voter id number",
            "registered mobile number", "mother maiden name",
        ),
    },
    "loan_approval": {
        "PERS": (
            "PAN number", "aadhaar number", "date of birth", "card number",
            "CVV", "PAN card number", "aadhaar card number",
            "mother maiden name",
        ),
    },
    "kyc_freeze": {
        "AUTH": (
            "RBI", "reserve bank of india", "RBI verification cell",
            "banking ombudsman", "RBI head office", "cyber crime branch",
            "income tax department", "RBI compliance department",
        ),
    },
    "otp_theft": {
        "AUTH": (
            "RBI", "reserve bank of india", "cyber crime branch",
            "banking ombudsman", "RBI verification cell", "cyber cell",
            "income tax department", "RBI head office",
        ),
    },
}


# --------------------------------------------------------------------------
# Rendering: pattern string plus bindings to text, tokens and BIO
# --------------------------------------------------------------------------

_SLOT_RE = re.compile(r"\{([A-Z_0-9]+)\}")
_NO_SPACE_BEFORE = set(",.?!;:)]}%@")
_NO_SPACE_AFTER = set("([{@")


def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens back into readable text without breaking the tokenisation.

    Punctuation hugs the word to its left, brackets and the UPI at-sign hug
    both sides. The caller checks the round trip and falls back to a plain
    space join if anything would re-tokenise differently.
    """
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        prev = tokens[i - 1]
        if tok in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out)


class SlotBinder:
    """Holds the slot values chosen for one call.

    Sticky slots are drawn once and cached, so a call keeps one bank, one
    caller name and one OTP code from start to finish.
    """

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.bound: Dict[str, str] = {}
        self.gender: Optional[int] = None      # 0 masculine, 1 feminine

    def options(self, slot: str) -> Sequence[str]:
        override = SCENARIO_SLOTS.get(self.scenario, {}).get(slot)
        if override:
            return override
        if slot in GENDERED_SLOTS and self.gender is not None:
            return GENDERED_SLOTS[slot][self.gender]
        return SLOTS[slot].options

    def value(self, slot: str, rng: random.Random) -> Tuple[str, Optional[str]]:
        try:
            spec = SLOTS[slot]
        except KeyError:  # pragma: no cover - caught by validate_grammar
            raise KeyError(f"unknown slot {{{slot}}}") from None
        if spec.sticky and slot in self.bound:
            return self.bound[slot], spec.entity
        if slot in GENDERED_SLOTS and self.gender is None:
            # First gendered slot in the call fixes the caller gender, and
            # every later one follows it.
            self.gender = rng.randrange(2)
        value = rng.choice(list(self.options(slot)))
        if spec.sticky:
            self.bound[slot] = value
        return value, spec.entity


@dataclass
class TurnDraft:
    """One rendered turn, before it becomes a schema.Turn."""

    act: str
    speaker: str
    text: str
    tokens: List[str]
    bio: List[str]
    template_id: str
    slots: List[Tuple[str, str, Optional[str]]] = field(default_factory=list)


def render_pattern(
    pattern: str,
    binder: SlotBinder,
    rng: random.Random,
) -> Tuple[str, List[str], List[str], List[Tuple[str, str, Optional[str]]]]:
    """Fill a pattern and return (text, tokens, bio, slots_used).

    Literal stretches tokenise to O tags. A filled slot contributes its own
    tokens and, when the slot has an entity type, exactly one B- tag followed
    by I- tags over the rest. This is where the gold BIO comes from.
    """
    tokens: List[str] = []
    bio: List[str] = []
    used: List[Tuple[str, str, Optional[str]]] = []
    pos = 0

    for match in _SLOT_RE.finditer(pattern):
        for tok in tokenize(pattern[pos:match.start()]):
            tokens.append(tok)
            bio.append("O")
        slot = match.group(1)
        value, etype = binder.value(slot, rng)
        slot_tokens = tokenize(value)
        if not slot_tokens:
            raise ValueError(f"slot {slot} produced no tokens from {value!r}")
        if etype:
            bio.append(f"B-{etype}")
            bio.extend([f"I-{etype}"] * (len(slot_tokens) - 1))
        else:
            bio.extend(["O"] * len(slot_tokens))
        tokens.extend(slot_tokens)
        used.append((slot, value, etype))
        pos = match.end()

    for tok in tokenize(pattern[pos:]):
        tokens.append(tok)
        bio.append("O")

    if tokens and tokens[0][:1].isalpha():
        # A pattern that opens with a slot would otherwise start lowercase.
        # The token list is capitalised too, so BIO offsets stay aligned.
        tokens[0] = tokens[0][0].upper() + tokens[0][1:]
    text = detokenize(tokens)
    if tokenize(text) != tokens:
        # Any pattern that would re-tokenise differently falls back to the
        # safe join. The self-test below reports these so they can be fixed.
        text = " ".join(tokens)
    return text, tokens, bio, used


# --------------------------------------------------------------------------
# Shared pools: lines that work in any scenario
# --------------------------------------------------------------------------

#: Caller lines that are not scenario specific. Used when a scenario does not
#: define its own pool for an act.
SHARED_CALLER: Dict[str, Tuple[str, ...]] = {
    "GREET": (
        "{GREET_WORD}, {CUST} baat kar rahe hain?",
        "Hello, {CUST} se baat ho rahi hai kya?",
        "{GREET_WORD}, ek minute baat kar sakte hain?",
        "Hello? {FILLER}, {CUST} hain line pe?",
        "{GREET_WORD}, awaaz aa rahi hai meri?",
    ),
    "REASSURE": (
        "Ghabraiye mat, aapka paisa bilkul safe hai, bas verification ki baat hai.",
        "Main hoon na, do minute ka kaam hai bas.",
        "{FILLER}, tension mat lijiye, main step by step bata {DETA} hoon.",
        "Aapko kuch pay nahi karna hai, ye sirf record update hai.",
    ),
    "ISOLATE": (
        "Aur haan, call mat kato, warna process beech me fail ho jayega.",
        "Ye internal process hai, kisi ko mat batana, family ko bhi nahi.",
        "Line pe rahiye, phone mat rakho, main hold pe daal {RAHA} hoon.",
        "Ghar me kisi ko batane ki zaroorat nahi, ye confidential matter hai.",
    ),
    "PRESSURE_ESCALATE": (
        "Sir main aapki help kar {RAHA} hoon, aap samajh nahi rahe, line pe aur log wait kar rahe hain.",
        "Dekhiye phir baad me branch ke chakkar kaatne padenge, mujhe mat boliyega.",
        "Aap ab bataiye, ya main file reject kar doon?",
        "Mera shift khatam ho raha hai, uske baad koi kuch nahi kar payega.",
        "Aap last warning samajh lijiye, main teesri baar bol {RAHA} hoon.",
    ),
    "CLOSE": (
        "Thik hai sir, verification ho gaya, dhanyavaad.",
        "Ho gaya ji, koi problem aaye to isi number pe call kar lena.",
        "Chaliye, aapka kaam ho gaya, rakhta hoon phone.",
        "Bas itna hi tha, aapka time lene ke liye sorry, dhanyavaad.",
    ),
    "DEADLINE": (
        "Aapke paas sirf {WINDOW} hai, uske baad main kuch nahi kar paunga.",
        "{DEADLINE} process complete karna hoga, warna file band ho jayegi.",
        "Ye {WINDOW} ka window hai, system khud band kar dega uske baad.",
        "{DEADLINE} nahi hua to main kuch nahi kar paunga, ye samajh lijiye.",
    ),
}

#: Callee lines that fit any scam scenario.
SHARED_CALLEE: Dict[str, Tuple[str, ...]] = {
    "CONFIRM": (
        "{ACK}, bol rahe hain.",
        "{ACK}, kaun bol raha hai?",
        "{ACK}, boliye.",
        "Ji main hi hoon, boliye kya baat hai.",
        "{ACK}, sunai de raha hai, boliye.",
    ),
    "VICTIM_QUESTION": (
        "Ek minute, aap exactly kaun bol rahe ho?",
        "Lekin mujhe to koi message nahi aaya iske baare me?",
        "Ye sab phone pe hi karna zaroori hai kya?",
        "Aapko mera number kahan se mila?",
        "Achha, aur ye kab ka mamla hai?",
        "Matlab main karoon kya ab, thoda samjhaiye.",
    ),
    "VICTIM_RESIST": (
        "Nahi nahi, main aise phone pe kuch nahi bataunga.",
        "Dekhiye main khud jaake pata kar leta hoon, aap rehne dijiye.",
        "Mujhe ye thik nahi lag raha, main baad me call karta hoon.",
        "Ek minute, main apne bete se pooch leta hoon pehle.",
        "Aap apna office ka number dijiye, main wapas call karta hoon.",
    ),
    "VICTIM_COMPLY": (
        "{ACK}, ruko, aaya hai message. {OTP_CODE}.",
        "Achha likh lijiye, {OTP_CODE} hai.",
        "{ACK}, thik hai, jaise aap keh rahe ho waise karta hoon.",
        "Haan aa gaya, {OTP_CODE} likha hai ismein.",
        "Thik hai bhai, kar deta hoon, par jaldi khatam karo.",
    ),
    "SMALLTALK": (
        "{ACK}, {EXCUSE}, thoda zor se boliye.",
        "Haan boliye, main sun raha hoon.",
        "{EXCUSE}, ek second.",
    ),
}

#: Mild scam lines. No threat, no deadline, one innocent-sounding question.
#: These are the calls the text branch misses, so the voice branch has to
#: carry them. Deliberately free of BENIGN_MARKERS phrases: the marker
#: dictionary has to stay a clean signal for genuinely benign calls.
MILD_SCAM: Dict[str, Tuple[str, ...]] = {
    "INFORM": (
        "Kuch karna nahi hai aapko, bas ek routine record check chal raha hai.",
        "Aapke {BANK_ITEM} ka annual record update ho raha hai, isliye ek chhota sa confirmation chahiye.",
        "Ye sirf ek verification call hai, do minute me khatam.",
        "Hamare system me aapki details thodi purani hain, bas cross check kar {RAHA} hoon.",
    ),
    "REASSURE": (
        "Nahi nahi, koi {OTP_WORD} nahi chahiye, aur na hi koi payment.",
        "Aapko kuch bhejna nahi hai, main sirf record match kar {RAHA} hoon.",
        "{FILLER}, ye standard process hai, sabko call ja rahi hai.",
        "Aap aaram se bataiye, koi jaldi ki baat nahi hai.",
    ),
    "REQUEST_SENSITIVE": (
        "Bas confirm kar dijiye, aapki {SOFT_PERS} kya hai record me?",
        "Ek chhoti si cheez, aapke {BANK_ITEM} ke last four digit {LAST4} hain na?",
        "Aap apna {SOFT_PERS} bata dijiye, main file me tick laga {DETA} hoon.",
        "Sirf ek line ka confirmation, aapka {SOFT_PERS} match kar raha hai kya?",
    ),
    "CONFIRM": (
        "{ACK}, wahi hai.",
        "Haan ji sahi hai.",
        "{ACK}, thik hai.",
    ),
    "CLOSE": (
        "Bas ho gaya, aapka record update kar diya, dhanyavaad.",
        "Thank you sir, aur kuch nahi chahiye, rakhta hoon.",
        "Ho gaya ji, aapka din achha rahe.",
    ),
    "VICTIM_QUESTION": (
        "Ye kis cheez ke liye pooch rahe ho?",
        "Aap {BANK} se hi bol rahe ho na?",
        "Achha, aur koi problem to nahi hai na?",
    ),
    "VICTIM_COMPLY": (
        "{ACK}, wahi hai jo aap bol rahe ho.",
        "Haan sahi hai, likh lijiye.",
        "{ACK}, correct hai.",
    ),
}

#: Caller lines for benign calls. Without these a real bank reminder ends up
#: borrowing the scam pool and telling the customer that their money is safe,
#: which is a scam tell, not a benign one.
BENIGN_CALLER: Dict[str, Tuple[str, ...]] = {
    "GREET": (
        "{GREET_WORD}, {CUST}? Ek chhoti si baat karni thi.",
        "{GREET_WORD} ji, do minute mil sakte hain aapke?",
        "{GREET_WORD}, {CUST} se baat ho rahi hai?",
        "Hello, {CUST}? Main disturb to nahi kar {RAHA} hoon?",
        "{GREET_WORD}, abhi baat kar sakte hain ya baad me call karoon?",
    ),
    "REASSURE": (
        "Koi jaldi nahi hai, aap aaram se dekh lijiyega.",
        "Aapki marzi hai, koi zabardasti nahi hai ismein.",
        "Aapko abhi kuch nahi karna, sirf jankari ke liye bataya hai.",
        "Agar koi doubt ho to aap official website pe check kar lijiye.",
        "Jab time mile tab kar lijiyega, koi problem nahi.",
    ),
    "CLOSE": (
        "Bas itna hi tha, thank you for your time.",
        "Chaliye ji, dhanyavaad, have a good day.",
        "Thik hai, aur koi help chahiye to bata dijiyega, dhanyavaad.",
        "Aapka din achha rahe, dhanyavaad.",
        "Chaliye, aapka time lene ke liye dhanyavaad.",
    ),
    "INSTRUCT": (
        "Aap bas ek baar dekh lijiyega, aur kuch nahi karna hai.",
        "Aane se pehle ek call kar dijiyega, taki wait na karna pade.",
        "Koi confusion ho to helpline pe pooch lijiyega.",
    ),
    "INFORM": (
        "Ye call sirf jankari ke liye hai, aur kuch nahi.",
        "Aapko yaad dilane ke liye call kiya hai, bas itna hi.",
    ),
}

#: Callee lines for benign calls: relaxed, no suspicion.
BENIGN_CALLEE: Dict[str, Tuple[str, ...]] = {
    "CONFIRM": (
        "{ACK}, thik hai, note kar liya.",
        "Achha achha, samajh gaya.",
        "{ACK}, main dekh leta hoon.",
        "Thik hai ji, dhanyavaad.",
        "Haan bilkul, main dekh lunga.",
        "Thik hai, aa jaunga main.",
    ),
    "VICTIM_QUESTION": (
        "Achha, aur kitne baje tak time hai?",
        "Ek baat batao, iske liye kuch lekar aana padega?",
        "Ye online bhi ho jayega ya jaana padega?",
        "Thik hai, par mujhe yaad nahi tha, kab ki baat hai?",
    ),
    "VICTIM_COMPLY": (
        "{ACK}, {OTP_CODE} hai.",
        "Haan aaya hai, {OTP_CODE}.",
        "{ACK}, kar deta hoon.",
    ),
    "VICTIM_RESIST": (
        "Nahi bhai, abhi nahi chahiye, phir kabhi dekhenge.",
        "{ACK}, mujhe interest nahi hai, dhanyavaad.",
        "Abhi rehne dijiye, mera pehle se hi chal raha hai.",
        "Nahi ji, filhaal zaroorat nahi hai.",
    ),
    "SMALLTALK": (
        "{ACK}, {EXCUSE}, ek minute.",
        "Arre haan, boliye boliye.",
        "{ACK}, sab thik, aap sunao.",
    ),
}


# --------------------------------------------------------------------------
# Scenario template pools
#
# SCENARIO_ACTS[scenario][act] is the pool of patterns for that act in that
# scenario. Anything missing falls back to SHARED_CALLER / SHARED_CALLEE, so a
# scenario only writes the lines that are actually specific to it.
#
# An arc is a space separated act sequence. "ACT:callee" flips the speaker for
# one turn when the default in ACT_SPEAKER is wrong for that scenario.
# --------------------------------------------------------------------------

SCENARIO_ACTS: Dict[str, Dict[str, Tuple[str, ...]]] = {}
SCENARIO_ARCS: Dict[str, Tuple[str, ...]] = {}

# -- 1. kyc_freeze ---------------------------------------------------------

SCENARIO_ACTS["kyc_freeze"] = {
    "IDENTIFY_SELF": (
        "Main {BANK} ke {DEPT} se {NAME} bol {RAHA} hoon.",
        "Ji mera naam {NAME} hai, {BANK} head office se call kar {RAHA} hoon.",
        "{BANK} customer care, {NAME} bol {RAHA} hoon, aapki service me.",
        "Main {NAME}, {BANK} ki {DEPT} se, aapke {BANK_ITEM} ke baare me baat karni thi.",
    ),
    "PROBLEM_STATE": (
        "Dekhiye, aapki {BANK_ITEM} system me update nahi hui hai, isliye call kiya.",
        "Aapke {BANK_ITEM} pe ek hold laga hua hai, {FILLER} pending verification ki wajah se.",
        "Sir aapki {BANK_ITEM} expire ho chuki hai, record me mismatch aa raha hai.",
        "Aapke {BANK_ITEM} ka document last year wala hai, wo ab valid nahi hai.",
    ),
    "AUTHORITY_ASSERT": (
        "Ye {AUTH} ka naya rule hai, hum bas usko follow kar rahe hain.",
        "{AUTH} ne saaf bola hai ki jinka record update nahi hai unka {BANK_ITEM} hold hoga.",
        "Ye order {AUTH} se aaya hai, hamare haath me kuch nahi hai.",
    ),
    "THREAT": (
        "Agar aaj update nahi hua to aapka {BANK_ITEM} {THREAT}.",
        "{FILLER}, mujhe bhi bura lagta hai bolte hue, par {DEADLINE} ke baad {BANK_ITEM} {THREAT}.",
        "System automatic hai sir, time pe nahi kiya to {THREAT}, phir main kuch nahi kar sakta.",
        "Aapka {BANK_ITEM} {THREAT} aur phir poora process dobara karna padega.",
    ),
    "INSTRUCT": (
        "Aap ek kaam kijiye, phone haath me rakhiye aur jo main {BOLTA} hoon wahi kariye.",
        "Main abhi ek {PAY} bhej {RAHA} hoon, usko open karke form bhar dijiye.",
        "Jo message aaye usko delete mat kariye, mujhe padhkar suna dijiye.",
    ),
    "REQUEST_SENSITIVE": (
        "Aapke number pe {OTP_WORD} aaya hoga, wo bata dijiye, main verify kar {DETA} hoon.",
        "Aapke {CARD} ke peeche jo {PERS} likha hai wo bhi bol dijiye, warna form submit nahi hoga.",
        "Ek aur cheez, aapka {PERS} confirm kar dijiye, file me mismatch hai.",
        "{OTP_WORD} aaya? Wahi number bataiye, main entry kar {RAHA} hoon.",
    ),
    "INFORM": (
        "Aapke {BANK_ITEM} ka record hamare paas hai, bas last step reh gaya hai.",
        "Ye process har customer ke liye ho raha hai, aap akele nahi hain.",
    ),
}
SCENARIO_ARCS["kyc_freeze"] = (
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_QUESTION AUTHORITY_ASSERT THREAT DEADLINE VICTIM_RESIST PRESSURE_ESCALATE INSTRUCT REQUEST_SENSITIVE VICTIM_COMPLY ISOLATE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF AUTHORITY_ASSERT PROBLEM_STATE THREAT VICTIM_QUESTION REASSURE DEADLINE INSTRUCT REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF CONFIRM PROBLEM_STATE INFORM VICTIM_QUESTION THREAT DEADLINE ISOLATE INSTRUCT REQUEST_SENSITIVE VICTIM_COMPLY CLOSE",
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_RESIST AUTHORITY_ASSERT THREAT PRESSURE_ESCALATE REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE CLOSE",
)

# -- 2. otp_theft ----------------------------------------------------------

SCENARIO_ACTS["otp_theft"] = {
    "IDENTIFY_SELF": (
        "Main {BANK} ke {DEPT} se {NAME} bol {RAHA} hoon, transaction ke baare me.",
        "{NAME} bol {RAHA} hoon {BANK} se, aapke {BANK_ITEM} pe ek alert aaya hai.",
        "Ji main {NAME}, {BANK} fraud desk se, isliye call kiya hai.",
    ),
    "PROBLEM_STATE": (
        "Aapke {BANK_ITEM} se abhi {MONEY} ka transaction try hua hai {CITY} me.",
        "Sir ek galat transaction ho gaya hai aapke {BANK_ITEM} se, {MONEY} ka.",
        "Hamare system me aapke naam pe ek suspicious payment dikh raha hai, {MONEY}.",
        "Aapne {MONEY} ki koi payment ki hai abhi? Nahi ki to problem hai.",
    ),
    "THREAT": (
        "Agar humne abhi cancel nahi kiya to {MONEY} kat jayega aur wapas nahi aayega.",
        "{DEADLINE} reverse nahi kiya to paisa gaya, phir claim ka lamba process hai.",
        "Aapka {BANK_ITEM} {THREAT} agar humne ise abhi hold nahi kiya.",
    ),
    "INSTRUCT": (
        "Cancel karne ke liye ek code aayega, wahi chahiye mujhe.",
        "Main transaction reverse kar {RAHA} hoon, aap phone rakhiye mat.",
        "Aap apna message box khol lijiye, abhi ek code aa raha hai.",
    ),
    "REQUEST_SENSITIVE": (
        "{OTP_WORD} aa gaya? Bas wahi bata dijiye, main reverse kar {DETA} hoon.",
        "Jaldi bataiye {OTP_WORD}, warna reversal window nikal jayegi.",
        "{OTP_WORD} ke saath {PERS} bhi confirm kar dijiye, dono match karne hain.",
        "Message me jo number hai wahi bol dijiye, main sun {RAHA} hoon.",
    ),
    "REASSURE": (
        "Ye paisa aapka hi hai, main sirf usko rok {RAHA} hoon.",
        "Hum kabhi paisa nahi maangte, main sirf reversal kar {RAHA} hoon.",
    ),
    "INFORM": (
        "Reversal ke liye system ek code bhejta hai, wo aapke registered number pe hi aayega.",
    ),
}
SCENARIO_ARCS["otp_theft"] = (
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_QUESTION THREAT DEADLINE INSTRUCT REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE VICTIM_COMPLY ISOLATE CLOSE",
    "GREET IDENTIFY_SELF CONFIRM PROBLEM_STATE THREAT VICTIM_QUESTION REASSURE INFORM INSTRUCT REQUEST_SENSITIVE VICTIM_COMPLY PRESSURE_ESCALATE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_RESIST THREAT DEADLINE ISOLATE REQUEST_SENSITIVE VICTIM_COMPLY CLOSE",
)

# -- 3. digital_arrest -----------------------------------------------------

SCENARIO_ACTS["digital_arrest"] = {
    "IDENTIFY_SELF": (
        "Main {AUTH} se inspector {NAME} bol {RAHA} hoon.",
        "Officer {NAME} this side, {AUTH}, {CITY} unit.",
        "Main sub inspector {NAME}, {AUTH} se, ye call record ho rahi hai.",
    ),
    "AUTHORITY_ASSERT": (
        "Ye call {AUTH} headquarters se ja rahi hai, aur poori recording ho rahi hai.",
        "{AUTH} ke under ye investigation chal rahi hai, aapka naam ismein aaya hai.",
        "Main {AUTH} ka officer hoon, aap jo bologe wo record me jayega.",
    ),
    "PROBLEM_STATE": (
        "Aapke {ID_DOC} pe ek parcel mila hai jisme illegal item tha, {CITY} airport pe.",
        "Aapke naam pe ek {BANK_ITEM} se {MONEY} ka money laundering hua hai.",
        "Aapka number ek trafficking case me use hua hai, {CITY} me.",
        "Ek arrested aadmi ne poochtach me aapka naam liya hai.",
    ),
    "THREAT": (
        "Aapke naam ka {THREAT}, aaj hi.",
        "Agar cooperate nahi kiya to {THREAT} aur {AUTH} ki team aapke ghar aa jayegi.",
        "Ye non bailable section hai, {THREAT}, samajh rahe hain aap?",
    ),
    "ISOLATE": (
        "Ye judicial matter hai, kisi ko mat batana, family ko bhi nahi.",
        "Call mat kato, aap digital custody me ho, line pe rahoge to hi bail ho payegi.",
        "Camera on rakhiye aur kisi aur kamre me mat jaiye, aap surveillance me hain.",
        "Phone mat rakho, disconnect karoge to {THREAT}.",
    ),
    "INSTRUCT": (
        "Aap ek kaam kijiye, video call pe aa jaiye aur camera on rakhiye.",
        "Apna {ID_DOC} haath me lijiye aur camera ke saamne dikhaiye.",
        "Ab aap apne {BANK_ITEM} ka poora balance safe custody me transfer kijiye.",
    ),
    "REQUEST_SENSITIVE": (
        "Verification ke liye apna {PERS} bataiye, main FIR me check {KARTA} hoon.",
        "Aapka {PERS} aur {BANK_ITEM} ki detail chahiye, court me submit hogi.",
        "Ye paisa {PAY_ID} pe transfer kijiye, {AUTH} verify karke wapas kar degi.",
        "{MONEY} abhi transfer kijiye {PAY} pe, ye refundable custody amount hai.",
    ),
    "PRESSURE_ESCALATE": (
        "Aap samajh nahi rahe, main aapko bacha {RAHA} hoon, warna aaj raat lockup me honge.",
        "Main senior officer ko line pe de {RAHA} hoon, phir baat mat badalna.",
        "Ye last chance hai, uske baad file court chali jayegi.",
    ),
}
SCENARIO_ARCS["digital_arrest"] = (
    "GREET CONFIRM IDENTIFY_SELF AUTHORITY_ASSERT PROBLEM_STATE VICTIM_QUESTION THREAT ISOLATE DEADLINE INSTRUCT REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF AUTHORITY_ASSERT PROBLEM_STATE THREAT VICTIM_QUESTION ISOLATE INSTRUCT DEADLINE REQUEST_SENSITIVE VICTIM_COMPLY PRESSURE_ESCALATE ISOLATE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_RESIST AUTHORITY_ASSERT THREAT ISOLATE PRESSURE_ESCALATE REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE CLOSE",
)

# -- 4. lottery_prize ------------------------------------------------------

SCENARIO_ACTS["lottery_prize"] = {
    "IDENTIFY_SELF": (
        "Main {JOB_COMPANY} ke lucky draw department se {NAME} bol {RAHA} hoon.",
        "{NAME} bol {RAHA} hoon, {JOB_COMPANY} prize distribution cell se.",
        "Ji main {NAME}, hamari company ka annual draw hua tha, uske liye call kiya hai.",
    ),
    "INFORM": (
        "Congratulations sir, aapka number lucky draw me select hua hai.",
        "Aapko {MONEY} ka bumper prize laga hai, {CITY} region se aap winner ho.",
        "Aapka mobile number computer se select hua hai, {MONEY} ka gift.",
        "Aapke naam pe ek car ka coupon bhi hai, ya cash le lijiye, {MONEY}.",
    ),
    "PROBLEM_STATE": (
        "Bas ek chhota sa issue hai, prize release karne ke liye tax clearance pending hai.",
        "Prize amount seedha aapke {BANK_ITEM} me transfer hoga, uske liye ek processing entry karni padegi.",
    ),
    "REQUEST_SENSITIVE": (
        "Aap sirf {FEE} processing fee {PAY} pe bhej dijiye, baaki hum kar denge.",
        "{FEE} tax lagta hai, wo {PAY_ID} pe bhej dijiye, prize ke saath wapas mil jayega.",
        "Apna {PAY} aur {PERS} bhej dijiye, transfer wahin karenge.",
        "{OTP_WORD} bhej {RAHA} hoon, wo bataiye taki claim confirm ho jaye.",
    ),
    "DEADLINE": (
        "Claim {DEADLINE} karna hoga, uske baad prize agle winner ko chala jayega.",
        "Ye offer {DEADLINE} valid hai, company ka rule hai.",
    ),
    "PRESSURE_ESCALATE": (
        "Sir itna bada prize log chhodte nahi hain, aap soch kya rahe ho?",
        "Aapse pehle wale customer ne 10 minute me claim kar liya tha.",
    ),
    "REASSURE": (
        "Fee refundable hai, prize ke saath hi aa jayegi, likhit me milega.",
        "Company registered hai, aap chinta mat kariye.",
    ),
}
SCENARIO_ARCS["lottery_prize"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION PROBLEM_STATE REQUEST_SENSITIVE VICTIM_RESIST REASSURE DEADLINE PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF INFORM CONFIRM PROBLEM_STATE DEADLINE REQUEST_SENSITIVE VICTIM_QUESTION REASSURE PRESSURE_ESCALATE VICTIM_COMPLY ISOLATE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_RESIST REASSURE PROBLEM_STATE REQUEST_SENSITIVE DEADLINE VICTIM_RESIST PRESSURE_ESCALATE CLOSE",
)

# -- 5. fake_relative ------------------------------------------------------

SCENARIO_ACTS["fake_relative"] = {
    "GREET": (
        "Hello, uncle ji? {GREET_WORD}, sun rahe hain?",
        "Hello, {CUST}? Ek zaroori baat karni thi, jaldi.",
        "{GREET_WORD}, aap {CUST} bol rahe hain na? Suniye zara.",
    ),
    "IDENTIFY_SELF": (
        "Main {NAME} bol {RAHA} hoon, {RELATIVE} ka dost hoon.",
        "Main {NAME}, hum {RELATIVE} ke saath the abhi, {CITY} me.",
        "Ji main {NAME} bol {RAHA} hoon, {HOSPITAL} se call kar {RAHA} hoon.",
    ),
    "PROBLEM_STATE": (
        "{RELATIVE} ka accident ho gaya hai, abhi {HOSPITAL} me hai.",
        "Dekhiye ghabraiye mat, par {RELATIVE} thoda injured hai, hospital le aaye hain.",
        "{RELATIVE} {AUTH} me hai, ek jhagde me naam aa gaya hai.",
        "{RELATIVE} ka phone toot gaya hai isliye main call kar {RAHA} hoon.",
    ),
    "THREAT": (
        "Doctor bol rahe hain operation abhi karna padega, warna haath me problem ho jayegi.",
        "Agar {DEADLINE} paisa jama nahi hua to admission cancel kar denge.",
        "Yahan case ban jayega, {THREAT}, main settle karwa {RAHA} hoon.",
    ),
    "DEADLINE": (
        "{DEADLINE} paisa jama karna hai, counter band ho jayega.",
        "Bas {WINDOW} ka time hai mere paas, uske baad main nikal jaunga.",
    ),
    "REQUEST_SENSITIVE": (
        "{MONEY} bhej dijiye {PAY} pe, main yahin counter pe khada hoon.",
        "Aap {MONEY} {PAY_ID} pe daal dijiye, main baaki apni jeb se laga {DETA} hoon.",
        "Mera {PAY} note kar lijiye, {PAY_ID}, ismein bhej dijiye jaldi.",
    ),
    "ISOLATE": (
        "Aunty ko mat batana abhi, {RELATIVE} ne mana kiya hai, tabiyat kharab ho jayegi.",
        "Ghar me kisi ko mat batana, {RELATIVE} khud baad me baat karega.",
        "Phone mat rakho, main doctor se baat karwa {RAHA} hoon.",
    ),
    "PRESSURE_ESCALATE": (
        "Uncle jaldi kijiye, yahan halat kharab hai, main akela kya karoon?",
        "Aap paise ki soch rahe hain, yahan {RELATIVE} andar hai.",
    ),
    "REASSURE": (
        "Main yahin hoon, aap paisa bhejiye, baaki main sambhal {LETA} hoon.",
        "Aap tension mat lijiye, doctor achhe hain.",
    ),
}
SCENARIO_ARCS["fake_relative"] = (
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_QUESTION THREAT DEADLINE ISOLATE REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF PROBLEM_STATE CONFIRM VICTIM_QUESTION REASSURE THREAT REQUEST_SENSITIVE ISOLATE PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_RESIST PRESSURE_ESCALATE THREAT DEADLINE REQUEST_SENSITIVE ISOLATE VICTIM_RESIST CLOSE",
)

# -- 6. electricity_disconnect --------------------------------------------

SCENARIO_ACTS["electricity_disconnect"] = {
    "IDENTIFY_SELF": (
        "Main {UTIL} se {NAME} bol {RAHA} hoon, meter section se.",
        "{NAME} bol {RAHA} hoon, {UTIL} ka junior engineer hoon.",
        "Ji main {NAME}, {UTIL} ke {DEPT} se, bill ke silsile me call kiya hai.",
    ),
    "PROBLEM_STATE": (
        "Aapka pichhla month ka bill system me update nahi hua hai.",
        "Aapke meter ki reading pending dikha raha hai, payment show nahi ho rahi.",
        "Sir aapka bill {FEE} ka bakaya pada hai, kisi ne submit nahi kiya.",
        "Hamare record me aapka connection defaulter list me chala gaya hai.",
    ),
    "AUTHORITY_ASSERT": (
        "Ye {AUTH} ka order hai, humein list mili hai upar se.",
        "{AUTH} ne aaj hi disconnection list nikali hai, usme aapka naam hai.",
    ),
    "THREAT": (
        "Aaj raat 9 baje {THREAT}.",
        "Agar aaj payment nahi hui to {THREAT}, phir naya connection lena padega.",
        "{DEADLINE} {THREAT} aur reconnection charge alag lagega.",
    ),
    "INSTRUCT": (
        "Aap ek kaam kijiye, main aapko ek {PAY} bhej {RAHA} hoon, usse pay kar dijiye.",
        "Aapko ek app install karni padegi, {PERS}, usse main meter reset kar dunga.",
        "Aap {SMALL_MONEY} ka test payment kijiye, usse hi verification ho jayega.",
    ),
    "REQUEST_SENSITIVE": (
        "Aap {FEE} abhi {PAY_ID} pe daal dijiye, main disconnection rukwa {DETA} hoon.",
        "{PAY} pe bhej dijiye aur uske baad {OTP_WORD} bata dijiye confirmation ke liye.",
        "App khol kar {PERS} ka number bataiye, main remote se update kar dunga.",
    ),
    "PRESSURE_ESCALATE": (
        "Sir mera aadmi wahin khada hai line kaatne ko, main usko rok {RAHA} hoon.",
        "Ab aap decide kijiye, mujhe aage report karni hai 5 minute me.",
    ),
    "DEADLINE": (
        "Bas {WINDOW} hai mere paas, uske baad file office chali jayegi.",
        "{DEADLINE} payment ho jani chahiye, warna kuch nahi ho payega.",
    ),
}
SCENARIO_ARCS["electricity_disconnect"] = (
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_QUESTION THREAT DEADLINE INSTRUCT REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF PROBLEM_STATE AUTHORITY_ASSERT THREAT CONFIRM VICTIM_QUESTION DEADLINE INSTRUCT REQUEST_SENSITIVE VICTIM_COMPLY ISOLATE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_RESIST THREAT PRESSURE_ESCALATE DEADLINE REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE CLOSE",
)

# -- 7. courier_customs ----------------------------------------------------

SCENARIO_ACTS["courier_customs"] = {
    "IDENTIFY_SELF": (
        "Main {COMPANY} se {NAME} bol {RAHA} hoon, international parcel section.",
        "{NAME} bol {RAHA} hoon {COMPANY} se, aapke naam ka ek parcel hai.",
        "Ji main {NAME}, {COMPANY} ke {DEPT} se, ek urgent matter hai.",
    ),
    "PROBLEM_STATE": (
        "Aapke naam ka parcel {CITY} airport pe rok liya gaya hai.",
        "Parcel number {ORDER_ID} aapke {ID_DOC} pe book hua tha, usme problem hai.",
        "Aapke naam ka jo parcel hai usme kuch aisi cheezein mili hain jo allowed nahi hain.",
        "Parcel {CITY} se aaya tha, {AUTH} ne scan me kuch pakda hai.",
    ),
    "AUTHORITY_ASSERT": (
        "Main aapko {AUTH} ke officer se connect kar {RAHA} hoon, wo baat karenge.",
        "Ye matter ab {AUTH} ke paas hai, hum bas inform kar rahe hain.",
        "{AUTH} se bol {RAHA} hoon, ye call official line pe record ho rahi hai.",
    ),
    "THREAT": (
        "Agar clearance nahi hui to aapke khilaf {THREAT}.",
        "Parcel aapke naam pe hai, isliye {THREAT}, ye samajh lijiye.",
        "{DEADLINE} clear nahi hua to {THREAT} aur aapko station aana padega.",
    ),
    "REQUEST_SENSITIVE": (
        "Clearance fee {FEE} hai, wo {PAY_ID} pe bhej dijiye, receipt mil jayegi.",
        "Verification ke liye aapka {PERS} chahiye, file me lagana hai.",
        "{FEE} {PAY} pe transfer kijiye, parcel aaj hi release ho jayega.",
        "Ek {OTP_WORD} aayega aapke phone pe, wo bata dijiye, tabhi case band hoga.",
    ),
    "ISOLATE": (
        "Ye {AUTH} ka matter hai, kisi ko mat batana jab tak clear na ho jaye.",
        "Line pe rahiye, call mat kato, officer abhi aa rahe hain.",
    ),
    "INSTRUCT": (
        "Aap apna {ID_DOC} ready rakhiye, officer poochhenge.",
        "Ek form bhejta hoon, usme details bhar dijiye, warna clearance nahi hoga.",
    ),
    "PRESSURE_ESCALATE": (
        "Sir parcel aapke naam pe hai, ab main kya kar {SAKTA} hoon, jaldi decide kijiye.",
        "Aap mana kar rahe ho, phir main file {AUTH} ko forward kar {DETA} hoon.",
    ),
}
SCENARIO_ARCS["courier_customs"] = (
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_QUESTION AUTHORITY_ASSERT THREAT DEADLINE REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE VICTIM_COMPLY ISOLATE CLOSE",
    "GREET IDENTIFY_SELF PROBLEM_STATE CONFIRM AUTHORITY_ASSERT THREAT VICTIM_QUESTION INSTRUCT REQUEST_SENSITIVE ISOLATE VICTIM_COMPLY CLOSE",
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_RESIST AUTHORITY_ASSERT THREAT PRESSURE_ESCALATE DEADLINE REQUEST_SENSITIVE VICTIM_RESIST CLOSE",
)

# -- 8. job_offer ----------------------------------------------------------

SCENARIO_ACTS["job_offer"] = {
    "IDENTIFY_SELF": (
        "Main {JOB_COMPANY} ke HR department se {NAME} bol {RAHA} hoon.",
        "{NAME} bol {RAHA} hoon, {JOB_COMPANY} ki recruitment team se.",
        "Ji main {NAME}, {JOB_COMPANY} se, aapke resume ke baare me baat karni thi.",
    ),
    "INFORM": (
        "Aapka resume hamare portal pe shortlist hua hai, {CITY} office ke liye.",
        "Aapke liye ek opening hai, salary {MONEY} monthly, work from home.",
        "Interview ki zaroorat nahi hai, direct joining letter ban raha hai.",
        "Position back office executive ki hai, {CITY} branch me.",
    ),
    "PROBLEM_STATE": (
        "Bas ek formality hai, registration ke bina file aage nahi badhti.",
        "Company ka rule hai ki candidate registration ke baad hi documents banate hain.",
    ),
    "REQUEST_SENSITIVE": (
        "Registration fee {FEE} hai, wo {PAY_ID} pe bhej dijiye, joining ke din wapas.",
        "Aap {FEE} {PAY} pe daal dijiye, offer letter aaj hi mail kar dungi.",
        "Apna {PERS} bhej dijiye, background verification ke liye chahiye.",
        "Ek {OTP_WORD} aayega, wo bata dijiye, tabhi profile lock hogi.",
    ),
    "DEADLINE": (
        "Seat {DEADLINE} hold hai, uske baad next candidate ko de denge.",
        "{DEADLINE} confirm kar dijiye, warna offer expire ho jayega.",
    ),
    "PRESSURE_ESCALATE": (
        "Dekhiye, itni achhi package wali job roz nahi milti, soch lijiye.",
        "Aapke baad 40 candidates line me hain, main kitna wait karoon?",
    ),
    "REASSURE": (
        "Fee refundable hai, joining ke pehle salary me adjust ho jayegi.",
        "Company registered hai, aap website pe dekh sakte hain.",
    ),
}
SCENARIO_ARCS["job_offer"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION PROBLEM_STATE REQUEST_SENSITIVE VICTIM_RESIST REASSURE DEADLINE PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF INFORM CONFIRM PROBLEM_STATE REQUEST_SENSITIVE DEADLINE VICTIM_QUESTION REASSURE PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION REASSURE PROBLEM_STATE REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE DEADLINE CLOSE",
)

# -- 9. loan_approval ------------------------------------------------------

SCENARIO_ACTS["loan_approval"] = {
    "IDENTIFY_SELF": (
        "Main {BANK} ke loan department se {NAME} bol {RAHA} hoon.",
        "{NAME} bol {RAHA} hoon {BANK} se, aapke pre approved offer ke baare me.",
        "Ji main {NAME}, {BANK} ke {DEPT} se, ek achhi khabar hai aapke liye.",
    ),
    "INFORM": (
        "Aapka {MONEY} ka loan pre approved hai, koi document nahi chahiye.",
        "Aapke CIBIL pe {MONEY} tak ka offer generate hua hai.",
        "Interest sirf 4 percent hai, aur EMI aapke hisaab se set kar denge.",
        "Paisa {DEADLINE} aapke {BANK_ITEM} me aa jayega.",
    ),
    "PROBLEM_STATE": (
        "Bas file processing pending hai, uske bina disbursal nahi hota.",
        "Aapki file me insurance charge lagta hai, wo pehle jama karna padta hai.",
    ),
    "REQUEST_SENSITIVE": (
        "Processing fee {FEE} hai, wo {PAY_ID} pe bhej dijiye, loan wale din adjust ho jayegi.",
        "Aap apna {PERS} aur {BANK_ITEM} number bata dijiye, file complete kar {DETA} hoon.",
        "Ek {OTP_WORD} bheja hai, wo bataiye, tabhi application submit hogi.",
        "{FEE} {PAY} pe daal dijiye, main disbursal same day karwa dunga.",
    ),
    "DEADLINE": (
        "Offer {DEADLINE} valid hai, uske baad rate badal jayega.",
        "{DEADLINE} fee jama kijiye, warna file cancel ho jayegi.",
    ),
    "PRESSURE_ESCALATE": (
        "Sir aisa offer dobara nahi milta, main aapka bhala soch {RAHA} hoon.",
        "Aap decide nahi kar pa rahe to main file close kar {DETA} hoon.",
    ),
    "REASSURE": (
        "Fee sirf ek baar lagti hai, loan me adjust ho jati hai.",
        "Aapko kahin jaana nahi hai, sab kuch phone pe ho jayega.",
    ),
}
SCENARIO_ARCS["loan_approval"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION PROBLEM_STATE REQUEST_SENSITIVE VICTIM_RESIST REASSURE DEADLINE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF INFORM CONFIRM PROBLEM_STATE DEADLINE REQUEST_SENSITIVE VICTIM_QUESTION REASSURE PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM PROBLEM_STATE VICTIM_RESIST REASSURE REQUEST_SENSITIVE PRESSURE_ESCALATE DEADLINE VICTIM_RESIST CLOSE",
)

# -- 10. sim_block ---------------------------------------------------------

SCENARIO_ACTS["sim_block"] = {
    "IDENTIFY_SELF": (
        "Main {AUTH} se {NAME} bol {RAHA} hoon, mobile verification ke liye.",
        "{NAME} bol {RAHA} hoon, {TELCO} ki {DEPT} se, {AUTH} ke instruction pe.",
        "Ji main {NAME}, {AUTH} se, aapke number ke baare me baat karni hai.",
    ),
    "PROBLEM_STATE": (
        "Aapke number ka {BANK_ITEM} document verify nahi hua hai.",
        "Aapke naam pe do sim chal rahi hain, ek fraud me use ho rahi hai.",
        "Sir aapke number pe complaint aayi hai, isliye verification chal rahi hai.",
        "Aapka mobile document {CITY} circle me pending dikha raha hai.",
    ),
    "AUTHORITY_ASSERT": (
        "Ye {AUTH} ka circular hai, sabhi operators ko bheja gaya hai.",
        "{AUTH} ke rule ke hisaab se verification zaroori hai, warna number band.",
    ),
    "THREAT": (
        "Aapka number {DEADLINE} {THREAT}.",
        "Verification nahi hui to sim {THREAT} aur wahi number dobara nahi milega.",
        "{THREAT}, aur uske baad saare {OTP_WORD} bhi band ho jayenge.",
    ),
    "INSTRUCT": (
        "Aap apne phone se 401 wala code dial kijiye, verification start ho jayega.",
        "Main ek verification link bhej {RAHA} hoon, usme details bhar dijiye.",
        "Aap phone kaan se hata kar screen dekhiye, ek code aaya hoga.",
    ),
    "REQUEST_SENSITIVE": (
        "Aapke phone pe {OTP_WORD} aaya hai, wo bata dijiye, verification complete ho jayegi.",
        "Apna {PERS} bataiye, wahi document se link karna hai.",
        "{OTP_WORD} jaldi bataiye, ye 2 minute me expire ho jata hai.",
    ),
    "DEADLINE": (
        "Aapke paas sirf {WINDOW} hai, uske baad system khud block kar dega.",
        "{DEADLINE} verification kar lijiye, main file open rakhta hoon.",
    ),
}
SCENARIO_ARCS["sim_block"] = (
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE VICTIM_QUESTION AUTHORITY_ASSERT THREAT DEADLINE INSTRUCT REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF AUTHORITY_ASSERT PROBLEM_STATE CONFIRM THREAT VICTIM_QUESTION INSTRUCT DEADLINE REQUEST_SENSITIVE VICTIM_COMPLY ISOLATE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF PROBLEM_STATE THREAT VICTIM_RESIST PRESSURE_ESCALATE DEADLINE REQUEST_SENSITIVE VICTIM_RESIST PRESSURE_ESCALATE CLOSE",
)

#: Mild scam arcs. Same scenarios, no threat and no deadline, one small
#: request that sounds like ordinary paperwork. Lexically these sit inside the
#: benign cloud, which is the point.
MILD_ARCS: Tuple[str, ...] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION REASSURE REQUEST_SENSITIVE VICTIM_COMPLY CLOSE",
    "GREET IDENTIFY_SELF INFORM CONFIRM REQUEST_SENSITIVE VICTIM_QUESTION REASSURE VICTIM_COMPLY CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM REQUEST_SENSITIVE CONFIRM REASSURE CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM VICTIM_QUESTION REASSURE REQUEST_SENSITIVE CONFIRM CLOSE",
)

# --------------------------------------------------------------------------
# Benign scenarios
#
# Five of these are hard negatives on purpose. They talk about accounts, KYC,
# OTPs and due dates, which is exactly the vocabulary the scam calls use, and
# they carry the anti-fraud markers a genuine caller actually says.
# --------------------------------------------------------------------------

# -- 1. bank_reminder (hard negative) --------------------------------------

SCENARIO_ACTS["bank_reminder"] = {
    "GREET": (
        "{GREET_WORD}, {CUST} se baat ho rahi hai?",
        "{GREET_WORD}, {CUST}? Ek chhoti si jankari deni thi.",
        "Hello ji, {CUST} bol rahe hain na?",
    ),
    "IDENTIFY_SELF": (
        "Main {BANK} ki customer care team se {NAME} bol {RAHA} hoon.",
        "{NAME} bol {RAHA} hoon {BANK} se, ye ek reminder call hai.",
        "Ji main {NAME}, {BANK} branch se, sirf yaad dilane ke liye call kiya hai.",
    ),
    "INFORM": (
        "Aapki {BANK_ITEM} agle mahine expire ho rahi hai, isliye reminder bhej rahe hain.",
        "Aapke {BANK_ITEM} ki details {TIME} tak update karwa lijiyega, aaram se.",
        "Aap chahe to branch me aaiye, ya app se khud update kar lijiye.",
        "Hum kabhi {OTP_LIT} nahi mangte, aur na koi link bhejte hain, ye dhyan rakhiyega.",
        "Aapko koi payment nahi karni hai, no action required abhi.",
        "Aapki {BANK_ITEM} ke liye sirf ek document chahiye hota hai, branch me jama ho jata hai.",
        "Ye call sirf jankari ke liye hai, koi form nahi bharna abhi.",
    ),
    "REASSURE": (
        "Koi jaldi nahi hai, jab time mile tab kar lijiyega.",
        "Aapki marzi hai, aap online kar lijiye ya branch aa jaiye.",
        "Agar koi doubt ho to aapke {CARD} ke peeche wala toll free number use kar lijiye.",
        "Aap official website pe bhi ye sab dekh sakte hain.",
    ),
    "INSTRUCT": (
        "Branch me aaiye to ek photo ID le aaiyega, bas itna kaafi hai.",
        "App me profile section me jaakar aap khud update kar sakte hain.",
    ),
    "CLOSE": (
        "Bas itna hi tha, aapka time lene ke liye dhanyavaad, have a good day.",
        "Thank you for your time sir, koi doubt ho to branch me poochh lijiyega.",
        "Chaliye ji, dhanyavaad, aapka din achha rahe.",
    ),
}
SCENARIO_ARCS["bank_reminder"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION INFORM REASSURE CONFIRM CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM INFORM VICTIM_QUESTION REASSURE INSTRUCT CONFIRM CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM REASSURE VICTIM_QUESTION INFORM CONFIRM CLOSE",
)

# -- 2. delivery_otp (hard negative) ---------------------------------------

SCENARIO_ACTS["delivery_otp"] = {
    "GREET": (
        "{GREET_WORD}, {CUST}? Main niche gate pe khada hoon.",
        "Hello sir, delivery ke liye call kiya hai, aap ghar pe hain?",
        "{GREET_WORD} ji, parcel leke aaya hoon, do minute me aata hoon upar.",
    ),
    "IDENTIFY_SELF": (
        "Main {COMPANY} se {NAME} bol {RAHA} hoon, delivery boy.",
        "{NAME} bol {RAHA} hoon {COMPANY} se, aapka order {ORDER_ID} {LAYA} hoon.",
        "Ji main {COMPANY} wala, {NAME}, aapka parcel hai mere paas.",
    ),
    "INFORM": (
        "Aapka order {ORDER_ID} hai, prepaid hai, koi payment nahi leni hai.",
        "Parcel chhota sa hai, aapke naam pe hai, address bhi wahi hai.",
        "Sir main gate pe hoon, guard bol raha hai aapse pooch kar bhejenge.",
        "Aap chahe to app me bhi status check kar lijiye, live dikh raha hoga.",
    ),
    "REQUEST_SENSITIVE": (
        "Aapke phone pe delivery ka {OTP_WORD} aaya hoga, wahi bata dijiye.",
        "Ek {OTP_WORD} aaya hai message me, wo bol dijiye to main close kar doon.",
        "Delivery confirm karne ke liye {OTP_WORD} chahiye, bas wahi.",
    ),
    "REASSURE": (
        "Sirf delivery ka code hai sir, koi payment nahi, koi link nahi.",
        "Aap chahe to parcel dekh kar hi code dijiye, koi jaldi nahi.",
        "Main yahin ruka hoon, aap aaram se dekh lijiye.",
    ),
    "CLOSE": (
        "Ho gaya sir, parcel de diya, thank you for your time.",
        "Thik hai ji, aapka parcel gate pe guard ko de diya hai, dhanyavaad.",
        "Chaliye sir, dhanyavaad, have a good day.",
    ),
    "INSTRUCT": (
        "Aap ek baar box check kar lijiye, phir code dijiyega.",
        "Guard ko bol dijiye, main unke paas hi khada hoon.",
    ),
}
SCENARIO_ARCS["delivery_otp"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM REQUEST_SENSITIVE VICTIM_COMPLY REASSURE CONFIRM CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM VICTIM_QUESTION REASSURE REQUEST_SENSITIVE VICTIM_COMPLY CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INSTRUCT INFORM REQUEST_SENSITIVE VICTIM_COMPLY REASSURE CLOSE",
)

# -- 3. family_call --------------------------------------------------------

SCENARIO_ACTS["family_call"] = {
    "CONFIRM": (
        "Arre haan, boliye, kaise ho aap?",
        "Haan ji main hi hoon, sab thik?",
        "Kuch nahi, bas baithe hain, aap sunao.",
        "Haan bolo, kaisa hai sab wahan?",
    ),
    "VICTIM_QUESTION": (
        "Achha, aur wahan sab kaise hain?",
        "Kab aa rahe ho phir?",
        "Khana khaya aapne?",
        "Aur batao, kya chal raha hai aajkal?",
    ),
    "GREET": (
        "{GREET_WORD} beta, kya kar rahe ho?",
        "Haan hello, {CUST}? Kaise ho?",
        "{GREET_WORD}, khana kha liya?",
    ),
    "IDENTIFY_SELF": (
        "Main hoon, {NAME}, pehchana nahi kya?",
        "Arre main bol {RAHA} hoon, {NAME}, naya number hai mera.",
        "{NAME} hoon beta, mummy ka phone se kar {RAHA} hoon.",
    ),
    "SMALLTALK": (
        "Aur sab thik? Ghar pe sab theek hain?",
        "Kal baarish bahut hui yahan, aapke wahan kaisa mausam hai?",
        "{RELATIVE} ka result aa gaya, achha hua hai.",
        "Aaj bahut garmi hai, AC bhi kaam nahi kar raha.",
    ),
    "INFORM": (
        "Main {TIME} aa {RAHA} hoon, ticket ho gaya hai.",
        "{RELATIVE} ki chhutti {TIME} se hai, sab aa jayenge.",
        "Maa ki tabiyat ab thik hai, {DOCTOR} ne dawai badal di hai.",
        "Main {CITY} me hi hoon abhi, kaam thoda lamba ho gaya.",
    ),
    "REASSURE": (
        "Tension mat lo, sab sambhal jayega.",
        "Koi jaldi nahi, jab time mile tab kar lena.",
    ),
    "CLOSE": (
        "Achha chalo rakhta hoon, khana khaakar so jana.",
        "Thik hai beta, apna khayal rakhna, baad me baat karte hain.",
        "Chalo phir, {TIME} baat karte hain, bye.",
    ),
}
SCENARIO_ARCS["family_call"] = (
    "GREET CONFIRM SMALLTALK SMALLTALK:caller INFORM VICTIM_QUESTION INFORM CONFIRM CLOSE",
    "GREET SMALLTALK IDENTIFY_SELF SMALLTALK:caller INFORM CONFIRM REASSURE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION SMALLTALK:caller CONFIRM CLOSE",
)

# -- 4. telemarketing (hard negative, sales pressure but no fraud) ---------

SCENARIO_ACTS["telemarketing"] = {
    "IDENTIFY_SELF": (
        "Main {TELCO} se {NAME} bol {RAHA} hoon, ek offer ke baare me.",
        "{NAME} bol {RAHA} hoon {JOB_COMPANY} se, do minute le {SAKTA} hoon aapke?",
        "Ji main {NAME}, {BANK} ke sales department se, ek scheme batani thi.",
    ),
    "INFORM": (
        "Hamara naya {PRODUCT} aaya hai, {FEE} monthly, pehle teen mahine free.",
        "Aapke area me ab {PRODUCT} available hai, rate bhi pehle se kam hai.",
        "Aapke {BANK_ITEM} pe ek lifetime free offer chal raha hai abhi.",
        "Ismein koi hidden charge nahi hai, sab bill me likha aayega.",
    ),
    "REASSURE": (
        "Aapki marzi hai, abhi decide karne ki koi zaroorat nahi.",
        "Koi jaldi nahi hai, aap soch kar bata dijiyega.",
        "Main sirf information de {RAHA} hoon, aap chahe to mana kar dijiye.",
        "Aap official website pe bhi ye plan dekh sakte hain.",
    ),
    "INSTRUCT": (
        "Agar interest ho to main ek SMS bhej {DETA} hoon, usme details hongi.",
        "Aap toll free number pe call karke bhi ye plan le sakte hain.",
    ),
    "CLOSE": (
        "Koi baat nahi sir, thank you for your time, have a good day.",
        "Thik hai ji, agar mann kare to bata dijiyega, dhanyavaad.",
        "Chaliye, aapka time lene ke liye sorry, dhanyavaad.",
    ),
}
SCENARIO_ARCS["telemarketing"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION INFORM REASSURE CONFIRM CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM REASSURE VICTIM_RESIST REASSURE CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM INFORM VICTIM_QUESTION INSTRUCT REASSURE CLOSE",
)

# -- 5. appointment_reminder ----------------------------------------------

SCENARIO_ACTS["appointment_reminder"] = {
    "IDENTIFY_SELF": (
        "Main {HOSPITAL} se {NAME} bol {RAHA} hoon, appointment reminder ke liye.",
        "{NAME} bol {RAHA} hoon {HOSPITAL} ke front desk se.",
        "Ji main {NAME}, {DOCTOR} ke clinic se call kar {RAHA} hoon.",
    ),
    "INFORM": (
        "Aapka appointment {DOCTOR} ke saath {TIME} ka hai, bas yaad dilane ke liye call kiya.",
        "{DOCTOR} {TIME} available hain, aapka number {ORDER_ID} hai.",
        "Reports ready hain, aap {TIME} aa kar le lijiyega.",
        "Aane se pehle kuch khana mat, blood test hai.",
    ),
    "REASSURE": (
        "Koi jaldi nahi, agar nahi aa paye to hum reschedule kar denge.",
        "Aapki marzi, jo time thik lage bata dijiyega.",
        "Payment counter pe hi hoti hai, phone pe kuch nahi lete hum.",
    ),
    "INSTRUCT": (
        "Purani file aur reports saath le aaiyega.",
        "Reception pe naam bata dijiyega, wo aage bhej denge.",
    ),
    "CLOSE": (
        "Thik hai ji, {TIME} milte hain, dhanyavaad.",
        "Bas itna hi tha, thank you for your time.",
        "Chaliye, apna khayal rakhiye, dhanyavaad.",
    ),
}
SCENARIO_ARCS["appointment_reminder"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION INSTRUCT CONFIRM CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM REASSURE VICTIM_QUESTION INFORM CONFIRM CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM INSTRUCT REASSURE CONFIRM CLOSE",
)

# -- 6. customer_support (hard negative: real complaint follow up and a real
#       electricity bill notice with a due date) ---------------------------

SCENARIO_ACTS["customer_support"] = {
    "IDENTIFY_SELF": (
        "Main {BANK} ke customer care se {NAME} bol {RAHA} hoon, aapki complaint ke baare me.",
        "{NAME} bol {RAHA} hoon {UTIL} se, aapke bill ke silsile me.",
        "Ji main {NAME}, {TELCO} customer care se, aapka ticket {TICKET} hai.",
    ),
    "INFORM": (
        "Aapki complaint number {TICKET} resolve ho gayi hai, {MONEY} refund ho gaya hai.",
        "Aapka bill {FEE} ka hai, due date {TIME} hai, ye sirf reminder hai.",
        "Aapke {BANK_ITEM} pe jo galat charge laga tha wo reverse kar diya gaya hai.",
        "Aap chahe to online pay kar lijiye ya counter pe jama kar dijiye.",
        "Hum kabhi {OTP_LIT} nahi mangte, agar koi mange to samajh jaiye ki fraud hai.",
        "Koi action nahi chahiye abhi, ye sirf jankari ke liye call hai.",
    ),
    "REASSURE": (
        "Koi jaldi nahi hai, due date tak kabhi bhi kar sakte hain.",
        "Aap chahe to customer care par call karke confirm kar lijiye.",
        "Aapki marzi hai, hum bas inform kar rahe hain.",
        "Agar late ho gaya to bhi koi connection nahi katega, bas late fee lagegi.",
    ),
    "INSTRUCT": (
        "App me bill section me jaakar aap pura detail dekh sakte hain.",
        "Receipt sambhal ke rakhiyega, ticket number {TICKET} likha hoga usme.",
    ),
    "CLOSE": (
        "Thank you for your time sir, aur koi help chahiye to bataiyega.",
        "Chaliye ji, aapki problem solve ho gayi, dhanyavaad.",
        "Bas itna hi tha, have a good day.",
    ),
}
SCENARIO_ARCS["customer_support"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION INFORM REASSURE CONFIRM CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM REASSURE VICTIM_QUESTION INSTRUCT CONFIRM CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM INFORM REASSURE VICTIM_QUESTION INFORM CONFIRM CLOSE",
)

# -- 7. survey_call --------------------------------------------------------

SCENARIO_ACTS["survey_call"] = {
    "IDENTIFY_SELF": (
        "Main {BANK} ke feedback team se {NAME} bol {RAHA} hoon.",
        "{NAME} bol {RAHA} hoon, hum {SURVEY_TOPIC} ka ek chhota survey kar rahe hain.",
        "Ji main {NAME}, {JOB_COMPANY} se, ek do sawaal poochhne the.",
    ),
    "INFORM": (
        "Bas do minute lagenge, sirf {SURVEY_TOPIC} ke baare me feedback chahiye.",
        "Ismein koi personal detail nahi poochhenge, sirf rating.",
        "Ye survey sirf jankari ke liye hai, koi selling nahi hai.",
        "Aapka feedback service improve karne me use hoga.",
    ),
    "INSTRUCT": (
        "Aap {SURVEY_TOPIC} ko 1 se 5 me kitna denge?",
        "Pichhli baar branch gaye the to staff ka behaviour kaisa laga, bata dijiye.",
        "Ek line me bata dijiye, {SURVEY_TOPIC} me kya sudhar chahiye?",
    ),
    "REASSURE": (
        "Aapki marzi hai, aap chahe to skip kar sakte hain.",
        "Koi jaldi nahi, jitna time de sakein utna hi kaafi hai.",
        "Aap chahe to abhi mana kar dijiye, hum dobara call nahi karenge.",
    ),
    "CLOSE": (
        "Thank you for your time ji, aapka feedback note kar liya.",
        "Bas itna hi, dhanyavaad, have a good day.",
        "Chaliye, aapka bahut dhanyavaad.",
    ),
}
SCENARIO_ARCS["survey_call"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM INSTRUCT VICTIM_COMPLY INSTRUCT CONFIRM CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM REASSURE INSTRUCT CONFIRM INFORM CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION REASSURE INSTRUCT CONFIRM CLOSE",
)

# -- 8. school_notice (hard negative: fee due date) ------------------------

SCENARIO_ACTS["school_notice"] = {
    "IDENTIFY_SELF": (
        "Main {SCHOOL} se {NAME} bol {RAHA} hoon, class teacher.",
        "{NAME} bol {RAHA} hoon {SCHOOL} ke office se.",
        "Ji main {NAME}, {SCHOOL} se, {RELATIVE} ke baare me baat karni thi.",
    ),
    "INFORM": (
        "Parent teacher meeting {TIME} hai, aap aa jaiyega.",
        "{RELATIVE} ka {SUBJECT} thoda kamzor hai, thoda dhyan dijiyega ghar pe.",
        "Fees ki last date {TIME} hai, ye sirf reminder hai, koi jaldi nahi.",
        "Fees office counter pe hi jama hoti hai, hum phone pe payment nahi lete.",
        "Kal school jaldi chhut jayega, sports day ki practice hai.",
        "Hum kabhi {OTP_LIT} nahi mangte aur na hi koi link bhejte hain, dhyan rakhiyega.",
    ),
    "REASSURE": (
        "Koi jaldi nahi hai, aap jab aayein tab jama kar dijiyega.",
        "Aap chahe to office aakar mil lijiye, principal se baat ho jayegi.",
        "Aapki marzi, jo time thik lage.",
    ),
    "INSTRUCT": (
        "Diary me bhi likh diya hai, ek baar dekh lijiyega.",
        "Aane se pehle office me naam likhwa dijiyega.",
    ),
    "CLOSE": (
        "Thik hai ji, {TIME} milte hain, dhanyavaad.",
        "Bas itna hi tha, thank you for your time.",
        "Chaliye, {RELATIVE} ko bol dijiyega, dhanyavaad.",
    ),
}
SCENARIO_ARCS["school_notice"] = (
    "GREET CONFIRM IDENTIFY_SELF INFORM VICTIM_QUESTION INFORM REASSURE CONFIRM CLOSE",
    "GREET IDENTIFY_SELF CONFIRM INFORM INSTRUCT VICTIM_QUESTION REASSURE CONFIRM CLOSE",
    "GREET CONFIRM IDENTIFY_SELF INFORM REASSURE INFORM VICTIM_QUESTION REASSURE CONFIRM CLOSE",
)

# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

#: Which callee pool to fall back to for a given call kind.
_CALLEE_POOLS = {
    "hard": (SHARED_CALLEE,),
    "mild": (MILD_SCAM, SHARED_CALLEE),
    "flat": (BENIGN_CALLEE, SHARED_CALLEE),
}
_CALLER_POOLS = {
    "hard": (SHARED_CALLER,),
    "mild": (MILD_SCAM, SHARED_CALLER),
    "flat": (BENIGN_CALLER, SHARED_CALLER),
}


def parse_arc(arc: str) -> List[Tuple[str, str]]:
    """Turn an arc string into [(act, speaker), ...].

    "SMALLTALK:caller" overrides the default speaker for that one turn.
    """
    out: List[Tuple[str, str]] = []
    for item in arc.split():
        if ":" in item:
            act, speaker = item.split(":", 1)
        else:
            act, speaker = item, ACT_SPEAKER.get(item, "caller")
        if act not in DIALOGUE_ACTS:
            raise ValueError(f"unknown dialogue act {act!r} in arc")
        if speaker not in ("caller", "callee"):
            raise ValueError(f"unknown speaker {speaker!r} in arc")
        out.append((act, speaker))
    return out


#: Scenarios that can be run as a soft "just confirming a record" call. A
#: relative in hospital or a lottery win cannot be delivered that way, and
#: forcing them into the mild frame produces calls that make no sense.
MILD_CAPABLE_SCENARIOS: Tuple[str, ...] = (
    "kyc_freeze",
    "otp_theft",
    "sim_block",
    "loan_approval",
    "job_offer",
    "courier_customs",
    "electricity_disconnect",
)

#: Acts a mild scam never borrows from its scenario pool. The mild call keeps
#: the scenario subject matter (bank, SIM, loan) but has to stay lexically
#: soft, so the pressure lines and the hard sensitive requests are off limits.
MILD_EXCLUDED_ACTS: Tuple[str, ...] = (
    "THREAT",
    "DEADLINE",
    "PRESSURE_ESCALATE",
    "ISOLATE",
    "REQUEST_SENSITIVE",
    "INSTRUCT",
    "REASSURE",
)

#: How often a turn takes its scenario specific line rather than a generic
#: shared one. High enough that calls sound like their scenario, low enough
#: that the shared lines still vary the corpus.
SCENARIO_PATTERN_BIAS = 0.7


def arcs_for(scenario: str, style: str) -> Tuple[str, ...]:
    if style == "mild":
        return MILD_ARCS
    return SCENARIO_ARCS[scenario]


def _pattern_pool(scenario: str, act: str, speaker: str, style: str) -> List[Tuple[str, str]]:
    """Every candidate pattern for one (scenario, act, speaker, style).

    Returns (template_id, pattern) pairs. The scenario pool comes first, then
    the style fallbacks, then the generic shared pools. Mild scam turns look at
    MILD_SCAM before the scenario pool, because the whole point of the mild
    style is that the scenario keeps its subject matter but drops the pressure.
    The first pool that has the act wins outright: a benign call must not
    borrow the pressure lines or the suspicious victim lines written for scam
    calls, and a mild scam must not borrow the hard ones.
    """
    out: List[Tuple[str, str]] = []
    seen = set()

    def add(source: str, patterns: Sequence[str]) -> None:
        for i, pat in enumerate(patterns):
            if pat in seen:
                continue
            seen.add(pat)
            out.append((f"{source}/{act}/{i}", pat))

    if style == "mild" and act in MILD_SCAM:
        add("mild", MILD_SCAM[act])
    scenario_pool = SCENARIO_ACTS.get(scenario, {}).get(act)
    if scenario_pool and not (style == "mild" and act in MILD_EXCLUDED_ACTS):
        add(scenario, scenario_pool)
    pools = _CALLEE_POOLS[style] if speaker == "callee" else _CALLER_POOLS[style]
    for pool in pools:
        if act in pool:
            add("shared", pool[act])
            break
    return out


def build_dialogue(
    scenario: str,
    rng: random.Random,
    style: str = "hard",
) -> Tuple[List[TurnDraft], Dict[str, object]]:
    """Generate one whole call.

    style is "hard" (full scam pressure ramp), "mild" (soft scam) or "flat"
    (benign). Returns the rendered turns plus the metadata that makes the call
    reproducible and auditable: which arc, which templates, which slot values.
    """
    if style not in ("hard", "mild", "flat"):
        raise ValueError(f"unknown style {style!r}")
    arc = rng.choice(list(arcs_for(scenario, style)))
    binder = SlotBinder(scenario)
    drafts: List[TurnDraft] = []
    used_ids: set = set()

    for act, speaker in parse_arc(arc):
        pool = _pattern_pool(scenario, act, speaker, style)
        if not pool:
            raise ValueError(f"no pattern for {scenario}/{act}/{speaker}/{style}")
        # A callee reads out a code only when the caller has just asked for
        # one. Without this the victim starts reciting digits in the middle of
        # a call about an aadhaar number, which no annotator would accept.
        if act == "VICTIM_COMPLY":
            last_caller = next(
                (d for d in reversed(drafts) if d.speaker == "caller"), None
            )
            asked_otp = bool(
                last_caller and any(tag.endswith("-OTP") for tag in last_caller.bio)
            )
            with_code = [p for p in pool if "{OTP_CODE}" in p[1]]
            without_code = [p for p in pool if "{OTP_CODE}" not in p[1]]
            if asked_otp and with_code:
                pool = with_code
            elif without_code:
                pool = without_code
        # The first thing the callee says after the greeting is "yes, speaking",
        # not a mid-call acknowledgement, so that turn always uses the shared
        # identity-confirmation lines.
        if act == "CONFIRM" and drafts and drafts[-1].act == "GREET":
            own = SCENARIO_ACTS.get(scenario, {}).get("CONFIRM", ())
            pool = [(f"{scenario}/CONFIRM/{i}", p) for i, p in enumerate(own)]
            pool += [
                (f"shared/CONFIRM/{i}", p)
                for i, p in enumerate(SHARED_CALLEE["CONFIRM"])
            ]
        fresh = [p for p in pool if p[0] not in used_ids]
        candidates = fresh if fresh else pool
        specific = [p for p in candidates if not p[0].startswith("shared/")]
        generic = [p for p in candidates if p[0].startswith("shared/")]
        if specific and generic:
            side = specific if rng.random() < SCENARIO_PATTERN_BIAS else generic
        else:
            side = specific or generic
        template_id, pattern = rng.choice(side)
        used_ids.add(template_id)
        text, tokens, bio, slots = render_pattern(pattern, binder, rng)
        drafts.append(
            TurnDraft(
                act=act,
                speaker=speaker,
                text=text,
                tokens=tokens,
                bio=bio,
                template_id=template_id,
                slots=slots,
            )
        )

    meta: Dict[str, object] = {
        "arc": arc,
        "style": style,
        "templates": [d.template_id for d in drafts],
        "slots": dict(binder.bound),
    }
    return drafts, meta


# --------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------


def template_counts() -> Dict[str, int]:
    """How many distinct turn templates each scenario can draw on.

    Counted the way the plan asks for it: scenario specific patterns plus the
    shared fallbacks the arcs actually reach, plus the arcs themselves.
    """
    counts: Dict[str, int] = {}
    for scenario in list(SCAM_SCENARIOS) + list(BENIGN_SCENARIOS):
        scam = scenario in SCAM_SCENARIOS
        styles = ("hard", "mild") if scam else ("flat",)
        seen = set()
        for style in styles:
            for arc in arcs_for(scenario, style):
                for act, speaker in parse_arc(arc):
                    for tid, pat in _pattern_pool(scenario, act, speaker, style):
                        seen.add(pat)
        counts[scenario] = len(seen)
    return counts


#: Surface forms that must never appear in the literal part of a pattern.
#: The annotation guideline for this corpus is simple: every entity mention is
#: produced by a slot, so the gold BIO is complete by construction. A literal
#: "OTP" or "account" sitting outside a slot would be a silent false negative
#: in the gold, and a tagger trained on it would see the same word tagged in
#: one turn and untagged in the next. This list plus the check below is what
#: stops that from creeping back in.
_LITERAL_FORBIDDEN: Tuple[str, ...] = (
    "otp", "one time password", "verification code", "account", "khata",
    "kyc", "netbanking", "net banking", "aadhaar", "aadhar", "pan card",
    "card", "cvv", "mpin", "atm pin", "upi", "vpa", "qr code",
    "payment link", "police", "cyber cell", "cyber crime", "crime branch",
    "cbi", "rbi", "trai", "customs", "narcotics", "interpol", "magistrate",
    "rupees", "rupaye", "lakh", "crore", "hazaar",
)


def literal_entity_leaks() -> List[str]:
    """Patterns whose literal text mentions an entity outside a slot."""
    leaks: List[str] = []
    pools: List[Tuple[str, str, Sequence[str]]] = []
    for scenario, acts in SCENARIO_ACTS.items():
        for act, patterns in acts.items():
            pools.append((scenario, act, patterns))
    for name, pool in (
        ("shared_caller", SHARED_CALLER),
        ("shared_callee", SHARED_CALLEE),
        ("mild", MILD_SCAM),
        ("benign_callee", BENIGN_CALLEE),
    ):
        for act, patterns in pool.items():
            pools.append((name, act, patterns))

    for source, act, patterns in pools:
        for pattern in patterns:
            literal = " " + _SLOT_RE.sub(" ", pattern).lower() + " "
            for form in _LITERAL_FORBIDDEN:
                if re.search(r"(?<![a-z])" + re.escape(form) + r"(?![a-z])", literal):
                    leaks.append(f"{source}/{act}: literal {form!r} in {pattern!r}")
    return leaks


def validate_grammar() -> Dict[str, object]:
    """Render every template once and report anything broken.

    Checks that every slot exists, every pattern produces tokens, every BIO
    sequence is well formed, that the detokenised text re-tokenises to the same
    tokens, and that no entity surface form appears outside a slot. Returns a
    report rather than raising, so the self-test can print all the problems at
    once.
    """
    rng = random.Random(7)
    problems: List[str] = []
    rendered = 0
    detok_fallbacks = 0
    slot_use: Dict[str, int] = {}

    all_pools: List[Tuple[str, str, Sequence[str]]] = []
    for scenario, acts in SCENARIO_ACTS.items():
        for act, patterns in acts.items():
            all_pools.append((scenario, act, patterns))
    for name, pool in (
        ("shared_caller", SHARED_CALLER),
        ("shared_callee", SHARED_CALLEE),
        ("mild", MILD_SCAM),
        ("benign_callee", BENIGN_CALLEE),
    ):
        for act, patterns in pool.items():
            all_pools.append((name, act, patterns))

    for scenario, act, patterns in all_pools:
        if act not in DIALOGUE_ACTS:
            problems.append(f"{scenario}/{act}: not a dialogue act")
        for pattern in patterns:
            for slot in _SLOT_RE.findall(pattern):
                slot_use[slot] = slot_use.get(slot, 0) + 1
                if slot not in SLOTS:
                    problems.append(f"{scenario}/{act}: unknown slot {slot}")
            binder = SlotBinder(scenario if scenario in SCENARIO_ARCS else "kyc_freeze")
            try:
                text, tokens, bio, _ = render_pattern(pattern, binder, rng)
            except Exception as exc:  # pragma: no cover - reported, not raised
                problems.append(f"{scenario}/{act}: render failed: {exc}")
                continue
            rendered += 1
            if len(tokens) != len(bio):
                problems.append(f"{scenario}/{act}: token and BIO length differ")
            if tokenize(text) != tokens:
                detok_fallbacks += 1
                problems.append(f"{scenario}/{act}: detokenise round trip failed: {text!r}")
            prev = "O"
            for tag in bio:
                if tag.startswith("I-") and prev not in (tag, "B-" + tag[2:]):
                    problems.append(f"{scenario}/{act}: dangling {tag} in {text!r}")
                prev = tag

    for scenario in list(SCAM_SCENARIOS) + list(BENIGN_SCENARIOS):
        if scenario not in SCENARIO_ARCS:
            problems.append(f"{scenario}: no arcs defined")
        if scenario not in SCENARIO_ACTS:
            problems.append(f"{scenario}: no act pool defined")
    for name, spec in SLOTS.items():
        # Gendered verb slots have exactly two forms by definition, and
        # OTP_LIT has exactly one on purpose, so the eight-option floor does
        # not apply to either.
        if name in GENDERED_SLOTS and name != "NAME":
            continue
        if name in FIXED_FORM_SLOTS:
            continue
        if len(spec.options) < 8:
            problems.append(f"slot {name}: only {len(spec.options)} options")
    for scenario, overrides in SCENARIO_SLOTS.items():
        for slot, options in overrides.items():
            if slot not in SLOTS:
                problems.append(f"{scenario}: override for unknown slot {slot}")
            if len(options) < 8:
                problems.append(f"{scenario}/{slot}: only {len(options)} override options")

    leaks = literal_entity_leaks()
    problems.extend(leaks)

    for scenario, arcs in list(SCENARIO_ARCS.items()) + [("mild", MILD_ARCS)]:
        for arc in arcs:
            pairs = parse_arc(arc)
            for (act_a, spk_a), (act_b, _spk_b) in zip(pairs, pairs[1:]):
                if act_a == "VICTIM_QUESTION" and _spk_b == "callee":
                    problems.append(
                        f"{scenario}: {act_b} follows a callee question with "
                        f"no answer in between"
                    )

    counts = template_counts()
    for scenario, n in counts.items():
        if n < 12:
            problems.append(f"{scenario}: only {n} reachable templates, need 12")

    return {
        "patterns_rendered": rendered,
        "problems": problems,
        "literal_entity_leaks": len(leaks),
        "detokenise_fallbacks": detok_fallbacks,
        "templates_per_scenario": counts,
        "n_slots": len(SLOTS),
        "unused_slots": sorted(set(SLOTS) - set(slot_use)),
    }


def _demo(scenario: str, style: str, seed: int = 3) -> None:
    rng = random.Random(seed)
    drafts, meta = build_dialogue(scenario, rng, style)
    print(f"\n=== {scenario} [{style}] arc={meta['arc'][:60]} ...")
    for d in drafts:
        ents = [
            (t, b) for t, b in zip(d.tokens, d.bio) if b != "O"
        ]
        marker = "  <" + ", ".join(f"{t}:{b}" for t, b in ents) + ">" if ents else ""
        print(f"  [{d.speaker:6s}] {d.act:18s} {d.text}{marker}")


if __name__ == "__main__":
    report = validate_grammar()
    print(f"patterns rendered : {report['patterns_rendered']}")
    print(f"slots             : {report['n_slots']}")
    print(f"unused slots      : {report['unused_slots']}")
    counts = report["templates_per_scenario"]
    print("templates per scenario:")
    for k in sorted(counts):
        print(f"  {k:24s} {counts[k]}")
    problems = report["problems"]
    print(f"problems          : {len(problems)}")
    for p in problems[:20]:
        print("  !", p)

    _demo("kyc_freeze", "hard")
    _demo("digital_arrest", "hard", seed=11)
    _demo("otp_theft", "mild", seed=5)
    _demo("bank_reminder", "flat", seed=2)
    _demo("delivery_otp", "flat", seed=9)
