"""Hinglish domain vocabulary for fraud call analysis.

Three jobs:

1. Per-token language identification. Romanised Hindi and English share the
   Latin alphabet, so a plain script check cannot separate them. Word lists
   plus a small set of orthographic rules do the job well enough on the
   telephone vocabulary this system sees.
2. Gazetteers for the CRF tagger. Classical sequence taggers gain a lot from
   membership features, and fraud vocabulary is small and closed enough that a
   hand-built list covers most of it.
3. Weighted term dictionaries for the interpretable rule baseline and for the
   lexical-arousal side of the prosody-intent mismatch feature.

Everything is lowercase. Look-ups go through the helper functions at the
bottom, which handle the spelling variation that romanised Hindi always has
(kijiye / kijie / kijiyega).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# 1. Language identification word lists
# --------------------------------------------------------------------------

#: Romanised Hindi. Function words first (they carry most of the signal for
#: language ID), then the content words that show up on phone calls.
HINDI_WORDS: Set[str] = set(
    """
    aap aapka aapki aapke aapko aapse apna apne apni ap hum hume humein humara humari
    hamara hamari mai main mein mera meri mere mujhe mujhko tum tumhara tera teri
    tu wo woh ye yeh is us isi usi inka unka inke unke iska uska iski uski
    kya kyu kyun kyon kaise kaisa kaisi kab kahan kaun kitna kitni kitne kaunsa
    hai hain ho hoga hogi honge hota hoti hote hua hui hue tha thi the raha rahi rahe
    kar karo kare karna karne karta karti karte kiya kiye ki ka ke ko se par pe
    aur ya lekin magar phir bhi to toh na nahi nahin mat bilkul haan han ji
    abhi turant jaldi foran fauran der late samay waqt time din raat subah shaam
    aaj kal parso ghanta ghante minute mint second
    paisa paise rupaye rupees rupya lakh lakhon crore karod hazaar hazar
    bank khata khate account number
    band bandh block hoga jayega jaayega ho jaega
    police thana adalat court warrant giraftar giraftari
    galat sahi thik theek achha accha bura
    bol bolo bolna bolta bolti bolte raha rahi rahe hoon hun hu
    bataiye bataye batao batana kahiye kahna sunna suniye suno
    dijiye dijiyega dena dena de do diya deni
    lijiye lena lo liya
    dekhiye dekho dekhna dekhiyega
    samajh samjhe samjha samajhiye
    chahiye zaroorat zarurat jarurat
    problem dikkat pareshani samasya
    madad help sahayata
    naam pata address ghar makan
    beta beti bhai behen behan maa mata pita papa mummy daddy chacha mama nana dada
    sahab saheb sahib sir madam ji
    kripya kripaya kripayaa meherbani
    dhanyavaad dhanyawad shukriya
    namaste namaskar
    haanji jee bilkul theekhai
    varna warna nahi to otherwise
    agar jab tab tak taak
    sab sabhi sabse kuch kuchh koi kisi
    apni apna khud swayam
    andar bahar upar niche aage peeche
    bahut bohot bahot zyada jyada kam thoda thodi
    pura pura poora poori sara saara
    ek do teen char chaar paanch panch chhe chah saat aath nau das
    gyarah barah bees tees chalis pachas saath sattar assi nabbe sau
    pehla pehle dusra doosra teesra aakhri last
    milega milegi milenge mila mili
    lagega lagegi lagta lagti
    chalega chalegi chal chalo
    aayega aayegi aana aaya aayi jana jaana gaya gayi
    rakho rakhiye rakhna rakhe
    khatam khatm bandh chalu shuru
    sach jhoot dhoka fraud thagi
    seedha sidha direct
    matlab yani arth
    wapas vapas dobara phir se
    dhyan khayal savdhan saavdhan
    khatra khatre risk
    jaruri zaruri zaroori important
    officer adhikari karmchari
    verify verification pramaan praman
    file case mamla
    signature dastakhat
    dastavez document kagaz kaagaz
    bharo bharna bhara
    khol kholo kholiye kholna
    """.split()
)

#: English words that turn up in Indian phone conversations. Anything not in
#: either list falls back to the orthographic rules.
ENGLISH_WORDS: Set[str] = set(
    """
    the a an and or but if then so that this these those there here
    is are was were be been being am do does did done doing have has had
    will would shall should can could may might must
    i you he she it we they me him her us them my your his their our
    to from with without for of on in at by about into over under
    not no yes okay ok please thanks thank sorry
    account bank branch card credit debit netbanking banking balance
    ifsc neft imps rtgs upi vpa gpay phonepe paytm bhim wallet qr scan
    otp mpin cvv atm sms apk anydesk teamviewer
    transaction transfer payment pay paid amount rupees rupee money fund funds
    otp password pin cvv code number verification verify verified
    kyc aadhaar aadhar pan passport license licence document documents
    customer service support helpline executive officer department
    call calling caller phone mobile sim network telecom
    police cyber crime cell station court warrant arrest arrested legal notice
    case complaint fir investigation officer inspector commissioner
    rbi reserve trai cbi customs income tax narcotics
    block blocked blocking suspend suspended suspension freeze frozen deactivate
    urgent urgently immediately immediate quickly fast now today tomorrow
    minute minutes hour hours day days
    security secure safe unsafe risk fraud fraudulent scam suspicious
    link click download install app application software remote access
    screen share sharing anydesk teamviewer
    delivery deliver package parcel courier order shipment tracking
    address pincode location
    prize lottery winner won win lucky draw gift offer scheme
    loan approved approval eligible eligibility interest emi
    job offer interview salary position vacancy company
    electricity bill meter connection disconnect disconnection reading
    insurance policy premium claim maturity
    appointment schedule reminder confirm confirmation booking
    survey feedback rating question questions
    school teacher parent student class exam result fee fees
    family father mother brother sister son daughter uncle aunt
    hospital doctor medical emergency accident
    please kindly sir madam
    name email
    yes right correct wrong true false
    understand understood listen listening hear hearing tell told say said
    give given take taken send sent receive received
    open close start stop continue wait waiting hold holding
    problem issue error mistake
    system server portal website online offline
    process procedure step steps instruction instructions
    minutes seconds
    government official department ministry
    """.split()
)

#: Words that belong to both languages or to neither. They must never count as
#: a code switch, otherwise the switch-point statistics become noise.
#: Discourse particles and address terms that both languages use identically.
#: Acronyms like OTP and UPI are deliberately NOT here: they are English
#: tokens, and the entity-language alignment feature depends on counting them
#: on the English side.
UNIVERSAL_WORDS: Set[str] = set(
    """
    ok okay hmm hm uh um ah oh hello bye
    sir madam ji sahab
    a b c d e f g h i j k l m n o p q r s t u v w x y z
    """.split()
)

#: The Hindi list above is written by hand and picks up English loanwords that
#: Hindi speakers use constantly (account, block, police, case). For
#: code-switch accounting those tokens belong to the English side: they are
#: exactly the English financial and legal vocabulary that the
#: entity-language alignment feature is looking for. Resolving the overlaps
#: here, once, keeps the three lists disjoint no matter how they are edited
#: later. Universal wins over both, then English wins over Hindi.
HINDI_WORDS -= UNIVERSAL_WORDS
ENGLISH_WORDS -= UNIVERSAL_WORDS
BORROWED_WORDS: Set[str] = HINDI_WORDS & ENGLISH_WORDS
HINDI_WORDS -= BORROWED_WORDS

# Orthographic hints for words in neither list. Romanised Hindi has a
# distinctive shape: doubled vowels, retroflex digraphs, and endings that
# English words rarely take.
_HINDI_SUFFIXES: Tuple[str, ...] = (
    "iye", "iyega", "ayega", "ayegi", "ayenge", "aoge", "aogi",
    "unga", "ungi", "enge", "engi", "oge", "ogi",
    "wala", "wali", "wale", "kar", "ke", "ka", "ki", "ko",
    "ega", "egi", "ta", "ti", "te", "na", "ne", "ni", "hai", "hain",
)
_HINDI_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"aa|ee|oo|ii"),                 # doubled vowels
    re.compile(r"(?<![sc])h[aeiou]"),           # aspirated consonant plus vowel,
                                                # excluding English sh- and ch-
    re.compile(r"^(kya|kyu|kah|keh|kar|ban|hoj|nah|jaa|aay|bat|dij|kij|leej)"),
    re.compile(r"(ao|ai|au)$"),                 # batao, karvai, banao
)
_ENGLISH_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(tion|sion|ment|ness|able|ible|ance|ence|ing|ed|ly|ity|ous)$"),
    re.compile(r"^(un|re|de|pre|dis|inter|trans|con|com|pro|sub)"),
    re.compile(r"[qwxz]"),
    re.compile(r"(ch|sh|th|ck|ph)$"),           # branch, cash, month, block, graph
    re.compile(r"^[bcdfghjklmnpqrstvwxyz]{2}"), # English tolerates initial clusters
)


# --------------------------------------------------------------------------
# 2. Weighted term dictionaries
# --------------------------------------------------------------------------

#: Urgency. Weight is how strongly the term compresses the victim's decision
#: time. These weights feed the interpretable rule baseline and the lexical
#: arousal signal, so they are deliberately hand-set and inspectable.
URGENCY_TERMS: Dict[str, float] = {
    "abhi": 0.55, "turant": 0.90, "foran": 0.85, "fauran": 0.85,
    "jaldi": 0.70, "immediately": 0.90, "immediate": 0.80, "urgent": 0.85,
    "urgently": 0.90, "right now": 0.85, "now": 0.40, "quickly": 0.70,
    "hurry": 0.75, "fast": 0.55, "asap": 0.85, "last chance": 0.90,
    "aakhri": 0.75, "aakhri mauka": 0.95, "der": 0.45, "time nahi": 0.85,
    "waqt nahi": 0.85, "samay nahi": 0.85, "abhi ke abhi": 0.95,
    "isi waqt": 0.90, "der mat": 0.85, "wait mat": 0.80,
    "within": 0.45, "before": 0.35, "deadline": 0.75, "expire": 0.70,
    "expiring": 0.80, "final": 0.60, "final notice": 0.90,
}

#: Threat. What happens if the victim does not comply.
THREAT_TERMS: Dict[str, float] = {
    "block": 0.75, "blocked": 0.80, "block ho": 0.85, "band ho": 0.85,
    "bandh": 0.70, "suspend": 0.80, "suspended": 0.85, "suspension": 0.80,
    "freeze": 0.85, "frozen": 0.85, "seize": 0.90, "seal": 0.85,
    "deactivate": 0.75, "disconnect": 0.70, "disconnection": 0.75,
    "cancel": 0.55, "terminate": 0.75, "penalty": 0.70, "fine": 0.60,
    "arrest": 0.95, "arrested": 0.95, "giraftar": 0.95, "giraftari": 0.95,
    "warrant": 0.95, "jail": 0.90, "case": 0.55, "fir": 0.85,
    "legal action": 0.85, "legal notice": 0.80, "court": 0.70,
    "adalat": 0.70, "police": 0.70, "raid": 0.85, "action": 0.45,
    "khatam": 0.60, "kho": 0.50, "loss": 0.55, "nuksan": 0.65,
    "problem ho": 0.55, "dikkat ho": 0.55,
}

#: Authority impersonation. Any of these plus a threat is the digital-arrest
#: pattern that I4C keeps issuing advisories about.
AUTHORITY_TERMS: Dict[str, float] = {
    "police": 0.85, "cyber cell": 0.95, "cyber crime": 0.95, "crime branch": 0.95,
    "cbi": 0.95, "ed": 0.70, "enforcement directorate": 0.95,
    "rbi": 0.90, "reserve bank": 0.90, "trai": 0.85, "income tax": 0.85,
    "customs": 0.85, "narcotics": 0.90, "ncb": 0.90, "interpol": 0.95,
    "court": 0.75, "adalat": 0.75, "magistrate": 0.85, "judge": 0.80,
    "inspector": 0.85, "commissioner": 0.85, "officer": 0.55, "adhikari": 0.60,
    "thana": 0.80, "station": 0.45, "department": 0.40, "vibhag": 0.45,
    "government": 0.50, "sarkar": 0.50, "sarkari": 0.55, "ministry": 0.60,
    "headquarters": 0.60, "sub inspector": 0.85, "si": 0.30,
}

#: Bank and account vocabulary.
BANK_TERMS: Dict[str, float] = {
    "account": 0.60, "khata": 0.60, "khate": 0.60, "bank": 0.50,
    "savings": 0.40, "current": 0.30, "netbanking": 0.65, "net banking": 0.65,
    "debit card": 0.65, "credit card": 0.65, "card": 0.45, "atm": 0.50,
    "kyc": 0.80, "cheque": 0.40, "passbook": 0.40, "ifsc": 0.55,
    "branch": 0.35, "balance": 0.45, "statement": 0.35,
    "transaction": 0.55, "transfer": 0.55, "neft": 0.50, "imps": 0.50, "rtgs": 0.50,
    "loan": 0.40, "emi": 0.40, "policy": 0.30, "insurance": 0.30,
}

#: Payment handles: how the money would actually leave.
PAYMENT_TERMS: Dict[str, float] = {
    "upi": 0.75, "upi id": 0.85, "vpa": 0.75, "gpay": 0.65, "google pay": 0.65,
    "phonepe": 0.65, "paytm": 0.65, "bhim": 0.65, "wallet": 0.55,
    "qr": 0.70, "qr code": 0.80, "scan": 0.55, "link": 0.60,
    "payment link": 0.85, "account number": 0.80, "khata number": 0.80,
    "beneficiary": 0.75, "transfer kar": 0.85, "paise bhej": 0.85,
    "deposit": 0.60, "processing fee": 0.85, "registration fee": 0.85,
    "security deposit": 0.85, "refundable": 0.65, "advance": 0.55,
}

#: Requests for information a genuine institution already has.
PERSONAL_TERMS: Dict[str, float] = {
    "otp": 0.95, "one time password": 0.95, "pin": 0.85, "mpin": 0.90,
    "cvv": 0.95, "password": 0.85, "expiry": 0.70, "expiry date": 0.80,
    "aadhaar": 0.75, "aadhar": 0.75, "pan": 0.70, "pan card": 0.80,
    "date of birth": 0.70, "dob": 0.70, "janm": 0.60,
    "mother name": 0.75, "maiden name": 0.80,
    "full number": 0.70, "card number": 0.85, "sixteen digit": 0.85,
    "screen share": 0.90, "anydesk": 0.95, "teamviewer": 0.95,
    "remote access": 0.90, "install": 0.55, "download": 0.50,
    "app download": 0.75, "apk": 0.85,
}

#: Isolation tactics. A real institution never asks for these.
ISOLATION_TERMS: Dict[str, float] = {
    "kisi ko mat batana": 0.95, "kisi ko mat bataiye": 0.95,
    "do not tell": 0.90, "dont tell": 0.90, "confidential": 0.65,
    "line mat kato": 0.90, "call mat kato": 0.90, "phone mat rakho": 0.90,
    "line pe raho": 0.90, "stay on the line": 0.90, "disconnect mat": 0.95,
    "family ko mat": 0.90, "ghar walo ko mat": 0.90,
    "secret": 0.70, "private": 0.45, "alone": 0.55, "akele": 0.60,
}

#: Phrases that argue against fraud. Without these the intent branch flags
#: every bank call, which is exactly the hard-negative problem in the plan.
BENIGN_MARKERS: Dict[str, float] = {
    "branch visit": 0.70, "branch aakar": 0.70, "branch me aaiye": 0.75,
    "hum otp nahi mangte": 0.95, "we never ask": 0.95,
    "never ask for otp": 0.95, "kabhi otp nahi": 0.95,
    "no action required": 0.80, "koi action nahi": 0.75,
    "information ke liye": 0.60, "sirf jankari": 0.65,
    "aapki suvidha": 0.55, "reminder": 0.50, "yaad dilane": 0.55,
    "appointment": 0.55, "feedback": 0.60, "survey": 0.55,
    "aap chahe to": 0.60, "aapki marzi": 0.70, "if you wish": 0.65,
    "no hurry": 0.80, "koi jaldi nahi": 0.85, "jab time mile": 0.75,
    "customer care par call": 0.70, "official website": 0.70,
    "toll free": 0.60, "helpline": 0.45,
    "thank you for": 0.40, "have a good day": 0.45,
}

# --------------------------------------------------------------------------
# 3. Gazetteers, keyed by entity type
# --------------------------------------------------------------------------

GAZETTEERS: Dict[str, Set[str]] = {
    "OTP": {
        "otp", "o t p", "one time password", "verification code", "code",
        "six digit", "6 digit", "chhe ank", "sms code", "pin",
    },
    "BANK_ENTITY": {
        "account", "khata", "khate", "bank", "netbanking", "net banking",
        "debit card", "credit card", "atm card", "kyc", "passbook",
        "savings account", "current account", "ifsc", "branch", "cheque book",
        "card", "atm", "internet banking",
    },
    "AUTHORITY_CLAIM": {
        "police", "cyber cell", "cyber crime", "crime branch", "cbi",
        "enforcement directorate", "rbi", "reserve bank of india", "trai",
        "income tax department", "customs", "narcotics control bureau", "ncb",
        "interpol", "magistrate", "sub inspector", "inspector", "commissioner",
        "police station", "thana", "cyber police", "delhi police",
    },
    "THREAT_DEADLINE": {
        "block ho jayega", "band ho jayega", "suspend ho jayega",
        "freeze ho jayega", "deactivate ho jayega", "arrest warrant",
        "legal action", "fir darj", "case darj", "seize",
    },
    "PAYMENT_HANDLE": {
        "upi id", "upi", "vpa", "gpay", "phonepe", "paytm", "bhim",
        "qr code", "payment link", "account number", "beneficiary",
        "processing fee", "registration fee", "security deposit",
    },
    "PERSONAL_INFO_REQ": {
        "cvv", "mpin", "atm pin", "card number", "expiry date", "aadhaar number",
        "aadhar number", "pan number", "date of birth", "mother maiden name",
        "password", "anydesk", "teamviewer", "screen share", "remote access",
    },
    "MONEY_AMOUNT": {
        "rupees", "rupaye", "rs", "inr", "lakh", "crore", "hazaar", "thousand",
    },
}

#: Surface forms that are strong single-token cues for an entity type. The CRF
#: uses these as membership features.
TOKEN_GAZETTEER: Dict[str, str] = {}
for _etype, _forms in GAZETTEERS.items():
    for _f in _forms:
        for _tok in _f.split():
            TOKEN_GAZETTEER.setdefault(_tok, _etype)

# --------------------------------------------------------------------------
# 4. Look-up helpers
# --------------------------------------------------------------------------

_SPELLING_NORMALISERS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"aa+"), "a"),
    (re.compile(r"ee+"), "i"),
    (re.compile(r"oo+"), "u"),
    (re.compile(r"ii+"), "i"),
    (re.compile(r"uu+"), "u"),
    (re.compile(r"(.)\1{2,}"), r"\1"),
)

#: Common romanisation variants collapsed to one canonical spelling. Applied
#: before every dictionary look-up, which recovers several F1 points on noisy
#: transcripts (see Plan B, section 2, point 3).
HINGLISH_VARIANTS: Dict[str, str] = {
    "kripaya": "kripya", "kripayaa": "kripya",
    "nahin": "nahi", "nahiin": "nahi",
    "haan": "han", "haa": "han", "haanji": "han",
    "theek": "thik", "theeek": "thik", "teek": "thik",
    "achha": "acha", "accha": "acha", "achchha": "acha",
    "bohot": "bahut", "bahot": "bahut", "bohut": "bahut",
    "zyada": "jyada", "zaada": "jyada",
    "zaroorat": "jarurat", "zarurat": "jarurat", "zaruri": "jaruri",
    "zaroori": "jaruri",
    "foran": "turant", "fauran": "turant",
    "aadhar": "aadhaar", "adhaar": "aadhaar", "adhar": "aadhaar",
    "paise": "paisa", "rupaye": "rupees", "rupya": "rupees", "rupaiya": "rupees",
    "khaata": "khata", "khaate": "khate",
    "giraftaar": "giraftar",
    "jaayega": "jayega", "jaega": "jayega", "jaegi": "jayegi",
    "bandh": "band",
    "dhanyawad": "dhanyavaad", "shukriya": "dhanyavaad",
    "namaskar": "namaste",
    "bataiye": "batao", "bataye": "batao", "batana": "batao", "bata": "batao",
    "dijiyega": "dijiye", "dijie": "dijiye", "dena": "dijiye",
    "kijiye": "karo", "kijie": "karo", "kijiyega": "karo", "karna": "karo",
    "suniye": "suno", "sunna": "suno",
    "dekhiye": "dekho", "dekhna": "dekho", "dekhiyega": "dekho",
    "aadmi": "admi",
    "mein": "me", "mai": "main",
}


def canonical(token: str) -> str:
    """Collapse a romanised token to the spelling the dictionaries use."""
    t = token.lower().strip()
    if not t:
        return t
    if t in HINGLISH_VARIANTS:
        return HINGLISH_VARIANTS[t]
    squeezed = t
    for pat, rep in _SPELLING_NORMALISERS:
        squeezed = pat.sub(rep, squeezed)
    if squeezed in HINGLISH_VARIANTS:
        return HINGLISH_VARIANTS[squeezed]
    if squeezed in HINDI_WORDS or squeezed in ENGLISH_WORDS:
        return squeezed
    return t


def token_language(token: str) -> str:
    """Language tag for one token: 'hi', 'en' or 'univ'.

    Order matters. Explicit membership wins, then the universal set, then the
    orthographic rules, and anything still unresolved is called 'univ' rather
    than guessed, so the code-mixing statistics stay honest.
    """
    t = token.lower().strip()
    if not t:
        return "univ"
    if not t.isalpha():
        return "univ"          # digits, punctuation, mixed alphanumerics
    if t in UNIVERSAL_WORDS:
        return "univ"
    if t in HINDI_WORDS:
        return "hi"
    if t in ENGLISH_WORDS:
        return "en"

    c = canonical(t)
    if c in HINDI_WORDS:
        return "hi"
    if c in ENGLISH_WORDS:
        return "en"

    hi_score = sum(1 for p in _HINDI_PATTERNS if p.search(t))
    hi_score += 1 if any(t.endswith(s) for s in _HINDI_SUFFIXES) else 0
    en_score = sum(1 for p in _ENGLISH_PATTERNS if p.search(t))
    if hi_score > en_score:
        return "hi"
    if en_score > hi_score:
        return "en"
    return "univ"


def _phrase_hits(text: str, table: Dict[str, float]) -> List[Tuple[str, float]]:
    """Find weighted terms in a text, longest phrase first, no double counting."""
    low = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "
    hits: List[Tuple[str, float]] = []
    taken: List[Tuple[int, int]] = []
    for term in sorted(table, key=len, reverse=True):
        pattern = re.compile(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])")
        for m in pattern.finditer(low):
            span = (m.start(), m.end())
            if any(span[0] < e and s < span[1] for s, e in taken):
                continue
            taken.append(span)
            hits.append((term, table[term]))
    return hits


def lexicon_hits(text: str) -> Dict[str, List[Tuple[str, float]]]:
    """Every weighted-dictionary hit in a piece of text, grouped by dictionary.

    This is what the interpretable baseline reports and what the evidence
    panel highlights, so it returns the matched surface form, not just a count.
    """
    return {
        "urgency": _phrase_hits(text, URGENCY_TERMS),
        "threat": _phrase_hits(text, THREAT_TERMS),
        "authority": _phrase_hits(text, AUTHORITY_TERMS),
        "bank": _phrase_hits(text, BANK_TERMS),
        "payment": _phrase_hits(text, PAYMENT_TERMS),
        "personal": _phrase_hits(text, PERSONAL_TERMS),
        "isolation": _phrase_hits(text, ISOLATION_TERMS),
        "benign": _phrase_hits(text, BENIGN_MARKERS),
    }


#: Which dictionaries push toward fraud and how much each is trusted. Used by
#: the rule baseline and by the lexical-arousal term in the PIM feature.
PRESSURE_WEIGHTS: Dict[str, float] = {
    "urgency": 1.00,
    "threat": 1.30,
    "authority": 1.10,
    "personal": 1.40,
    "payment": 1.20,
    "isolation": 1.50,
    "bank": 0.45,
    "benign": -1.20,
}


def gazetteer_type(token: str) -> Optional[str]:
    """Entity type this single token hints at, if any."""
    t = token.lower()
    return TOKEN_GAZETTEER.get(t) or TOKEN_GAZETTEER.get(canonical(t))


def word_shape(token: str) -> str:
    """Classic CRF shape feature: Xxxx, dddd, xxx-dd and so on."""
    s = re.sub(r"[A-Z]", "X", token)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def is_digit_run(token: str, min_len: int = 4) -> bool:
    """A run of four or more digits is almost always an OTP or an account."""
    return token.isdigit() and len(token) >= min_len


LEXICON_TABLES: Dict[str, Dict[str, float]] = {
    "urgency": URGENCY_TERMS,
    "threat": THREAT_TERMS,
    "authority": AUTHORITY_TERMS,
    "bank": BANK_TERMS,
    "payment": PAYMENT_TERMS,
    "personal": PERSONAL_TERMS,
    "isolation": ISOLATION_TERMS,
    "benign": BENIGN_MARKERS,
}


def stats() -> Dict[str, int]:
    return {
        "hindi_words": len(HINDI_WORDS),
        "english_words": len(ENGLISH_WORDS),
        "universal_words": len(UNIVERSAL_WORDS),
        "urgency_terms": len(URGENCY_TERMS),
        "threat_terms": len(THREAT_TERMS),
        "authority_terms": len(AUTHORITY_TERMS),
        "bank_terms": len(BANK_TERMS),
        "payment_terms": len(PAYMENT_TERMS),
        "personal_terms": len(PERSONAL_TERMS),
        "isolation_terms": len(ISOLATION_TERMS),
        "benign_markers": len(BENIGN_MARKERS),
        "gazetteer_types": len(GAZETTEERS),
        "gazetteer_tokens": len(TOKEN_GAZETTEER),
        "spelling_variants": len(HINGLISH_VARIANTS),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(stats(), indent=2))
    demo = "Sir main cyber crime branch se bol raha hoon, aapka account turant block ho jayega, OTP 445566 abhi batao"
    toks = demo.split()
    print("\nlanguage tags:")
    print("  " + "  ".join(f"{t}/{token_language(t)}" for t in toks))
    print("\nlexicon hits:")
    for k, v in lexicon_hits(demo).items():
        if v:
            print(f"  {k:10s} {v}")
    benign = "Namaste, ye SBI ka reminder call hai. Hum kabhi OTP nahi mangte. Aap chahe to branch aakar mil sakte hain."
    print("\nbenign example hits:")
    for k, v in lexicon_hits(benign).items():
        if v:
            print(f"  {k:10s} {v}")
