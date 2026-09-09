"""Tests for the scam-intent NLP branch.

Plain asserts, so this runs under pytest and also as a script:

    .venv\\Scripts\\python.exe tests\\test_text.py

The fixture corpus is hand-built here rather than imported, for two reasons.
The corpus generator is written by a different module and these tests must not
wait for it or break when it changes. And the text modules ship their own small
demo corpus for their self-tests, so scoring them on that same data would only
prove the data is memorisable. The templates below are written independently:
different scripts, different wording, same label conventions.

The gradient check is the important one. Everything else measures a model; that
test proves the CRF's analytic gradient is actually the derivative of its
objective, which is the part of a hand-written CRF that fails silently.
"""

from __future__ import annotations

import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarkavach.config import SETTINGS
from swarkavach.corpus.lexicon import token_language
from swarkavach.schema import BIO_LABELS, Call, COERCION_RANK, DIALOGUE_ACTS, Turn, tokenize
from swarkavach.text import coercion as coercion_mod
from swarkavach.text.asr import ASR, aggregate_wer, word_error_rate
from swarkavach.text.coercion import (
    ActHMM,
    act_classifier,
    coercion_features,
    evaluate_act_classifier,
    predict_acts,
)
from swarkavach.text.crf import LinearChainCRF, numerical_gradient
from swarkavach.text.intent_rules import RuleIntent
from swarkavach.text.intent_tfidf import TfidfIntent
from swarkavach.text.langid import LANG_FEATURE_KEYS, code_mixing_features, tag_languages
from swarkavach.text.ner_bilstm import BiLSTMTagger, majority_baseline, most_frequent_tag_baseline
from swarkavach.text.ner_crf import CRFTagger, bio_spans, entity_prf
from swarkavach.text.normalize import normalize_text, normalize_tokens
from swarkavach.text.pos import pos_tag

# --------------------------------------------------------------------------
# Fixture corpus
# --------------------------------------------------------------------------

_SLOTS: Dict[str, List[str]] = {
    "name": ["arjun", "kavita", "rohit", "neha"],
    "bank": ["sbi", "hdfc", "icici", "kotak"],
    "agency": ["cbi", "cyber crime branch", "narcotics control bureau"],
    "otp": ["338211", "907643", "451209"],
    "amount": ["25000", "78000", "13500"],
    "minutes": ["10", "20", "30"],
    "upi": ["refund@okhdfc", "clear@ybl", "settle@paytm"],
    "brand": ["amazon", "flipkart", "myntra"],
    "telecom": ["airtel", "jio", "bsnl"],
    "app": ["anydesk", "teamviewer"],
}

# (speaker, gold act, text template, [(entity phrase template, type), ...])
_SCAM: List[Tuple[str, List[Tuple[str, str, str, List[Tuple[str, str]]]]]] = [
    ("kyc_freeze", [
        ("caller", "GREET", "hello main {bank} bank ke verification team se baat kar raha hoon",
         [("{bank} bank", "BANK_ENTITY")]),
        ("callee", "VICTIM_QUESTION", "haan kya problem hai", []),
        ("caller", "PROBLEM_STATE", "aapke debit card par ek galat transaction hua hai",
         [("debit card", "BANK_ENTITY")]),
        ("caller", "DEADLINE", "agar {minutes} minute ke andar verify nahi kiya to card block ho jayega",
         [("card", "BANK_ENTITY"), ("block ho jayega", "THREAT_DEADLINE")]),
        ("callee", "VICTIM_RESIST", "mujhe aap par bharosa nahi hai", []),
        ("caller", "REQUEST_SENSITIVE", "card ka cvv aur expiry date jaldi bataiye",
         [("card", "BANK_ENTITY"), ("cvv", "PERSONAL_INFO_REQ"), ("expiry date", "PERSONAL_INFO_REQ")]),
        ("callee", "VICTIM_COMPLY", "thik hai likh lijiye", []),
        ("caller", "CLOSE", "shukriya aapka card safe ho gaya", [("card", "BANK_ENTITY")]),
    ]),
    ("digital_arrest", [
        ("caller", "IDENTIFY_SELF", "main {agency} ka officer bol raha hoon", [("{agency}", "AUTHORITY_CLAIM")]),
        ("caller", "PROBLEM_STATE", "aapke naam par ek illegal courier pakda gaya hai", []),
        ("caller", "THREAT", "hamare paas aapke khilaf giraftari ka warrant hai",
         [("warrant", "THREAT_DEADLINE")]),
        ("callee", "VICTIM_QUESTION", "ye sab kaise ho sakta hai", []),
        ("caller", "ISOLATE", "ye jaanch confidential hai kisi ko mat bataiye", []),
        ("caller", "REQUEST_SENSITIVE", "apna pan number aur date of birth abhi bataiye",
         [("pan number", "PERSONAL_INFO_REQ"), ("date of birth", "PERSONAL_INFO_REQ")]),
        ("caller", "INSTRUCT", "ab {amount} rupees is {upi} par bhej dijiye",
         [("{amount} rupees", "MONEY_AMOUNT"), ("{upi}", "PAYMENT_HANDLE")]),
        ("caller", "PRESSURE_ESCALATE", "jaldi kijiye warna raid ho jayegi", []),
    ]),
    ("otp_theft", [
        ("caller", "GREET", "namaste main {brand} customer support se bol raha hoon", []),
        ("caller", "PROBLEM_STATE", "aapka order galat address par chala gaya hai", []),
        ("caller", "INSTRUCT", "refund ke liye ye link click kijiye aur form bhariye",
         [("link", "PAYMENT_HANDLE")]),
        ("callee", "VICTIM_QUESTION", "isme kitna time lagega", []),
        ("caller", "REASSURE", "chinta mat kijiye ye bilkul safe process hai", []),
        ("caller", "REQUEST_SENSITIVE", "us page par apna card number aur otp daaliye",
         [("card number", "PERSONAL_INFO_REQ"), ("otp", "OTP")]),
        ("callee", "VICTIM_COMPLY", "ok otp {otp} hai", [("otp {otp}", "OTP")]),
        ("caller", "CLOSE", "dhanyavaad refund process ho gaya hai", []),
    ]),
    ("electricity_disconnect", [
        ("caller", "IDENTIFY_SELF", "main bijli board ke office se baat kar raha hoon", []),
        ("caller", "PROBLEM_STATE", "aapka pichla bill abhi tak unpaid hai", []),
        ("caller", "DEADLINE", "aaj raat ko aapka connection disconnect ho jayega",
         [("disconnect ho jayega", "THREAT_DEADLINE")]),
        ("callee", "VICTIM_RESIST", "maine to bill pehle hi bhar diya tha", []),
        ("caller", "INSTRUCT", "aap ye {app} app install kijiye aur screen dikhaiye",
         [("{app}", "PERSONAL_INFO_REQ")]),
        ("caller", "REQUEST_SENSITIVE", "ab apna atm pin type kijiye", [("atm pin", "PERSONAL_INFO_REQ")]),
        ("caller", "PRESSURE_ESCALATE", "turant kijiye warna line kat jayegi", []),
    ]),
    ("lottery_prize", [
        ("caller", "GREET", "badhai ho aapne {amount} rupees ka lucky draw jeeta hai",
         [("{amount} rupees", "MONEY_AMOUNT")]),
        ("callee", "VICTIM_QUESTION", "kaun bol raha hai aap", []),
        ("caller", "IDENTIFY_SELF", "main {brand} lucky draw team se bol raha hoon", []),
        ("caller", "INSTRUCT", "inaam ke liye {amount} rupees registration fee bhejiye",
         [("{amount} rupees", "MONEY_AMOUNT"), ("registration fee", "PAYMENT_HANDLE")]),
        ("caller", "REQUEST_SENSITIVE", "apna account number aur otp bhi bhej dijiye",
         [("account number", "PAYMENT_HANDLE"), ("otp", "OTP")]),
        ("caller", "DEADLINE", "ye offer aaj raat tak hi valid hai warna cancel ho jayega",
         [("cancel ho jayega", "THREAT_DEADLINE")]),
    ]),
    ("loan_approval", [
        ("caller", "GREET", "namaste aapka {amount} rupees ka loan approve ho gaya hai",
         [("{amount} rupees", "MONEY_AMOUNT")]),
        ("callee", "VICTIM_RESIST", "maine to koi loan apply hi nahi kiya ye galat hai", []),
        ("caller", "REASSURE", "chinta mat kijiye ye pre approved offer hai", []),
        ("caller", "INSTRUCT", "processing fee {upi} par transfer kijiye",
         [("processing fee", "PAYMENT_HANDLE"), ("{upi}", "PAYMENT_HANDLE")]),
        ("caller", "REQUEST_SENSITIVE", "aur apna aadhaar number bhi bataiye",
         [("aadhaar number", "PERSONAL_INFO_REQ")]),
        ("caller", "PRESSURE_ESCALATE", "jaldi kijiye offer khatam ho raha hai", []),
    ]),
    ("sim_block", [
        ("caller", "IDENTIFY_SELF", "main {telecom} ke customer department se bol rahi hoon", []),
        ("caller", "PROBLEM_STATE", "aapke aadhaar par ek galat sim active hai",
         [("aadhaar", "PERSONAL_INFO_REQ")]),
        ("caller", "DEADLINE", "kal subah tak number band ho jayega",
         [("band ho jayega", "THREAT_DEADLINE")]),
        ("callee", "VICTIM_QUESTION", "main kya karun ab", []),
        ("caller", "REQUEST_SENSITIVE", "sms wala otp mujhe bata dijiye", [("otp", "OTP")]),
        ("callee", "VICTIM_COMPLY", "otp {otp} likh lijiye", [("otp {otp}", "OTP")]),
        ("caller", "CLOSE", "dhanyavaad aapka sim verify ho gaya", []),
    ]),
    ("fake_relative", [
        ("caller", "GREET", "hello beta main tumhare mama bol raha hoon", []),
        ("caller", "PROBLEM_STATE", "mera accident ho gaya hai aur bahut problem hai", []),
        ("caller", "INSTRUCT", "abhi {amount} rupees is {upi} par bhejo",
         [("{amount} rupees", "MONEY_AMOUNT"), ("{upi}", "PAYMENT_HANDLE")]),
        ("callee", "VICTIM_QUESTION", "aap kaun hain sach batao", []),
        ("caller", "ISOLATE", "ghar me kisi ko mat batana ye baat", []),
        ("caller", "PRESSURE_ESCALATE", "jaldi karo warna bahut nuksan ho jayega", []),
    ]),
]

_BENIGN: List[Tuple[str, List[Tuple[str, str, str, List[Tuple[str, str]]]]]] = [
    ("bank_reminder", [
        ("caller", "GREET", "namaste main {bank} bank se bol raha hoon", [("{bank} bank", "BANK_ENTITY")]),
        ("caller", "INFORM", "aapka monthly statement email par bhej diya gaya hai", []),
        ("caller", "REASSURE", "hum kabhi otp ya password nahi mangte",
         [("otp", "OTP"), ("password", "PERSONAL_INFO_REQ")]),
        ("callee", "VICTIM_COMPLY", "thik hai dhanyavaad", []),
        ("caller", "CLOSE", "aapka din shubh ho dhanyavaad", []),
    ]),
    ("delivery_otp", [
        ("caller", "GREET", "hello main {brand} delivery se bol raha hoon", []),
        ("caller", "INFORM", "aapka parcel building ke gate par hai", []),
        ("caller", "REQUEST_SENSITIVE", "delivery ke liye otp bataiye", [("otp", "OTP")]),
        ("callee", "VICTIM_COMPLY", "haan otp {otp} hai", [("otp {otp}", "OTP")]),
        ("caller", "CLOSE", "dhanyavaad parcel mil gaya", []),
    ]),
    ("family_call", [
        ("caller", "GREET", "hello {name} kaise ho tum", []),
        ("callee", "INFORM", "main bilkul thik hoon", []),
        ("caller", "SMALLTALK", "ghar par sab kaise hain bacche kya kar rahe hain", []),
        ("callee", "INFORM", "sab badhiya hai aap sunao", []),
        ("caller", "CLOSE", "chalo phir baad me baat karte hain apna khyal rakhna", []),
    ]),
    ("appointment_reminder", [
        ("caller", "GREET", "namaste main city clinic se bol rahi hoon", []),
        ("caller", "INFORM", "kal shaam paanch baje aapka appointment hai", []),
        ("caller", "CONFIRM", "kya ye samay aapke liye sahi rahega", []),
        ("callee", "VICTIM_COMPLY", "haan sahi hai", []),
        ("caller", "CLOSE", "dhanyavaad kal milte hain", []),
    ]),
    ("customer_support", [
        ("caller", "GREET", "namaste main {bank} bank customer care se bol raha hoon",
         [("{bank} bank", "BANK_ENTITY")]),
        ("callee", "INFORM", "meri netbanking login nahi ho rahi", [("netbanking", "BANK_ENTITY")]),
        ("caller", "INSTRUCT", "aap official website par login karo aur reset karo", []),
        ("caller", "REASSURE", "hum aapse kabhi password nahi puchenge",
         [("password", "PERSONAL_INFO_REQ")]),
        ("callee", "VICTIM_COMPLY", "thik hai main try karta hoon", []),
        ("caller", "CLOSE", "aur koi madad chahiye to bataiye dhanyavaad", []),
    ]),
    ("survey_call", [
        ("caller", "GREET", "namaste hum ek chota sa survey kar rahe hain", []),
        ("caller", "INFORM", "isme sirf do minute lagenge aur koi jankari nahi chahiye", []),
        ("callee", "VICTIM_QUESTION", "kis cheez ka survey hai", []),
        ("caller", "INFORM", "hamari service ke baare me aapki ray chahiye", []),
        ("callee", "VICTIM_COMPLY", "thik hai puchiye", []),
        ("caller", "CLOSE", "aapke samay ke liye dhanyavaad", []),
    ]),
]


def _make_turn(index: int, speaker: str, act: str, text: str,
               ents: Sequence[Tuple[str, str]], t0: float) -> Turn:
    """Build a Turn with gold BIO derived from entity phrases, never by hand."""
    tokens = tokenize(text)
    low = [t.lower() for t in tokens]
    bio = ["O"] * len(tokens)
    for phrase, etype in ents:
        ptoks = [t.lower() for t in tokenize(phrase)]
        k = len(ptoks)
        for i in range(len(tokens) - k + 1):
            if low[i:i + k] == ptoks and all(b == "O" for b in bio[i:i + k]):
                bio[i] = f"B-{etype}"
                for j in range(i + 1, i + k):
                    bio[j] = f"I-{etype}"
                break
        else:
            raise AssertionError(f"fixture entity {phrase!r} not found in {text!r}")
    dur = 1.1 + 0.3 * len(tokens)
    return Turn(index=index, speaker=speaker, text=text, act=act, tokens=tokens,
                bio=bio, lang=[token_language(t) for t in tokens],
                t_start=t0, t_end=t0 + dur)


def _fixture_calls(n_repeat: int = 3) -> List[Call]:
    """42 hand-built calls: 8 scam scripts and 6 benign, three slot fills each."""
    rng = random.Random(SETTINGS.pipeline.seed)
    calls: List[Call] = []
    scripts = [(1, s) for s in _SCAM] + [(0, s) for s in _BENIGN]
    for rep in range(n_repeat):
        for label, (scenario, spec) in scripts:
            slots = {k: rng.choice(v) for k, v in _SLOTS.items()}
            turns: List[Turn] = []
            t = 0.0
            for i, (speaker, act, text, ents) in enumerate(spec):
                filled = [(p.format(**slots), et) for p, et in ents]
                turn = _make_turn(i, speaker, act, text.format(**slots), filled, t)
                t = turn.t_end + 0.3
                turns.append(turn)
            idx = len(calls)
            calls.append(Call(
                call_id=f"fix_{idx:03d}", turns=turns, label_scam=label,
                label_voice="synthetic" if idx % 3 == 0 else "human",
                scenario=scenario, speaker_id=f"spk_{idx % 6:02d}",
                split="train" if rep < n_repeat - 1 else "test",
            ))
    return calls


# --------------------------------------------------------------------------
# Shared state, so the models are trained once for the whole file
# --------------------------------------------------------------------------

_CACHE: Dict[str, object] = {}
MEASURED: Dict[str, float] = {}


def _calls() -> List[Call]:
    if "calls" not in _CACHE:
        _CACHE["calls"] = _fixture_calls()
    return _CACHE["calls"]  # type: ignore[return-value]


def _split() -> Tuple[List[Call], List[Call]]:
    calls = _calls()
    return ([c for c in calls if c.split == "train"],
            [c for c in calls if c.split == "test"])


def _crf_tagger() -> CRFTagger:
    if "crf" not in _CACHE:
        train, _ = _split()
        t0 = time.time()
        _CACHE["crf"] = CRFTagger(l2=0.5, max_iter=120).fit(train)
        MEASURED["crf_fit_seconds"] = time.time() - t0
    return _CACHE["crf"]  # type: ignore[return-value]


def _tfidf() -> TfidfIntent:
    if "tfidf" not in _CACHE:
        train, _ = _split()
        _CACHE["tfidf"] = TfidfIntent().fit(train)
    return _CACHE["tfidf"]  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Fixture sanity
# --------------------------------------------------------------------------


def test_fixture_is_well_formed():
    calls = _calls()
    assert 30 <= len(calls) <= 60, len(calls)
    n_turns = 0
    types = set()
    for c in calls:
        assert c.turns, c.call_id
        for t in c.turns:
            n_turns += 1
            assert len(t.bio) == len(t.tokens) == len(t.lang)
            assert t.act in DIALOGUE_ACTS, t.act
            for tag in t.bio:
                assert tag in BIO_LABELS, tag
            for s, e, ty in bio_spans(t.bio):
                types.add(ty)
    assert len(types) == 7, sorted(types)
    assert sum(c.label_scam for c in calls) * 2 > len(calls) * 0.5
    MEASURED["n_calls"] = len(calls)
    MEASURED["n_turns"] = n_turns
    print(f"    fixture: {len(calls)} calls, {n_turns} turns, {len(types)} entity types")


# --------------------------------------------------------------------------
# CRF
# --------------------------------------------------------------------------


def test_crf_gradient_matches_numerical():
    """The analytic gradient must equal a central finite-difference gradient.

    This is the whole correctness argument for the hand-written CRF: if the
    gradient is wrong, L-BFGS still runs, still reports convergence and still
    produces a model that tags things, just a worse one, with nothing in the
    output to say so.
    """
    rng = np.random.default_rng(7)
    vocab = ["otp", "batao", "account", "445566", "sir", "abhi"]
    X, Y = [], []
    for _ in range(5):
        n = int(rng.integers(3, 8))
        toks = list(rng.choice(vocab, size=n))
        tags, prev = [], "O"
        for w in toks:
            if w == "otp":
                tag = "B-OTP"
            elif w == "445566" and prev in ("B-OTP", "I-OTP"):
                tag = "I-OTP"
            else:
                tag = "O"
            tags.append(tag)
            prev = tag
        X.append([[f"w={w}", "bias"] for w in toks])
        Y.append(tags)

    crf = LinearChainCRF(l2=0.7, labels=["O", "B-OTP", "I-OTP"])
    crf._build_index(X, Y)
    data = crf.prepare(X, Y)
    w = rng.normal(scale=0.5, size=crf.n_params)

    loss, analytic = crf.loss_and_grad(w, data)
    numeric = numerical_gradient(crf, w, data, eps=1e-5)
    max_abs = float(np.max(np.abs(analytic - numeric)))
    denom = np.maximum(1.0, np.abs(analytic) + np.abs(numeric))
    max_rel = float(np.max(np.abs(analytic - numeric) / denom))
    MEASURED["crf_grad_max_abs_err"] = max_abs
    print(f"    {crf.n_params} params, loss={loss:.4f}, "
          f"max abs err={max_abs:.3e}, max rel err={max_rel:.3e}")
    assert np.isfinite(loss)
    assert max_abs < 1e-6, max_abs
    assert max_rel < 1e-6, max_rel


def test_crf_gradient_with_constraints():
    """Masked transitions must not break the gradient or leak NaNs."""
    from swarkavach.text.ner_crf import bio_constraints

    labels = ["O", "B-OTP", "I-OTP"]
    rng = np.random.default_rng(11)
    X = [[["w=otp", "bias"], ["w=445566", "bias"], ["w=sir", "bias"]],
         [["w=sir", "bias"], ["w=otp", "bias"]]]
    Y = [["B-OTP", "I-OTP", "O"], ["O", "B-OTP"]]
    crf = LinearChainCRF(l2=0.3, labels=labels, constraints=bio_constraints)
    crf._build_index(X, Y)
    data = crf.prepare(X, Y)
    w = rng.normal(scale=0.4, size=crf.n_params)
    loss, analytic = crf.loss_and_grad(w, data)
    numeric = numerical_gradient(crf, w, data, eps=1e-5)
    err = float(np.max(np.abs(analytic - numeric)))
    print(f"    constrained CRF: loss={loss:.4f}, max abs err={err:.3e}")
    assert np.isfinite(loss) and np.all(np.isfinite(analytic))
    assert err < 1e-6, err


def test_crf_viterbi_returns_valid_bio():
    """Decoded sequences must be legal BIO, on real turns and on noise."""
    tagger = _crf_tagger()
    _, test = _split()
    checked = 0
    for c in test:
        for t in c.turns:
            bio = tagger.predict_turn(t.tokens)
            assert len(bio) == len(t.tokens)
            _assert_valid_bio(bio)
            checked += 1
    # Random token soup as well: a valid decode must not depend on the input
    # looking like training data.
    rng = random.Random(3)
    words = [w for c in test for t in c.turns for w in t.tokens]
    for _ in range(30):
        toks = [rng.choice(words) for _ in range(rng.randint(1, 14))]
        _assert_valid_bio(tagger.predict_turn(toks))
        checked += 1
    print(f"    {checked} decoded sequences, all valid BIO")


def _assert_valid_bio(bio: Sequence[str]) -> None:
    prev = "O"
    for tag in bio:
        assert tag in BIO_LABELS, tag
        if tag.startswith("I-"):
            etype = tag[2:]
            assert prev in (f"B-{etype}", f"I-{etype}"), f"{prev} -> {tag} in {list(bio)}"
        prev = tag


def test_crf_tagger_entity_f1():
    tagger = _crf_tagger()
    train, test = _split()
    res = tagger.evaluate(test)
    MEASURED["crf_entity_f1"] = res["f1"]
    MEASURED["crf_entity_precision"] = res["precision"]
    MEASURED["crf_entity_recall"] = res["recall"]
    MEASURED["crf_token_accuracy"] = res["token_accuracy"]
    print(f"    CRF  P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f} "
          f"on {res['support']} gold spans, fit in "
          f"{MEASURED.get('crf_fit_seconds', 0):.1f}s")
    for t, v in sorted(res["per_type"].items()):
        if v["support"]:
            print(f"       {t:18s} F1={v['f1']:.2f}  n={v['support']}")
    assert res["f1"] > 0.80, res["f1"]
    assert res["support"] > 20


def test_generalisation_to_unseen_scripts():
    """The harder split: hold out whole scenarios rather than only slot fills.

    The main split shares templates between train and test, so a near-perfect
    score there only says the templates are learnable. Holding out four entire
    scripts asks the real question, and the gap between the two numbers is what
    belongs in the report.
    """
    held = {"sim_block", "fake_relative", "survey_call", "delivery_otp"}
    calls = _calls()
    train = [c for c in calls if c.scenario not in held]
    test = [c for c in calls if c.scenario in held]
    assert train and test

    crf = CRFTagger(l2=0.5, max_iter=120).fit(train)
    ner = crf.evaluate(test)
    intent = TfidfIntent().fit(train).evaluate(test)
    MEASURED["unseen_crf_entity_f1"] = ner["f1"]
    MEASURED["unseen_tfidf_turn_accuracy"] = intent["turn_accuracy"]
    MEASURED["unseen_tfidf_call_accuracy"] = intent["call_accuracy"]
    print(f"    unseen scripts: CRF F1={ner['f1']:.3f} (P={ner['precision']:.3f} "
          f"R={ner['recall']:.3f}, {ner['support']} spans), token acc="
          f"{ner['token_accuracy']:.3f}")
    print(f"    unseen scripts: intent turn acc={intent['turn_accuracy']:.3f} "
          f"(auc {intent['turn_auc']:.3f}), call acc={intent['call_accuracy']:.3f}")
    for t, v in sorted(ner["per_type"].items()):
        if v["support"]:
            print(f"       {t:18s} F1={v['f1']:.2f}  n={v['support']}")
    assert ner["f1"] > 0.50, ner["f1"]
    assert intent["call_accuracy"] >= 0.75, intent["call_accuracy"]


def test_crf_save_and_load_roundtrip():
    tagger = _crf_tagger()
    _, test = _split()
    toks = test[0].turns[0].tokens
    before = tagger.predict_turn(toks)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "crf.joblib"
        tagger.save(path)
        loaded = CRFTagger.load(path)
        after = loaded.predict_turn(toks)
    assert before == after
    print(f"    round-trip stable on {len(toks)} tokens")


def test_entity_prf_is_exact_span_match():
    """The scorer itself: a partial span must score zero, not partial credit."""
    gold = [["B-OTP", "I-OTP", "O"]]
    exact = [["B-OTP", "I-OTP", "O"]]
    partial = [["B-OTP", "O", "O"]]
    wrong_type = [["B-BANK_ENTITY", "I-BANK_ENTITY", "O"]]
    assert entity_prf(gold, exact)["f1"] == 1.0
    assert entity_prf(gold, partial)["f1"] == 0.0
    assert entity_prf(gold, wrong_type)["f1"] == 0.0
    # Token accuracy is the number that would have hidden the failure.
    assert entity_prf(gold, partial)["token_accuracy"] > 0.6
    print("    exact-match scoring confirmed (partial span scores 0.0, "
          f"token accuracy {entity_prf(gold, partial)['token_accuracy']:.2f})")


# --------------------------------------------------------------------------
# BiLSTM
# --------------------------------------------------------------------------


def test_bilstm_beats_majority_baseline():
    train, test = _split()
    base = majority_baseline(test)
    freq = most_frequent_tag_baseline(train, test)
    t0 = time.time()
    tagger = BiLSTMTagger(epochs=20, seed=SETTINGS.pipeline.seed).fit(train)
    fit_s = time.time() - t0
    res = tagger.evaluate(test)
    MEASURED["bilstm_entity_f1"] = res["f1"]
    MEASURED["bilstm_token_accuracy"] = res["token_accuracy"]
    MEASURED["bilstm_fit_seconds"] = fit_s
    MEASURED["majority_f1"] = base["f1"]
    MEASURED["most_frequent_tag_f1"] = freq["f1"]
    print(f"    majority(all O)   F1={base['f1']:.3f}  token acc={base['token_accuracy']:.3f}")
    print(f"    per-word baseline F1={freq['f1']:.3f}  token acc={freq['token_accuracy']:.3f}")
    print(f"    BiLSTM+CRF        F1={res['f1']:.3f}  token acc={res['token_accuracy']:.3f}  "
          f"({fit_s:.1f}s, {tagger.history_['epochs']} epochs, "
          f"{tagger.history_['n_params']} params)")
    assert res["f1"] > base["f1"]
    assert res["token_accuracy"] > base["token_accuracy"]
    assert fit_s < 120.0, fit_s
    for c in test[:3]:
        for t in c.turns:
            _assert_valid_bio(tagger.predict_turn(t.tokens))


# --------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------


def test_rule_intent_scores_scam_above_benign():
    ri = RuleIntent()
    calls = _calls()
    scam, benign = [], []
    for c in calls:
        for t in c.turns:
            if t.speaker != "caller":
                continue
            (scam if c.label_scam else benign).append(ri.score_turn(t.tokens))
    m_s, m_b = float(np.mean(scam)), float(np.mean(benign))
    MEASURED["rule_scam_mean"] = m_s
    MEASURED["rule_benign_mean"] = m_b
    print(f"    mean rule score: scam {m_s:.3f} (n={len(scam)}) vs "
          f"benign {m_b:.3f} (n={len(benign)})")
    assert m_s > m_b
    assert m_s - m_b > 0.15, (m_s, m_b)


def test_rule_intent_explains_its_matches():
    ri = RuleIntent()
    toks = tokenize("OTP turant batao warna account block ho jayega")
    hits = ri.explain(toks, turn_index=4)
    assert hits
    kinds = {h["type"] for h in hits}
    assert "imperative_sensitive" in kinds, kinds
    for h in hits:
        assert 0 <= h["tok_start"] < h["tok_end"] <= len(toks)
        assert h["text"] == " ".join(toks[h["tok_start"]:h["tok_end"]])
        assert h["turn_index"] == 4
        assert set(h) >= {"turn_index", "tok_start", "tok_end", "type", "text", "weight", "source"}
    print(f"    {len(hits)} evidence spans, kinds={sorted(kinds)}")


def test_tfidf_intent_accuracy():
    model = _tfidf()
    train, test = _split()
    res = model.evaluate(test)
    MEASURED["tfidf_turn_accuracy"] = res["turn_accuracy"]
    MEASURED["tfidf_call_accuracy"] = res["call_accuracy"]
    MEASURED["tfidf_turn_auc"] = res["turn_auc"]
    print(f"    turn acc={res['turn_accuracy']:.3f} (auc {res['turn_auc']:.3f}) on "
          f"{res['n_turns']} caller turns;  call acc={res['call_accuracy']:.3f} "
          f"(auc {res['call_auc']:.3f}) on {res['n_calls']} calls")
    assert res["turn_accuracy"] > 0.85, res["turn_accuracy"]
    assert res["call_accuracy"] >= 0.85, res["call_accuracy"]


def test_tfidf_call_features_match_the_contract():
    model = _tfidf()
    _, test = _split()
    keys = {"intent_score", "intent_max_turn", "urgency_density", "ent_otp",
            "ent_authority", "ent_threat", "ent_payment", "ent_personal", "ent_bank"}
    scam = [c for c in test if c.label_scam][0]
    out = model.score_call(scam)
    assert keys <= set(out), sorted(keys - set(out))
    for k in keys:
        assert isinstance(out[k], float) and np.isfinite(out[k])
    assert 0.0 <= out["intent_score"] <= 1.0
    assert out["intent_max_turn"] >= max(out["per_turn"]) - 1e-9
    print(f"    {sorted(keys)} present; intent_score={out['intent_score']:.3f}")


# --------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------


def test_code_mixing_features():
    mono_en = tokenize("please confirm your account number for the delivery today")
    feats = code_mixing_features(mono_en)
    for k in LANG_FEATURE_KEYS:
        assert k in feats, k
    assert feats["cmi"] == 0.0, feats["cmi"]
    assert feats["switch_points"] == 0.0
    assert feats["switch_entropy"] == 0.0

    mono_hi = tokenize("aap apna naam aur pata bataiye phir hum aage badhenge")
    assert code_mixing_features(mono_hi)["cmi"] == 0.0

    mixed = tokenize("sir aapka account turant block ho jayega otp abhi batao")
    mf = code_mixing_features(mixed)
    assert mf["cmi"] > 0.0
    assert mf["switch_points"] > 0
    assert 0.0 <= mf["switch_entropy"] <= 1.0
    assert abs(mf["hi_ratio"] + mf["en_ratio"] + mf["univ_ratio"] - 1.0) < 1e-9

    # ent_lang_align needs BIO and stays 0.0 without it.
    toks = tokenize("aapka account turant band ho jayega")
    bio = ["O", "B-BANK_ENTITY", "O", "B-THREAT_DEADLINE", "I-THREAT_DEADLINE", "I-THREAT_DEADLINE"]
    with_bio = code_mixing_features(toks, None, bio)
    assert 0.0 <= with_bio["ent_lang_align"] <= 1.0
    assert with_bio["ent_tokens"] > 0
    assert code_mixing_features(toks)["ent_lang_align"] == 0.0
    MEASURED["cmi_mixed_turn"] = mf["cmi"]
    print(f"    monolingual CMI=0.0; mixed CMI={mf['cmi']:.3f}, "
          f"switches={mf['switch_points']:.0f}, entropy={mf['switch_entropy']:.3f}, "
          f"ent_lang_align={with_bio['ent_lang_align']:.2f}")


def test_language_tags_align_with_tokens():
    for c in _calls()[:5]:
        for t in c.turns:
            langs = tag_languages(t.tokens)
            assert len(langs) == len(t.tokens)
            assert set(langs) <= {"hi", "en", "univ"}


# --------------------------------------------------------------------------
# Dialogue acts and coercion
# --------------------------------------------------------------------------


def test_act_classifier_accuracy():
    calls = _calls()
    res = evaluate_act_classifier(calls)
    MEASURED["act_accuracy"] = res["accuracy"]
    print(f"    act accuracy {res['accuracy']:.3f} over {res['n_turns']} turns")
    worst = sorted(res["per_act"].items(), key=lambda kv: kv[1]["accuracy"])[:4]
    print("       weakest acts:", [(a, round(v["accuracy"], 2), v["support"]) for a, v in worst])
    if res["top_confusions"]:
        print("       confusions:", [(c["gold"], c["pred"], c["n"]) for c in res["top_confusions"][:4]])
    assert res["accuracy"] > 0.70, res["accuracy"]
    for c in calls[:4]:
        acts = predict_acts(c)
        assert len(acts) == len(c.turns)
        assert all(a in DIALOGUE_ACTS for a in acts)
        assert acts == [act_classifier(t) for t in c.turns]
        # Every predicted act has to have a rung on the coercion ladder,
        # otherwise the trajectory features would silently read it as zero.
        assert all(a in COERCION_RANK for a in acts)


def test_coercion_slope_higher_for_scam():
    train, test = _split()
    hmm = ActHMM().fit(train)
    coercion_mod.set_act_hmm(hmm)
    try:
        rows = [(c.label_scam, coercion_features(c)) for c in test]
    finally:
        coercion_mod.set_act_hmm(None)

    scam = [f for l, f in rows if l == 1]
    benign = [f for l, f in rows if l == 0]
    assert scam and benign

    def mean(sel, key):
        return float(np.mean([f[key] for f in sel]))

    for key in ("coercion_slope", "coercion_peak", "act_scam_llr", "callee_resist"):
        MEASURED[f"{key}_scam"] = mean(scam, key)
        MEASURED[f"{key}_benign"] = mean(benign, key)
        print(f"    {key:16s} scam {mean(scam, key):+.3f}   benign {mean(benign, key):+.3f}")

    assert mean(scam, "coercion_slope") > mean(benign, "coercion_slope")
    assert mean(scam, "coercion_peak") > mean(benign, "coercion_peak")
    assert mean(scam, "act_scam_llr") > mean(benign, "act_scam_llr")
    for _, f in rows:
        assert -1.0 <= f["coercion_slope"] <= 1.0
        assert 0.0 <= f["coercion_peak"] <= 1.0
        assert -10.0 <= f["act_scam_llr"] <= 10.0
        assert 0.0 <= f["callee_resist"] <= 1.0


def test_act_hmm_llr_direction():
    """A scripted escalation must score above a flat, ordinary call."""
    train, _ = _split()
    hmm = ActHMM().fit(train)
    escalating = ["GREET", "IDENTIFY_SELF", "PROBLEM_STATE", "THREAT", "DEADLINE",
                  "ISOLATE", "REQUEST_SENSITIVE"]
    flat = ["GREET", "INFORM", "CONFIRM", "VICTIM_COMPLY", "CLOSE"]
    hi, lo = hmm.llr(escalating), hmm.llr(flat)
    MEASURED["llr_escalating"] = hi
    MEASURED["llr_flat"] = lo
    print(f"    llr(escalating)={hi:+.3f}  llr(flat)={lo:+.3f}  "
          f"trained on {hmm.counts_}")
    assert hi > lo
    assert hmm.llr([]) == 0.0


def test_coercion_features_use_predicted_acts_by_default():
    """No gold leakage: the default path must not read Turn.act."""
    _, test = _split()
    call = test[0]
    baseline = coercion_features(call)
    for t in call.turns:
        t.act = "SMALLTALK"          # scribble over the gold labels
    after = coercion_features(call)
    assert baseline["coercion_peak"] == after["coercion_peak"]
    gold_driven = coercion_features(call, acts=["SMALLTALK"] * len(call.turns))
    assert gold_driven["coercion_peak"] == 0.0
    _CACHE.pop("calls", None)        # the fixture was mutated, rebuild it
    print("    default path ignores Turn.act, explicit acts are honoured")


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------


def test_word_error_rate():
    ref = "aapka account 10 minute me band ho jayega"
    assert word_error_rate(ref, ref) == 0.0
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("ek do teen", "seven eight nine") == 1.0
    assert word_error_rate(ref, "") == 1.0
    one_sub = word_error_rate(ref, ref.replace("band", "blocked"))
    assert abs(one_sub - 1 / 8) < 1e-9, one_sub

    noisy = "apka akaunt 10 minute me bandh ho jayega"
    plain = word_error_rate(ref, noisy)
    canon = word_error_rate(ref, noisy, hinglish=True)
    MEASURED["wer_noisy"] = plain
    MEASURED["wer_noisy_canonical"] = canon
    print(f"    identical=0.0, mismatch=1.0, one substitution={one_sub:.3f}, "
          f"noisy={plain:.3f} (canonicalised {canon:.3f})")
    assert canon <= plain

    agg = aggregate_wer([ref, "otp bataiye"], [noisy, "otp bataiye"])
    assert 0.0 < agg["wer"] < 1.0
    assert agg["n_ref_words"] == 10


def test_asr_gold_backend_reads_sidecar():
    call = _calls()[0]
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio" / f"{call.call_id}_g711u.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"placeholder, the gold backend never opens the audio")
        (Path(tmp) / "calls").mkdir(parents=True, exist_ok=True)
        call.save(str(Path(tmp) / "calls" / f"{call.call_id}.json"))

        res = ASR(backend="gold").transcribe(str(audio))
    assert set(res) >= {"text", "segments", "backend", "turns"}
    assert res["backend"] == "gold"
    assert len(res["turns"]) == len(call.turns)
    assert res["turns"][0].text == call.turns[0].text
    assert word_error_rate(call.full_text(), res["text"]) == 0.0
    print(f"    sidecar found through a channel suffix, {len(res['turns'])} turns, WER 0.0")


def test_asr_manual_and_auto_backends():
    manual = ASR(backend="manual").transcribe(
        text="caller: sir aapka account band ho jayega\ncallee: kya baat hai"
    )
    assert manual["backend"] == "manual"
    assert [t.speaker for t in manual["turns"]] == ["caller", "callee"]
    assert manual["turns"][0].lang and len(manual["turns"][0].lang) == len(manual["turns"][0].tokens)
    resolved = ASR(backend="auto").resolve(None, text="hello")
    assert resolved in ("whisper", "manual")
    print(f"    manual backend parses speaker prefixes; auto resolves to {resolved!r} here")


# --------------------------------------------------------------------------
# Normalisation and POS
# --------------------------------------------------------------------------


def test_normalisation_preserves_token_count():
    for c in _calls()[:6]:
        for t in c.turns:
            assert len(normalize_tokens(t.tokens)) == len(t.tokens)
    assert normalize_text("Kripayaa TURAAAANT O.T.P. bataiye") == "kripya turant otp. batao"
    assert "445566" in normalize_text("otp 4 4 5 5 6 6 hai")
    print("    spelling variants collapse and token counts hold")


def test_pos_tagger_shapes():
    for c in _calls()[:6]:
        for t in c.turns:
            tags = pos_tag(t.tokens)
            assert len(tags) == len(t.tokens)
    toks = tokenize("aapka account 10 minute mein block ho jayega")
    tags = pos_tag(toks)
    assert tags[toks.index("10")] == "NUM"
    assert tags[toks.index("block")] == "VERB"
    print("    " + " ".join(f"{w}/{t}" for w, t in zip(toks, tags)))


# --------------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------------


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    t0 = time.time()
    for name, fn in tests:
        print(f"\n{name}")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL: {exc}")
        except Exception as exc:                     # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 68)
    print(f"{len(tests) - len(failures)}/{len(tests)} passed in {time.time() - t0:.1f}s")
    if MEASURED:
        print("\nmeasured values")
        for k, v in sorted(MEASURED.items()):
            if isinstance(v, float) and 0 < abs(v) < 1e-3:
                print(f"  {k:30s} {v:.3e}")
            elif isinstance(v, float):
                print(f"  {k:30s} {v:.4f}")
            else:
                print(f"  {k:30s} {v}")
    for name, exc in failures:
        print(f"  FAILED {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
