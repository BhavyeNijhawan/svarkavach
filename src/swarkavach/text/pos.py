"""Coarse part-of-speech tagging for code-mixed Hinglish.

No Hinglish treebank ships with this project and none may be downloaded, so
this is a lexicon-plus-suffix tagger over a small universal-style tagset. That
is enough for the two jobs it has:

1. A feature for the CRF tagger. POS is the classic way to tell the CRF that
   "block" in "block ho jayega" behaves like a verb while "block" in "block
   number" behaves like a noun, without giving it a separate word feature for
   every context.
2. Construction detection for the rule baseline. The patterns that matter in
   fraud speech are syntactic, not lexical: an imperative verb governing a
   sensitive noun ("OTP batao") is the giveaway, and neither word on its own
   is. Matching on POS sequences catches the construction across the many
   spellings of the verb.

Tagging order is: punctuation, numerals, closed-class look-up on the raw form,
closed-class look-up on the canonical form, named-entity list, suffix rules,
capitalisation, then NOUN as the default. Closed classes come first because
they are the tags a wrong guess would hurt most.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

from ..corpus.lexicon import canonical, gazetteer_type, token_language
from ..schema import tokenize

__all__ = ["POS_TAGS", "pos_tag", "pos_tag_text", "pos_ngrams", "tag_distribution"]

#: Coarse tagset, a cut-down universal set. Kept small on purpose: with a
#: hand-built lexicon, fine distinctions would be guesses.
POS_TAGS: Tuple[str, ...] = (
    "NOUN", "PROPN", "VERB", "AUX", "PRON", "DET", "ADP", "ADJ",
    "ADV", "NUM", "CONJ", "PART", "INTJ", "PUNCT", "X",
)


def _words(s: str) -> Set[str]:
    return set(s.split())


# -- closed classes -------------------------------------------------------
# Hindi and English entries live in the same set. The tagger is coarse and the
# two languages do not collide much on function words, apart from "to" (English
# preposition, Hindi discourse particle) and "do" (English auxiliary, Hindi
# "two" and "give"), which are resolved by the ordering below.

_PRON = _words("""
    main mai hum ham aap tum tu wo woh ye yeh mera meri mere hamara hamari humara
    aapka aapki aapke aapko mujhe mujhko hume humein tumhara tera teri iska uska
    iski uski inka unka apna apni apne khud
    i you he she it we they me him her us them my your his their our mine yours
    myself yourself himself herself itself themselves
""")

_AUX = _words("""
    hai hain ho hoon hun hu tha thi hoga hogi honge hota hoti hote hua hui hue
    raha rahi rahe sakta sakti sakte chahiye chahe chahta chahte padega padegi
    is are was were am be been being will would shall should can could may might
    must has have had does
""")
# 'the' is deliberately absent above: it is both a Hindi past-tense auxiliary
# (wo the) and the English article. The article is far more frequent in these
# transcripts, so it is left to the DET list.

_ADP = _words("""
    me mein par pe se ko ka ki ke tak liye saath bina andar bahar upar niche aage
    peeche taraf baare
    to from with without for of on in at by about into over under between during
    after before through across against
""")

_CONJ = _words("""
    aur ya lekin magar kyunki kyuki agar warna varna jab tab phir toh
    and or but because if unless while whereas so then although though
""")

_PART = _words("""
    nahi nahin na mat bhi hi ji please kripya kripaya kindly zara
    not no never neither nor
""")

_INTJ = _words("""
    namaste namaskar hello hi hey arre arey oh ah oye haan han ha ji sorry
    thanks thank shukriya dhanyavaad dhanyawad welcome bye alvida
""")

_DET = _words("""
    ek koi kuch kuchh sab sabhi har yah is us in un woh ye
    the a an this that these those some any every each all both few many
""")

_ADV = _words("""
    abhi turant foran fauran jaldi aaj kal parso ab phir dobara wapas kabhi hamesha
    bahut bohot
    bahot zyada jyada kam thoda thodi bilkul sirf keval jarur zaroor pehle baad
    now today tomorrow yesterday soon immediately immediately urgently quickly
    already still just very really only always never often again here there
""")

_ADJ = _words("""
    galat sahi thik theek achha accha bura pura poora sara jaruri zaroori naya
    purana bada chota khali
    urgent important legal illegal valid invalid suspicious fake fraudulent safe
    unsafe pending active inactive final last new old free full
""")

# Verbs are listed by stem and by the imperative forms that scam scripts use,
# because the imperative is the construction the rule baseline looks for.
_VERB = _words("""
    kar karo karna karne karta karti karte kiya kijiye kije
    bol bolo bolna bolta bolti bolte
    bata batao bataiye bataye batana
    de do dijiye dijiyega dena diya dedo
    le lo lijiye lena liya
    bhej bhejo bhejiye bhejna bheja
    dekh dekho dekhiye dekhna dekha
    sun suno suniye sunna suna
    ja jao jaiye jana jaye jayega jaayega jaegi
    aa aao aaiye aana aaya aayega
    rakh rakho rakhiye rakhna
    khol kholo kholiye kholna
    dab dabao dabaiye dabana
    bhar bharo bharna bhara
    mil milega milegi milenge mila
    lag lagega lagegi lagta
    chal chalo chalega
    samajh samjhiye samajhna
    bharo verify confirm check update install download click share send tell give
    pay transfer deposit block suspend freeze cancel deactivate disconnect arrest
    call calling press open close start stop wait hold continue follow complete
    provide submit register apply receive
""")

_NUM_WORDS = _words("""
    ek do teen tin char chaar paanch panch chhe chah saat aath nau das gyarah
    barah bees tees chalis pachas saath sattar assi nabbe sau hazaar hazar lakh
    crore karod
    one two three four five six seven eight nine ten eleven twelve twenty thirty
    forty fifty hundred thousand million
""")

#: Organisations, agencies and brands that show up as fraud props. Tagged
#: PROPN so the CRF gets a clean "this is a named institution" signal even when
#: the transcript is lowercase, which ASR output always is.
_PROPN = _words("""
    sbi hdfc icici axis kotak pnb canara idbi indusind
    paytm gpay phonepe bhim googlepay amazonpay mobikwik freecharge
    anydesk teamviewer quicksupport
    cbi rbi trai ncb ed interpol nia
    delhi mumbai kolkata chennai bengaluru bangalore hyderabad pune noida gurgaon
    airtel jio vodafone bsnl
    india indian bharat
    amazon flipkart myntra swiggy zomato blinkit
""")

# Suffix rules for words that are in no list. English derivational endings are
# reliable; the Hindi ones below are the verb and adjective endings that
# romanisation preserves.
_SUFFIX_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(tion|sion|ment|ness|ance|ence|ity|ship|hood|ism|ist|er|or)$"), "NOUN"),
    (re.compile(r"(able|ible|ful|less|ous|ive|al|ic|ish)$"), "ADJ"),
    (re.compile(r"ly$"), "ADV"),
    (re.compile(r"(ing|ed|ise|ize|ify|ate)$"), "VERB"),
    (re.compile(r"(iye|iyega|aiye|aiyega|aao|aoge|unga|ungi|enge|oge)$"), "VERB"),
    (re.compile(r"(ega|egi|enge|ayega|ayegi)$"), "VERB"),
    (re.compile(r"(wala|wali|wale)$"), "ADJ"),
)

_HI_VERB_END = re.compile(r"(te|ta|ti|na|ne|ao|iye|oge|enge)$")
_PUNCT_RE = re.compile(r"^[^\w\s]+$", re.UNICODE)
_NUM_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def _tag_one(token: str, index: int) -> str:
    if _PUNCT_RE.match(token):
        return "PUNCT"
    if _NUM_RE.match(token):
        return "NUM"

    low = token.lower()
    can = canonical(low)

    for form in (low, can):
        if form in _PRON:
            return "PRON"
        if form in _AUX:
            return "AUX"
        if form in _PART:
            return "PART"
        if form in _INTJ:
            return "INTJ"
        if form in _CONJ:
            return "CONJ"
        if form in _ADP:
            return "ADP"
        if form in _DET:
            return "DET"
        if form in _NUM_WORDS:
            return "NUM"
        if form in _ADV:
            return "ADV"
        if form in _ADJ:
            return "ADJ"
        if form in _VERB:
            return "VERB"
        if form in _PROPN:
            return "PROPN"

    # Gazetteer members are institution or instrument names: nouns, except the
    # threat phrases, which are verb phrases and are handled by the suffix
    # rules below.
    if gazetteer_type(low) in ("BANK_ENTITY", "PAYMENT_HANDLE", "PERSONAL_INFO_REQ", "OTP"):
        return "NOUN"
    if gazetteer_type(low) == "AUTHORITY_CLAIM":
        return "PROPN"

    # Hindi verb endings, gated on the language tag. Ungated, this rule would
    # eat English nouns such as 'minute' and 'state'.
    if token_language(low) == "hi" and _HI_VERB_END.search(low):
        return "VERB"

    for pat, tag in _SUFFIX_RULES:
        if pat.search(low):
            return tag

    if index > 0 and token[:1].isupper() and token[1:].islower():
        return "PROPN"
    if token.isupper() and len(token) > 1:
        return "PROPN"          # acronyms: OTP, KYC, UPI in a cased transcript
    if not low.isalpha():
        return "X"
    return "NOUN"


def pos_tag(tokens: Sequence[str]) -> List[str]:
    """Coarse POS tag per token. Always the same length as `tokens`."""
    return [_tag_one(t, i) for i, t in enumerate(tokens)]


def pos_tag_text(text: str) -> List[Tuple[str, str]]:
    """Convenience wrapper: tokenise with the shared tokeniser, then tag."""
    toks = tokenize(text)
    return list(zip(toks, pos_tag(toks)))


def pos_ngrams(tags: Sequence[str], n: int = 2) -> List[str]:
    """POS n-grams, used as a shape feature for the intent classifier.

    "VERB NOUN" and "NOUN VERB AUX" separate an instruction from a statement
    even when the two share every content word.
    """
    if n < 1 or len(tags) < n:
        return []
    return ["_".join(tags[i:i + n]) for i in range(len(tags) - n + 1)]


def tag_distribution(tags: Sequence[str]) -> Dict[str, float]:
    """Normalised tag counts, every tag in POS_TAGS present."""
    total = max(1, len(tags))
    counts = Counter(tags)
    return {t: counts.get(t, 0) / total for t in POS_TAGS}


if __name__ == "__main__":
    demos = [
        "Sir main cyber crime branch se bol raha hoon",
        "aapka account 10 minute mein block ho jayega",
        "OTP batao warna FIR darj ho jayegi",
        "Hum kabhi OTP nahi mangte, aap chahe to branch aakar mil sakte hain",
        "Your Amazon parcel will be delivered tomorrow, please confirm the address",
    ]
    for d in demos:
        pairs = pos_tag_text(d)
        print("  ".join(f"{w}/{t}" for w, t in pairs))
        print("   bigrams:", pos_ngrams([t for _, t in pairs])[:6])
    toks = tokenize(demos[1])
    assert len(pos_tag(toks)) == len(toks)
    print("\ntag distribution of turn 2:",
          {k: round(v, 2) for k, v in tag_distribution(pos_tag(toks)).items() if v})
