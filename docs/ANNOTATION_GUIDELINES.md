# Annotation guidelines

The fraud entity tagset, the dialogue act set, and the rules an annotator (or
the generator) follows. This document is the reference an evaluator checks the
labels against, and it is dated evidence of how the tagset was designed.

Version 1.0. Tokenisation is always `swarkavach.schema.tokenize`, so BIO tags
and tokens are guaranteed to line up.

---

## 1. Why this tagset and not a general-purpose one

Standard NER gives you PERSON, LOCATION and ORGANISATION. None of those help
decide whether a phone call is a fraud. A caller naming a city is not
suspicious; a caller naming a police department while attaching a ten-minute
deadline to your bank account is.

So the seven types below are chosen to answer questions a fraud analyst
actually asks. Each one corresponds to a step in the extraction scripts that
Indian cyber-crime advisories describe: who the caller claims to be, what they
claim is wrong, what happens if you refuse, and what they want you to hand
over.

## 2. Entity types

### OTP
A one-time password or verification code: the mention, the request, or the
digits themselves.

- Tag the code word and the digits as one span when they are adjacent:
  `[OTP OTP 445566]`.
- Tag the request even with no digits present: `[OTP OTP]  batao`.
- Six-digit runs in a code context are part of the span. A six-digit run in an
  account context is not; that is `BANK_ENTITY`.
- Includes: `otp`, `one time password`, `verification code`, `code`,
  `chhe ank ka code`, `sms code`.

### BANK_ENTITY
References to a banking instrument or relationship.

- Includes: `account`, `khata`, `card`, `debit card`, `credit card`,
  `netbanking`, `KYC`, `passbook`, `IFSC`, `branch`, `ATM`.
- Tag the head noun and its modifier together: `[BANK_ENTITY savings account]`.
- The bank's invented name is part of the span when adjacent:
  `[BANK_ENTITY Metro Bank account]`.

### AUTHORITY_CLAIM
A claim to institutional power, whether or not it is plausible.

- Includes: `police`, `cyber cell`, `cyber crime branch`, `CBI`, `RBI`,
  `TRAI`, `income tax department`, `customs`, `narcotics control bureau`,
  `magistrate`, `inspector`, `commissioner`.
- Tag the full institutional name: `[AUTHORITY_CLAIM cyber crime branch]`, not
  just `branch`.
- A rank alone counts when it is used to assert standing:
  `main [AUTHORITY_CLAIM inspector] bol raha hoon`.
- A genuine bank agent saying "customer care" is **not** an authority claim.
  Authority means state or regulatory power.

### THREAT_DEADLINE
A consequence bound to a clock. Both halves must be present in the span or in
the immediate context.

- `[THREAT_DEADLINE das minute mein block ho jayega]`.
- `[THREAT_DEADLINE account will be suspended today]`.
- A threat with no time bound is still tagged if the consequence is
  institutional (arrest, FIR, seizure). A deadline with no threat is not
  tagged; a delivery slot is not a threat.
- This is the type annotators disagree on most. The test: would a reasonable
  person feel they had to act before a specific moment or lose something?

### PAYMENT_HANDLE
Where money or control is meant to go.

- Includes: `UPI ID`, `VPA`, `account number` given as a destination,
  `QR code`, `payment link`, `processing fee`, `registration fee`,
  `security deposit`, `beneficiary`.
- The destination string itself is inside the span when present.
- An account number the caller claims is *yours* is `BANK_ENTITY`. An account
  number they want money sent *to* is `PAYMENT_HANDLE`. Direction decides.

### PERSONAL_INFO_REQ
A request for something a real institution already holds or would never ask
for by phone.

- Includes: `CVV`, `MPIN`, `ATM PIN`, `card number`, `expiry date`,
  `Aadhaar number`, `PAN number`, `date of birth`, `mother's maiden name`,
  `password`, and remote-access tooling (`AnyDesk`, `TeamViewer`,
  `screen share`).
- Remote-access apps are here rather than in their own type because
  functionally they are a request for everything at once.
- OTP is its own type because it is frequent enough to be worth counting
  separately.

### MONEY_AMOUNT
A sum, in digits or words. `[MONEY_AMOUNT saade teen lakh]`,
`[MONEY_AMOUNT 4,500 rupees]`. Include the currency word when adjacent.

## 3. BIO conventions

- `B-TYPE` on the first token of a span, `I-TYPE` on every following token,
  `O` elsewhere.
- Spans never overlap. When two types compete, the more specific one wins:
  OTP beats BANK_ENTITY, PERSONAL_INFO_REQ beats BANK_ENTITY.
- Spans never cross a turn boundary. A threat completed in the next turn is
  tagged in whichever turn carries the head.
- Punctuation is a token and is tagged `O` unless it sits inside a span, which
  in practice happens only inside amounts.
- Maximum span length is six tokens. Longer than that means the annotator has
  captured a clause rather than an entity.

## 4. Dialogue acts

One act per turn, chosen for the turn's primary function. The full set is in
`schema.DIALOGUE_ACTS`; these are the ones that carry the fraud signal.

| Act | What it does | Pressure rank |
| --- | --- | --- |
| `GREET`, `SMALLTALK` | opening pleasantries | 0.00 |
| `IDENTIFY_SELF` | states a name or a company | 0.10 |
| `INFORM`, `CONFIRM` | gives or checks a fact | 0.10 to 0.15 |
| `PROBLEM_STATE` | asserts something is wrong | 0.35 |
| `REASSURE` | promises the problem is fixable | 0.30 |
| `INSTRUCT` | tells the callee to do something | 0.50 |
| `AUTHORITY_ASSERT` | claims state or regulatory standing | 0.55 |
| `THREAT` | states a consequence | 0.80 |
| `DEADLINE` | attaches a clock to it | 0.85 |
| `ISOLATE` | tells the callee not to hang up or tell anyone | 0.90 |
| `REQUEST_SENSITIVE` | asks for the OTP, PIN, or remote access | 0.95 |
| `PRESSURE_ESCALATE` | repeats the demand with more force | 1.00 |
| `VICTIM_QUESTION` | callee asks for clarification | n/a |
| `VICTIM_RESIST` | callee pushes back or refuses | n/a |
| `VICTIM_COMPLY` | callee agrees or supplies information | n/a |

The pressure ranks are hand-set from the script structure described in I4C and
RBI advisories rather than learned, because the coercion-slope feature has to
stay interpretable in the evidence panel. A learned ranking would score better
and explain worse.

Callee acts get no rank. They are counted separately as the resistance ratio,
since a callee who keeps asking questions is being pressured, and that is
evidence about the caller.

## 5. Per-token language tags

`hi`, `en` or `univ`, from `corpus.lexicon.token_language`.

- `univ` is for digits, punctuation, and the discourse particles both
  languages use identically (`ji`, `sir`, `madam`, `ok`, `hmm`).
- Established English loanwords used inside Hindi sentences (`account`,
  `block`, `police`, `OTP`) are tagged `en`, not `univ`. They are the English
  financial and legal vocabulary the entity-language alignment feature exists
  to count, and folding them into `univ` would erase the signal.
- Never guess. A token the lexicon and the orthographic rules cannot resolve
  is `univ`, which keeps the code-mixing statistics honest at the cost of a
  slightly lower switch count.

## 6. Hard cases, decided

These are the calls that separate a working system from one that flags every
bank call.

**A real bank reminder that mentions KYC and an account.** Tag
`BANK_ENTITY`. Do not tag `THREAT_DEADLINE` for a due date with no
consequence. The call is benign and must be labelled benign. If the model
cannot tell this from a KYC scam, the model is not finished.

**A genuine delivery call asking the customer to read out an OTP.** Tag the
`OTP`. Label the call benign. This is the single hardest negative in the
corpus and it is there on purpose: a text-only classifier fails it, and the
system is supposed to clear it because the delivery context, the absence of
threat and deadline acts, and the natural prosody all argue against fraud.

**A soft scam that asks one innocent-sounding verification question.** Tag
whatever entities appear, which may be only `BANK_ENTITY`. Label the call
fraudulent. The text branch will probably miss it. The voice branch is
supposed to catch it, and that is the point of having two.

**A cloned voice reading a benign script.** Every entity and act is tagged as
if the voice were human, because the transcript does not change. The voice
label is separate metadata (`label_voice`), never an entity.

## 7. Quality control

The generated corpus emits gold labels by construction: a template slot of
type OTP produces exactly the BIO tags over exactly the tokens it inserted, so
there is no annotator drift to control for. `tests/test_corpus.py` asserts
that every BIO sequence is well formed, that gold entities decode back to the
surface strings the slots inserted, and that splits are speaker-disjoint.

For any hand-annotated material added later, the procedure is two independent
passes with Cohen's kappa computed per entity type, and adjudication of every
disagreement against the rules above. Report kappa in the write-up. Anything
below 0.75 on a type means the rule for that type is not clear enough yet and
this document needs another revision, not that the annotators need more
practice.
