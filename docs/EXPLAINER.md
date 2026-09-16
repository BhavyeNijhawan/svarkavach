# SwarKavach in plain words

A short guide to what this project is, how it works, and what to say about it.

---

## The problem in one paragraph

A fraud call in India follows a script: someone pretends to be from a bank, a courier company, the police, or a relative in trouble; they announce a problem; they put a clock on it; they tell you not to talk to anyone; and only then do they ask for an OTP, a card number, or a transfer. Everyone has heard this pattern. It still works because the call is built to take away the minute in which you would remember it. Two things made it worse recently. Automated dialling means one script reaches thousands of people a day. And voice cloning means the script can now be read by a machine in a Hindi voice that sounds like a person on a phone line.

## What SwarKavach does

It listens to a call as it happens and, after every sentence the caller says, gives a risk score from 0 to 1 with a band: low, elevated, high, critical. When the score crosses 0.65 it raises an alert and explains why. On the test calls it raises that alert at a median of 36 seconds, before the caller has asked for anything sensitive.

It works on Hinglish, the mix of Hindi and English that real Indian calls are actually spoken in: *"Aapka account block ho jayega, OTP bata dijiye."*

## How it works: four things it listens for at once

Think of four people listening to the same call, each with one job, and a fifth person who combines what they say.

**1. Is the voice real?** The voice branch looks at the fine structure of the sound: how the spectrum is shaped, how steady the pitch is, how the loudness wobbles. Machine voices are too smooth. This branch was trained on 600 real Hindi field recordings, each paired with a machine-made copy of the same sentence.

**2. What is being asked for?** The text branch reads the words. A tagger marks seven kinds of dangerous things: OTPs, bank names, claims of authority ("RBI se bol raha hoon"), threats and deadlines, payment handles, requests for personal details, amounts of money. A classifier scores how much the words sound like a scam.

**3. In what order?** The structure branch watches the sequence. A real bank reminder goes: greet, identify, inform, close. A scam goes: greet, identify, problem, authority, threat, deadline, isolate, then the request. The branch measures whether the pressure is climbing.

**4. Do the words and the voice agree?** This is the part that is new. A human reading a threat gets worked up. A machine reading a threat sounds exactly the same as a machine reading a greeting. So if the words are pressing hard and the voice is flat, that mismatch is itself a signal.

**5. Combine.** A small, calibrated model takes 25 numbers from the four branches and turns them into one probability. It was trained on speakers the four branches had never heard, so it cannot cheat by memorising them.

## What the numbers say

| What was measured | Result | What it means |
|---|---|---|
| 120 test calls, all four branches | AUC 1.000, F1 0.983 | On the test corpus it gets almost everything right |
| Time to alert | 57 of 60 scams, median 35.8 s, 6 turns | Flags before the OTP is asked for |
| 1,413 real recorded robocalls it never saw | 78.7 percent caught, 5 percent false alarms | It generalises to real fraud, even in English |
| Real Hindi voice vs cloned Hindi voice | Near-zero error | The voice branch separates them cleanly |
| Entity tagging | F1 0.970 on 324 spans | It finds the OTPs, threats and bank names |
| Calibration | Brier 0.011 | When it says 90 percent, it is right about 90 percent of the time |

The single most important number is the third row. Everything else was measured on calls we generated ourselves. The robocall recordings are real, from another country, in another language, and the system had never seen them. It still caught nearly four in five.

## What is honest to admit

Three things, and it is better to say them before being asked.

**The test corpus is generated, not real.** No one will give you 480 real fraud calls with consent. So the calls were written by a template system and read by neural voices. The upside is that every label is exact. The downside is that on scripted data the words alone are nearly enough, so the corpus cannot prove that the voice branch adds much. The voice branch is proven separately, on the 600 real recordings.

**The cloned voices come from two synthesisers.** Microsoft ships two Hindi voices, so every fake in the voice test comes from one of two systems. The detector only has to recognise those two. A real-world detector would meet dozens. The near-zero error is a fact about our test set, not a claim about voice cloning in general.

**No speech recogniser was run.** The text branch read the reference transcript. A deployed system would read a recogniser's output, which has errors, so the text numbers are an upper bound.

## What we found out along the way, which is half the project

Twice, the system scored perfectly for the wrong reason, and both times we caught it and fixed it.

The first time, the text classifier hit AUC 1.000 even after we deleted every fraud word from the transcripts. It was recognising the *topic* of a call, because every scam topic was different from every benign topic. We rebuilt the corpus so that five subjects (a parcel, a bank KYC call, a transaction alert, a relative in trouble, a loan offer) each have a scam version and a benign version that say the same things and differ only in what they ask for.

The second time, the voice detector hit zero error. A review found that the real recordings were telephone quality with nothing above 3 kHz, and the fake ones were studio quality. The detector was hearing the microphone, not the voice. We put both sides through the same band-pass, matched their noise, and built an audit that runs ten simple checks on every build and refuses to continue if any one of them can tell the classes apart.

If a teacher asks "what was the hardest part," this is the answer.

## The two-minute version

"Phone scams follow a script, and now the script can be read by a cloned voice. Existing tools either check the voice or read the words, never both. SwarKavach does both, on a live Hinglish call, and adds two things nobody else measures: whether the pressure in the conversation is climbing the way a script climbs, and whether the emotion in the voice matches the pressure in the words. It flags scams six turns in, before the OTP is requested. On real recorded scam calls it had never seen, it caught 79 percent with 5 percent false alarms. And along the way we found and fixed two ways a detector like this can score perfectly while learning nothing, which we think is as useful as the detector."

## Likely questions and short answers

**Why generate the corpus instead of collecting real calls?** Consent and labels. Real fraud calls cannot be collected ethically at scale, and hand-labelling entities in 5,000 turns would take weeks and still have errors. Generation gives exact labels, and we tested on real recordings separately.

**Why not just use a large language model on the transcript?** One recent paper does exactly that, and it works. But it reads only text, so a cloned voice reading a benign script looks identical to a real one, and it needs an LLM call per sentence, which is slow and expensive on a phone. Our text branch is a small classifier that runs in a quarter of a second.

**Does it work in real time?** Yes, per turn. The whole analysis of a 74-second call takes about 19 seconds on a laptop with no GPU, most of it recomputing over the whole call each turn. Making it incremental is the first engineering item on the list.

**What if the scammer is a real person?** Then the voice branch says "real" and the other three branches carry the decision. Half of the scam calls in the test set are in human voices, and they are all caught.

**What would you do next?** Test the voice branch against IndicSynth, a new dataset with 989 speakers across 12 Indian languages, so the "two synthesisers" limitation goes away. And run a Hindi speech recogniser so the text branch is tested on real transcripts.

## Where things are

- Full report: `docs/PROJECT_REPORT.docx`
- Code: `src/swarkavach/` (corpus generator, four branches, fusion, console)
- Results: `data/results/*.json`
- Console: run `python -m swarkavach.cli serve` and open http://127.0.0.1:7860. The Upload tab with a real recording and its cloned partner is the best live demonstration of the voice branch.
