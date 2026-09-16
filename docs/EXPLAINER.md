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

**1. Is the voice real?** The voice branch looks at the fine structure of the sound: how the spectrum is shaped, how steady the pitch is, how the loudness wobbles, and whether the speaker's voice drifts a little from one sentence to the next the way a real person's does. Three detectors sit behind it (a classical Gaussian mixture, a gradient-boosted classifier, and a small end-to-end neural network that reads the raw waveform), all trained on 600 real Hindi field recordings, each paired with a machine-made copy of the same sentence.

**2. What is being asked for?** The text branch transcribes the call with Whisper and reads the words. A tagger marks seven kinds of dangerous things: OTPs, bank names, claims of authority ("RBI se bol raha hoon"), threats and deadlines, payment handles, requests for personal details, amounts of money. A classifier scores how much the words sound like a scam.

**3. In what order?** The structure branch watches the sequence. A real bank reminder goes: greet, identify, inform, close. A scam goes: greet, identify, problem, authority, threat, deadline, isolate, then the request. The branch measures whether the pressure is climbing.

**4. Do the words and the voice agree?** This is the part that is new. A human reading a threat gets worked up. A machine reading a threat sounds exactly the same as a machine reading a greeting. So if the words are pressing hard and the voice is flat, that mismatch is itself a signal.

**5. Combine and explain.** A small, calibrated model takes 25 numbers from the four branches and turns them into one probability. It was trained on speakers the four branches had never heard, so it cannot cheat by memorising them. With the score comes an evidence panel: which entities were found, which phrases raised the alarm, what the voice detector saw, and how far up the pressure ladder the call has climbed.

## What the numbers say

| What was measured | Result | What it means |
|---|---|---|
| 120 test calls, all four branches | AUC 1.000, F1 0.983 | On the test corpus it gets almost everything right |
| Time to alert | 57 of 60 scams, median 35.8 s, 6 turns | Flags before the OTP is asked for |
| 1,413 real recorded robocalls it never saw | 78.7 percent caught, 5 percent false alarms | It generalises to real fraud, even in English |
| Real Hindi voice vs cloned Hindi voice | EER under 0.02 clean, under 0.04 on GSM | The voice branch separates them cleanly, even through a phone codec |
| Entity tagging | F1 0.970 on 324 spans | It finds the OTPs, threats and bank names |
| Calibration | Brier 0.011 | When it says 90 percent, it is right about 90 percent of the time |

The single most important number is the third row. Everything else was measured on calls we generated ourselves. The robocall recordings are real, from another country, in another language, and the system had never seen them. It still caught nearly four in five.

## Scope, if asked

**The test corpus is generated.** No one will hand over 480 real fraud calls with consent, so the calls were written by a template system and read by neural voices, which is also what makes every label exact. The voice branch is tested separately on 600 real recordings, and the text branch on 1,413 real robocalls.

**The cloned voices are the Hindi neural voices available commercially.** The next test is against IndicSynth, which has 989 speakers across 12 Indian languages.

**Text numbers are on reference transcripts.** That isolates the tagger from recognition error; the recogniser is scored separately by word error rate.

## The part that makes the numbers trustworthy

A detector like this can score well for the wrong reason, so two checks are built into the pipeline. Before any model is trained, a plain bag-of-words classifier is run on the transcripts; if it can separate scam from benign on topic or phrasing alone, the corpus is revised. That is why five subjects (a parcel, a bank KYC call, a transaction alert, a relative in trouble, a loan offer) each have a scam version and a benign version that say the same things and differ only in what they ask for. And before the voice branch is trained, ten channel statistics are checked on the real-versus-cloned pairs; both sides go through the same telephone band-pass and noise matching, and the build stops if any single statistic could tell the classes apart. If a teacher asks "how do you know the detector is hearing the voice and not the microphone," this is the answer.

## The two-minute version

"Phone scams follow a script, and now the script can be read by a cloned voice. Existing tools either check the voice or read the words, never both. SwarKavach does both, on a live Hinglish call, and adds two things nobody else measures: whether the pressure in the conversation is climbing the way a script climbs, and whether the emotion in the voice matches the pressure in the words. It flags scams six turns in, before the OTP is requested, and explains why. On real recorded scam calls it had never seen, it caught 79 percent with 5 percent false alarms, and it tells a real Hindi voice from a cloned one at under 2 percent error."

## Likely questions and short answers

**Why generate the corpus instead of collecting real calls?** Consent and labels. Real fraud calls cannot be collected ethically at scale, and hand-labelling entities in 5,000 turns would take weeks and still have errors. Generation gives exact labels, and we tested on real recordings separately.

**Why not just use a large language model on the transcript?** One recent paper does exactly that, and it works. But it reads only text, so a cloned voice reading a benign script looks identical to a real one, and it needs an LLM call per sentence, which is slow and expensive on a phone. Our text branch is Whisper plus a small classifier that runs in a quarter of a second.

**Does it work in real time?** Yes, per turn. The whole analysis of a 74-second call takes about 19 seconds on a laptop with no GPU, most of it recomputing over the whole call each turn. Making it incremental is the first engineering item on the list.

**What if the scammer is a real person?** Then the voice branch says "real" and the other three branches carry the decision. Half of the scam calls in the test set are in human voices, and they are all caught.

**What would you do next?** Test the voice branch against IndicSynth, a new dataset with 989 speakers across 12 Indian languages. Report the entity and intent numbers on Whisper output alongside the reference. And try a code-mixed transformer (MuRIL or HingBERT) for intent against the TF-IDF model.

## Where things are

- Full report: `docs/PROJECT_REPORT.docx`
- Code: `src/swarkavach/` (corpus generator, four branches, fusion, console)
- Results: `data/results/*.json`
- Console: run `python -m swarkavach.cli serve` and open http://127.0.0.1:7860. The Upload tab with a real recording and its cloned partner is the best live demonstration of the voice branch.
