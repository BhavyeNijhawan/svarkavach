# Ethics, misuse and limitations

This document is deliberately blunt. A detector that oversells itself is worse
than no detector, because somebody eventually trusts it.

## What this system is

A defensive tool. It reads a recorded call and produces a risk score with the
evidence behind it. It does not place calls, generate scam scripts, clone a
stranger's voice on request, or automate any part of an attack.

## Voices

**The offline audio is not a clone of anybody.** `corpus/synth.py` is a
source-filter synthesiser: a glottal pulse train through formant resonators.
It produces signals with the acoustic statistics that separate real speech
from vocoder output, which is what the anti-spoofing branch needs to learn
from, and it produces nothing resembling a specific human being. That is
stated in the module docstring and again in the console, because a reader who
assumes otherwise would draw the wrong conclusion from every number in the
report.

**The Colab cloning notebook is restricted.** It clones a speaker's own voice,
with that speaker's written consent, for the purpose of populating the
synthetic cells of the evaluation grid. The consent form template is in
`docs/CONSENT_FORM.md`. Cloned audio stays inside the project and is deleted
after the assessment. No third party's voice is cloned under any
circumstances, including public figures.

## Content

No real bank name, no real UPI handle, no real phone number, no real case or
FIR number and no real person's name appears in the generated corpus. Invented
institutions ("Metro Bank", "XYZ Bank") are used throughout.

The scripts are written to be recognisable as fraud patterns for a detector,
not to be usable as fraud scripts. They are shorter, more explicit and less
adaptive than real attacks, which is a limitation of the corpus and is stated
as one rather than hidden.

## Where this fails

**False positives on legitimate synthetic voices carry real cost.** Bank IVR,
appointment reminders, delivery notifications and accessibility software are
all synthetic and all legitimate. An audio-only detector flags every one of
them. The intent branch exists precisely to clear these, and the
cloned-voice-plus-benign-script cell of the evaluation grid is where that gets
measured. If that cell scores badly, the system is not deployable, whatever
the headline AUC says.

**Offline numbers come from generated audio.** They demonstrate that the
method separates the four cells. They are not a claim about performance on
real telephone calls. The Colab notebooks run the same code on ASVspoof 2019
LA and the In-the-Wild set for that, and the console's provenance table marks
which numbers came from which source. Expect the In-the-Wild numbers to be
much worse than the ASVspoof ones. That gap is a known result in the
literature and analysing it is part of the report, not a failure to hide.

**Hinglish transcription is the weakest link.** Word error rate on code-mixed
telephone speech is high, and entity F1 falls with it. The system therefore
supports a transcript-in mode and reports the gold-versus-ASR gap as a
measured quantity. Production telephony systems use carrier-grade ASR, so
taking the transcript as an input is honest rather than evasive, but it does
mean the end-to-end claim depends on a component this project does not own.

**The corpus is generated, so it is regular.** A template grammar produces
more consistent phrasing than real people do. Models trained on it will look
better than they should. This is why the hard negatives were built in
deliberately and why the ablation is reported per cell rather than as a single
aggregate number: a single number would hide exactly the failure mode that
matters.

**A determined attacker can defeat this.** A human scammer with natural
prosody defeats the voice branch. A cloned voice reading a script with no
fraud vocabulary defeats the intent branch. The argument of this project is
that defeating both at once is harder, not that it is impossible.

## Deployment cautions

If anything like this were deployed on live calls:

- Recording and analysing a call is a surveillance capability. It needs
  informed consent from both parties and a legal basis, neither of which this
  project addresses.
- The score must never auto-terminate a call. It should surface a warning to
  the person receiving it and leave the decision with them.
- The evidence panel is not decoration. A score without checkable reasons
  cannot be appealed, and a system that cannot be appealed will eventually
  harm someone who did nothing wrong.
- Error rates differ across speakers, accents and channel conditions. Any
  deployment needs per-group evaluation, which this project does not have the
  data to do.

## Reporting

Problems with this system, including cases where it misclassifies in a way
that could harm someone, should go to the author. Nothing here is production
software.
