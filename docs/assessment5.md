NAME - BHAVYE NIJHAWAN

REG. NO. - 23BAI0100

SPEECH AND LANGUAGE PROCESSING LAB

LAB ASSESSMENT - 5

INTRODUCTION AND OBJECTIVES OF THE WORK

**Voice Clone Fraud Call Detection: Combining Acoustic Anti Spoofing with Scam Intent Analysis for Hindi-English Calls**

**1. Introduction**

A fraud call is a performance with a fixed script. The caller says they are from a bank, a courier company, a government office or the police, or they pretend to be a relative who is in trouble. They announce a problem, attach a deadline to it, tell the person not to discuss it with anyone, and only after all of that do they ask for the thing they actually want: an OTP, a card number, a UPI PIN or a transfer. The pattern is not a secret. Banks print it on the back of debit cards and regulators repeat it in advertisements. It keeps working because the call is designed to remove the few seconds in which a person would remember the warning.

Two developments have changed the scale of the problem. Automated dialling lets one script reach thousands of numbers in a day, so a very low success rate is still profitable. And neural text to speech can now produce Hindi speech that a listener on a phone line cannot reliably distinguish from a person. The line itself helps the attacker: telephone audio is band limited and compressed, so many of the small imperfections that would give a synthetic voice away in a studio recording are stripped out before the listener hears it. The caller no longer needs to speak the target's language well, stay calm under questions, or even be present on the call.

There is a third feature of Indian fraud calls that makes them harder for software to handle than for a person. They are not in Hindi and not in English but in both at the same time, switching inside a sentence, as in "Aapka account block ho jayega, abhi OTP bata dijiye." Tokenisers, dictionaries and entity taggers built for one language mishandle the other half, and most published systems are built and tested on English only.

Assessment 4 of this lab surveyed the literature on both sides of the problem and found that the two relevant research communities do not talk to each other. Anti spoofing work asks whether a voice is synthetic and evaluates on isolated utterances. Scam detection work reads the transcript and never hears the voice. Neither is evaluated on code mixed Indian telephone conversation, and neither considers the case that matters most in practice: a benign call in a cloned voice, or a scam delivered by a real person. This document states the problem precisely, sets the scope, and lists the objectives that the implementation was built to meet.

**2. Motivation**

Three observations motivate the design.

First, the voice and the words carry different information and neither is sufficient alone. A bank really does send automated reminders in a synthetic voice, so "synthetic" cannot mean "fraud". A human scammer with a real voice walks past an audio only detector. What is suspicious is the combination: a request for a sensitive credential, delivered under pressure, possibly in a voice that is not a person.

Second, a fraud call has structure in time. The request for the OTP is only alarming because of the threat that came before it and the instruction not to hang up that comes after it. A detector that scores each sentence on its own will miss the ramp; a detector that reads the order of dialogue acts can see it being built.

Third, the delivery and the content should agree. A person reading a threat sounds different from a person reading a greeting. A machine reading a script sounds the same on both. When the words press hard and the voice stays flat, the mismatch itself is evidence, and it is evidence that neither a pure acoustic detector nor a pure text classifier can produce.

**3. Problem Statement**

To design and evaluate a system that, given a Hindi-English code mixed phone call as audio with its transcript, produces after every caller turn a calibrated fraud risk probability and a human readable set of reasons, by combining (a) a synthetic voice detector trained on real Hindi recordings paired with cloned renderings of the same sentences, (b) a fraud entity tagger and scam intent classifier for code mixed text, (c) a model of the dialogue act sequence that measures how coercive pressure grows across the call, and (d) a measure of the mismatch between the pressure in the words and the arousal in the voice, and to show through controlled experiments what each part contributes, with the evaluation data built and audited so that the numbers measure the voice and the intent rather than the recording conditions.

**4. Scope of the Work**

The work covers the following.

- A generated corpus of 480 Hinglish calls in which voice authenticity and scam content are varied independently, giving four balanced cells: human voice with benign content, human voice with scam content, cloned voice with benign content, cloned voice with scam content. Every entity span, dialogue act and per token language tag is produced by the template slot that produced the text, so the annotation is exact rather than annotated by hand afterwards.
- A set of 600 real Hindi utterances from the public GramVaani corpus, each paired with a neural rendering of its own transcript, with the recording channel matched on level, spectral colour and bandwidth so that the detector cannot learn the microphone instead of the voice.
- Four analysis branches and a fusion layer, implemented in Python and running on a laptop CPU without a GPU.
- A web console that plays a call, shows the risk band updating turn by turn, highlights the entities found, and explains each alert.
- An evaluation that reports every number per cell, on a deliberately hard subset, on real fraud recordings from outside the training data, and under telephone codec degradation.

The following are outside the scope. No real fraud victims' calls were recorded or used; all scam scripts are fictional and all institution names in them are invented. The cloned voices are the Hindi neural voices available commercially; evaluation against the wider set of synthesis systems in IndicSynth is planned as the next step.

**5. Objectives**

The objectives are written so that each one is checked by a specific experiment in the final report.

1. **Develop a code mixed Hindi-English call corpus with exact annotations.** Design a template grammar that produces scam and benign dialogues, renders them with neural voices, and emits entity, dialogue act and language labels from the same slots that produce the words. Measure whether the corpus leaks its labels through topic or phrasing before any model is trained, and revise the grammar until a bag of words classifier can no longer separate the classes on authorship alone.

2. **Build a voice authenticity branch on real Hindi speech.** Pair real field recordings with cloned renderings of the same transcripts, match the two sides on the recording channel, and audit the pair set with simple channel statistics before reporting any error rate. Compare five cepstral front ends (LFCC, MFCC, GFCC, CQCC, LPCC) with a Gaussian mixture back end, a gradient boosted back end and a RawNet style end to end detector, add a TDNN speaker embedding consistency measure, and report EER and minimum t-DCF under clean and codec degraded conditions.

3. **Build a text branch for fraud entities and scam intent.** Transcribe the call with Whisper, implement a linear chain conditional random field that tags seven entity types (OTP, bank entity, authority claim, threat or deadline, payment handle, personal information request, money amount) in code mixed text with a BiLSTM-CRF counterpart for comparison, and a TF-IDF intent classifier over word and character n grams with a lexicon rule baseline.

4. **Build structural and cross modal features that neither branch can produce alone.** Fit a dialogue act sequence model and derive the slope and peak of coercive pressure across the call; compute a prosody intent mismatch score that compares lexical pressure with acoustic arousal turn by turn; compute code mixing statistics and callee resistance.

5. **Fuse the branches into one calibrated, streaming risk score with an evidence panel.** Train the fusion layer on speakers that no branch was trained on, calibrate it so that the probability can be read as a probability, map it to four operational bands, and present each alert with the entities, phrases, voice cues and coercion step behind it.

6. **Evaluate against baselines and under realistic conditions.** Run an ablation over audio only, text only, late fusion and full arms on the same test split; report per cell and on the hard subset; measure entity F1 by type; measure time to detection relative to the sensitive request; sweep G.711 and GSM codecs; and test the intent model on 1,413 real recorded robocalls that it never saw.

**6. Expected Outcomes**

- A working console that accepts a call, streams a risk band and reasons per turn, and accepts an uploaded recording for analysis.
- An annotated Hinglish call corpus with cloned audio, reusable by later work, together with the grammar that generated it.
- A matched Hindi anti spoofing pair set with its confound audit, and models trained on it.
- Numbers for each objective: ablation by arm and by cell, entity F1 by type, anti spoofing EER by front end and codec, out of domain recall on real robocalls, time to detection, and calibration.
- A leak check and a channel audit built into the pipeline, so that the reported numbers measure the voice and the intent rather than the recording conditions or the authorship of the scripts.

**7. Ethical Considerations**

All scam scripts are fictional. No real bank, company, hospital or school is named; the slot tables contain invented names only. Regulator and police names appear because impersonating them is the fraud pattern being modelled. Real speech comes from GramVaani, a public research corpus, and no private individual's voice was cloned. The synthetic voices are commercial text to speech voices used within their terms. The robocall recordings are a public dataset collected for research. The system is a detector, not a generator; nothing in it produces fraudulent content for use.

**8. References**

1. Wang X., Delgado H., Tak H., et al. (2024). ASVspoof 5: Crowdsourced speech data, deepfakes, and adversarial attacks at scale. Proc. ASVspoof Workshop 2024.
2. Zhu Y., Koppisetti S., Tran T., Bharaj G. (2024). SLIM: Style-linguistics mismatch model for generalized audio deepfake detection. Proc. NeurIPS 2024.
3. Sharma D. V., Ekbote V., Gupta A. (2025). IndicSynth: A large-scale multilingual synthetic speech dataset for low-resource Indian languages. Proc. ACL 2025.
4. Shen Z., Yan S., Zhang Y., Luo X., Ngai G., Fu E. Y. (2025). "It warned me just at the right moment": Exploring LLM-based real-time detection of phone scams. arXiv:2502.03964.
5. Ma Z., Wang P., Huang M., et al. (2025). TeleAntiFraud-28k: An audio-text slow-thinking dataset for telecom fraud detection. arXiv:2503.24115.
6. Yi J., Wang C., Tao J., Zhang X., Zhang C. Y., Zhao Y. (2023). Audio deepfake detection: A survey. arXiv:2308.14970.
