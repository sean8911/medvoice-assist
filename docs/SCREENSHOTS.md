# App screenshots

No app screenshots are committed yet. Automatic Windows browser capture was
blocked because the computer-use tool could not verify the current browser URL.
The app was launched locally, but that does not constitute a completed visual
inspection. No mock UI or fabricated model result is substituted for a capture.

## Capture these three views

1. **Welcome screen:** show the full interface, text/voice controls and the
   verified model status before a consultation begins.
2. **Fictional consultation:** show the actual submitted description, recognised
   symptoms and the real accepted or withheld classification response.
3. **Research details:** open the panel after the consultation, expand BioBERT
   output and stage timings, and keep the canonical input/status readable.

Use the same browser size and zoom for consistent captures. Hide browser account
details, unrelated tabs and desktop information. Do not include a real patient's
description, participant name, email, credentials or confidential logs.

## Suggested fictional scenario

Initial message:

```text
I have a headache.
```

Follow-up:

```text
I also feel nauseous, have vomited twice, and saw flashing lights before the
headache. I do not have weakness, confusion, fainting, or slurred speech.
```

Keep whatever the real app returns. Do not change symptoms or edit a displayed
score just to make a prediction look stronger. One successful example is not an
accuracy benchmark, and model scores are not clinical probabilities.

## Add genuine captures

After reviewing the images, store them under `docs/assets/` as:

```text
app-welcome.png
app-consultation.png
app-research-details.png
```

Add relative image links with descriptive alternative text to the README's Demo
section. Record the capture date and whether each example is typed or transcribed.
If there is no speech capture, do not label a typed demonstration as a complete
voice-pipeline demonstration. Do not add broken image placeholders in advance.
