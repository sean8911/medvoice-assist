# Rule-Based Urgency Gate

## Purpose

The urgency gate runs after symptom extraction and before BioBERT disease
classification. Its purpose is to prevent a routine disease prediction from
overriding clear warning signs. It is a research prototype, not a complete
clinical triage system.

## Decision Flow

```text
Whisper transcript
  -> narrative-to-symptom extraction
  -> urgency rules evaluate narrative + asserted current symptoms
       -> emergency: show emergency action; do not run BioBERT
       -> urgent: show same-day assessment action; do not run BioBERT
       -> routine: continue to confirmation and BioBERT
```

Negated, historical, and other-person mentions are excluded. Uncertain raw
emergency phrases are treated conservatively as actionable.

## Implemented Rules

| Rule group | Examples of evidence | Result |
|---|---|---|
| Severe breathing/unresponsive | gasping, choking, blue lips, not responding | Emergency |
| Immediate self-harm/violence | explicit immediate intent, plan, or overdose | Emergency |
| Stroke warning signs | facial droop, slurred speech, one-sided weakness, sudden severe headache | Emergency |
| Dangerous chest pain | chest pain with breathlessness, sweating, fainting, nausea, arm pain, or rhythm symptoms | Emergency |
| Possible anaphylaxis | airway swelling/tightness with breathing, wheezing, rash, or allergy signs | Emergency |
| Seizure emergency | first seizure, over 5 minutes, repeated without recovery, seizure now | Emergency |
| Major bleeding/shock | heavy bleeding or bleeding with faintness, weakness, or abnormal breathing | Emergency |
| Pregnancy emergency | bleeding with severe abdominal/shoulder pain, dizziness, or fainting | Emergency |
| Fever and stiff neck | fever, neck stiffness, and another neurological warning sign | Emergency |
| Prompt assessment | chest pain, breathing difficulty, reported seizure, bleeding, neurological deficit, persistent vomiting, pregnancy bleeding, urinary retention | Urgent |

## Authoritative Sources

- [Malaysia Government: MERS 999 Emergency Line](https://www.malaysia.gov.my/en/categories/safety-and-community/public-safety/mers-999-emergency-line)
- [NHS: Heart attack](https://www.nhs.uk/conditions/heart-attack/)
- [CDC: Signs and Symptoms of Stroke](https://www.cdc.gov/stroke/signs-symptoms/)
- [NHS: Anaphylaxis](https://www.nhs.uk/conditions/anaphylaxis/)
- [NHS: Epilepsy](https://www.nhs.uk/conditions/epilepsy/)
- [NHS: Meningitis](https://www.nhs.uk/conditions/meningitis/)
- [NHS: Vaginal bleeding in pregnancy](https://www.nhs.uk/pregnancy/common-symptoms/vaginal-bleeding/)
- [NHS: Urgent mental-health help](https://www.nhs.uk/nhs-services/mental-health-services/where-to-get-urgent-help-for-mental-health/)
- [NHS: Blood in urine](https://www.nhs.uk/symptoms/blood-in-urine/)
- [NHS: Coughing up blood](https://www.nhs.uk/symptoms/coughing-up-blood/)
- [NHS: Rectal bleeding](https://www.nhs.uk/symptoms/bleeding-from-the-bottom-rectal-bleeding/)
- [NHS: Diarrhoea and vomiting](https://www.nhs.uk/symptoms/diarrhoea-and-vomiting/)

## Required Validation

The rules must be reviewed by a qualified clinician before any real-world use.
Evaluate them on an independently annotated test set and report emergency-rule
sensitivity, specificity, precision, recall, F1 score, false-negative rate, and
false-positive rate. False negatives should receive the greatest attention.

A `routine` result means only that no configured rule matched. It must never be
presented as confirmation that the user is medically safe.
