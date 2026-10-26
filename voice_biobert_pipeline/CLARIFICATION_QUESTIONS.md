# Clarification-Question Stage

## Purpose

This stage reduces unsupported disease-class output when the patient's first
message is incomplete, uncertain, or produces a weak BioBERT result. It does not
diagnose, replace clinical history taking, or provide a complete triage protocol.

## Decision order

1. Run narrative-to-symptom extraction.
2. Run urgency detection on the complete user narrative.
3. If urgency is `emergency` or `urgent`, show the safety action, defer questions,
   and bypass BioBERT.
4. If symptoms are absent, insufficient, or uncertain, return up to three
   clarification questions and do not load BioBERT.
5. Otherwise, classify the canonical symptom list with BioBERT.
6. Withhold the class and ask questions when the top score is below the configured
   threshold or the top two classes are too close.
7. Display a class only when `classification.display_prediction` is `true`.

## Question selection

The generator first chooses questions that address the reason for clarification.
Safety questions are then prioritized for the extracted symptom group, followed
by onset, duration, severity, and associated-symptom questions. Current groups
cover headache, abdominal, urinary, respiratory, fever, gastrointestinal,
allergy/rash, neurological, and mental-health presentations.

Question output is structured for a later chatbot interface:

```json
{
  "question_id": "headache_red_flags",
  "question": "Did the headache start suddenly ...?",
  "category": "safety",
  "purpose": "Check for time-critical neurological warning signs.",
  "answer_type": "yes_no_with_details",
  "safety_related": true
}
```

## Conversation handling

The application should store the initial narrative and question IDs in its session.
When the user answers, append the answer to the original narrative and rerun symptom
extraction and urgency detection. Do not send only the short answer to BioBERT,
because answers such as "yes" have no useful symptom context.

Limit one clarification round initially. If the combined narrative still cannot
produce enough asserted symptoms, return a non-diagnostic message recommending
professional assessment instead of repeatedly questioning the user.

## Evaluation

Report the clarification stage separately from disease-classifier accuracy. Useful
measures include trigger precision, red-flag question recall on scripted scenarios,
average number of questions, percentage of cases resolved after one round, and the
change in classification accuracy and weighted F1 after clarification.

The score and margin defaults are research heuristics, not clinically calibrated
cutoffs. Select them on a validation set and have the question wording and rule
coverage reviewed by a qualified healthcare professional before user deployment.
