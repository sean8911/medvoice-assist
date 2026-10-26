# Rule-Based Response Generation

## Purpose

The response stage turns the pipeline decision into deterministic chatbot text.
It runs after urgency detection, clarification, and BioBERT classification. It
does not infer a new condition or provide a disease-specific treatment plan.

No additional training dataset is required. The content is a small curated rule
base, and each supported self-care group records its source URL in the output.

## Response states

| State | Behaviour |
|---|---|
| `emergency` | Show the configured emergency action; no prediction or self-care |
| `urgent` | Recommend prompt assessment; no prediction or self-care |
| `clarification` | Show follow-up questions; hide any classifier result |
| `routine_prediction` | Explain the accepted class as a research result and add matching guidance |
| `no_prediction` | Give a professional-assessment fallback without naming a disease |

The response never exposes a predicted class unless
`classification.display_prediction` is `true`.

## Supported guidance

The first version contains conservative symptom-level rules for:

- Headache
- Respiratory or cold-like symptoms
- Nausea, vomiting, or diarrhoea
- Fever or chills
- Dizziness
- Itchy skin or rash

Medication names, dosages, and antibiotic recommendations are deliberately
excluded. Unsupported symptoms and any unsupported disease class use the general
professional-assessment fallback.

## Sources

- [Malaysia Government: MERS 999 Emergency Line](https://www.malaysia.gov.my/en/categories/safety-and-community/public-safety/mers-999-emergency-line)
- [NHS: Headaches](https://www.nhs.uk/symptoms/headaches/)
- [NHS: Respiratory tract infections](https://www.nhs.uk/conditions/respiratory-tract-infection/)
- [NHS: Diarrhoea and vomiting](https://www.nhs.uk/conditions/diarrhoea-and-vomiting/)
- [NHS: High temperature in adults](https://www.nhs.uk/symptoms/fever-in-adults/)
- [NHS: Dizziness](https://www.nhs.uk/symptoms/dizziness/)
- [NHS: Itchy skin](https://www.nhs.uk/symptoms/itchy-skin/)

## Chatbot integration

The UI can display `rule_based_response.response_text` directly or render its
structured arrays separately. The original pipeline JSON should be retained for
research logging, but hidden predictions must never be read from the internal
classification object and shown to the user.

The rules should remain deterministic and version controlled. Changes should
record the source, access date, reviewer, reason, and affected test scenarios.

## Evaluation and limitations

Evaluate response-state correctness, source coverage, hidden-label leakage,
emergency and urgent override correctness, template consistency, and clinician
ratings for safety and usefulness. This content is not a validated clinical
triage protocol. A qualified healthcare professional must review the rule logic
and wording before testing with real users.

General self-care may not apply to children, pregnancy, older adults, immune
suppression, major long-term conditions, or prescribed fluid restrictions. The
chatbot should collect these characteristics in a later user-profile stage and
avoid self-care output when applicability is uncertain.
