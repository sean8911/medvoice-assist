#!/usr/bin/env python
"""Generate structured clarification questions before or after BioBERT.

Questions are selected from asserted symptoms and missing context. Emergency and
urgent actions always take priority and are never delayed by clarification.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from symptom_extractor import SymptomExtractor, normalize_text
from urgency_detector import detect_urgency


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    priority: int
    category: str
    purpose: str
    answer_type: str
    safety_related: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.text,
            "category": self.category,
            "purpose": self.purpose,
            "answer_type": self.answer_type,
            "safety_related": self.safety_related,
        }


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def _has_onset_or_duration(text: str) -> bool:
    return _contains_any(
        text,
        (
            r"\b(?:started|began|onset|since|today|yesterday|tonight|this morning|"
            r"last night|suddenly|gradually|recently)\b",
            r"\bfor\s+(?:\d+|one|two|three|four|five|six|seven|several|a few)\s+"
            r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b",
        ),
    )


def _has_severity(text: str) -> bool:
    return _contains_any(
        text,
        (
            r"\b(?:mild|moderate|severe|extreme|unbearable|worst)\b",
            r"\b(?:pain|severity|score)\s*(?:is|of|at|:)??\s*\d{1,2}\s*(?:/|out of)\s*10\b",
            r"\b(?:about\s+)?\d{1,2}\s*(?:/|out of)\s*10\b",
        ),
    )


def _join_symptoms(symptoms: Iterable[str], limit: int = 4) -> str:
    values = list(symptoms)[:limit]
    if not values:
        return "the symptom"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" and {values[-1]}"


def _add(candidates: list[Question], question: Question) -> None:
    if not any(existing.question_id == question.question_id for existing in candidates):
        candidates.append(question)


def _add_targeted_questions(candidates: list[Question], symptoms: set[str]) -> None:
    if symptoms & {
        "headache",
        "frontal headache",
        "spots or clouds in vision",
        "diminished vision",
        "double vision",
    }:
        _add(
            candidates,
            Question(
                "headache_red_flags",
                (
                    "Did the headache start suddenly or feel like the worst headache you have "
                    "experienced, and do you have weakness, confusion, fainting, slurred speech, "
                    "or a new loss of vision?"
                ),
                10,
                "safety",
                "Check for time-critical neurological warning signs.",
                "yes_no_with_details",
                True,
            ),
        )
        _add(
            candidates,
            Question(
                "headache_characteristics",
                "Where is the headache, when did it start, and how severe is it from 0 to 10?",
                30,
                "symptom_detail",
                "Collect location, onset, and severity for the headache.",
                "free_text_and_scale",
            ),
        )

    if symptoms & {"sharp abdominal pain", "burning abdominal pain", "lower abdominal pain", "upper abdominal pain", "side pain", "pelvic pain"}:
        _add(
            candidates,
            Question(
                "abdominal_red_flags",
                (
                    "Is the abdominal pain severe or rapidly worsening, and is there repeated "
                    "vomiting, blood, fainting, a high fever, or a possibility of pregnancy?"
                ),
                10,
                "safety",
                "Check for abdominal and pregnancy warning signs.",
                "yes_no_with_details",
                True,
            ),
        )
        _add(
            candidates,
            Question(
                "abdominal_characteristics",
                (
                    "Where exactly is the abdominal pain, when did it start, and is it constant "
                    "or does it come and go?"
                ),
                30,
                "symptom_detail",
                "Clarify pain location, onset, and pattern.",
                "free_text",
            ),
        )

    if symptoms & {"painful urination", "frequent urination", "blood in urine", "retention of urine", "side pain", "suprapubic pain"}:
        _add(
            candidates,
            Question(
                "urinary_red_flags",
                (
                    "Can you still pass urine, and do you have fever, chills, side or back pain, "
                    "vomiting, or visible blood in the urine?"
                ),
                10,
                "safety",
                "Check for urinary obstruction, bleeding, or infection warning signs.",
                "yes_no_with_details",
                True,
            ),
        )

    if symptoms & {"cough", "coughing up sputum", "wheezing", "nasal congestion", "sore throat", "hoarse voice"}:
        _add(
            candidates,
            Question(
                "respiratory_red_flags",
                (
                    "Do you have trouble breathing, chest pain, blue or grey lips, confusion, or "
                    "are you coughing up blood?"
                ),
                10,
                "safety",
                "Check for severe respiratory warning signs.",
                "yes_no_with_details",
                True,
            ),
        )
        _add(
            candidates,
            Question(
                "cough_characteristics",
                "How long have you had the cough, and is it dry or producing mucus?",
                30,
                "symptom_detail",
                "Clarify cough duration and characteristics.",
                "free_text",
            ),
        )

    if "fever" in symptoms:
        _add(
            candidates,
            Question(
                "fever_red_flags",
                (
                    "What temperature did you measure, and do you have a stiff neck, a rash that "
                    "does not fade when pressed, confusion, severe sleepiness, or trouble breathing?"
                ),
                10,
                "safety",
                "Check temperature and serious infection warning signs.",
                "free_text",
                True,
            ),
        )

    if symptoms & {"vomiting", "diarrhea", "nausea", "decreased appetite"}:
        _add(
            candidates,
            Question(
                "gastrointestinal_red_flags",
                (
                    "Can you keep fluids down, and is there blood in the vomit or stool, black "
                    "stool, severe abdominal pain, fainting, or very little urine?"
                ),
                10,
                "safety",
                "Check for dehydration, bleeding, and severe abdominal warning signs.",
                "yes_no_with_details",
                True,
            ),
        )

    if symptoms & {"skin rash", "skin swelling", "allergic reaction", "itching of skin", "lip swelling"}:
        _add(
            candidates,
            Question(
                "rash_allergy_red_flags",
                (
                    "Did the rash or swelling appear suddenly, and do you have swelling of the "
                    "lips, tongue, or throat, trouble breathing, fainting, or a rash that does not "
                    "fade when pressed?"
                ),
                10,
                "safety",
                "Check for anaphylaxis or serious infection warning signs.",
                "yes_no_with_details",
                True,
            ),
        )

    if symptoms & {"dizziness", "fainting", "loss of sensation", "arm weakness", "leg weakness", "difficulty speaking", "slurring words"}:
        _add(
            candidates,
            Question(
                "neurological_red_flags",
                (
                    "Did this begin suddenly, and do you have one-sided weakness or numbness, "
                    "facial drooping, slurred speech, chest pain, or loss of consciousness?"
                ),
                10,
                "safety",
                "Check for stroke, cardiac, or loss-of-consciousness warning signs.",
                "yes_no_with_details",
                True,
            ),
        )

    if symptoms & {"depression", "anxiety and nervousness", "depressive or psychotic symptoms", "delusions or hallucinations"}:
        _add(
            candidates,
            Question(
                "mental_health_safety",
                (
                    "Are you currently thinking about harming yourself or someone else, or do "
                    "you feel unable to keep yourself safe?"
                ),
                10,
                "safety",
                "Check for immediate mental-health danger.",
                "yes_no",
                True,
            ),
        )


def _single_conversational_question(
    present: list[str],
    uncertain: list[str],
    model_uncertainty: bool,
) -> Question:
    """Create one natural follow-up that combines confirmation and useful detail."""
    understood = _join_symptoms(present)
    symptom_set = set(present) | set(uncertain)

    if not present:
        text = (
            "In your own words, what is the main symptom you have right now, and is "
            "there anything else happening with it? Mention severe breathing trouble, "
            "severe chest pain, sudden weakness, heavy bleeding, or fainting immediately."
        )
    elif uncertain:
        text = (
            f"I understood {understood}, but I was unsure about "
            f"{_join_symptoms(uncertain)}. Tell me naturally which symptoms you have "
            "right now and which you do not."
        )
    elif model_uncertainty:
        text = (
            f"I understood that you currently have {understood}. Is that correct, and "
            "did I miss another current symptom? Reply naturally, or say 'no other "
            "symptoms'."
        )
    elif symptom_set & {
        "cough",
        "coughing up sputum",
        "wheezing",
        "nasal congestion",
        "sore throat",
        "hoarse voice",
    }:
        text = (
            f"I understood {understood}. Besides these, do you currently have fever, "
            "wheezing, shortness of breath, chest pain, mucus, heartburn, or another "
            "symptom? Reply naturally or say 'no other symptoms', and mention severe "
            "breathing difficulty or coughing blood immediately."
        )
    elif symptom_set & {
        "headache",
        "frontal headache",
        "spots or clouds in vision",
        "diminished vision",
        "double vision",
    }:
        text = (
            f"I understood {understood}. Did the headache start suddenly, and do you "
            "also have nausea, vomiting, fever, vision changes, weakness, confusion, "
            "or speech difficulty? Reply naturally or say 'no other symptoms'."
        )
    elif symptom_set & {
        "sharp abdominal pain",
        "burning abdominal pain",
        "lower abdominal pain",
        "upper abdominal pain",
        "side pain",
        "pelvic pain",
    }:
        text = (
            f"I understood {understood}. Where is the discomfort, and do you also have "
            "vomiting, fever, diarrhea, bleeding, fainting, urinary changes, or another "
            "symptom? Reply naturally or say 'no other symptoms'."
        )
    elif symptom_set & {
        "painful urination",
        "frequent urination",
        "blood in urine",
        "retention of urine",
        "suprapubic pain",
    }:
        text = (
            f"I understood {understood}. Do you also have fever, chills, side or back "
            "pain, vomiting, visible blood, difficulty passing urine, or another symptom? "
            "Reply naturally or say 'no other symptoms'."
        )
    else:
        text = (
            f"I understood {understood}. What other symptom, if any, is happening with "
            "this? Reply naturally in your own words, or say 'no other symptoms'."
        )

    return Question(
        "natural_followup",
        text,
        0,
        "natural_followup",
        "Confirm the extracted symptoms and collect one concise free-form update.",
        "free_text",
        True,
    )


def generate_clarification(
    narrative: str,
    symptom_extraction: dict[str, Any] | None,
    urgency: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    max_questions: int = 3,
) -> dict[str, Any]:
    """Select a small set of questions and decide whether prediction must pause."""
    if max_questions <= 0:
        raise ValueError("max_questions must be greater than zero.")

    if urgency is not None and urgency.get("urgency_level") in {"emergency", "urgent"}:
        return {
            "clarification_required": False,
            "should_pause_classification": True,
            "should_withhold_prediction": True,
            "deferred_due_to_urgency": True,
            "reason_codes": ["urgency_override"],
            "questions": [],
            "question_count": 0,
            "message": "Clarification is deferred because the urgency action takes priority.",
        }

    if symptom_extraction is None:
        return {
            "clarification_required": False,
            "should_pause_classification": False,
            "should_withhold_prediction": False,
            "deferred_due_to_urgency": False,
            "reason_codes": ["symptom_extraction_disabled"],
            "questions": [],
            "question_count": 0,
            "message": "Clarification was not evaluated because symptom extraction is disabled.",
        }

    present = list(symptom_extraction.get("present_symptoms", []))
    uncertain = list(symptom_extraction.get("uncertain_symptoms", []))
    minimum = int(symptom_extraction.get("minimum_present_symptoms", 2))
    reasons: list[str] = []
    if not present:
        reasons.append("no_current_symptoms")
    elif len(present) < minimum:
        reasons.append("insufficient_current_symptoms")
    if uncertain:
        reasons.append("uncertain_symptoms")

    model_uncertainty_status = (
        classification.get("status") if classification is not None else None
    )
    model_uncertainty = bool(
        classification
        and classification.get("predicted_disease") is not None
        and model_uncertainty_status
        in {"low_score_review_symptoms", "ambiguous_prediction_review_symptoms"}
    )
    if model_uncertainty_status == "low_score_review_symptoms":
        reasons.append("low_model_score")
    elif model_uncertainty_status == "ambiguous_prediction_review_symptoms":
        reasons.append("ambiguous_model_prediction")

    preclassification_reasons = {
        "no_current_symptoms",
        "insufficient_current_symptoms",
        "uncertain_symptoms",
    }
    pause = bool(preclassification_reasons & set(reasons))
    required = bool(reasons)
    if not required:
        return {
            "clarification_required": False,
            "should_pause_classification": False,
            "should_withhold_prediction": False,
            "deferred_due_to_urgency": False,
            "reason_codes": [],
            "questions": [],
            "question_count": 0,
            "message": "No clarification trigger was detected.",
        }

    candidates: list[Question] = []
    if not present:
        _add(
            candidates,
            Question(
                "describe_main_symptom",
                (
                    "Please describe the main symptom you are experiencing, where it is, and "
                    "what it feels like."
                ),
                0,
                "missing_information",
                "Obtain at least one current symptom in the user's own words.",
                "free_text",
            ),
        )
        _add(
            candidates,
            Question(
                "general_emergency_check",
                (
                    "Are you currently having severe trouble breathing, severe or persistent "
                    "chest pain, sudden weakness or speech difficulty, heavy bleeding, a seizure, "
                    "or thoughts of immediate self-harm?"
                ),
                10,
                "safety",
                "Give the user another opportunity to report an emergency warning sign.",
                "yes_no_with_details",
                True,
            ),
        )

    if uncertain:
        _add(
            candidates,
            Question(
                "confirm_uncertain_symptoms",
                (
                    f"Are you currently experiencing {_join_symptoms(uncertain)}, or are you "
                    "unsure whether these symptoms are present?"
                ),
                0,
                "confirmation",
                "Confirm symptoms that were expressed with uncertainty.",
                "yes_no_with_details",
            ),
        )

    symptom_set = set(present) | set(uncertain)
    _add_targeted_questions(candidates, symptom_set)

    if model_uncertainty and present:
        _add(
            candidates,
            Question(
                "confirm_extracted_symptoms",
                (
                    f"I understood that you currently have {_join_symptoms(present)}. Is that "
                    "correct, and did I miss any important symptom?"
                ),
                20,
                "confirmation",
                "Verify the model input after a low-score classification.",
                "yes_no_with_details",
            ),
        )

    candidate_ids = {question.question_id for question in candidates}
    targeted_onset_questions = {
        "headache_characteristics",
        "abdominal_characteristics",
        "cough_characteristics",
    }
    targeted_severity_questions = {"headache_characteristics"}

    if (
        present
        and not _has_onset_or_duration(narrative)
        and not candidate_ids.intersection(targeted_onset_questions)
    ):
        _add(
            candidates,
            Question(
                "onset_and_duration",
                "When did the symptoms start, and are they constant, intermittent, improving, or worsening?",
                40,
                "timeline",
                "Collect onset, duration, and progression.",
                "free_text",
            ),
        )

    if (
        present
        and not _has_severity(narrative)
        and not candidate_ids.intersection(targeted_severity_questions)
    ):
        _add(
            candidates,
            Question(
                "severity",
                "How severe are the symptoms from 0 to 10, and are they stopping normal activities?",
                50,
                "severity",
                "Collect symptom severity and functional impact.",
                "scale_and_free_text",
            ),
        )

    if present:
        _add(
            candidates,
            Question(
                "additional_symptoms",
                (
                    "What other symptoms occur with this, such as fever, pain, vomiting, weakness, "
                    "breathing changes, bleeding, or a rash?"
                ),
                60,
                "associated_symptoms",
                "Collect additional symptoms that may improve classification and safety review.",
                "free_text",
            ),
        )

    if max_questions == 1:
        selected = [
            _single_conversational_question(present, uncertain, model_uncertainty)
        ]
    else:
        selected = sorted(candidates, key=lambda item: (item.priority, item.question_id))[
            :max_questions
        ]
    return {
        "clarification_required": True,
        "should_pause_classification": pause,
        "should_withhold_prediction": True,
        "deferred_due_to_urgency": False,
        "reason_codes": reasons,
        "questions": [question.as_dict() for question in selected],
        "question_count": len(selected),
        "max_questions": max_questions,
        "context": {
            "present_symptoms": present,
            "uncertain_symptoms": uncertain,
            "onset_or_duration_detected": _has_onset_or_duration(narrative),
            "severity_detected": _has_severity(narrative),
        },
        "message": (
            "More information is needed before showing a disease prediction. "
            "After the user answers, append the answers to the narrative and rerun symptom "
            "extraction and urgency detection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clarification questions.")
    parser.add_argument("text", help="Patient narrative.")
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument(
        "--simulate-low-score",
        action="store_true",
        help="Simulate a low BioBERT score to test post-classification questions.",
    )
    args = parser.parse_args()

    extraction = SymptomExtractor().extract(args.text)
    urgency = detect_urgency(args.text, extraction)
    classification = None
    if args.simulate_low_score:
        classification = {
            "predicted_disease": "research-placeholder",
            "model_score": 0.10,
            "status": "low_score_review_symptoms",
        }
    result = generate_clarification(
        args.text,
        extraction,
        urgency=urgency,
        classification=classification,
        max_questions=args.max_questions,
    )
    print(
        json.dumps(
            {
                "symptom_extraction": extraction,
                "urgency": urgency,
                "clarification": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
