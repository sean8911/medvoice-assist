#!/usr/bin/env python
"""Generate conservative, deterministic responses for the medical-chatbot prototype.

The engine uses pipeline state and asserted symptoms. It does not infer a new
diagnosis, prescribe medication, or create disease-specific treatment plans.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from clarification_questions import generate_clarification
from symptom_extractor import SymptomExtractor
from urgency_detector import detect_urgency


DISCLAIMER = (
    "This chatbot is a research prototype. It cannot diagnose a condition or "
    "replace assessment by a qualified healthcare professional. General self-care "
    "may not be suitable for children, pregnancy, older adults, immune suppression, "
    "major long-term conditions, or people following a prescribed fluid restriction."
)


@dataclass(frozen=True)
class GuidanceRule:
    rule_id: str
    symptoms: frozenset[str]
    self_care: tuple[str, ...]
    safety_net: tuple[str, ...]
    source_title: str
    source_url: str


GUIDANCE_RULES = (
    GuidanceRule(
        "headache_guidance",
        frozenset(
            {
                "headache",
                "frontal headache",
                "spots or clouds in vision",
                "diminished vision",
                "double vision",
            }
        ),
        (
            "Drink water, rest, avoid alcohol, and avoid skipping meals while monitoring the headache.",
            "Record when the headache occurs, how long it lasts, its severity, and possible triggers.",
        ),
        (
            "Call the configured emergency service if a headache becomes sudden and extremely painful, or occurs with a seizure, weakness, confusion, speech difficulty, or loss of vision.",
        ),
        "NHS: Headaches",
        "https://www.nhs.uk/symptoms/headaches/",
    ),
    GuidanceRule(
        "respiratory_guidance",
        frozenset(
            {
                "cough",
                "coughing up sputum",
                "nasal congestion",
                "sore throat",
                "hoarse voice",
                "sneezing",
            }
        ),
        (
            "Rest and drink water regularly while monitoring the respiratory symptoms.",
            "Avoid smoking and use good hand and cough hygiene to reduce irritation and possible spread of infection.",
        ),
        (
            "Call the configured emergency service for severe breathing difficulty, blue or grey lips, confusion, or coughing up a significant amount of blood.",
            "Seek prompt medical assessment if breathlessness, chest pain, or rapid worsening develops.",
        ),
        "NHS: Respiratory tract infections",
        "https://www.nhs.uk/conditions/respiratory-tract-infection/",
    ),
    GuidanceRule(
        "gastrointestinal_guidance",
        frozenset({"vomiting", "diarrhea", "nausea", "decreased appetite"}),
        (
            "Rest and drink fluids; take small sips if nausea makes drinking difficult.",
            "Eat only when you feel able and monitor for signs of dehydration, such as very little urine.",
        ),
        (
            "Seek prompt medical assessment if you cannot keep fluids down, have very little urine, or symptoms are worsening.",
            "Call the configured emergency service for vomiting blood, black or bloody stool with fainting, confusion, or severe rapidly worsening abdominal pain.",
        ),
        "NHS: Diarrhoea and vomiting",
        "https://www.nhs.uk/conditions/diarrhoea-and-vomiting/",
    ),
    GuidanceRule(
        "fever_guidance",
        frozenset({"fever", "chills"}),
        (
            "Rest, drink water regularly, and record the measured temperature if a thermometer is available.",
        ),
        (
            "Seek prompt medical assessment if the fever is not improving, is worsening, or occurs in someone who is pregnant, immunocompromised, very young, older, or living with a serious long-term condition.",
            "Call the configured emergency service if fever occurs with confusion, severe breathing difficulty, a seizure, a stiff neck, or a rash that does not fade when pressed.",
        ),
        "NHS: High temperature in adults",
        "https://www.nhs.uk/symptoms/fever-in-adults/",
    ),
    GuidanceRule(
        "dizziness_guidance",
        frozenset({"dizziness", "fainting"}),
        (
            "Sit or lie down until the dizziness passes, get up slowly, rest, and drink water.",
            "Do not drive, climb, or operate machinery while dizzy.",
        ),
        (
            "Seek medical assessment if dizziness does not go away, keeps returning, or occurs with hearing, vision, pulse, fainting, headache, nausea, or vomiting changes.",
            "Call the configured emergency service for sudden one-sided weakness, facial drooping, speech difficulty, loss of consciousness, or severe chest pain.",
        ),
        "NHS: Dizziness",
        "https://www.nhs.uk/symptoms/dizziness/",
    ),
    GuidanceRule(
        "itchy_skin_guidance",
        frozenset({"itching of skin", "skin rash"}),
        (
            "Avoid scratching; a cool damp towel and loose clothing may reduce irritation.",
            "Avoid newly introduced perfumed skin products until a pharmacist or clinician has assessed the reaction.",
        ),
        (
            "Seek medical assessment if the rash is severe, widespread, keeps returning, affects daily life, or occurs during pregnancy.",
            "Call the configured emergency service for swelling of the lips, tongue, or throat, breathing difficulty, fainting, or a rash that does not fade when pressed.",
        ),
        "NHS: Itchy skin",
        "https://www.nhs.uk/symptoms/itchy-skin/",
    ),
)


EMERGENCY_SOURCE = {
    "source_id": "malaysia_mers_999",
    "title": "Malaysia Government: MERS 999 Emergency Line",
    "url": "https://www.malaysia.gov.my/en/categories/safety-and-community/public-safety/mers-999-emergency-line",
    "applies_to": "emergency_contact",
}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _compose_response(
    headline: str,
    message: str,
    actions: list[str],
    self_care: list[str],
    safety_net: list[str],
    questions: list[dict[str, Any]],
) -> str:
    sections = [headline, message]
    if actions:
        sections.append("Recommended next steps:\n- " + "\n- ".join(actions))
    if self_care:
        sections.append("General self-care:\n- " + "\n- ".join(self_care))
    if safety_net:
        sections.append("Get more help if:\n- " + "\n- ".join(safety_net))
    if questions:
        sections.append(
            "Please answer:\n- "
            + "\n- ".join(str(question["question"]) for question in questions)
        )
    sections.append(DISCLAIMER)
    return "\n\n".join(sections)


def generate_rule_based_response(
    narrative: str,
    symptom_extraction: dict[str, Any] | None,
    urgency: dict[str, Any] | None,
    clarification: dict[str, Any] | None,
    classification: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a user-facing response selected only from deterministic rules."""
    del narrative  # Reserved for future demographic and duration rules.

    urgency_level = urgency.get("urgency_level", "routine") if urgency else "routine"
    emergency_number = urgency.get("emergency_number", "999") if urgency else "999"
    present = set(
        symptom_extraction.get("present_symptoms", []) if symptom_extraction else []
    )
    clarification_questions = (
        list(clarification.get("questions", [])) if clarification else []
    )

    response_type: str
    headline: str
    message: str
    actions: list[str] = []
    self_care: list[str] = []
    safety_net: list[str] = []
    sources: list[dict[str, str]] = []
    displayed_prediction: dict[str, Any] | None = None
    matched_rule_ids: list[str] = []

    if urgency_level == "emergency":
        response_type = "emergency"
        headline = "Possible emergency warning signs detected"
        message = urgency["action_message"]
        actions = [
            f"Call {emergency_number} now and describe the symptoms and location.",
            "Do not wait for a disease prediction and do not drive yourself.",
        ]
        sources = [EMERGENCY_SOURCE]
    elif urgency_level == "urgent":
        response_type = "urgent"
        headline = "Prompt medical assessment recommended"
        message = urgency["action_message"]
        actions = [
            f"Contact {urgency['urgent_service']} now and report the symptoms clearly.",
            f"If symptoms become severe or rapidly worsen, call {emergency_number}.",
        ]
        sources = [EMERGENCY_SOURCE]
    elif clarification and (
        clarification.get("clarification_required")
        or clarification.get("should_withhold_prediction")
    ):
        response_type = "clarification"
        headline = "More symptom information is needed"
        message = (
            "I cannot show a reliable disease-class result from the information available. "
            "Please answer one short follow-up in your own words before classification continues."
        )
        actions = [
            "Reply naturally; no special format is required, and you may say 'no other symptoms'.",
            "Mention any new severe or rapidly worsening symptom immediately.",
        ]
        safety_net = [
            f"Call {emergency_number} for severe breathing difficulty, severe or persistent chest pain, sudden weakness or speech difficulty, heavy bleeding, a seizure, loss of consciousness, or immediate risk of self-harm."
        ]
        sources = [EMERGENCY_SOURCE]
    elif classification and classification.get("display_prediction"):
        response_type = "routine_prediction"
        headline = "Research model result"
        disease = str(classification["predicted_disease"])
        score = float(classification["model_score"])
        displayed_prediction = {
            "disease_class": disease,
            "model_score": score,
            "interpretation": "Similarity to a training-dataset class; not a diagnosis.",
        }
        message = (
            f"The reported symptom pattern is most similar to the '{disease}' class in the "
            "research dataset. This result must be confirmed by a healthcare professional."
        )
        actions = [
            "Arrange an assessment with a qualified healthcare professional to confirm the cause.",
            "Keep a record of symptom onset, duration, severity, changes, and any known triggers.",
            "Tell the clinician about current medicines, allergies, pregnancy, and long-term conditions.",
        ]
        for rule in GUIDANCE_RULES:
            if not present.intersection(rule.symptoms):
                continue
            matched_rule_ids.append(rule.rule_id)
            self_care.extend(rule.self_care)
            safety_net.extend(rule.safety_net)
            sources.append(
                {
                    "source_id": rule.rule_id,
                    "title": rule.source_title,
                    "url": rule.source_url,
                    "applies_to": rule.rule_id,
                }
            )
        if not safety_net:
            safety_net.append(
                f"Call {emergency_number} if severe or rapidly worsening symptoms, breathing difficulty, severe chest pain, sudden weakness, heavy bleeding, a seizure, or loss of consciousness develops."
            )
            sources.append(EMERGENCY_SOURCE)
        safety_net.append(
            "Seek medical assessment if symptoms persist, worsen, recur, or interfere with normal activities."
        )
    else:
        response_type = "no_prediction"
        classification_status = (
            classification.get("status", "") if classification else ""
        )
        minimum_symptoms = int(
            symptom_extraction.get("minimum_present_symptoms", 2)
            if symptom_extraction
            else 2
        )
        if classification_status == "clarification_limit_reached":
            symptom_count = len(present)
            symptom_list = ", ".join(sorted(present))
            if symptom_count:
                recognized = (
                    f"The chatbot recognized {symptom_count} current symptom"
                    f"{'s' if symptom_count != 1 else ''}: {symptom_list}."
                )
            else:
                recognized = "The chatbot did not recognize a current symptom."
            inference_performed = bool(
                classification.get("inference_performed")
                or classification.get("top_predictions")
            )
            if inference_performed:
                headline = "BioBERT result withheld because confidence was low"
                message = (
                    f"{recognized} BioBERT evaluated these symptoms, but its result did "
                    "not meet the configured confidence and class-separation thresholds. "
                    "The disease class is therefore hidden rather than presented as a "
                    "potentially misleading result."
                )
                actions = [
                    "Seek professional medical assessment if the symptoms persist, worsen, or concern you.",
                    "Start a new assessment only if symptoms change or additional symptoms are genuinely present.",
                ]
            else:
                headline = "Not enough distinct symptoms for classification"
                message = (
                    f"{recognized} Details such as duration, timing, severity, and "
                    "triggers provide useful context but do not count as separate "
                    "symptom labels. The BioBERT classifier was not run because this "
                    f"prototype requires at least {minimum_symptoms} current symptoms."
                )
                actions = [
                    "If other symptoms are genuinely present, start a new assessment and include them in the first description.",
                    "If this is the only symptom, seek professional medical assessment if it persists, worsens, or concerns you.",
                ]
            for rule in GUIDANCE_RULES:
                if not present.intersection(rule.symptoms):
                    continue
                matched_rule_ids.append(rule.rule_id)
                self_care.extend(rule.self_care)
                safety_net.extend(rule.safety_net)
                sources.append(
                    {
                        "source_id": rule.rule_id,
                        "title": rule.source_title,
                        "url": rule.source_url,
                        "applies_to": rule.rule_id,
                    }
                )
            safety_net.append(
                f"Call {emergency_number} for severe breathing difficulty, severe chest pain, sudden weakness or speech difficulty, heavy bleeding, a seizure, loss of consciousness, or immediate risk of self-harm."
            )
            sources.append(EMERGENCY_SOURCE)
        else:
            headline = "No disease-class result is available"
            message = (
                "The pipeline did not produce a disease class that should be shown. "
                "A healthcare professional can assess the symptoms directly."
            )
            actions = [
                "Seek professional medical assessment if symptoms are concerning, persistent, or worsening."
            ]
            safety_net = [
                f"Call {emergency_number} for severe breathing difficulty, severe chest pain, sudden weakness or speech difficulty, heavy bleeding, a seizure, loss of consciousness, or immediate risk of self-harm."
            ]
            sources = [EMERGENCY_SOURCE]

    actions = _unique(actions)
    self_care = _unique(self_care)
    safety_net = _unique(
        [
            item.replace("the configured emergency service", emergency_number)
            for item in safety_net
        ]
    )
    unique_sources: list[dict[str, str]] = []
    seen_source_ids: set[str] = set()
    for source in sources:
        if source["source_id"] not in seen_source_ids:
            unique_sources.append(source)
            seen_source_ids.add(source["source_id"])

    return {
        "response_type": response_type,
        "headline": headline,
        "message": message,
        "displayed_prediction": displayed_prediction,
        "recommended_actions": actions,
        "self_care_guidance": self_care,
        "safety_net_guidance": safety_net,
        "clarification_questions": clarification_questions,
        "matched_guidance_rule_ids": matched_rule_ids,
        "sources": unique_sources,
        "disclaimer": DISCLAIMER,
        "requires_clinical_review": True,
        "response_text": _compose_response(
            headline,
            message,
            actions,
            self_care,
            safety_net,
            clarification_questions,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test rule-based chatbot responses.")
    parser.add_argument("text", help="Patient symptom narrative.")
    args = parser.parse_args()

    extraction = SymptomExtractor().extract(args.text)
    urgency = detect_urgency(args.text, extraction)
    clarification = generate_clarification(args.text, extraction, urgency=urgency)
    classification = {
        "predicted_disease": None,
        "model_score": 0.0,
        "display_prediction": False,
        "status": "standalone_response_test",
    }
    response = generate_rule_based_response(
        args.text,
        extraction,
        urgency,
        clarification,
        classification,
    )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
