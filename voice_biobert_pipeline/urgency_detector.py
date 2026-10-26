#!/usr/bin/env python
"""Rule-based urgency gate for the voice medical-chatbot prototype.

This module detects a deliberately limited set of red-flag combinations before
disease classification. It does not diagnose a condition or prove that an input
is safe when no rule matches.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from symptom_extractor import SymptomExtractor, assertion_status, normalize_text


SOURCE_CATALOG = {
    "malaysia_emergency": {
        "title": "Malaysia Government - MERS 999 Emergency Line",
        "url": "https://www.malaysia.gov.my/en/categories/safety-and-community/public-safety/mers-999-emergency-line",
    },
    "heart_attack": {
        "title": "NHS - Heart attack",
        "url": "https://www.nhs.uk/conditions/heart-attack/",
    },
    "stroke": {
        "title": "CDC - Signs and Symptoms of Stroke",
        "url": "https://www.cdc.gov/stroke/signs-symptoms/",
    },
    "anaphylaxis": {
        "title": "NHS - Anaphylaxis",
        "url": "https://www.nhs.uk/conditions/anaphylaxis/",
    },
    "seizure": {
        "title": "NHS - Epilepsy",
        "url": "https://www.nhs.uk/conditions/epilepsy/",
    },
    "pregnancy_bleeding": {
        "title": "NHS - Vaginal bleeding in pregnancy",
        "url": "https://www.nhs.uk/pregnancy/common-symptoms/vaginal-bleeding/",
    },
    "mental_health": {
        "title": "NHS - Where to get urgent help for mental health",
        "url": "https://www.nhs.uk/nhs-services/mental-health-services/where-to-get-urgent-help-for-mental-health/",
    },
    "meningitis": {
        "title": "NHS - Meningitis",
        "url": "https://www.nhs.uk/conditions/meningitis/",
    },
    "vomiting": {
        "title": "NHS - Diarrhoea and vomiting",
        "url": "https://www.nhs.uk/symptoms/diarrhoea-and-vomiting/",
    },
    "blood_in_urine": {
        "title": "NHS - Blood in urine",
        "url": "https://www.nhs.uk/symptoms/blood-in-urine/",
    },
    "coughing_blood": {
        "title": "NHS - Coughing up blood",
        "url": "https://www.nhs.uk/symptoms/coughing-up-blood/",
    },
    "rectal_bleeding": {
        "title": "NHS - Rectal bleeding",
        "url": "https://www.nhs.uk/symptoms/bleeding-from-the-bottom-rectal-bleeding/",
    },
    "urinary_retention": {
        "title": "NHS - Prostatitis urgent symptoms",
        "url": "https://www.nhs.uk/conditions/prostatitis/",
    },
}

LEVEL_PRIORITY = {"routine": 0, "urgent": 1, "emergency": 2}


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    level: str
    title: str
    reason: str
    evidence: tuple[str, ...]
    source_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "title": self.title,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "sources": [SOURCE_CATALOG[source_id] for source_id in self.source_ids],
        }


def _literal_pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in normalize_text(phrase).split()]
    body = r"\s+".join(words)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _asserted_phrase_hits(text: str, phrases: Iterable[str]) -> list[dict[str, str]]:
    """Return present or uncertain red-flag phrases, excluding clear denials."""
    normalized = normalize_text(text)
    hits: list[dict[str, str]] = []
    occupied: set[tuple[int, int, str]] = set()
    for phrase in sorted(set(phrases), key=len, reverse=True):
        for match in _literal_pattern(phrase).finditer(normalized):
            status, reason = assertion_status(normalized, match.start(), match.end())
            if status not in {"present", "uncertain"}:
                continue
            key = (match.start(), match.end(), normalize_text(phrase))
            if key in occupied:
                continue
            occupied.add(key)
            hits.append(
                {
                    "phrase": match.group(0),
                    "status": status,
                    "reason": reason,
                }
            )
    return sorted(hits, key=lambda item: normalized.find(item["phrase"]))


def _evidence(*groups: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    for group in groups:
        for value in group:
            if value not in values:
                values.append(value)
    return tuple(values)


def detect_urgency(
    narrative: str,
    symptom_extraction: dict[str, Any],
    emergency_number: str = "999",
    urgent_service: str = "a same-day medical service",
) -> dict[str, Any]:
    """Evaluate asserted red flags and return an urgency override decision."""
    present = set(symptom_extraction.get("present_symptoms", []))
    matches: list[RuleMatch] = []

    def raw(phrases: Iterable[str]) -> list[dict[str, str]]:
        return _asserted_phrase_hits(narrative, phrases)

    def phrases(hits: list[dict[str, str]]) -> tuple[str, ...]:
        return tuple(f"phrase: {item['phrase']} ({item['status']})" for item in hits)

    severe_breathing = raw(
        (
            "not breathing",
            "stopped breathing",
            "struggling to breathe",
            "gasping for air",
            "gasping for breath",
            "choking",
            "cannot get words out",
            "can't get words out",
            "blue lips",
            "lips turned blue",
            "turning blue",
        )
    )
    unresponsive = raw(
        (
            "unconscious",
            "unresponsive",
            "cannot wake up",
            "can't wake up",
            "not responding",
            "passed out and not waking up",
        )
    )
    if severe_breathing or unresponsive:
        matches.append(
            RuleMatch(
                "life_threatening_breathing_or_response",
                "emergency",
                "Severe breathing difficulty or unresponsiveness",
                "The narrative contains an asserted sign of severe breathing difficulty or unresponsiveness.",
                _evidence(phrases(severe_breathing), phrases(unresponsive)),
                ("heart_attack", "anaphylaxis", "malaysia_emergency"),
            )
        )

    immediate_self_harm = raw(
        (
            "i am going to kill myself",
            "i'm going to kill myself",
            "i want to kill myself",
            "end my life",
            "i have a suicide plan",
            "about to harm myself",
            "about to hurt myself",
            "taken an overdose",
            "took an overdose",
            "i overdosed",
            "i am going to kill someone",
            "i want to kill someone",
        )
    )
    if immediate_self_harm:
        matches.append(
            RuleMatch(
                "immediate_self_harm_or_violence_risk",
                "emergency",
                "Immediate danger of self-harm or harm to another person",
                "The narrative indicates possible immediate danger and requires emergency human support.",
                phrases(immediate_self_harm),
                ("mental_health", "malaysia_emergency"),
            )
        )

    mental_health_urgent = raw(
        (
            "suicidal thoughts",
            "thinking about suicide",
            "thoughts of suicide",
            "thoughts of self harm",
            "thinking about self harm",
            "want to hurt myself",
        )
    )
    if mental_health_urgent:
        matches.append(
            RuleMatch(
                "urgent_mental_health_risk",
                "urgent",
                "Urgent mental-health concern",
                "The narrative contains asserted suicidal or self-harm thoughts without an explicit immediate action statement.",
                phrases(mental_health_urgent),
                ("mental_health",),
            )
        )

    stroke_phrases = raw(
        (
            "face drooping",
            "facial droop",
            "one side of my face is drooping",
            "one sided weakness",
            "weakness on one side",
            "slurred speech",
            "speech became slurred",
            "speech is slurred",
            "speech suddenly became slurred",
            "one arm feels weak",
            "one arm is weak",
            "sudden trouble speaking",
            "sudden confusion",
            "sudden loss of vision",
            "sudden severe headache",
            "worst headache of my life",
            "thunderclap headache",
        )
    )
    stroke_symptoms = present & {"slurring words", "focal weakness"}
    stroke_combination = (
        "difficulty speaking" in present
        and bool(
            present
            & {
                "arm weakness",
                "leg weakness",
                "loss of sensation",
                "double vision",
                "dizziness",
            }
        )
    )
    if stroke_phrases or stroke_symptoms or stroke_combination:
        symptom_evidence = tuple(f"symptom: {item}" for item in sorted(stroke_symptoms))
        if stroke_combination:
            symptom_evidence += ("symptom combination: difficulty speaking with a neurological symptom",)
        matches.append(
            RuleMatch(
                "possible_stroke_signs",
                "emergency",
                "Possible stroke warning signs",
                "Sudden speech, facial, one-sided weakness, vision, or severe-headache signs require emergency assessment.",
                _evidence(phrases(stroke_phrases), symptom_evidence),
                ("stroke", "malaysia_emergency"),
            )
        )

    chest_pain = present & {"sharp chest pain", "burning chest pain", "chest tightness"}
    cardiac_companions = present & {
        "shortness of breath",
        "difficulty breathing",
        "sweating",
        "nausea",
        "vomiting",
        "dizziness",
        "fainting",
        "arm pain",
        "neck pain",
        "palpitations",
        "irregular heartbeat",
    }
    dangerous_chest_phrases = raw(
        (
            "crushing chest pain",
            "squeezing chest pain",
            "pressure in my chest",
            "chest pain spreading to my arm",
            "chest pain spreading to my jaw",
            "chest pain spreading to my neck",
            "chest pain that will not go away",
            "chest pain that won't go away",
        )
    )
    if dangerous_chest_phrases or (chest_pain and cardiac_companions):
        matches.append(
            RuleMatch(
                "dangerous_chest_pain_pattern",
                "emergency",
                "Dangerous chest-pain pattern",
                "Chest discomfort with breathing, sweating, fainting, nausea, radiating pain, or rhythm symptoms may need emergency care.",
                _evidence(
                    phrases(dangerous_chest_phrases),
                    (f"symptom: {item}" for item in sorted(chest_pain | cardiac_companions)),
                ),
                ("heart_attack", "malaysia_emergency"),
            )
        )

    airway_symptoms = present & {
        "throat swelling",
        "throat feels tight",
        "difficulty in swallowing",
        "lip swelling",
    }
    allergy_companions = present & {
        "difficulty breathing",
        "shortness of breath",
        "wheezing",
        "breathing fast",
        "allergic reaction",
        "skin rash",
    }
    sudden_airway = raw(
        (
            "tongue is swelling",
            "tongue swelling",
            "mouth suddenly swollen",
            "lips suddenly swollen",
            "throat suddenly swollen",
        )
    )
    if sudden_airway or (airway_symptoms and allergy_companions):
        matches.append(
            RuleMatch(
                "possible_anaphylaxis",
                "emergency",
                "Possible severe allergic reaction",
                "Airway swelling or tightness with breathing, wheezing, rash, or allergy signs may indicate anaphylaxis.",
                _evidence(
                    phrases(sudden_airway),
                    (f"symptom: {item}" for item in sorted(airway_symptoms | allergy_companions)),
                ),
                ("anaphylaxis", "malaysia_emergency"),
            )
        )

    seizure_emergency = raw(
        (
            "first seizure",
            "seizure lasting more than five minutes",
            "seizure lasting more than 5 minutes",
            "seizure for more than five minutes",
            "seizure for more than 5 minutes",
            "another seizure without recovering",
            "repeated seizures without recovering",
            "having a seizure now",
            "seizing now",
        )
    )
    if seizure_emergency:
        matches.append(
            RuleMatch(
                "seizure_emergency",
                "emergency",
                "Seizure emergency",
                "A first, prolonged, repeated-without-recovery, or current seizure requires emergency help.",
                phrases(seizure_emergency),
                ("seizure", "malaysia_emergency"),
            )
        )

    bleeding_symptoms = present & {
        "vomiting blood",
        "hemoptysis",
        "melena",
        "rectal bleeding",
        "blood in stool",
        "blood in urine",
        "vaginal bleeding after menopause",
        "spotting or bleeding during pregnancy",
    }
    shock_symptoms = present & {
        "fainting",
        "dizziness",
        "weakness",
        "breathing fast",
        "increased heart rate",
    }
    heavy_bleeding = raw(
        (
            "heavy bleeding",
            "bleeding heavily",
            "uncontrolled bleeding",
            "bleeding will not stop",
            "bleeding won't stop",
            "soaking through a pad",
            "soaking a period pad",
            "lost a lot of blood",
        )
    )
    if heavy_bleeding or (bleeding_symptoms and shock_symptoms):
        matches.append(
            RuleMatch(
                "major_bleeding_or_shock",
                "emergency",
                "Possible major bleeding or shock",
                "Heavy or uncontrolled bleeding, or bleeding with faintness, weakness, or abnormal breathing, requires emergency care.",
                _evidence(
                    phrases(heavy_bleeding),
                    (f"symptom: {item}" for item in sorted(bleeding_symptoms | shock_symptoms)),
                ),
                (
                    "pregnancy_bleeding",
                    "coughing_blood",
                    "rectal_bleeding",
                    "malaysia_emergency",
                ),
            )
        )

    pregnancy = bool(raw(("i am pregnant", "i'm pregnant", "during my pregnancy", "weeks pregnant")))
    pregnancy_bleeding = bool(
        present & {"spotting or bleeding during pregnancy", "intermenstrual bleeding"}
        or raw(("vaginal bleeding", "bleeding during pregnancy", "bleeding while pregnant"))
    )
    severe_pregnancy_signs = bool(
        present
        & {
            "sharp abdominal pain",
            "lower abdominal pain",
            "shoulder pain",
            "fainting",
            "dizziness",
        }
        or raw(("severe abdominal pain", "severe tummy pain", "heavy bleeding"))
    )
    if pregnancy and pregnancy_bleeding and severe_pregnancy_signs:
        matches.append(
            RuleMatch(
                "pregnancy_bleeding_emergency",
                "emergency",
                "Pregnancy bleeding with severe warning signs",
                "Bleeding during pregnancy with severe pain, shoulder pain, dizziness, fainting, or heavy bleeding needs emergency assessment.",
                ("pregnancy context", "bleeding context", "severe associated sign"),
                ("pregnancy_bleeding", "malaysia_emergency"),
            )
        )

    meningitis_pattern = (
        "fever" in present
        and "neck stiffness or tightness" in present
        and bool(present & {"headache", "skin rash", "difficulty speaking", "sleepiness"})
    )
    if meningitis_pattern:
        matches.append(
            RuleMatch(
                "fever_neck_stiffness_red_flag",
                "emergency",
                "Fever with neck stiffness and another neurological warning sign",
                "This combination can indicate a time-critical illness and requires emergency assessment.",
                tuple(f"symptom: {item}" for item in sorted(present & {"fever", "neck stiffness or tightness", "headache", "skin rash", "difficulty speaking", "sleepiness"})),
                ("meningitis", "malaysia_emergency"),
            )
        )

    # Same-day urgent rules are intentionally broad enough to avoid silently
    # passing concerning symptoms into an ordinary disease response.
    if chest_pain and not any(match.rule_id == "dangerous_chest_pain_pattern" for match in matches):
        matches.append(
            RuleMatch(
                "chest_pain_needs_assessment",
                "urgent",
                "Chest pain needs prompt assessment",
                "Chest pain without a matched emergency combination should still be assessed promptly.",
                tuple(f"symptom: {item}" for item in sorted(chest_pain)),
                ("heart_attack",),
            )
        )

    breathing_concerns = present & {"difficulty breathing", "shortness of breath", "wheezing"}
    if breathing_concerns and not (severe_breathing or airway_symptoms and allergy_companions):
        matches.append(
            RuleMatch(
                "breathing_problem_needs_assessment",
                "urgent",
                "Breathing problem needs prompt assessment",
                "Breathing difficulty that does not match a severe emergency rule still needs prompt medical review.",
                tuple(f"symptom: {item}" for item in sorted(breathing_concerns)),
                ("heart_attack", "anaphylaxis"),
            )
        )

    if bleeding_symptoms and not (heavy_bleeding or bleeding_symptoms and shock_symptoms):
        matches.append(
            RuleMatch(
                "bleeding_needs_assessment",
                "urgent",
                "Bleeding symptom needs prompt assessment",
                "Blood in vomit, sputum, stool, urine, or pregnancy-related bleeding should not receive an ordinary chatbot response.",
                tuple(f"symptom: {item}" for item in sorted(bleeding_symptoms)),
                (
                    "pregnancy_bleeding",
                    "blood_in_urine",
                    "coughing_blood",
                    "rectal_bleeding",
                ),
            )
        )

    if "seizures" in present and not seizure_emergency:
        matches.append(
            RuleMatch(
                "reported_seizure_needs_assessment",
                "urgent",
                "Reported seizure needs prompt assessment",
                "A reported seizure needs clinician assessment even when emergency duration or recovery details are unavailable.",
                ("symptom: seizures",),
                ("seizure",),
            )
        )

    urgent_neurological = present & {"difficulty speaking", "arm weakness", "leg weakness", "double vision"}
    if urgent_neurological and not any(match.rule_id == "possible_stroke_signs" for match in matches):
        matches.append(
            RuleMatch(
                "neurological_symptom_needs_assessment",
                "urgent",
                "Neurological symptom needs prompt assessment",
                "New speech, limb weakness, or double-vision symptoms need prompt medical review.",
                tuple(f"symptom: {item}" for item in sorted(urgent_neurological)),
                ("stroke",),
            )
        )

    persistent_vomiting = raw(
        (
            "cannot stop vomiting",
            "can't stop vomiting",
            "vomiting continuously",
            "cannot keep fluids down",
            "can't keep fluids down",
        )
    )
    if persistent_vomiting:
        matches.append(
            RuleMatch(
                "persistent_vomiting_needs_assessment",
                "urgent",
                "Persistent vomiting needs prompt assessment",
                "Continuous vomiting or inability to keep fluids down can cause serious dehydration.",
                phrases(persistent_vomiting),
                ("vomiting",),
            )
        )

    if pregnancy and pregnancy_bleeding and not severe_pregnancy_signs:
        matches.append(
            RuleMatch(
                "pregnancy_bleeding_needs_assessment",
                "urgent",
                "Bleeding during pregnancy needs prompt assessment",
                "Pregnancy-related bleeding should be assessed by a maternity or urgent-care service.",
                ("pregnancy context", "bleeding context"),
                ("pregnancy_bleeding",),
            )
        )

    if "retention of urine" in present:
        matches.append(
            RuleMatch(
                "urinary_retention_needs_assessment",
                "urgent",
                "Inability to urinate needs prompt assessment",
                "Current inability to urinate should receive prompt medical assessment.",
                ("symptom: retention of urine",),
                ("urinary_retention",),
            )
        )

    level = max((match.level for match in matches), key=LEVEL_PRIORITY.get, default="routine")
    bypass = level in {"emergency", "urgent"}
    if level == "emergency":
        action = (
            f"Possible emergency warning signs were detected. Call {emergency_number} now. "
            "Do not rely on the chatbot's disease prediction or drive yourself."
        )
    elif level == "urgent":
        action = (
            f"Concerning symptoms were detected. Contact {urgent_service} now for prompt "
            f"assessment. If symptoms become severe or rapidly worsen, call {emergency_number}."
        )
    else:
        action = (
            "No configured red-flag rule matched. This does not prove the symptoms are safe; "
            "continue with symptom confirmation and classification."
        )

    highest_matches = [match for match in matches if match.level == level]
    return {
        "urgency_level": level,
        "should_bypass_classification": bypass,
        "action_message": action,
        "emergency_number": emergency_number,
        "urgent_service": urgent_service,
        "evaluated_present_symptoms": sorted(present),
        "matched_rules": [match.as_dict() for match in matches],
        "highest_priority_rule_ids": [match.rule_id for match in highest_matches],
        "scope_notice": (
            "Rule-based research prototype. A routine result only means that no configured "
            "rule matched; it is not proof that care is unnecessary."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based urgency gate.")
    parser.add_argument("text", help="Patient narrative.")
    parser.add_argument("--emergency-number", default="999")
    parser.add_argument("--urgent-service", default="a same-day medical service")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    extraction = SymptomExtractor().extract(args.text)
    result = detect_urgency(
        args.text,
        extraction,
        emergency_number=args.emergency_number,
        urgent_service=args.urgent_service,
    )
    payload = {"symptom_extraction": extraction, "urgency": result}
    print(json.dumps(payload, indent=2))
    if args.output_json:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
