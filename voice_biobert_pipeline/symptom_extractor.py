#!/usr/bin/env python
"""Deterministic narrative-to-canonical-symptom extraction.

The trained disease classifier expects short comma-separated symptom lists. This
module maps conversational patient wording onto the same symptom vocabulary and
separates present symptoms from negated, uncertain, historical, or other-person
mentions. It does not diagnose disease or infer symptoms that were not stated.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_VOCABULARY_PATH = (
    Path(__file__).resolve().parent / "resources" / "symptom_vocabulary.json"
)

# Canonical vocabulary item -> common patient/ASR expressions. Exact canonical
# phrases are added automatically, so this table only contains useful variants.
ALIASES: dict[str, tuple[str, ...]] = {
    "abdominal distention": ("distended abdomen", "abdominal swelling"),
    "abnormal involuntary movements": (
        "involuntary movement",
        "involuntary movements",
        "uncontrolled movement",
        "uncontrolled movements",
        "shaking uncontrollably",
        "tremor",
        "tremors",
    ),
    "ache all over": ("body ache", "body aches", "aching all over"),
    "allergic reaction": ("allergy reaction",),
    "anxiety and nervousness": ("anxious", "feeling anxious", "feeling nervous"),
    "apnea": ("stopped breathing during sleep", "pauses in breathing during sleep"),
    "blood in stool": ("bloody stool", "blood in my stool", "blood in poop"),
    "blood in urine": (
        "bloody urine",
        "blood in my urine",
        "blood in pee",
        "pink urine",
        "urine looked pink",
        "urine is pink",
        "urine was pink",
    ),
    "breathing fast": ("fast breathing", "breathing rapidly", "rapid breathing"),
    "burning abdominal pain": (
        "burning stomach pain",
        "burning pain in my stomach",
        "burning pain in my abdomen",
    ),
    "burning chest pain": ("burning pain in my chest",),
    "chest tightness": ("tight chest", "tightness in my chest", "chest feels tight"),
    "chills": ("shivering", "having chills", "feeling shivery"),
    "constipation": ("constipated", "difficulty passing stool"),
    "coryza": ("runny nose", "running nose", "nasal discharge"),
    "cough": (
        "coughing",
        "persistent cough",
        "dry cough",
        "keep coughing",
        "cannot stop coughing",
        "can't stop coughing",
        "night cough",
    ),
    "coughing up sputum": (
        "coughing up phlegm",
        "coughing up mucus",
        "phlegmy cough",
        "productive cough",
    ),
    "cramps and spasms": ("muscle spasms", "spasms and cramps"),
    "decreased appetite": ("loss of appetite", "poor appetite", "no appetite"),
    "decreased heart rate": ("slow heart rate", "slow heartbeat"),
    "depression": ("feeling depressed", "depressed mood"),
    "diarrhea": ("diarrhoea", "loose stool", "loose stools", "watery stool", "watery stools"),
    "difficulty breathing": (
        "trouble breathing",
        "hard to breathe",
        "difficulty in breathing",
        "cannot breathe",
        "can't breathe",
    ),
    "difficulty in swallowing": (
        "trouble swallowing",
        "hard to swallow",
        "difficulty swallowing",
        "cannot swallow",
        "can't swallow",
    ),
    "difficulty speaking": (
        "trouble speaking",
        "difficulty talking",
        "cannot speak",
        "can't speak",
    ),
    "diminished hearing": (
        "hearing loss",
        "reduced hearing",
        "cannot hear",
        "can't hear",
    ),
    "diminished vision": (
        "blurred vision",
        "blurry vision",
        "reduced vision",
        "poor vision",
    ),
    "disturbance of memory": ("memory problem", "memory problems", "memory loss", "forgetful"),
    "dizziness": ("dizzy", "lightheaded", "light headed", "feeling lightheaded"),
    "double vision": ("seeing double",),
    "ear pain": ("earache", "ear ache", "pain in my ear", "pain in the ear"),
    "excessive urination at night": (
        "peeing a lot at night",
        "urinating often at night",
        "waking up to urinate",
    ),
    "eye redness": ("red eye", "red eyes", "bloodshot eye", "bloodshot eyes"),
    "fainting": ("fainted", "passed out", "passing out", "blacked out"),
    "fatigue": ("fatigued", "very tired", "extremely tired", "lack of energy"),
    "fever": ("feverish", "high temperature", "high body temperature", "having a fever"),
    "flatulence": ("excessive gas", "passing gas"),
    "focal weakness": ("one sided weakness", "weakness on one side", "weak on one side"),
    "foot or toe pain": ("foot pain", "toe pain", "pain in my foot", "pain in my toe"),
    "frequent urination": (
        "peeing often",
        "urinating often",
        "need to pee often",
        "need to urinate often",
        "pee every hour",
        "urinate every hour",
        "need to pee every hour",
        "need to urinate every hour",
        "going to the toilet often",
    ),
    "frontal headache": ("forehead headache", "pain in my forehead"),
    "hand or finger pain": ("hand pain", "finger pain", "pain in my hand", "pain in my finger"),
    "headache": ("head ache", "headaches", "head hurts", "head pain", "pain in my head"),
    "heartburn": ("acid reflux", "burning indigestion"),
    "hemoptysis": ("coughing blood", "coughing up blood", "blood when coughing"),
    "hoarse voice": ("hoarseness", "voice is hoarse", "lost my voice"),
    "hurts to breath": ("hurts to breathe", "pain when breathing", "painful breathing"),
    "increased heart rate": ("fast heart rate", "fast heartbeat", "racing heart", "heart racing"),
    "insomnia": (
        "cannot sleep",
        "can't sleep",
        "difficulty sleeping",
        "trouble sleeping",
        "not sleeping",
    ),
    "irregular heartbeat": ("irregular heart beat", "uneven heartbeat"),
    "itchiness of eye": ("itchy eye", "itchy eyes", "eyes are itchy"),
    "itching of skin": ("itchy skin", "skin is itchy", "skin itching"),
    "joint pain": ("joint ache", "aching joints", "joints hurt"),
    "lacrimation": ("watery eye", "watery eyes", "eyes watering", "tearing eyes"),
    "loss of sensation": ("numb", "numbness", "cannot feel", "can't feel"),
    "lower abdominal pain": (
        "lower belly pain",
        "lower stomach pain",
        "pain in my lower abdomen",
    ),
    "melena": ("black stool", "black stools", "tarry stool", "tarry stools"),
    "mouth dryness": ("dry mouth", "mouth feels dry"),
    "mouth ulcer": ("mouth ulcers", "canker sore", "canker sores"),
    "muscle pain": ("muscle ache", "muscle aches", "aching muscles", "muscles hurt"),
    "nasal congestion": (
        "blocked nose",
        "stuffy nose",
        "congested nose",
        "nose is blocked",
        "nose feels blocked",
    ),
    "nausea": ("nauseous", "queasy", "feel nauseous", "feeling nauseous", "sick to my stomach"),
    "neck stiffness or tightness": ("stiff neck", "neck feels stiff"),
    "nosebleed": ("nose bleed", "bloody nose"),
    "pain during intercourse": ("painful intercourse", "pain during sex"),
    "pain in eye": ("eye pain", "eyes hurt", "pain in my eye"),
    "pain in testicles": ("testicle pain", "testicular pain", "pain in my testicles"),
    "painful urination": (
        "burning urination",
        "burns when i pee",
        "burning when i pee",
        "pain when i pee",
        "pain when urinating",
        "burns when urinating",
    ),
    "palpitations": ("heart palpitations", "pounding heart", "heart pounding", "skipped beats"),
    "paresthesia": ("tingling", "pins and needles", "prickling sensation"),
    "plugged feeling in ear": ("blocked ear", "plugged ear", "ear feels blocked"),
    "rectal bleeding": ("bleeding from rectum", "rectum is bleeding"),
    "retention of urine": (
        "cannot urinate",
        "can't urinate",
        "cannot pee",
        "can't pee",
        "unable to urinate",
    ),
    "ringing in ear": ("ringing in my ear", "ringing in ears", "tinnitus"),
    "seizures": ("seizure", "convulsion", "convulsions", "having fits"),
    "sharp abdominal pain": (
        "abdominal pain",
        "stomach pain",
        "belly pain",
        "sharp stomach pain",
        "sharp belly pain",
        "sharp pain in my abdomen",
    ),
    "sharp chest pain": ("chest pain", "sharp pain in my chest"),
    "shortness of breath": (
        "short of breath",
        "breathless",
        "out of breath",
        "cannot catch my breath",
        "can't catch my breath",
    ),
    "sinus congestion": ("blocked sinuses", "congested sinuses"),
    "skin rash": ("rash", "rashes", "skin rashes", "hives"),
    "sleepiness": ("sleepy", "drowsy", "drowsiness"),
    "slurring words": ("slurred speech", "words are slurred"),
    "sore throat": (
        "throat pain",
        "throat hurts",
        "scratchy throat",
        "throat feels scratchy",
        "irritated throat",
    ),
    "spots or clouds in vision": (
        "flashing lights",
        "flashes of light",
        "zig zag lines",
        "zigzag lines",
        "blind spots",
        "spots in my vision",
        "clouds in my vision",
        "visual aura",
    ),
    "stomach bloating": ("bloated stomach", "stomach feels bloated", "feeling bloated"),
    "sweating": ("sweaty", "sweating a lot", "excessive sweating"),
    "swollen lymph nodes": ("swollen glands", "enlarged lymph nodes"),
    "throat feels tight": ("tight throat", "throat is tight"),
    "unusual color or odor to urine": (
        "dark urine",
        "cloudy urine",
        "smelly urine",
        "strong smelling urine",
    ),
    "upper abdominal pain": ("upper belly pain", "upper stomach pain", "pain in my upper abdomen"),
    "vaginal discharge": ("unusual vaginal discharge",),
    "vomiting": (
        "vomit",
        "vomited",
        "throw up",
        "throwing up",
        "threw up",
        "puking",
    ),
    "vomiting blood": ("vomiting blood", "vomited blood", "blood in my vomit"),
    "weakness": ("feeling weak", "generally weak"),
    "wheezing": ("wheeze", "wheezes", "wheezy", "wheezing sound"),
}

# These are positive symptom idioms even though they contain a normal negation
# cue. They must not be interpreted as a denial of the mapped symptom.
POSITIVE_NEGATION_ALIASES = {
    "no appetite",
    "cannot breathe",
    "can't breathe",
    "cannot swallow",
    "can't swallow",
    "cannot speak",
    "can't speak",
    "cannot hear",
    "can't hear",
    "cannot sleep",
    "can't sleep",
    "not sleeping",
    "cannot feel",
    "can't feel",
    "cannot urinate",
    "can't urinate",
    "cannot pee",
    "can't pee",
    "cannot catch my breath",
    "can't catch my breath",
}

UNUSABLE_CANONICAL_MATCHES = {"or roughness", "or spasms", "regurgitation.1"}


@dataclass(frozen=True)
class PatternEntry:
    canonical: str
    alias: str
    pattern: re.Pattern[str]
    source: str
    positive_idiom: bool


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _literal_pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in phrase.split()]
    separator = r"\s+(?:(?:also|still|really|very)\s+)?"
    body = separator.join(words)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


class SymptomExtractor:
    """Extract asserted symptoms and format them for the trained BioBERT model."""

    def __init__(self, vocabulary_path: Path = DEFAULT_VOCABULARY_PATH) -> None:
        vocabulary_path = vocabulary_path.expanduser().resolve()
        payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        self.vocabulary_path = vocabulary_path
        self.vocabulary = tuple(item["canonical"] for item in payload["symptoms"])
        vocabulary_set = set(self.vocabulary)

        invalid = sorted(set(ALIASES) - vocabulary_set)
        if invalid:
            raise ValueError(f"Aliases refer to unknown canonical symptoms: {invalid}")

        alias_owner: dict[str, str] = {}
        entries: list[PatternEntry] = []
        for canonical in self.vocabulary:
            if canonical in UNUSABLE_CANONICAL_MATCHES:
                continue
            phrases = [(canonical, "canonical")]
            phrases.extend((alias, "alias") for alias in ALIASES.get(canonical, ()))
            for phrase, source in phrases:
                alias = normalize_text(phrase)
                previous = alias_owner.get(alias)
                if previous is not None and previous != canonical:
                    raise ValueError(
                        f"Alias {alias!r} maps to both {previous!r} and {canonical!r}."
                    )
                alias_owner[alias] = canonical
                entries.append(
                    PatternEntry(
                        canonical=canonical,
                        alias=alias,
                        pattern=_literal_pattern(alias),
                        source=source,
                        positive_idiom=alias in POSITIVE_NEGATION_ALIASES,
                    )
                )

        # Longer aliases win when, for example, "frontal headache" and
        # "headache" begin inside the same phrase.
        self.patterns = tuple(
            sorted(entries, key=lambda item: (-len(item.alias), item.canonical))
        )

    @staticmethod
    def _assertion_status(
        text: str,
        start: int,
        end: int,
        positive_idiom: bool,
    ) -> tuple[str, str]:
        before = text[:start]
        after = text[end:]

        # Sentence punctuation and contrast terms reset the scope of assertion
        # cues. Commas do not reset it so "no fever, cough, or nausea" works.
        boundaries = [match.end() for match in re.finditer(r"[.!?;]", before)]
        boundaries.extend(
            match.end()
            for match in re.finditer(
                r"\b(?:but|however|although|though|yet|except)\b", before
            )
        )
        boundary = max(boundaries, default=0)
        context = before[boundary:]
        context_words = re.findall(r"[a-z']+", context)[-14:]
        context_tail = " ".join(context_words)

        if re.search(
            r"\b(?:my|the)\s+(?:mother|father|mom|mum|dad|sister|brother|wife|"
            r"husband|partner|friend|child|son|daughter|patient)\b",
            context_tail,
        ) or re.search(r"\bfamily history of\b", context_tail):
            return "other_subject", "The symptom refers to another person or family history."

        if re.search(
            r"\b(?:used to|previously|in the past|history of|last year|years ago|"
            r"as a child|once had)\b",
            context_tail,
        ):
            return "historical", "A historical-time cue occurs before the symptom."

        post_clause = re.split(r"[,.!?;]", after, maxsplit=1)[0]
        post_words = " ".join(re.findall(r"[a-z']+", post_clause)[:6])
        if re.search(r"\b(?:is|has|have|had)?\s*(?:gone|resolved|stopped)\b", post_words):
            return "historical", "The symptom is described as resolved."

        if not positive_idiom and re.search(
            r"\b(?:no|not|never|without|denies|deny|denied|cannot|can't|dont|don't|"
            r"doesnt|doesn't|isnt|isn't|wasnt|wasn't|havent|haven't|hasnt|hasn't|"
            r"didnt|didn't)\b",
            context_tail,
        ):
            return "negated", "A negation cue occurs before the symptom."

        if re.search(r"\b(?:no longer|not anymore)\b", post_words):
            return "negated", "A negation cue occurs after the symptom."

        if re.search(
            r"\b(?:maybe|perhaps|possibly|might|may|could|unsure|uncertain)\b",
            context_tail,
        ):
            return "uncertain", "An uncertainty cue occurs before the symptom."

        return "present", "The symptom is asserted as currently present."

    def extract(self, text: str, minimum_present_symptoms: int = 2) -> dict[str, Any]:
        if minimum_present_symptoms < 1:
            raise ValueError("minimum_present_symptoms must be at least 1.")

        normalized = normalize_text(text)
        candidates: list[tuple[int, int, PatternEntry, str]] = []
        for entry in self.patterns:
            for match in entry.pattern.finditer(normalized):
                candidates.append((match.start(), match.end(), entry, match.group(0)))

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2].canonical))
        accepted: list[tuple[int, int, PatternEntry, str]] = []
        occupied: list[tuple[int, int]] = []
        for candidate in candidates:
            start, end = candidate[0], candidate[1]
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            accepted.append(candidate)
            occupied.append((start, end))
        accepted.sort(key=lambda item: item[0])

        evidence: list[dict[str, Any]] = []
        latest_by_symptom: dict[str, dict[str, Any]] = {}
        first_start_by_symptom: dict[str, int] = {}
        for start, end, entry, matched_text in accepted:
            status, reason = self._assertion_status(
                normalized,
                start,
                end,
                positive_idiom=entry.positive_idiom,
            )
            item = {
                "canonical_symptom": entry.canonical,
                "matched_text": matched_text,
                "status": status,
                "reason": reason,
                "match_source": entry.source,
                "start": start,
                "end": end,
            }
            evidence.append(item)
            first_start_by_symptom.setdefault(entry.canonical, start)
            latest_by_symptom[entry.canonical] = item

        status_order = ("present", "negated", "uncertain", "historical", "other_subject")
        grouped: dict[str, list[str]] = {status: [] for status in status_order}
        for item in sorted(
            latest_by_symptom.values(),
            key=lambda value: first_start_by_symptom[value["canonical_symptom"]],
        ):
            grouped[item["status"]].append(item["canonical_symptom"])

        present = grouped["present"]
        return {
            "original_text": str(text),
            "normalized_text": normalized,
            "present_symptoms": present,
            "negated_symptoms": grouped["negated"],
            "uncertain_symptoms": grouped["uncertain"],
            "historical_symptoms": grouped["historical"],
            "other_subject_symptoms": grouped["other_subject"],
            "biobert_text": ", ".join(present),
            "present_symptom_count": len(present),
            "minimum_present_symptoms": minimum_present_symptoms,
            "needs_clarification": len(present) < minimum_present_symptoms,
            "evidence": evidence,
            "vocabulary_size": len(self.vocabulary),
            "vocabulary_path": str(self.vocabulary_path),
        }


def assertion_status(
    text: str,
    start: int,
    end: int,
    positive_idiom: bool = False,
) -> tuple[str, str]:
    """Public assertion helper shared by downstream safety-rule matching."""
    return SymptomExtractor._assertion_status(text, start, end, positive_idiom)


def extract_symptoms(
    text: str,
    vocabulary_path: Path = DEFAULT_VOCABULARY_PATH,
    minimum_present_symptoms: int = 2,
) -> dict[str, Any]:
    return SymptomExtractor(vocabulary_path).extract(text, minimum_present_symptoms)


def _format_lines(items: Iterable[str]) -> str:
    values = list(items)
    return ", ".join(values) if values else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a patient narrative to BioBERT symptoms.")
    parser.add_argument("text", help="Patient symptom narrative.")
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY_PATH)
    parser.add_argument("--minimum-symptoms", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Print the complete JSON result.")
    args = parser.parse_args()

    result = extract_symptoms(args.text, args.vocabulary, args.minimum_symptoms)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Present: {_format_lines(result['present_symptoms'])}")
    print(f"Negated: {_format_lines(result['negated_symptoms'])}")
    print(f"Uncertain: {_format_lines(result['uncertain_symptoms'])}")
    print(f"Historical: {_format_lines(result['historical_symptoms'])}")
    print(f"Other subject: {_format_lines(result['other_subject_symptoms'])}")
    print(f"BioBERT input: {result['biobert_text'] or '(none)'}")
    print(f"Needs clarification: {result['needs_clarification']}")


if __name__ == "__main__":
    main()
