from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from symptom_extractor import SymptomExtractor  # noqa: E402
from urgency_detector import detect_urgency  # noqa: E402


class UrgencyDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = SymptomExtractor()

    def detect(self, text: str) -> dict:
        extraction = self.extractor.extract(text)
        return detect_urgency(text, extraction)

    def test_dangerous_chest_pain_combination_is_emergency(self) -> None:
        result = self.detect(
            "I have crushing chest pain, I am short of breath and sweating."
        )
        self.assertEqual(result["urgency_level"], "emergency")
        self.assertTrue(result["should_bypass_classification"])
        self.assertIn(
            "dangerous_chest_pain_pattern", result["highest_priority_rule_ids"]
        )

    def test_negated_chest_and_breathing_symptoms_do_not_trigger(self) -> None:
        result = self.detect(
            "I have no chest pain or shortness of breath. I only have a headache."
        )
        self.assertEqual(result["urgency_level"], "routine")
        self.assertFalse(result["should_bypass_classification"])

    def test_stroke_phrase_is_emergency(self) -> None:
        result = self.detect("My speech suddenly became slurred and one arm feels weak.")
        self.assertEqual(result["urgency_level"], "emergency")
        self.assertIn("possible_stroke_signs", result["highest_priority_rule_ids"])

    def test_possible_anaphylaxis_is_emergency(self) -> None:
        result = self.detect(
            "My throat feels tight, my lips are swollen, and I am wheezing."
        )
        self.assertEqual(result["urgency_level"], "emergency")
        self.assertIn("possible_anaphylaxis", result["highest_priority_rule_ids"])

    def test_prolonged_seizure_is_emergency(self) -> None:
        result = self.detect("This is my first seizure and it lasted more than 5 minutes.")
        self.assertEqual(result["urgency_level"], "emergency")
        self.assertIn("seizure_emergency", result["highest_priority_rule_ids"])

    def test_blood_in_urine_is_urgent(self) -> None:
        result = self.detect("It burns when I pee and there is blood in my urine.")
        self.assertEqual(result["urgency_level"], "urgent")
        self.assertIn("bleeding_needs_assessment", result["highest_priority_rule_ids"])

    def test_immediate_self_harm_statement_is_emergency(self) -> None:
        result = self.detect("I want to kill myself tonight.")
        self.assertEqual(result["urgency_level"], "emergency")
        self.assertIn(
            "immediate_self_harm_or_violence_risk",
            result["highest_priority_rule_ids"],
        )

    def test_negated_self_harm_statement_does_not_trigger(self) -> None:
        result = self.detect(
            "I do not want to kill myself. I feel anxious and cannot sleep."
        )
        self.assertEqual(result["urgency_level"], "routine")

    def test_pregnancy_bleeding_with_severe_pain_is_emergency(self) -> None:
        result = self.detect(
            "I am 10 weeks pregnant and have vaginal bleeding with severe abdominal pain."
        )
        self.assertEqual(result["urgency_level"], "emergency")
        self.assertIn(
            "pregnancy_bleeding_emergency", result["highest_priority_rule_ids"]
        )

    def test_common_non_red_flag_symptoms_continue(self) -> None:
        result = self.detect("I have a runny nose, sore throat and mild cough.")
        self.assertEqual(result["urgency_level"], "routine")
        self.assertFalse(result["should_bypass_classification"])


if __name__ == "__main__":
    unittest.main()
