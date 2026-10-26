from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from clarification_questions import generate_clarification  # noqa: E402
from rule_based_response import GUIDANCE_RULES, generate_rule_based_response  # noqa: E402
from symptom_extractor import SymptomExtractor  # noqa: E402
from urgency_detector import detect_urgency  # noqa: E402


class RuleBasedResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = SymptomExtractor()

    def pipeline_state(
        self,
        text: str,
        classification: dict | None = None,
    ) -> tuple[dict, dict, dict, dict]:
        extraction = self.extractor.extract(text)
        urgency = detect_urgency(text, extraction)
        clarification = generate_clarification(
            text,
            extraction,
            urgency=urgency,
            classification=classification,
        )
        response = generate_rule_based_response(
            text,
            extraction,
            urgency,
            clarification,
            classification,
        )
        return extraction, urgency, clarification, response

    def test_emergency_response_uses_safety_action_and_hides_prediction(self) -> None:
        _, urgency, _, response = self.pipeline_state(
            "I have crushing chest pain, shortness of breath and sweating."
        )

        self.assertEqual(urgency["urgency_level"], "emergency")
        self.assertEqual(response["response_type"], "emergency")
        self.assertIsNone(response["displayed_prediction"])
        self.assertIn("999", response["response_text"])
        self.assertEqual(response["self_care_guidance"], [])

    def test_urgent_response_bypasses_routine_self_care(self) -> None:
        _, urgency, _, response = self.pipeline_state(
            "It burns when I urinate and there is blood in my urine."
        )

        self.assertEqual(urgency["urgency_level"], "urgent")
        self.assertEqual(response["response_type"], "urgent")
        self.assertIsNone(response["displayed_prediction"])
        self.assertEqual(response["self_care_guidance"], [])

    def test_clarification_response_includes_questions(self) -> None:
        _, _, clarification, response = self.pipeline_state("I have a headache.")

        self.assertTrue(clarification["clarification_required"])
        self.assertEqual(response["response_type"], "clarification")
        self.assertEqual(
            response["clarification_questions"], clarification["questions"]
        )
        self.assertIsNone(response["displayed_prediction"])

    def test_accepted_prediction_uses_matching_symptom_guidance(self) -> None:
        classification = {
            "predicted_disease": "migraine",
            "model_score": 0.91,
            "display_prediction": True,
            "status": "prediction_available",
        }
        _, _, clarification, response = self.pipeline_state(
            "I have a headache, nausea and vomiting since this morning.",
            classification,
        )

        self.assertFalse(clarification["clarification_required"])
        self.assertEqual(response["response_type"], "routine_prediction")
        self.assertEqual(
            response["displayed_prediction"]["disease_class"], "migraine"
        )
        self.assertIn("headache_guidance", response["matched_guidance_rule_ids"])
        self.assertIn(
            "gastrointestinal_guidance", response["matched_guidance_rule_ids"]
        )
        self.assertTrue(response["self_care_guidance"])
        self.assertTrue(all(source["url"].startswith("https://") for source in response["sources"]))
        self.assertNotIn("configured emergency service", response["response_text"])
        self.assertIn("Call 999", response["response_text"])

    def test_withheld_class_never_appears_in_user_facing_response(self) -> None:
        hidden_class = "do-not-display-this-class"
        classification = {
            "predicted_disease": hidden_class,
            "model_score": 0.12,
            "display_prediction": False,
            "status": "low_score_review_symptoms",
        }
        _, _, _, response = self.pipeline_state(
            "I have a headache and nausea.",
            classification,
        )

        self.assertEqual(response["response_type"], "clarification")
        self.assertIsNone(response["displayed_prediction"])
        self.assertNotIn(hidden_class, response["response_text"])

    def test_guidance_rules_do_not_prescribe_medication(self) -> None:
        guidance_text = " ".join(
            item
            for rule in GUIDANCE_RULES
            for item in (*rule.self_care, *rule.safety_net)
        ).lower()
        prohibited = ("paracetamol", "ibuprofen", "antibiotic", "dosage", "dose")

        for term in prohibited:
            self.assertNotIn(term, guidance_text)

    def test_every_response_requires_clinical_review(self) -> None:
        _, _, _, response = self.pipeline_state("I have a headache.")
        self.assertTrue(response["requires_clinical_review"])


if __name__ == "__main__":
    unittest.main()
