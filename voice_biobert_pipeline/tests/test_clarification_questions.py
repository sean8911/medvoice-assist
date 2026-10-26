from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from clarification_questions import generate_clarification  # noqa: E402
from symptom_extractor import SymptomExtractor  # noqa: E402
from urgency_detector import detect_urgency  # noqa: E402


class ClarificationQuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = SymptomExtractor()

    def generate(
        self,
        text: str,
        classification: dict | None = None,
        max_questions: int = 3,
    ) -> dict:
        extraction = self.extractor.extract(text)
        urgency = detect_urgency(text, extraction)
        return generate_clarification(
            text,
            extraction,
            urgency=urgency,
            classification=classification,
            max_questions=max_questions,
        )

    def test_no_symptom_requests_description_and_safety_check(self) -> None:
        result = self.generate("I do not feel well today.")
        question_ids = {item["question_id"] for item in result["questions"]}

        self.assertTrue(result["clarification_required"])
        self.assertTrue(result["should_pause_classification"])
        self.assertIn("no_current_symptoms", result["reason_codes"])
        self.assertIn("describe_main_symptom", question_ids)
        self.assertIn("general_emergency_check", question_ids)

    def test_one_headache_generates_targeted_questions(self) -> None:
        result = self.generate("I have a headache.")
        question_ids = {item["question_id"] for item in result["questions"]}

        self.assertTrue(result["should_pause_classification"])
        self.assertIn("insufficient_current_symptoms", result["reason_codes"])
        self.assertIn("headache_red_flags", question_ids)
        self.assertIn("headache_characteristics", question_ids)
        self.assertNotIn("onset_and_duration", question_ids)
        self.assertNotIn("severity", question_ids)
        self.assertLessEqual(result["question_count"], 3)

    def test_uncertain_symptom_pauses_even_with_two_present_symptoms(self) -> None:
        result = self.generate(
            "I have a headache and vomiting, and I might have nausea too."
        )
        question_ids = {item["question_id"] for item in result["questions"]}

        self.assertTrue(result["should_pause_classification"])
        self.assertIn("uncertain_symptoms", result["reason_codes"])
        self.assertIn("confirm_uncertain_symptoms", question_ids)

    def test_emergency_action_defers_questions(self) -> None:
        result = self.generate(
            "I have crushing chest pain, shortness of breath and sweating."
        )

        self.assertFalse(result["clarification_required"])
        self.assertTrue(result["should_pause_classification"])
        self.assertTrue(result["should_withhold_prediction"])
        self.assertTrue(result["deferred_due_to_urgency"])
        self.assertEqual(result["questions"], [])

    def test_low_score_withholds_prediction_and_asks_for_confirmation(self) -> None:
        classification = {
            "predicted_disease": "Migraine",
            "model_score": 0.21,
            "status": "low_score_review_symptoms",
        }
        result = self.generate(
            "I have a headache and nausea.",
            classification=classification,
        )
        question_ids = {item["question_id"] for item in result["questions"]}

        self.assertTrue(result["clarification_required"])
        self.assertFalse(result["should_pause_classification"])
        self.assertTrue(result["should_withhold_prediction"])
        self.assertIn("low_model_score", result["reason_codes"])
        self.assertIn("confirm_extracted_symptoms", question_ids)

    def test_high_score_with_sufficient_symptoms_needs_no_questions(self) -> None:
        classification = {
            "predicted_disease": "Migraine",
            "model_score": 0.91,
            "status": "prediction_available",
        }
        result = self.generate(
            "I have a headache, nausea and vomiting since this morning.",
            classification=classification,
        )

        self.assertFalse(result["clarification_required"])
        self.assertFalse(result["should_withhold_prediction"])
        self.assertEqual(result["questions"], [])

    def test_close_competing_classes_trigger_clarification(self) -> None:
        classification = {
            "predicted_disease": "Class A",
            "model_score": 0.53,
            "top_two_score_margin": 0.08,
            "status": "ambiguous_prediction_review_symptoms",
        }
        result = self.generate(
            "I have a headache, nausea and vomiting.",
            classification=classification,
        )

        self.assertTrue(result["clarification_required"])
        self.assertTrue(result["should_withhold_prediction"])
        self.assertIn("ambiguous_model_prediction", result["reason_codes"])

    def test_maximum_question_count_is_respected(self) -> None:
        result = self.generate("I have a headache.", max_questions=1)

        self.assertEqual(result["question_count"], 1)
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["question_id"], "natural_followup")
        self.assertIn("Reply naturally", result["questions"][0]["question"])

    def test_natural_followup_recognizes_plain_severity_expression(self) -> None:
        result = self.generate(
            "I have a dry cough that is about 4 out of 10.",
            max_questions=1,
        )

        self.assertTrue(result["context"]["severity_detected"])
        self.assertEqual(result["question_count"], 1)
        self.assertIn("no other symptoms", result["questions"][0]["question"])

    def test_invalid_question_limit_is_rejected(self) -> None:
        extraction = self.extractor.extract("I have a headache.")
        with self.assertRaises(ValueError):
            generate_clarification("I have a headache.", extraction, max_questions=0)


if __name__ == "__main__":
    unittest.main()
