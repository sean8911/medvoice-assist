from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from symptom_extractor import SymptomExtractor  # noqa: E402


class SymptomExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = SymptomExtractor()

    def extract(self, text: str) -> dict:
        return self.extractor.extract(text)

    def test_migraine_narrative_is_converted_to_training_style(self) -> None:
        result = self.extract(
            "I've had a severe headache since this morning. I feel nauseous and "
            "have vomited twice. Before the headache I saw flashing lights and "
            "zig-zag lines."
        )
        self.assertEqual(
            result["present_symptoms"],
            ["headache", "nausea", "vomiting", "spots or clouds in vision"],
        )
        self.assertEqual(
            result["biobert_text"],
            "headache, nausea, vomiting, spots or clouds in vision",
        )
        self.assertFalse(result["needs_clarification"])

    def test_uti_narrative_and_negated_symptoms(self) -> None:
        result = self.extract(
            "It burns when I pee and I need to urinate every hour. My urine looked "
            "pink, but I don't have a fever or cough."
        )
        self.assertEqual(
            result["present_symptoms"],
            ["painful urination", "frequent urination", "blood in urine"],
        )
        self.assertEqual(result["negated_symptoms"], ["fever", "cough"])

    def test_asthma_narrative(self) -> None:
        result = self.extract(
            "I am wheezing and feel short of breath, with tightness in my chest. "
            "No fever or cough."
        )
        self.assertEqual(
            result["present_symptoms"],
            ["wheezing", "shortness of breath", "chest tightness"],
        )
        self.assertEqual(result["negated_symptoms"], ["fever", "cough"])

    def test_free_form_respiratory_wording_maps_to_canonical_symptoms(self) -> None:
        result = self.extract(
            "I keep coughing at night, my throat feels scratchy, and my nose also feels blocked."
        )

        self.assertEqual(
            result["present_symptoms"],
            ["cough", "sore throat", "nasal congestion"],
        )
        self.assertEqual(
            result["biobert_text"],
            "cough, sore throat, nasal congestion",
        )

    def test_uncertain_historical_and_other_subject_are_excluded(self) -> None:
        result = self.extract(
            "My father has headaches. I used to have a cough, but now I have a "
            "fever. I might have nausea too."
        )
        self.assertEqual(result["present_symptoms"], ["fever"])
        self.assertEqual(result["other_subject_symptoms"], ["headache"])
        self.assertEqual(result["historical_symptoms"], ["cough"])
        self.assertEqual(result["uncertain_symptoms"], ["nausea"])
        self.assertTrue(result["needs_clarification"])

    def test_positive_idioms_are_not_treated_as_negation(self) -> None:
        result = self.extract("I can't sleep and I have no appetite.")
        self.assertEqual(result["present_symptoms"], ["insomnia", "decreased appetite"])
        self.assertEqual(result["negated_symptoms"], [])

    def test_latest_mention_controls_current_status(self) -> None:
        current = self.extract("I had no fever earlier, but now I have a fever and chills.")
        resolved = self.extract("I had fever earlier, but now I have no fever or chills.")
        self.assertEqual(current["present_symptoms"], ["fever", "chills"])
        self.assertEqual(resolved["present_symptoms"], [])
        self.assertEqual(resolved["negated_symptoms"], ["fever", "chills"])

    def test_direct_training_style_input_is_preserved(self) -> None:
        result = self.extract(
            "sharp abdominal pain, nausea, diarrhea, burning abdominal pain, "
            "fever, upper abdominal pain and chills"
        )
        self.assertEqual(
            result["present_symptoms"],
            [
                "sharp abdominal pain",
                "nausea",
                "diarrhea",
                "burning abdominal pain",
                "fever",
                "upper abdominal pain",
                "chills",
            ],
        )


if __name__ == "__main__":
    unittest.main()
