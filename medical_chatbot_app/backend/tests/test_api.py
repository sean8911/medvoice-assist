from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from medical_chatbot_app.backend.app import main
from medical_chatbot_app.backend.app.services.pipeline_service import PipelineService


class FakeRuntime:
    def __init__(self) -> None:
        self.device = "cpu"
        self.model_dir = Path("fake-trained-biobert").resolve()
        self.whisper_model_name = "openai/whisper-small"
        self.biobert_loaded = True
        self.whisper_loaded = False
        self.classification_calls = 0

    def classify(self, text: str, **_: object) -> dict:
        self.classification_calls += 1
        return {
            "predicted_disease": "migraine",
            "predicted_label_id": 401,
            "model_score": 0.91,
            "status": "prediction_available",
            "minimum_score_threshold": 0.5,
            "top_two_score_margin": 0.72,
            "minimum_margin_threshold": 0.1,
            "top_predictions": [
                {
                    "rank": 1,
                    "label_id": 401,
                    "disease": "migraine",
                    "model_score": 0.91,
                }
            ],
            "number_of_disease_classes": 773,
            "display_prediction": True,
        }


class MedicalChatbotApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_service = main.service
        self.runtime = FakeRuntime()
        main.service = PipelineService(runtime=self.runtime)
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        main.service = self.original_service

    def test_single_symptom_returns_clarification_without_loading_classifier(self) -> None:
        response = self.client.post(
            "/api/analyze/text",
            json={"text": "I have a headache."},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["rule_based_response"]["response_type"], "clarification")
        self.assertFalse(payload["classification"]["display_prediction"])
        self.assertEqual(self.runtime.classification_calls, 0)
        self.assertEqual(payload["clarification"]["question_count"], 1)
        self.assertEqual(
            payload["clarification"]["questions"][0]["question_id"],
            "natural_followup",
        )

    def test_free_form_followup_can_continue_to_classification(self) -> None:
        first_response = self.client.post(
            "/api/analyze/text",
            json={"text": "I keep coughing at night."},
        )

        self.assertEqual(first_response.status_code, 200)
        first_payload = first_response.json()
        self.assertEqual(first_payload["clarification"]["question_count"], 1)
        self.assertIn("Reply naturally", first_payload["clarification"]["questions"][0]["question"])

        followup_response = self.client.post(
            "/api/analyze/clarification",
            json={
                "original_narrative": "I keep coughing at night.",
                "answer": (
                    "My throat feels scratchy and my nose also feels blocked. It began "
                    "gradually last week, but I have no breathing difficulty or chest pain."
                ),
            },
        )

        self.assertEqual(followup_response.status_code, 200)
        followup_payload = followup_response.json()
        self.assertIn("cough", followup_payload["symptom_extraction"]["present_symptoms"])
        self.assertIn(
            "sore throat",
            followup_payload["symptom_extraction"]["present_symptoms"],
        )
        self.assertIn(
            "nasal congestion",
            followup_payload["symptom_extraction"]["present_symptoms"],
        )
        self.assertEqual(self.runtime.classification_calls, 1)
        self.assertEqual(
            followup_payload["rule_based_response"]["response_type"],
            "routine_prediction",
        )

    def test_emergency_bypasses_classifier(self) -> None:
        response = self.client.post(
            "/api/analyze/text",
            json={
                "text": "I have crushing chest pain, shortness of breath and sweating."
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["urgency"]["urgency_level"], "emergency")
        self.assertEqual(payload["rule_based_response"]["response_type"], "emergency")
        self.assertEqual(self.runtime.classification_calls, 0)

    def test_sufficient_symptoms_use_trained_model_contract(self) -> None:
        response = self.client.post(
            "/api/analyze/text",
            json={
                "text": (
                    "I have a headache, nausea, vomiting and spots in my vision "
                    "since this morning."
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["classification"]["predicted_disease"], "migraine")
        self.assertTrue(payload["classification"]["display_prediction"])
        self.assertEqual(payload["rule_based_response"]["response_type"], "routine_prediction")
        self.assertEqual(self.runtime.classification_calls, 1)

    def test_clarification_is_limited_to_one_round(self) -> None:
        response = self.client.post(
            "/api/analyze/clarification",
            json={
                "original_narrative": "I have a headache.",
                "answer": "It is behind my eyes and seven out of ten.",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["classification"]["status"], "clarification_limit_reached")
        self.assertEqual(payload["rule_based_response"]["response_type"], "no_prediction")
        self.assertEqual(payload["clarification"]["questions"], [])

    def test_single_cough_after_clarification_explains_why_model_was_not_run(self) -> None:
        response = self.client.post(
            "/api/analyze/clarification",
            json={
                "original_narrative": "I have a dry cough that is worse at night.",
                "answer": (
                    "I have had a dry cough for one week. It is worse at night and "
                    "is 4 out of 10. I do not have chest pain, weakness, numbness, "
                    "facial drooping, or slurred speech."
                ),
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        result = payload["rule_based_response"]
        self.assertEqual(payload["symptom_extraction"]["present_symptoms"], ["cough"])
        self.assertEqual(self.runtime.classification_calls, 0)
        self.assertEqual(
            result["headline"],
            "Not enough distinct symptoms for classification",
        )
        self.assertIn("requires at least 2 current symptoms", result["message"])
        self.assertIn("respiratory_guidance", result["matched_guidance_rule_ids"])

    def test_low_confidence_after_clarification_reports_correct_reason(self) -> None:
        low_confidence = {
            "predicted_disease": "hidden-low-confidence-class",
            "predicted_label_id": 717,
            "model_score": 0.13,
            "status": "low_score_review_symptoms",
            "minimum_score_threshold": 0.5,
            "top_two_score_margin": 0.004,
            "minimum_margin_threshold": 0.1,
            "top_predictions": [
                {
                    "rank": 1,
                    "label_id": 717,
                    "disease": "hidden-low-confidence-class",
                    "model_score": 0.13,
                },
                {
                    "rank": 2,
                    "label_id": 146,
                    "disease": "second-hidden-class",
                    "model_score": 0.126,
                },
            ],
            "number_of_disease_classes": 773,
            "display_prediction": False,
        }
        with patch.object(self.runtime, "classify", return_value=low_confidence):
            response = self.client.post(
                "/api/analyze/clarification",
                json={
                    "original_narrative": "I have a dry cough that is worse at night.",
                    "answer": (
                        "I also have a sore throat and nasal congestion. The cough "
                        "started one week ago and is 4 out of 10."
                    ),
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        classification = payload["classification"]
        result = payload["rule_based_response"]
        self.assertTrue(classification["inference_performed"])
        self.assertEqual(
            classification["pre_limit_status"],
            "low_score_request_clarification",
        )
        self.assertEqual(
            result["headline"],
            "BioBERT result withheld because confidence was low",
        )
        self.assertNotIn("hidden-low-confidence-class", result["response_text"])

    def test_audio_upload_contract(self) -> None:
        mocked_result = {
            "request_id": "audio-test",
            "transcript": "I have a headache.",
            "audio_preprocessing": {"processed_duration_seconds": 1.0},
            "whisper_model": "openai/whisper-small",
        }
        with patch.object(
            main.service,
            "transcribe_upload",
            return_value=mocked_result,
        ):
            response = self.client.post(
                "/api/transcribe",
                files={"audio": ("sample.wav", b"RIFF-test", "audio/wav")},
                data={"preprocess": "true", "language": "en"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["transcript"], "I have a headache.")

    def test_browser_webm_audio_is_decoded_to_mono_16khz(self) -> None:
        try:
            import av
        except ImportError:
            self.skipTest("PyAV is not installed")

        with NamedTemporaryFile(suffix=".webm", delete=False) as handle:
            webm_path = Path(handle.name)

        try:
            sample_rate = 48_000
            duration_seconds = 0.5
            samples = np.sin(
                2 * np.pi * 440 * np.arange(int(sample_rate * duration_seconds)) / sample_rate
            )
            pcm = (samples * 16_000).astype(np.int16).reshape(1, -1)

            with av.open(str(webm_path), mode="w") as container:
                stream = container.add_stream("libopus", rate=sample_rate)
                stream.layout = "mono"
                frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
                frame.sample_rate = sample_rate
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)

            decoded = PipelineService._load_uploaded_audio(webm_path)
            self.assertEqual(decoded.dtype, np.float32)
            self.assertGreater(decoded.size, 7_000)
            self.assertLess(decoded.size, 9_000)
        finally:
            webm_path.unlink(missing_ok=True)

    def test_empty_text_is_rejected(self) -> None:
        response = self.client.post("/api/analyze/text", json={"text": "   "})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
