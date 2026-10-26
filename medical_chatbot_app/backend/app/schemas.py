from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TextAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Symptom text cannot be empty.")
        return value


class ClarificationAnswerRequest(BaseModel):
    original_narrative: str = Field(min_length=1, max_length=5000)
    answer: str = Field(min_length=1, max_length=3000)

    @field_validator("original_narrative", "answer")
    @classmethod
    def clean_parts(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation text cannot be empty.")
        return value
