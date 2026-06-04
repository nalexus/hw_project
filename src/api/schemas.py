from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_text: StrictStr = Field(
        ...,
        description="Document text to classify.",
        examples=["The team won the championship after extra time."],
    )

    @field_validator("document_text")
    @classmethod
    def validate_document_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_text must not be empty")
        return value


class ClassificationResponse(BaseModel):
    message: str
    label: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
