from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .landmarks import LANDMARK_CODES

Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
ModelId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,119}$")]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LandmarkRequest(StrictSchema):
    analysisId: UUID | None = None
    imageUrl: str = Field(min_length=1, max_length=8192)
    imageSha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    modelVersion: ModelId | None = None


class Landmark(StrictSchema):
    code: str
    x: Unit
    y: Unit
    confidence: Unit

    @field_validator("code")
    @classmethod
    def known_code(cls, value):
        if value not in LANDMARK_CODES:
            raise ValueError("Unknown anatomical landmark code")
        return value


class ModelInfo(StrictSchema):
    name: ModelId
    version: ModelId


class ImageInfo(StrictSchema):
    width: int = Field(gt=0, le=16000, strict=True)
    height: int = Field(gt=0, le=16000, strict=True)


class Timings(StrictSchema):
    downloadMs: float = Field(ge=0, allow_inf_nan=False)
    preprocessMs: float = Field(ge=0, allow_inf_nan=False)
    inferenceMs: float = Field(ge=0, allow_inf_nan=False)
    totalMs: float = Field(ge=0, allow_inf_nan=False)


class LandmarkResponse(StrictSchema):
    model: ModelInfo
    image: ImageInfo
    landmarks: list[Landmark] = Field(min_length=1, max_length=len(LANDMARK_CODES))
    timings: Timings

    @field_validator("landmarks")
    @classmethod
    def unique_codes(cls, values):
        if len({p.code for p in values}) != len(values):
            raise ValueError("Duplicate landmark codes")
        return values


class ModelManifest(StrictSchema):
    """A reviewed, versioned adapter contract, never guessed from a model filename."""
    name: ModelId
    version: ModelId
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    rights_review: str = Field(min_length=1)
    landmark_definitions: str = Field(min_length=1)
    validation_report: str = Field(min_length=1)
    input_name: str
    output_name: str
    input_width: int = Field(ge=16, le=2048)
    input_height: int = Field(ge=16, le=2048)
    channels: Literal[1, 3]
    resize: Literal["letterbox-bilinear-v1"]
    scale: Literal["divide-by-255"]
    mean: list[float]
    std: list[float]
    pad_value: int = Field(ge=0, le=255)
    output_adapter: Literal["coordinates-xy-confidence-v1", "probability-heatmaps-v1"]
    confidence_definition: str = Field(min_length=1)
    codes: list[str] = Field(min_length=1, max_length=len(LANDMARK_CODES))

    @field_validator("codes")
    @classmethod
    def codes_valid(cls, codes):
        if len(set(codes)) != len(codes) or any(c not in LANDMARK_CODES for c in codes):
            raise ValueError("Invalid landmark mapping")
        return codes
