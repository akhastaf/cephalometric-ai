import os
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    internal_api_key: str = ""
    model_path: str = ""
    model_name: str = ""
    model_version: str = ""
    model_manifest_path: str = ""
    allowed_image_origins: tuple[str, ...] = ()
    allow_insecure_image_http: bool = False
    max_image_size_mb: int = Field(default=20, ge=1, le=50)
    max_image_pixels: int = Field(default=40_000_000, ge=1, le=80_000_000)
    request_timeout_seconds: int = Field(default=90, ge=1, le=180)
    inference_timeout_seconds: int = Field(default=60, ge=1, le=120)

    @model_validator(mode="after")
    def validate_origins(self):
        for origin in self.allowed_image_origins:
            url = urlsplit(origin)
            if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment or url.path:
                raise ValueError("ALLOWED_IMAGE_ORIGINS must contain exact origins without paths or trailing slashes")
            if url.scheme != "https" and not self.allow_insecure_image_http:
                raise ValueError("HTTP storage requires explicit ALLOW_INSECURE_IMAGE_HTTP=true (local development only)")
        return self

    @classmethod
    def from_env(cls):
        return cls(
            internal_api_key=os.getenv("CEPH_AI_INTERNAL_API_KEY", ""),
            model_path=os.getenv("MODEL_PATH", ""),
            model_name=os.getenv("MODEL_NAME", ""),
            model_version=os.getenv("MODEL_VERSION", ""),
            model_manifest_path=os.getenv("MODEL_MANIFEST_PATH", ""),
            allowed_image_origins=tuple(o.strip() for o in os.getenv("ALLOWED_IMAGE_ORIGINS", "").split(",") if o.strip()),
            allow_insecure_image_http=os.getenv("ALLOW_INSECURE_IMAGE_HTTP", "false").lower() == "true",
            max_image_size_mb=int(os.getenv("MAX_IMAGE_SIZE_MB", "20")),
            max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "40000000")),
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90")),
            inference_timeout_seconds=int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "60")),
        )

    @property
    def manifest_path(self) -> Path:
        return Path(self.model_manifest_path or f"{self.model_path}.json")
