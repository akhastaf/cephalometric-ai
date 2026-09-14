import asyncio
import hashlib
import hmac
from urllib.parse import urlsplit

import httpx

from .config import Settings
from .errors import ServiceError


def validate_url(value: str, settings: Settings):
    try:
        url = urlsplit(value)
        # Only administratively trusted exact storage origins. No wildcard or suffix matching.
        origin = f"{url.scheme}://{url.netloc}"
        if (url.scheme not in ("https", "http") or not url.hostname or url.username or url.password
                or url.fragment or "\\" in value or any(ord(c) < 32 for c in value)
                or origin not in settings.allowed_image_origins):
            raise ValueError("Untrusted image origin")
        if url.scheme == "http" and not settings.allow_insecure_image_http:
            raise ValueError("Insecure image URL")
    except ValueError:
        raise ServiceError("INVALID_IMAGE_URL") from None


async def download_image(url: str, settings: Settings, expected_sha256: str | None = None) -> bytes:
    validate_url(url, settings)
    maximum = settings.max_image_size_mb * 1024 * 1024
    try:
        async with asyncio.timeout(settings.request_timeout_seconds):
            async with httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                async with client.stream("GET", url, headers={"Accept": "image/jpeg,image/png"}) as response:
                    if response.status_code in (401, 403):
                        raise ServiceError("SIGNED_URL_EXPIRED", 502)
                    if response.status_code != 200:
                        raise ServiceError("STORAGE_ERROR", 502)
                    length = response.headers.get("Content-Length")
                    if length and int(length) > maximum:
                        raise ServiceError("IMAGE_TOO_LARGE", 413)
                    content = bytearray()
                    async for chunk in response.aiter_bytes(65536):
                        if len(content) + len(chunk) > maximum:
                            raise ServiceError("IMAGE_TOO_LARGE", 413)
                        content.extend(chunk)
                    image = bytes(content)
                    if expected_sha256 and not hmac.compare_digest(hashlib.sha256(image).hexdigest(), expected_sha256):
                        raise ServiceError("IMAGE_CHANGED", 409)
                    return image
    except (TimeoutError, httpx.TimeoutException):
        raise ServiceError("AI_TIMEOUT", 504) from None
    except (httpx.HTTPError, ValueError):
        raise ServiceError("STORAGE_ERROR", 502) from None
