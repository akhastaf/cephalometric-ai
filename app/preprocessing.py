from dataclasses import dataclass
from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ServiceError
from .schemas import ModelManifest


@dataclass(frozen=True)
class Transform:
    width: int
    height: int
    input_width: int
    input_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int

    def normalized_original(self, x: float, y: float) -> tuple[float, float]:
        # Coordinate convention: continuous edges, x=0 left, x=1 right, y increases down.
        return ((x * self.input_width - self.left) / self.resized_width,
                (y * self.input_height - self.top) / self.resized_height)


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in ("JPEG", "PNG") or getattr(image, "n_frames", 1) != 1:
                    raise ServiceError("UNSUPPORTED_IMAGE")
                if image.width * image.height > max_pixels or max(image.size) > 16000:
                    raise ServiceError("IMAGE_TOO_LARGE", 413)
                image.load()
                # Browser display, Nest dimensions, and inference share EXIF orientation.
                return ImageOps.exif_transpose(image).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ServiceError("IMAGE_TOO_LARGE", 413) from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise ServiceError("UNSUPPORTED_IMAGE") from None


def preprocess(image: Image.Image, manifest: ModelManifest) -> tuple[np.ndarray, Transform]:
    w, h = image.size
    iw, ih = manifest.input_width, manifest.input_height
    ratio = min(iw / w, ih / h)
    rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (iw - rw) // 2, (ih - rh) // 2
    mode = "L" if manifest.channels == 1 else "RGB"
    resized = image.convert(mode).resize((rw, rh), Image.Resampling.BILINEAR)
    canvas = Image.new(mode, (iw, ih), manifest.pad_value if mode == "L" else (manifest.pad_value,) * 3)
    canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    if manifest.channels == 1:
        pixels = pixels[:, :, None]
    mean = np.asarray(manifest.mean, dtype=np.float32)
    std = np.asarray(manifest.std, dtype=np.float32)
    tensor = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(tensor), Transform(w, h, iw, ih, rw, rh, left, top)
