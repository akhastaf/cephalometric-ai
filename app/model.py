import hashlib
import logging
from pathlib import Path
from threading import Timer

import numpy as np
import onnxruntime as ort

from .config import Settings
from .errors import ServiceError
from .schemas import ModelManifest

logger = logging.getLogger("cephalometry")


class ModelRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: ort.InferenceSession | None = None
        self.manifest: ModelManifest | None = None
        self.error_code = "MODEL_NOT_CONFIGURED"

    def load(self):
        if not self.settings.model_path or not self.settings.model_name or not self.settings.model_version:
            return
        try:
            manifest = ModelManifest.model_validate_json(self.settings.manifest_path.read_text())
            if manifest.name != self.settings.model_name or manifest.version != self.settings.model_version:
                raise ValueError("Model identity mismatch")
            if len(manifest.mean) != manifest.channels or len(manifest.std) != manifest.channels:
                raise ValueError("Invalid channel normalization")
            if not np.isfinite(manifest.mean).all() or not np.isfinite(manifest.std).all() or any(v <= 0 for v in manifest.std):
                raise ValueError("Invalid normalization values")
            path = Path(self.settings.model_path)
            with path.open("rb") as model_file:
                checksum = hashlib.file_digest(model_file, "sha256").hexdigest()
            if checksum != manifest.sha256:
                raise ValueError("Model checksum mismatch")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.log_severity_level = 3
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            if len(inputs) != 1 or inputs[0].name != manifest.input_name or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, manifest.channels, manifest.input_height, manifest.input_width]:
                raise ValueError("Unsupported ONNX input contract")
            outputs = session.get_outputs()
            if len(outputs) != 1 or outputs[0].name != manifest.output_name or outputs[0].type != "tensor(float)":
                raise ValueError("Unsupported ONNX output contract")
            self.manifest, self.session, self.error_code = manifest, session, ""
            logger.info("model_loaded name=%s version=%s", manifest.name, manifest.version)
        except Exception:
            # Filenames, external initializer paths and parser internals stay out of logs/responses.
            self.error_code = "MODEL_NOT_CONFIGURED"
            logger.error("model_configuration_invalid")

    @property
    def loaded(self):
        return self.session is not None and self.manifest is not None

    def run(self, tensor: np.ndarray) -> np.ndarray:
        if not self.loaded:
            raise ServiceError("MODEL_NOT_CONFIGURED", 503)
        options = ort.RunOptions()
        def terminate():
            options.terminate = True
        timer = Timer(self.settings.inference_timeout_seconds, terminate)
        timer.daemon = True
        timer.start()
        try:
            return self.session.run([self.manifest.output_name], {self.manifest.input_name: tensor}, options)[0]
        except Exception:
            raise ServiceError("AI_TIMEOUT" if options.terminate else "INVALID_AI_RESPONSE", 503) from None
        finally:
            timer.cancel()
