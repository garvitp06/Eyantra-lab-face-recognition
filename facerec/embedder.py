"""ArcFace-ResNet100 embedder on OpenVINO with an async request pool, so several faces
are embedded concurrently and the caller never blocks the detector."""
from __future__ import annotations

import os
import threading
import cv2
import numpy as np

try:
    import openvino as ov
except ImportError:  # very old OpenVINO
    from openvino.runtime import Core as _Core  # type: ignore

    class ov:  # type: ignore
        Core = _Core


class ArcFaceEmbedder:
    def __init__(self, model_path: str, device: str = "CPU", num_requests: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}\n"
                "Run:  python scripts/get_model.py   (downloads + converts arcfaceresnet100-8)")
        core = ov.Core()
        cache = os.path.join(os.path.dirname(model_path), ".ov_cache")
        try:
            core.set_property({"CACHE_DIR": cache})   # much faster 2nd start-up
        except Exception:
            pass
        model = core.read_model(model_path)
        inp = model.input(0)
        shape = list(inp.get_partial_shape().get_min_shape()) if inp.get_partial_shape().is_dynamic else list(inp.shape)
        self.h, self.w = int(shape[2]) or 112, int(shape[3]) or 112
        cfg = {"PERFORMANCE_HINT": "THROUGHPUT" if num_requests > 1 else "LATENCY"}
        if num_requests > 1:
            cfg["PERFORMANCE_HINT_NUM_REQUESTS"] = str(num_requests)
        try:
            self.compiled = core.compile_model(model, device, cfg)
        except Exception:
            self.compiled = core.compile_model(model, device)
        self.device = device
        self.output = self.compiled.output(0)
        self.num_requests = max(1, num_requests)
        self._requests = [self.compiled.create_infer_request() for _ in range(self.num_requests)]
        self._lock = threading.Lock()

    def _blob(self, chip_bgr: np.ndarray) -> np.ndarray:
        # ArcFace (insightface/ONNX zoo) expects RGB, 0..255, NCHW. Normalisation is inside the graph.
        if chip_bgr.shape[0] != self.h or chip_bgr.shape[1] != self.w:
            chip_bgr = cv2.resize(chip_bgr, (self.w, self.h))
        rgb = chip_bgr[:, :, ::-1]
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32))

    def embed(self, chips_bgr: list[np.ndarray], flip: bool = False, bgr_input: bool = False) -> np.ndarray:
        """Embed N aligned chips -> (N, 512) L2-normalised. Runs up to num_requests in parallel.
        bgr_input=True reproduces the old (buggy) channel order for legacy galleries."""
        if not chips_bgr:
            return np.zeros((0, 512), np.float32)
        items = []
        for c in chips_bgr:
            src = c[:, :, ::-1] if bgr_input else c
            items.append(self._blob(src))
            if flip:
                items.append(self._blob(src[:, ::-1]))
        out = [None] * len(items)
        with self._lock:
            for start in range(0, len(items), self.num_requests):
                batch = list(enumerate(items))[start:start + self.num_requests]
                reqs = self._requests[:len(batch)]
                for req, (_, blob) in zip(reqs, batch):
                    req.start_async({0: blob})
                for req, (i, _) in zip(reqs, batch):
                    req.wait()
                    out[i] = req.get_output_tensor(0).data.reshape(-1).copy()
        emb = np.stack(out).astype(np.float32)
        if flip:
            emb = emb.reshape(len(chips_bgr), 2, -1).sum(axis=1)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return emb
