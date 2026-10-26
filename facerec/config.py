"""Application settings, persisted as JSON next to the project."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, fields

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

# Where we look for the ArcFace network if the configured path does not exist.
MODEL_CANDIDATES = [
    "openvino_model/arcfaceresnet100-8.xml",
    "openvino_model/arcfaceresnet100-8.onnx",
    "converted_ir/new/arcfaceresnet100-8.xml",
    "model/arcfaceresnet100-8.xml",
    "model/arcfaceresnet100-8.onnx",
]


def abspath(p: str) -> str:
    if not p:
        return ""
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def relpath(p: str) -> str:
    """Store paths relative to the project when they live inside it."""
    if not p:
        return ""
    p = os.path.abspath(p)
    try:
        if os.path.commonpath([p, BASE_DIR]) == BASE_DIR:
            return os.path.relpath(p, BASE_DIR)
    except ValueError:  # different drives on Windows
        pass
    return p


@dataclass
class Config:
    # paths
    arcface_model: str = "openvino_model/arcfaceresnet100-8.xml"
    gallery_path: str = "model/model.pkl"
    dataset_dir: str = "dataset"
    attendance_csv: str = "attendance/attendance.csv"

    # camera
    camera_source: str = "0"          # "0", "1" or an RTSP/HTTP URL / video file
    camera_width: int = 1280
    camera_height: int = 720

    # inference
    device: str = "CPU"               # CPU / GPU / AUTO
    infer_requests: int = 2           # parallel OpenVINO requests (faces embedded concurrently)

    # Haar detector
    detect_width: int = 480           # frame is downscaled to this width for Haar (big speed win)
    scale_factor: float = 1.1
    min_neighbors: int = 6
    min_face_frac: float = 0.07       # min face width as a fraction of frame width
    detect_every: int = 1             # run Haar on every Nth frame (tracker fills the gaps)

    # recognition / decision
    threshold: float = 0.36           # cosine similarity required to accept an identity
    margin: float = 0.06              # best must beat 2nd-best person by this much
    votes_to_confirm: int = 3         # consistent hits before a name is shown as confirmed / logged
    reverify_seconds: float = 1.5     # confirmed tracks are re-checked this often
    flip_tta: bool = False            # also embed mirrored face at runtime (slower, slightly more robust)

    # registration
    samples_per_person: int = 40

    @classmethod
    def load(cls, path: str = SETTINGS_FILE) -> "Config":
        cfg = cls()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                names = {f.name for f in fields(cls)}
                for k, v in data.items():
                    if k in names:
                        setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception as e:  # corrupt settings should never block startup
                print(f"[config] ignoring unreadable settings: {e}")
        return cfg

    def save(self, path: str = SETTINGS_FILE) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def resolve_model(self) -> str:
        """Absolute path of the ArcFace model, falling back to known locations."""
        p = abspath(self.arcface_model)
        if os.path.exists(p):
            return p
        for c in MODEL_CANDIDATES:
            if os.path.exists(abspath(c)):
                return abspath(c)
        return p

    def camera(self):
        s = str(self.camera_source).strip()
        return int(s) if s.isdigit() else s
