"""Real-time face recognition engine: Haar cascade detection + ArcFace-ResNet100 (OpenVINO)
running in parallel pipeline stages."""

from .config import Config, BASE_DIR
from .detector import HaarDetector
from .align import align_face, CHIP_SIZE
from .gallery import Gallery
from .attendance import AttendanceLog
from .pipeline import RecognitionPipeline

__all__ = [
    "Config", "BASE_DIR", "HaarDetector", "align_face", "CHIP_SIZE",
    "Gallery", "AttendanceLog", "RecognitionPipeline",
]
