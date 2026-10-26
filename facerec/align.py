"""Face alignment to the 112x112 ArcFace template.

ArcFace was trained on faces warped so the eyes land on fixed pixel positions. Feeding it
raw Haar crops (as the old code did) costs a lot of accuracy, so every crop - at
registration and at recognition time - goes through the same similarity warp here."""
from __future__ import annotations

import cv2
import numpy as np

CHIP_SIZE = 112
# Standard ArcFace 5-point template (insightface), we use the two eye points.
_LEFT_EYE = np.array([38.2946, 51.6963])
_RIGHT_EYE = np.array([73.5318, 51.5014])

# Where the eyes sit inside a Haar 'frontalface_default' box, on average.
# Used when the eye cascade cannot find both eyes.
BOX_LEFT_EYE = (0.302, 0.392)
BOX_RIGHT_EYE = (0.678, 0.393)


def box_eyes(box) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = box
    return (np.array([x + BOX_LEFT_EYE[0] * w, y + BOX_LEFT_EYE[1] * h]),
            np.array([x + BOX_RIGHT_EYE[0] * w, y + BOX_RIGHT_EYE[1] * h]))


def _similarity(src_l, src_r, dst_l, dst_r) -> np.ndarray:
    """2x3 similarity transform mapping two source points onto two destination points."""
    s = complex(*(src_r - src_l))
    d = complex(*(dst_r - dst_l))
    m = d / s                                     # rotation + scale as a complex number
    a, b = m.real, m.imag
    t = complex(*dst_l) - m * complex(*src_l)
    return np.array([[a, -b, t.real], [b, a, t.imag]], dtype=np.float64)


def align_face(frame_bgr: np.ndarray, box, eyes=None) -> np.ndarray:
    """Return an aligned 112x112 BGR face chip."""
    bl, br = box_eyes(box)
    if eyes is not None:
        el, er = eyes
        # sanity: detected eyes must roughly agree with the box geometry
        w = box[2]
        if np.linalg.norm(el - bl) < 0.18 * w and np.linalg.norm(er - br) < 0.18 * w:
            bl, br = el, er
    M = _similarity(bl, br, _LEFT_EYE, _RIGHT_EYE)
    return cv2.warpAffine(frame_bgr, M, (CHIP_SIZE, CHIP_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def legacy_crop(frame_bgr: np.ndarray, box) -> np.ndarray:
    """Preprocessing used by the old model.pkl (raw Haar crop, squashed to 112)."""
    x, y, w, h = box
    crop = frame_bgr[max(0, y):y + h, max(0, x):x + w]
    return cv2.resize(crop, (CHIP_SIZE, CHIP_SIZE))


def sharpness(chip_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(chip_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
