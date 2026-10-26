"""Haar-cascade face detector tuned for speed (downscaled, equalised input) plus
Haar eye localisation used for alignment."""
from __future__ import annotations

import os
import cv2
import numpy as np


def _cascade(name: str) -> cv2.CascadeClassifier:
    candidates = []
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, name))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades", name))
    for p in candidates:
        if os.path.exists(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                return c
    raise FileNotFoundError(f"Haar cascade '{name}' not found (looked in {candidates})")


class HaarDetector:
    """Not thread-safe: create one instance per thread."""

    def __init__(self, detect_width: int = 480, scale_factor: float = 1.1,
                 min_neighbors: int = 6, min_face_frac: float = 0.07):
        self.face = _cascade("haarcascade_frontalface_default.xml")
        self.eyes = _cascade("haarcascade_eye.xml")
        self.detect_width = detect_width
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_frac = min_face_frac
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._clahe_det = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def detect(self, frame_bgr: np.ndarray, gray: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
        """Return face boxes (x, y, w, h) in full-resolution frame coordinates."""
        h, w = frame_bgr.shape[:2]
        if gray is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        s = min(1.0, self.detect_width / float(w))
        small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
        # local contrast (CLAHE): global equalizeHist breaks when a large dark/bright background
        # dominates the histogram (backlit rooms), CLAHE keeps faces detectable in both cases
        small = self._clahe_det.apply(small)
        mn = max(24, int(self.min_face_frac * small.shape[1]))
        rects = self.face.detectMultiScale(
            small, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(mn, mn), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        inv = 1.0 / s
        return [(int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)) for (x, y, bw, bh) in rects]

    def find_eyes(self, gray: np.ndarray, box) -> tuple[np.ndarray, np.ndarray] | None:
        """Locate (left_eye, right_eye) centres in image coordinates, or None.
        'left' means the eye on the left side of the image."""
        x, y, w, h = box
        H, W = gray.shape[:2]
        y0, y1 = max(0, y + int(0.15 * h)), min(H, y + int(0.58 * h))
        x0, x1 = max(0, x), min(W, x + w)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        k = 160.0 / roi.shape[1]
        roi = cv2.resize(roi, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        roi = self._clahe.apply(roi)
        eyes = self.eyes.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18), maxSize=(70, 70))
        if len(eyes) < 2:
            return None
        cx = 80.0
        left = [e for e in eyes if e[0] + e[2] / 2 < cx]
        right = [e for e in eyes if e[0] + e[2] / 2 >= cx]
        if not left or not right:
            return None
        # pick the pair with the most similar height and plausible spacing
        best, best_cost = None, 1e9
        for l in left:
            for r in right:
                lc = np.array([l[0] + l[2] / 2, l[1] + l[3] / 2])
                rc = np.array([r[0] + r[2] / 2, r[1] + r[3] / 2])
                d = rc[0] - lc[0]
                if not (0.28 * 160 <= d <= 0.62 * 160):
                    continue
                cost = abs(lc[1] - rc[1]) + 0.3 * abs(d - 0.40 * 160)
                if cost < best_cost:
                    best, best_cost = (lc, rc), cost
        if best is None:
            return None
        lc, rc = best
        if abs(lc[1] - rc[1]) > 0.35 * (rc[0] - lc[0]):   # > ~20 degrees roll: unreliable
            return None
        off = np.array([x0, y0], dtype=np.float64)
        return lc / k + off, rc / k + off
