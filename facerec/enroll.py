"""Registration (quality-gated, diverse sample capture) and gallery training."""
from __future__ import annotations

import os
import re
import time

import cv2
import numpy as np

from .align import CHIP_SIZE, align_face, sharpness
from .detector import HaarDetector
from .gallery import Gallery, GALLERY_VERSION

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\- ]+", "", name).strip()


class Enroller:
    """Feeds live frames, saves aligned 112x112 chips only when the face is
    big enough, sharp, and noticeably different from the last saved sample."""

    def __init__(self, detector: HaarDetector, dataset_dir: str, person: str, target: int = 40,
                 min_interval: float = 0.12, min_sharp: float = 35.0):
        self.det = detector
        self.person = safe_name(person)
        self.dir = os.path.join(dataset_dir, self.person)
        os.makedirs(self.dir, exist_ok=True)
        self.target = target
        self.min_interval = min_interval
        self.min_sharp = min_sharp
        self.count = 0
        self._last_t = 0.0
        self._last_thumb = None
        self.chips: list[np.ndarray] = []

    @property
    def done(self) -> bool:
        return self.count >= self.target

    @staticmethod
    def check(det: HaarDetector, frame: np.ndarray, min_sharp: float = 35.0):
        """Quality gate. -> (box | None, status_text, ok: bool, chip | None)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = det.detect(frame, gray)
        if not boxes:
            return None, "No face detected - look at the camera", False, None
        box = max(boxes, key=lambda b: b[2] * b[3])
        if box[2] < 0.16 * frame.shape[1]:
            return box, "Move closer to the camera", False, None
        chip = align_face(frame, box, det.find_eyes(gray, box))
        if sharpness(chip) < min_sharp:
            return box, "Hold still - image is blurry", False, None
        if len(boxes) > 1:
            return box, "Several faces in view - only one person please", False, None
        return box, "Face looks good", True, chip

    def process(self, frame: np.ndarray, capture: bool = True):
        """-> (box | None, status_text, ok_quality: bool, saved_chip | None)"""
        box, status, ok, chip = self.check(self.det, frame, self.min_sharp)
        if not ok or not capture or self.done:
            return box, status, ok, None
        now = time.monotonic()
        thumb = cv2.resize(cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY), (24, 24)).astype(np.float32)
        if now - self._last_t < self.min_interval:
            return box, "Capturing...", True, None
        if self._last_thumb is not None and np.abs(thumb - self._last_thumb).mean() < 4.0:
            return box, "Slowly turn your head / change expression", True, None
        self._last_t, self._last_thumb = now, thumb
        path = os.path.join(self.dir, f"{int(time.time() * 1000)}_{self.count:03d}.png")
        cv2.imwrite(path, chip)
        self.count += 1
        self.chips.append(chip)
        return box, f"Captured {self.count}/{self.target}", True, chip


def dataset_summary(dataset_dir: str) -> dict[str, list[str]]:
    out = {}
    if not os.path.isdir(dataset_dir):
        return out
    for p in sorted(os.listdir(dataset_dir)):
        d = os.path.join(dataset_dir, p)
        if os.path.isdir(d):
            files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXT)]
            if files:
                out[p] = files
    return out


def image_to_chip(img: np.ndarray, det: HaarDetector) -> np.ndarray | None:
    """Accept an aligned chip, a tight face crop or a full photo."""
    h, w = img.shape[:2]
    if (h, w) == (CHIP_SIZE, CHIP_SIZE):
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = det.detect(img, gray)
    if boxes:
        box = max(boxes, key=lambda b: b[2] * b[3])
        return align_face(img, box, det.find_eyes(gray, box))
    if 0.7 < w / float(h) < 1.4 and w <= 400:      # probably an old raw Haar crop
        return align_face(img, (0, 0, w, h), det.find_eyes(gray, (0, 0, w, h)))
    return None


def build_gallery(dataset_dir: str, embedder, progress=None, log=print) -> Gallery:
    """Embed every image in dataset/<person>/*.{png,jpg} (with flip augmentation)."""
    det = HaarDetector(detect_width=640, min_neighbors=5, min_face_frac=0.05)
    people = dataset_summary(dataset_dir)
    if not people:
        raise RuntimeError(f"No images found in {dataset_dir}. Register at least one person first.")
    total = sum(len(v) for v in people.values())
    done = 0
    all_enc, all_names = [], []
    for person, files in people.items():
        chips, skipped = [], 0
        for f in files:
            img = cv2.imread(f)
            chip = image_to_chip(img, det) if img is not None else None
            if chip is None:
                skipped += 1
            else:
                chips.append(chip)
            done += 1
            if progress and done % 5 == 0:
                progress(int(90 * done / total), f"Reading {person} ({done}/{total})")
        if not chips:
            log(f"[skip] {person}: no usable face in {len(files)} images")
            continue
        enc = []
        for i in range(0, len(chips), 16):
            enc.append(embedder.embed(chips[i:i + 16], flip=True))
        enc = np.concatenate(enc)
        kept = Gallery.clean(enc)
        all_enc.append(kept)
        all_names += [person] * len(kept)
        log(f"[ok] {person}: {len(kept)} samples"
            + (f" ({len(enc) - len(kept)} outliers removed)" if len(kept) < len(enc) else "")
            + (f", {skipped} images without a face" if skipped else ""))
        if progress:
            progress(int(90 * done / total), f"Encoded {person}")
    if not all_enc:
        raise RuntimeError("No usable faces found in the dataset.")
    g = Gallery(np.concatenate(all_enc), all_names,
                {"version": GALLERY_VERSION, "preprocess": "aligned_rgb", "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    # warn about look-alike identities
    if len(g.people) > 1:
        C = g.centroids @ g.centroids.T
        np.fill_diagonal(C, -1)
        i, j = np.unravel_index(np.argmax(C), C.shape)
        if C[i, j] > 0.45:
            log(f"[warn] '{g.people[i]}' and '{g.people[j]}' look very similar (sim {C[i, j]:.2f}); "
                "capture more varied samples for both.")
    return g
