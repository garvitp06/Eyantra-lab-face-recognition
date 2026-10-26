"""Enrolled identities and vectorised matching."""
from __future__ import annotations

import os
import pickle
import numpy as np

GALLERY_VERSION = 2


class Gallery:
    def __init__(self, encodings=None, names=None, meta=None):
        self.meta = meta or {"version": GALLERY_VERSION, "preprocess": "aligned_rgb"}
        self._set(np.asarray(encodings if encodings is not None else np.zeros((0, 512)), np.float32),
                  list(names or []))

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str) -> "Gallery":
        if not os.path.exists(path):
            return cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        meta = data.get("meta") or {"version": 1, "preprocess": "legacy"}
        return cls(data["encodings"], data["names"], meta)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"encodings": self.encodings, "names": self.names, "meta": self.meta}, f)

    @property
    def legacy(self) -> bool:
        return self.meta.get("preprocess") == "legacy"

    # ------------------------------------------------------------ internals
    def _set(self, enc: np.ndarray, names: list[str]):
        if len(enc):
            enc = enc / (np.linalg.norm(enc, axis=1, keepdims=True) + 1e-9)
        self.encodings, self.names = enc.astype(np.float32), names
        self.people = sorted(set(names))
        self._idx = {p: np.array([i for i, n in enumerate(names) if n == p]) for p in self.people}
        cents = []
        for p in self.people:
            c = enc[self._idx[p]].mean(axis=0)
            cents.append(c / (np.linalg.norm(c) + 1e-9))
        self.centroids = np.stack(cents).astype(np.float32) if cents else np.zeros((0, enc.shape[1] if enc.ndim == 2 else 512), np.float32)

    def counts(self) -> dict[str, int]:
        return {p: len(self._idx[p]) for p in self.people}

    def __len__(self):
        return len(self.people)

    # ------------------------------------------------------------- matching
    def scores(self, q: np.ndarray) -> np.ndarray:
        """(M, P) per-person score: blend of centroid similarity and mean of the top-3
        sample similarities (robust to pose variety and to a few bad samples)."""
        if len(self.people) == 0 or len(q) == 0:
            return np.zeros((len(q), len(self.people)), np.float32)
        cent = q @ self.centroids.T
        sims = q @ self.encodings.T
        out = np.empty_like(cent)
        for j, p in enumerate(self.people):
            s = sims[:, self._idx[p]]
            k = min(3, s.shape[1])
            top = np.partition(s, -k, axis=1)[:, -k:].mean(axis=1)
            out[:, j] = 0.5 * cent[:, j] + 0.5 * top
        return out

    def match(self, q: np.ndarray, threshold: float, margin: float):
        """-> list of (name | None, score, best_name) per query."""
        S = self.scores(q)
        res = []
        for row in S:
            if len(row) == 0:
                res.append((None, 0.0, None))
                continue
            order = np.argsort(row)[::-1]
            best = float(row[order[0]])
            second = float(row[order[1]]) if len(order) > 1 else -1.0
            name = self.people[order[0]]
            ok = best >= threshold and (best - second) >= margin
            res.append((name if ok else None, best, name))
        return res

    # ------------------------------------------------------------- editing
    @staticmethod
    def clean(enc: np.ndarray, min_sim: float = 0.25) -> np.ndarray:
        """Drop outlier samples (bad crops / wrong person) far from the person's centroid."""
        if len(enc) < 4:
            return enc
        c = enc.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-9
        keep = enc @ c >= min_sim
        return enc[keep] if keep.sum() >= max(3, len(enc) // 3) else enc
