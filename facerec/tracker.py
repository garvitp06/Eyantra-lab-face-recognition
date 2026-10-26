"""Lightweight IoU tracker with per-track identity voting.

Haar runs every frame; ArcFace only runs when a track actually needs (re)identification.
Votes across frames remove flicker and single-frame mistakes, which is where most of the
real-world accuracy gain comes from."""
from __future__ import annotations

import itertools
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


@dataclass
class Track:
    id: int
    box: np.ndarray
    hits: int = 1
    misses: int = 0
    born: float = field(default_factory=time.monotonic)
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    label: str | None = None          # decided identity (None = not yet / unknown)
    score: float = 0.0
    confirmed: bool = False
    unknown: bool = False
    pending: bool = False
    last_recog: float = 0.0
    logged: bool = False

    def view(self) -> dict:
        x, y, w, h = self.box
        return {"id": self.id, "box": (int(x), int(y), int(w), int(h)), "label": self.label,
                "score": self.score, "confirmed": self.confirmed, "unknown": self.unknown,
                "votes": sum(1 for n, _ in self.history if n is not None and n == self.label)}


class Tracker:
    def __init__(self, votes_to_confirm=3, reverify_seconds=1.5, max_misses=6, smooth=0.55):
        self.votes_to_confirm = votes_to_confirm
        self.reverify_seconds = reverify_seconds
        self.max_misses = max_misses
        self.smooth = smooth
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def update(self, dets) -> list[Track]:
        dets = [np.array(d, dtype=np.float64) for d in dets]
        pairs = sorted(((iou(t.box, d), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)),
                       reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < 0.25:
                break
            if ti in used_t or di in used_d:
                continue
            t = self.tracks[ti]
            a = self.smooth
            t.box = a * dets[di] + (1 - a) * t.box     # EMA: stops box jitter
            t.hits += 1
            t.misses = 0
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(next(self._ids), d))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

    def needs_recognition(self, t: Track, now: float) -> bool:
        if t.pending or t.misses > 0:
            return False
        if t.confirmed:
            return now - t.last_recog >= self.reverify_seconds
        if t.unknown:
            return now - t.last_recog >= 0.5
        return now - t.last_recog >= 0.06

    def add_result(self, track_id: int, name: str | None, score: float):
        t = next((t for t in self.tracks if t.id == track_id), None)
        if t is None:
            return None
        t.pending = False
        t.last_recog = time.monotonic()
        t.history.append((name, score))
        named = [n for n, _ in t.history if n is not None]
        if named:
            best, c = Counter(named).most_common(1)[0]
            ratio = c / len(t.history)
            t.score = float(np.mean([s for n, s in t.history if n == best]))
            if c >= self.votes_to_confirm and ratio >= 0.5:
                t.label, t.confirmed, t.unknown = best, True, False
            elif name == best:
                t.label, t.confirmed, t.unknown = best, False, False
            elif t.confirmed and ratio < 0.5:
                t.confirmed = False
        recent = list(t.history)[-3:]
        if len(recent) == 3 and all(n is None for n, _ in recent) and not t.confirmed:
            t.label, t.unknown = None, True
            t.score = float(np.mean([s for _, s in recent]))
        return t
