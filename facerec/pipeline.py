"""Parallel recognition pipeline.

    [capture thread] --latest frame--> [Haar detect + track thread] --face chips--> [ArcFace thread(s)]
                                              |                                          |
                                              +---- snapshot (frame, tracks) <-- results-+
                                                         |
                                                     UI / CLI

* Capture keeps only the newest frame, so a slow consumer never builds up lag.
* Haar detection never waits for ArcFace: it hands chips over a small queue and moves on.
* ArcFace runs OpenVINO requests concurrently and only for tracks that need an identity,
  so the expensive network runs a few times per person rather than on every frame.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from .align import align_face, legacy_crop
from .attendance import AttendanceLog
from .config import Config, abspath
from .detector import HaarDetector
from .embedder import ArcFaceEmbedder
from .gallery import Gallery
from .tracker import Tracker


class _Ema:
    def __init__(self, a=0.1):
        self.a, self.v = a, None

    def add(self, x):
        self.v = x if self.v is None else (1 - self.a) * self.v + self.a * x
        return self.v

    def get(self):
        return self.v or 0.0


class FrameSource:
    """Background frame grabber that always exposes only the newest frame."""

    def __init__(self, source, width=1280, height=720):
        self.source = source
        self.is_file = isinstance(source, str) and not source.lower().startswith(("rtsp", "http", "udp", "tcp"))
        api = cv2.CAP_FFMPEG if isinstance(source, str) else cv2.CAP_ANY
        if isinstance(source, int) and os.name == "nt":
            api = cv2.CAP_DSHOW                    # fast webcam open on Windows
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps_file = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cond = threading.Condition()
        self._frame, self._id, self._t = None, 0, 0.0
        self.ended = False
        self.fps = _Ema(0.05)
        self._running = True
        self._th = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._th.start()

    def _loop(self):
        last = time.monotonic()
        period = 1.0 / max(1.0, self.fps_file)
        while self._running:
            ok, frame = self.cap.read()
            now = time.monotonic()
            if not ok:
                if self.is_file:
                    self.ended = True
                    with self._cond:
                        self._cond.notify_all()
                    break
                time.sleep(0.01)
                continue
            if self.is_file:                       # play files in real time
                sleep = period - (now - last)
                if sleep > 0:
                    time.sleep(sleep)
                now = time.monotonic()
            self.fps.add(1.0 / max(1e-3, now - last))
            last = now
            with self._cond:
                self._frame, self._t = frame, now
                self._id += 1
                self._cond.notify_all()

    def wait_next(self, last_id: int, timeout=0.5):
        with self._cond:
            if self._id == last_id and not self.ended:
                self._cond.wait(timeout)
            return self._frame, self._id, self._t

    def release(self):
        self._running = False
        self._th.join(timeout=1.0)
        self.cap.release()


class RecognitionPipeline:
    def __init__(self, cfg: Config, embedder: ArcFaceEmbedder | None = None,
                 gallery: Gallery | None = None, attendance: AttendanceLog | None = None):
        self.cfg = cfg
        self.embedder = embedder or ArcFaceEmbedder(cfg.resolve_model(), cfg.device, cfg.infer_requests)
        self.gallery = gallery if gallery is not None else Gallery.load(abspath(cfg.gallery_path))
        self.attendance = attendance
        self.tracker = Tracker(cfg.votes_to_confirm, cfg.reverify_seconds)
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue(maxsize=max(4, 2 * cfg.infer_requests))
        self.events: deque = deque(maxlen=200)
        self._snapshot = (None, [], 0)
        self.stats = {k: _Ema(0.1) for k in ("detect_ms", "recog_ms", "latency_ms", "proc_fps")}
        self.source: FrameSource | None = None
        self._threads: list[threading.Thread] = []
        self.running = False
        self.error: str | None = None

    # ------------------------------------------------------------ control
    def start(self, source=None):
        self.source = FrameSource(source if source is not None else self.cfg.camera(),
                                  self.cfg.camera_width, self.cfg.camera_height)
        self.running = True
        self._threads = [threading.Thread(target=self._detect_loop, name="haar", daemon=True),
                         threading.Thread(target=self._recog_loop, name="arcface", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self):
        self.running = False
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        for t in self._threads:
            t.join(timeout=2.0)
        if self.source:
            self.source.release()
            self.source = None

    @property
    def finished(self) -> bool:
        return self.source is not None and self.source.ended

    def snapshot(self):
        """-> (frame, [track dicts], frame_id) of the newest processed frame."""
        with self._lock:
            return self._snapshot

    def metrics(self) -> dict:
        m = {k: v.get() for k, v in self.stats.items()}
        m["camera_fps"] = self.source.fps.get() if self.source else 0.0
        m["queue"] = self._jobs.qsize()
        return m

    # ------------------------------------------------------- stage 1: Haar
    def _detect_loop(self):
        cfg = self.cfg
        det = HaarDetector(cfg.detect_width, cfg.scale_factor, cfg.min_neighbors, cfg.min_face_frac)
        legacy = self.gallery.legacy
        last_id, n, last_t = 0, 0, time.monotonic()
        try:
            while self.running:
                frame, fid, t_cap = self.source.wait_next(last_id)
                if frame is None or fid == last_id:
                    if self.source.ended:
                        break
                    continue
                last_id = fid
                t0 = time.monotonic()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                n += 1
                with self._lock:
                    if n % max(1, cfg.detect_every) == 0 or not self.tracker.tracks:
                        boxes = det.detect(frame, gray)
                        self.tracker.update(boxes)
                    now = time.monotonic()
                    todo = [t for t in self.tracker.tracks
                            if t.hits >= 2 and self.tracker.needs_recognition(t, now)]
                    for t in todo:
                        t.pending = True
                    tracks_view = [t.view() for t in self.tracker.tracks if t.hits >= 2 or t.label]
                # alignment + hand-off to ArcFace happens outside the lock
                for t in todo:
                    box = tuple(int(v) for v in t.box)
                    if legacy:
                        chip = legacy_crop(frame, box)
                    else:
                        chip = align_face(frame, box, det.find_eyes(gray, box))
                    try:
                        self._jobs.put_nowait((t.id, chip, t_cap))
                    except queue.Full:
                        t.pending = False           # ArcFace busy: try again next frame
                dt = time.monotonic() - t0
                self.stats["detect_ms"].add(dt * 1000)
                self.stats["proc_fps"].add(1.0 / max(1e-3, time.monotonic() - last_t))
                last_t = time.monotonic()
                self.stats["latency_ms"].add((time.monotonic() - t_cap) * 1000)
                with self._lock:
                    self._snapshot = (frame, tracks_view, fid)
        except Exception as e:
            self.error = f"Detection stage crashed: {e}"
            self.running = False
            raise

    # ---------------------------------------------------- stage 2: ArcFace
    def _recog_loop(self):
        cfg = self.cfg
        legacy = self.gallery.legacy
        try:
            while self.running:
                job = self._jobs.get()
                if job is None:
                    break
                jobs = [job]
                while len(jobs) < self.embedder.num_requests:     # grab whatever else is waiting
                    try:
                        j = self._jobs.get_nowait()
                    except queue.Empty:
                        break
                    if j is None:
                        self.running = False
                        break
                    jobs.append(j)
                t0 = time.monotonic()
                emb = self.embedder.embed([j[1] for j in jobs], flip=cfg.flip_tta, bgr_input=legacy)
                self.stats["recog_ms"].add((time.monotonic() - t0) * 1000 / len(jobs))
                results = self.gallery.match(emb, cfg.threshold, cfg.margin)
                with self._lock:
                    for (tid, chip, _), (name, score, _best) in zip(jobs, results):
                        t = self.tracker.add_result(tid, name, score)
                        if t and t.confirmed and not t.logged and t.label:
                            t.logged = True
                            ts = self.attendance.mark(t.label, t.score) if self.attendance else None
                            self.events.append({"name": t.label, "score": t.score, "time": ts,
                                                "new": ts is not None, "chip": chip})
        except Exception as e:
            self.error = f"Recognition stage crashed: {e}"
            self.running = False
            raise
