"""Attendance CSV: one row per person per day, kept in memory so marking is O(1)."""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime

HEADER = ["Name", "Timestamp", "Confidence"]


class AttendanceLog:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._day = None
        self._today: dict[str, str] = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(HEADER)
        self._refresh_day()

    def rows(self) -> list[list[str]]:
        out = []
        try:
            with open(self.path, "r", newline="") as f:
                for r in csv.reader(f):
                    if len(r) >= 2 and r[0].strip() and r[0] != "Name" and r[1].strip():
                        out.append([r[0], r[1], r[2] if len(r) > 2 else ""])
        except FileNotFoundError:
            pass
        return out

    def _refresh_day(self):
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._day:
            self._day = day
            self._today = {r[0]: r[1] for r in self.rows() if r[1].startswith(day)}

    def present_today(self) -> dict[str, str]:
        with self._lock:
            self._refresh_day()
            return dict(self._today)

    def mark(self, name: str, confidence: float) -> str | None:
        """Log `name` once per day. Returns the timestamp if a new row was written."""
        with self._lock:
            self._refresh_day()
            if name in self._today:
                return None
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([name, ts, f"{confidence:.3f}"])
            self._today[name] = ts
            return ts
