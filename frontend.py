"""Facial Attendance System - desktop UI (PyQt5).

Run:  python frontend.py
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import QDate, QPointF, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QLinearGradient, QPainter, QPainterPath,
                         QPen, QPixmap)
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDateEdit,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                             QSlider, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from facerec import AttendanceLog, Config, Gallery, HaarDetector, RecognitionPipeline
from facerec.config import BASE_DIR, abspath, relpath

# --------------------------------------------------------------------------- theme
C = {
    "bg": "#0e1117", "panel": "#151a23", "card": "#1b2130", "card2": "#222a3b", "border": "#2a3346",
    "text": "#e7ebf3", "muted": "#8a94a8", "accent": "#5b8cff", "accent2": "#7c5bff",
    "green": "#34d399", "amber": "#fbbf24", "red": "#f87171",
}

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial; font-size: 13px; color: {C['text']}; }}
QMainWindow, QWidget#root {{ background: {C['bg']}; }}
QFrame#sidebar {{ background: {C['panel']}; border-right: 1px solid {C['border']}; }}
QLabel#brand {{ font-size: 19px; font-weight: 700; }}
QLabel#brandSub {{ color: {C['muted']}; font-size: 11px; letter-spacing: 1px; }}
QPushButton#nav {{ text-align: left; padding: 11px 16px; border: none; border-radius: 10px;
    background: transparent; color: {C['muted']}; font-size: 14px; font-weight: 600; }}
QPushButton#nav:hover {{ background: {C['card']}; color: {C['text']}; }}
QPushButton#nav:checked {{ background: {C['card2']}; color: {C['text']}; border-left: 3px solid {C['accent']}; }}
QLabel#h1 {{ font-size: 22px; font-weight: 700; }}
QLabel#sub {{ color: {C['muted']}; }}
QLabel#muted {{ color: {C['muted']}; }}
QLabel#section {{ color: {C['muted']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QFrame#card {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 14px; }}
QPushButton {{ background: {C['card2']}; border: 1px solid {C['border']}; border-radius: 9px; padding: 9px 16px;
    font-weight: 600; }}
QPushButton:hover {{ border-color: {C['accent']}; }}
QPushButton:disabled {{ color: {C['muted']}; background: {C['card']}; }}
QPushButton#primary {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {C['accent']}, stop:1 {C['accent2']});
    border: none; color: white; }}
QPushButton#primary:hover {{ background: {C['accent']}; }}
QPushButton#danger {{ background: {C['red']}; border: none; color: #1a0e0e; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit {{ background: {C['panel']}; border: 1px solid {C['border']};
    border-radius: 8px; padding: 7px 10px; selection-background-color: {C['accent']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus {{ border-color: {C['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0px; border: none; }}
QWidget#transparent {{ background: transparent; }}
QComboBox QAbstractItemView {{ background: {C['card']}; border: 1px solid {C['border']}; selection-background-color: {C['accent']}; }}
QProgressBar {{ background: {C['panel']}; border: none; border-radius: 6px; height: 12px; text-align: center;
    color: transparent; }}
QProgressBar::chunk {{ border-radius: 6px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {C['accent']}, stop:1 {C['green']}); }}
QTableWidget {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 12px;
    gridline-color: transparent; alternate-background-color: {C['panel']}; selection-background-color: #2b3a5e; }}
QTableWidget::item {{ padding: 6px; border: none; }}
QHeaderView::section {{ background: {C['card2']}; color: {C['muted']}; border: none; padding: 9px;
    font-weight: 700; font-size: 11px; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 8px; padding: 2px; }}
QListWidget::item:selected {{ background: {C['card2']}; }}
QPlainTextEdit {{ background: #0b0e14; border: 1px solid {C['border']}; border-radius: 10px;
    font-family: Consolas, "JetBrains Mono", monospace; font-size: 12px; color: #b8c4d9; padding: 8px; }}
QSlider::groove:horizontal {{ height: 6px; background: {C['panel']}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; width: 16px; margin: -6px 0; border-radius: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {C['border']}; background: {C['panel']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QStatusBar {{ background: {C['panel']}; color: {C['muted']}; border-top: 1px solid {C['border']}; }}
QToolTip {{ background: {C['card2']}; color: {C['text']}; border: 1px solid {C['border']}; padding: 6px; }}
"""

# --------------------------------------------------------------------------- helpers

def card(layout_cls=QVBoxLayout, margins=16, spacing=10):
    f = QFrame()
    f.setObjectName("card")
    lay = layout_cls(f)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    return f, lay


def label(text, obj=None, wrap=False):
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


def avatar(name: str, size=34) -> QPixmap:
    palette = ["#5b8cff", "#7c5bff", "#34d399", "#f59e0b", "#ec4899", "#06b6d4", "#f97316", "#a3e635"]
    col = QColor(palette[sum(map(ord, name)) % len(palette)])
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(col)
    p.setPen(Qt.NoPen)
    p.drawEllipse(0, 0, size, size)
    p.setPen(QColor("white"))
    f = QFont()
    f.setBold(True)
    f.setPixelSize(int(size * 0.42))
    p.setFont(f)
    initials = "".join(w[0] for w in name.split()[:2]).upper() or "?"
    p.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, initials)
    p.end()
    return pm


def chip_pixmap(chip_bgr: np.ndarray, size=64) -> QPixmap:
    rgb = np.ascontiguousarray(chip_bgr[:, :, ::-1])
    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
    src = QPixmap.fromImage(img).scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, size, size), 10, 10)
    p.setClipPath(path)
    p.drawPixmap(0, 0, src)
    p.end()
    return pm


def app_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    g = QLinearGradient(0, 0, 64, 64)
    g.setColorAt(0, QColor(C["accent"]))
    g.setColorAt(1, QColor(C["accent2"]))
    p.setBrush(g)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, 64, 64, 16, 16)
    p.setPen(QPen(QColor("white"), 4, cap=Qt.RoundCap))
    for (x1, y1, x2, y2) in [(14, 24, 14, 14), (14, 14, 24, 14), (50, 24, 50, 14), (50, 14, 40, 14),
                             (14, 40, 14, 50), (14, 50, 24, 50), (50, 40, 50, 50), (50, 50, 40, 50)]:
        p.drawLine(x1, y1, x2, y2)
    p.setBrush(QColor("white"))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(32, 30), 7, 7)
    p.drawRoundedRect(21, 39, 22, 9, 4, 4)
    p.end()
    return QIcon(pm)


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    h, w = frame.shape[:2]
    return QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)


# --------------------------------------------------------------------------- video widget
class VideoView(QWidget):
    """Aspect-correct video with antialiased overlays painted by Qt."""

    def __init__(self, placeholder="Camera is off"):
        super().__init__()
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img: QImage | None = None
        self._frame = None
        self.tracks: list[dict] = []
        self.placeholder = placeholder
        self.placeholder_sub = ""
        self.guide = None              # (ok: bool, text) -> registration face guide
        self.guide_box = None
        self.hud = ""
        self.threshold = 0.4

    def set_frame(self, frame, tracks=None):
        self._frame = frame            # keep buffer alive for the QImage
        self._img = bgr_to_qimage(frame) if frame is not None else None
        self.tracks = tracks or []
        self.update()

    def clear(self):
        self._img, self._frame, self.tracks, self.guide, self.hud = None, None, [], None, ""
        self.update()

    def _target(self):
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), s

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        outer = QPainterPath()
        outer.addRoundedRect(QRectF(self.rect()), 16, 16)
        p.fillPath(outer, QColor("#07090d"))
        if self._img is None:
            self._paint_placeholder(p)
            return
        r, s = self._target()
        clip = QPainterPath()
        clip.addRoundedRect(r, 16, 16)
        p.setClipPath(clip)
        p.drawImage(r, self._img)
        for t in self.tracks:
            self._paint_track(p, t, r, s)
        if self.guide_box is not None:
            self._paint_guide(p, r, s)
        if self.hud:
            self._pill(p, QPointF(r.left() + 14, r.top() + 14), self.hud, QColor(0, 0, 0, 150), QColor("white"))
        p.end()

    def _paint_placeholder(self, p):
        p.setPen(QColor(C["border"]))
        cx, cy = self.width() / 2, self.height() / 2 - 20
        p.setPen(QPen(QColor(C["border"]), 3))
        p.drawEllipse(QPointF(cx, cy - 8), 26, 26)
        p.drawArc(QRectF(cx - 46, cy + 24, 92, 60), 0, 180 * 16)
        f = QFont()
        f.setPixelSize(17)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(C["text"]))
        p.drawText(QRectF(0, cy + 62, self.width(), 26), Qt.AlignCenter, self.placeholder)
        f.setPixelSize(13)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(C["muted"]))
        p.drawText(QRectF(0, cy + 90, self.width(), 22), Qt.AlignCenter, self.placeholder_sub)

    def _pill(self, p, pt, text, bg, fg, bold=True):
        f = QFont()
        f.setPixelSize(13)
        f.setBold(bold)
        p.setFont(f)
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 18, fm.height() + 8
        rect = QRectF(pt.x(), pt.y(), w, h)
        path = QPainterPath()
        path.addRoundedRect(rect, h / 2, h / 2)
        p.fillPath(path, bg)
        p.setPen(fg)
        p.drawText(rect, Qt.AlignCenter, text)
        return rect

    def _paint_track(self, p, t, r, s):
        x, y, w, h = t["box"]
        box = QRectF(r.left() + x * s, r.top() + y * s, w * s, h * s)
        if t["label"]:
            col = QColor(C["green"] if t["confirmed"] else C["amber"])
            text = t["label"] if t["confirmed"] else f'{t["label"]}?'
        elif t["unknown"]:
            col, text = QColor(C["red"]), "Unknown"
        else:
            col, text = QColor("#cbd5e1"), "Identifying…"
        # soft fill + corner brackets
        fill = QColor(col)
        fill.setAlpha(28)
        p.fillRect(box, fill)
        p.setPen(QPen(col, 1.2))
        p.drawRoundedRect(box, 8, 8)
        L = min(box.width(), box.height()) * 0.22
        p.setPen(QPen(col, 3.5, cap=Qt.RoundCap))
        for (cx, cy, dx, dy) in [(box.left(), box.top(), 1, 1), (box.right(), box.top(), -1, 1),
                                 (box.left(), box.bottom(), 1, -1), (box.right(), box.bottom(), -1, -1)]:
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx * L, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * L))
        # label pill (name + score) with a confidence bar, kept inside the video area
        has_score = bool(t["label"] or t["unknown"])
        full = f'{text}  {t["score"]:.2f}' if has_score else text
        f = QFont()
        f.setPixelSize(13)
        f.setBold(True)
        p.setFont(f)
        pw = p.fontMetrics().horizontalAdvance(full) + 18
        px = min(max(box.left(), r.left() + 4), r.right() - pw - 4)
        top = box.top() - 38 if box.top() - 38 > r.top() else box.bottom() + 6
        bg = QColor(col)
        bg.setAlpha(240)
        rect = self._pill(p, QPointF(px, top), full, bg, QColor("#0b0f17"))
        if has_score:
            frac = max(0.0, min(1.0, (t["score"] - 0.15) / 0.65))
            bar = QRectF(rect.left() + 8, rect.bottom() + 3, rect.width() - 16, 4)
            back = QPainterPath()
            back.addRoundedRect(bar, 2, 2)
            p.fillPath(back, QColor(0, 0, 0, 170))
            fg = QPainterPath()
            fg.addRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * frac, bar.height()), 2, 2)
            p.fillPath(fg, col)

    def _paint_guide(self, p, r, s):
        ok, text = self.guide
        col = QColor(C["green"] if ok else C["amber"])
        x, y, w, h = self.guide_box
        box = QRectF(r.left() + x * s, r.top() + y * s, w * s, h * s)
        el = box.adjusted(-box.width() * 0.08, -box.height() * 0.18, box.width() * 0.08, box.height() * 0.12)
        p.setPen(QPen(col, 3, Qt.DashLine if not ok else Qt.SolidLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(el)
        bg = QColor(col)
        bg.setAlpha(235)
        f = QFont()
        f.setPixelSize(13)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(text) + 18
        self._pill(p, QPointF(r.center().x() - tw / 2, r.bottom() - 46), text, bg, QColor("#0b0f17"))


class MetricCard(QFrame):
    def __init__(self, title, unit=""):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(0)
        self.title = label(title.upper(), "section")
        self.value = QLabel("–")
        self.value.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.unit = unit
        lay.addWidget(self.title)
        lay.addWidget(self.value)

    def set(self, v, color=None):
        self.value.setText(f"{v}<span style='font-size:12px;color:{C['muted']}'> {self.unit}</span>")
        if color:
            self.value.setStyleSheet(f"font-size: 22px; font-weight: 700; color: {color};")


class PersonRow(QWidget):
    def __init__(self, name, right_text, pix=None):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 5, 6, 5)
        a = QLabel()
        a.setPixmap(pix or avatar(name))
        lay.addWidget(a)
        col = QVBoxLayout()
        col.setSpacing(0)
        n = QLabel(name)
        n.setStyleSheet("font-weight: 600;")
        col.addWidget(n)
        t = QLabel(right_text)
        t.setStyleSheet(f"color: {C['muted']}; font-size: 11px;")
        col.addWidget(t)
        lay.addLayout(col, 1)


def add_row(lst: QListWidget, widget: QWidget, top=False):
    it = QListWidgetItem()
    it.setSizeHint(QSize(0, 48))
    if top:
        lst.insertItem(0, it)
    else:
        lst.addItem(it)
    lst.setItemWidget(it, widget)


# --------------------------------------------------------------------------- engine
class ModelLoader(QThread):
    done = pyqtSignal(object, str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            from facerec.embedder import ArcFaceEmbedder
            t = time.monotonic()
            emb = ArcFaceEmbedder(self.cfg.resolve_model(), self.cfg.device, self.cfg.infer_requests)
            emb.embed([np.zeros((112, 112, 3), np.uint8)])      # warm-up
            self.done.emit(emb, f"ArcFace ready on {self.cfg.device} ({time.monotonic() - t:.1f}s)")
        except Exception as e:
            self.done.emit(None, str(e))


class Engine:
    def __init__(self):
        self.cfg = Config.load()
        self.embedder = None
        self.model_msg = "Loading ArcFace model…"
        self.gallery = Gallery.load(abspath(self.cfg.gallery_path))
        self.attendance = AttendanceLog(abspath(self.cfg.attendance_csv))

    def reload_gallery(self):
        self.gallery = Gallery.load(abspath(self.cfg.gallery_path))

    def reload_attendance(self):
        self.attendance = AttendanceLog(abspath(self.cfg.attendance_csv))


def source_combo(cfg: Config) -> QComboBox:
    cb = QComboBox()
    cb.setEditable(True)
    cb.addItems(["0", "1", "rtsp://user:pass@192.168.1.10:554/stream"])
    cb.setCurrentText(str(cfg.camera_source))
    cb.setMinimumWidth(200)
    cb.setToolTip("Camera index (0, 1…), RTSP/HTTP stream URL, or a video file path")
    return cb


def parse_source(text: str):
    t = text.strip()
    return int(t) if t.isdigit() else t


# --------------------------------------------------------------------------- pages
class Page(QWidget):
    def __init__(self, title, subtitle):
        super().__init__()
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(28, 22, 28, 22)
        self.root.setSpacing(16)
        head = QHBoxLayout()
        tl = QVBoxLayout()
        tl.setSpacing(2)
        tl.addWidget(label(title, "h1"))
        tl.addWidget(label(subtitle, "sub"))
        head.addLayout(tl)
        head.addStretch(1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        head.addLayout(self.actions)
        self.root.addLayout(head)

    def on_show(self):
        pass

    def on_hide(self):
        pass


class LivePage(Page):
    def __init__(self, eng: Engine, status):
        super().__init__("Live Recognition", "Haar cascade detection and ArcFace-ResNet100 identification run in parallel threads")
        self.eng, self.status = eng, status
        self.pipe: RecognitionPipeline | None = None
        self.last_fid = 0

        self.src = source_combo(eng.cfg)
        self.actions.addWidget(label("Source", "muted"))
        self.actions.addWidget(self.src)
        self.btn = QPushButton("▶  Start")
        self.btn.setObjectName("primary")
        self.btn.setMinimumWidth(130)
        self.btn.clicked.connect(self.toggle)
        self.actions.addWidget(self.btn)

        self.banner = label("", wrap=True)
        self.banner.setStyleSheet(f"background: #3a2a0c; color: {C['amber']}; border-radius: 10px; padding: 10px 14px;")
        self.banner.hide()
        self.root.addWidget(self.banner)

        body = QHBoxLayout()
        body.setSpacing(16)
        self.view = VideoView("Camera is off")
        self.view.placeholder_sub = "Press Start to begin recognising faces"
        body.addWidget(self.view, 1)

        side = QVBoxLayout()
        side.setSpacing(12)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.m_fps = MetricCard("FPS")
        self.m_lat = MetricCard("Latency", "ms")
        self.m_det = MetricCard("Haar", "ms")
        self.m_rec = MetricCard("ArcFace", "ms/face")
        self.m_faces = MetricCard("Faces")
        self.m_present = MetricCard("Present")
        for i, m in enumerate([self.m_fps, self.m_lat, self.m_det, self.m_rec, self.m_faces, self.m_present]):
            grid.addWidget(m, i // 2, i % 2)
        side.addLayout(grid)

        pc, pl = card(spacing=6)
        pl.addWidget(label("PRESENT TODAY", "section"))
        self.present = QListWidget()
        self.present.setSelectionMode(QAbstractItemView.NoSelection)
        pl.addWidget(self.present, 1)
        side.addWidget(pc, 3)

        ac, al = card(spacing=6)
        al.addWidget(label("ACTIVITY", "section"))
        self.activity = QListWidget()
        self.activity.setSelectionMode(QAbstractItemView.NoSelection)
        al.addWidget(self.activity, 1)
        side.addWidget(ac, 2)

        sw = QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(330)
        body.addWidget(sw)
        self.root.addLayout(body, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.mtimer = QTimer(self)
        self.mtimer.timeout.connect(self.tick_metrics)
        self.refresh_present()
        self.update_banner()

    def update_banner(self):
        msgs = []
        if self.eng.embedder is None:
            msgs.append(self.eng.model_msg)
        if len(self.eng.gallery) == 0:
            msgs.append("No one is registered yet - add people on the Register page, then Train.")
        elif self.eng.gallery.legacy:
            msgs.append("model.pkl was created by the old pipeline (BGR, unaligned crops). It still works, "
                        "but re-register + Train for much better accuracy.")
        self.banner.setText("   •   ".join(msgs))
        self.banner.setVisible(bool(msgs))

    def refresh_present(self):
        self.present.clear()
        today = self.eng.attendance.present_today()
        for name, ts in sorted(today.items(), key=lambda kv: kv[1], reverse=True):
            add_row(self.present, PersonRow(name, f"Checked in at {ts[11:16]}"))
        self.m_present.set(f"{len(today)}", C["green"])
        if not today:
            it = QListWidgetItem("No check-ins yet today")
            it.setForeground(QColor(C["muted"]))
            self.present.addItem(it)

    def toggle(self):
        if self.pipe:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.eng.embedder is None:
            QMessageBox.warning(self, "Model not ready", self.eng.model_msg)
            return
        self.eng.cfg.camera_source = self.src.currentText().strip()
        try:
            self.pipe = RecognitionPipeline(self.eng.cfg, self.eng.embedder, self.eng.gallery, self.eng.attendance)
            self.pipe.start(parse_source(self.src.currentText()))
        except Exception as e:
            self.pipe = None
            QMessageBox.critical(self, "Camera error",
                                 f"{e}\n\nCheck that the camera is connected, not used by another app, "
                                 "and that the source index / URL is correct.")
            return
        self.view.threshold = self.eng.cfg.threshold
        self.btn.setText("■  Stop")
        self.btn.setObjectName("danger")
        self.btn.setStyle(self.btn.style())
        self.timer.start(10)
        self.mtimer.start(300)
        self.status(f"Recognition running on source {self.src.currentText()}")

    def stop(self):
        self.timer.stop()
        self.mtimer.stop()
        if self.pipe:
            self.pipe.stop()
            self.pipe = None
        self.view.clear()
        self.view.placeholder, self.view.placeholder_sub = "Camera is off", "Press Start to begin recognising faces"
        self.btn.setText("▶  Start")
        self.btn.setObjectName("primary")
        self.btn.setStyle(self.btn.style())
        for m in (self.m_fps, self.m_lat, self.m_det, self.m_rec, self.m_faces):
            m.value.setText("–")
        self.status("Recognition stopped")

    def tick(self):
        p = self.pipe
        if not p:
            return
        if p.error or p.finished or not p.running:
            err = p.error
            self.stop()
            if err:
                QMessageBox.critical(self, "Pipeline error", err)
            return
        frame, tracks, fid = p.snapshot()
        if frame is not None and fid != self.last_fid:
            self.last_fid = fid
            self.view.set_frame(frame, tracks)
            self.m_faces.set(str(len(tracks)))
        changed = False
        while p.events:
            e = p.events.popleft()
            when = time.strftime("%H:%M:%S")
            txt = f"{'Checked in' if e['new'] else 'Seen again'} · {when} · {e['score']:.2f}"
            add_row(self.activity, PersonRow(e["name"], txt, chip_pixmap(e["chip"], 34)), top=True)
            if self.activity.count() > 60:
                self.activity.takeItem(self.activity.count() - 1)
            changed |= e["new"]
            if e["new"]:
                self.status(f"✔ {e['name']} marked present")
        if changed:
            self.refresh_present()

    def tick_metrics(self):
        if not self.pipe:
            return
        m = self.pipe.metrics()
        fps = m["proc_fps"]
        self.m_fps.set(f"{fps:.0f}", C["green"] if fps >= 20 else C["amber"])
        self.m_lat.set(f"{m['latency_ms']:.0f}", C["green"] if m["latency_ms"] < 50 else C["amber"])
        self.m_det.set(f"{m['detect_ms']:.1f}")
        self.m_rec.set(f"{m['recog_ms']:.0f}" if m["recog_ms"] else "–")

    def on_hide(self):
        pass


class RegisterWorker(QThread):
    frame = pyqtSignal(object, object, str, bool)
    captured = pyqtSignal(object, int, int)
    finished_capture = pyqtSignal(str, int)
    error = pyqtSignal(str)

    def __init__(self, cfg: Config, source):
        super().__init__()
        self.cfg, self.source = cfg, source
        self.running = True
        self.enroller = None
        self.pending = None

    def begin(self, person, count):
        self.pending = (person, count)

    def cancel(self):
        self.enroller = None

    def run(self):
        from facerec.enroll import Enroller
        det = HaarDetector(self.cfg.detect_width, self.cfg.scale_factor, self.cfg.min_neighbors, self.cfg.min_face_frac)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error.emit(f"Could not open camera source {self.source!r}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if self.pending:
                person, count = self.pending
                self.pending = None
                self.enroller = Enroller(det, abspath(self.cfg.dataset_dir), person, count)
            enr = self.enroller
            if enr:
                box, status, ok_q, chip = enr.process(frame)
                if chip is not None:
                    self.captured.emit(chip, enr.count, enr.target)
                if enr.done:
                    self.enroller = None
                    self.finished_capture.emit(enr.person, enr.count)
            else:
                box, status, ok_q, _ = Enroller.check(det, frame)
            self.frame.emit(frame, box, status, ok_q)
        cap.release()

    def stop(self):
        self.running = False
        self.wait(2000)


class RegisterPage(Page):
    request_train = pyqtSignal()

    def __init__(self, eng: Engine, status):
        super().__init__("Register Person", "Captures aligned, sharp and varied face samples automatically")
        self.eng, self.status = eng, status
        self.worker: RegisterWorker | None = None

        self.src = source_combo(eng.cfg)
        self.actions.addWidget(label("Source", "muted"))
        self.actions.addWidget(self.src)
        self.cam_btn = QPushButton("Restart camera")
        self.cam_btn.clicked.connect(self.restart_camera)
        self.actions.addWidget(self.cam_btn)

        body = QHBoxLayout()
        body.setSpacing(16)
        self.view = VideoView("Starting camera…")
        body.addWidget(self.view, 1)

        side = QVBoxLayout()
        side.setSpacing(12)
        fc, fl = card()
        fl.addWidget(label("PERSON", "section"))
        self.name = QLineEdit()
        self.name.setPlaceholderText("Full name, e.g. Garvit Sharma")
        fl.addWidget(self.name)
        row = QHBoxLayout()
        row.addWidget(label("Samples", "muted"))
        self.count = QSpinBox()
        self.count.setRange(10, 200)
        self.count.setValue(eng.cfg.samples_per_person)
        row.addWidget(self.count)
        fl.addLayout(row)
        self.auto_train = QCheckBox("Train model automatically when done")
        self.auto_train.setChecked(True)
        fl.addWidget(self.auto_train)
        self.go = QPushButton("●  Start capture")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self.toggle_capture)
        fl.addWidget(self.go)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        fl.addWidget(self.progress)
        self.progress_lbl = label("0 / 0", "muted")
        fl.addWidget(self.progress_lbl)
        side.addWidget(fc)

        tc, tl = card()
        tl.addWidget(label("TIPS", "section"))
        tl.addWidget(label("• Face the camera in good, even light\n• Slowly turn your head left/right and up/down\n"
                           "• Try a neutral face and a smile\n• Remove sunglasses; glasses can stay on", "muted", wrap=True))
        side.addWidget(tc)

        gc, gl = card(spacing=6)
        gl.addWidget(label("CAPTURED SAMPLES", "section"))
        self.thumbs = QListWidget()
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(56, 56))
        self.thumbs.setResizeMode(QListWidget.Adjust)
        self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setSpacing(4)
        gl.addWidget(self.thumbs, 1)
        side.addWidget(gc, 1)

        sw = QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(330)
        body.addWidget(sw)
        self.root.addLayout(body, 1)

    def on_show(self):
        self.start_camera()

    def on_hide(self):
        self.stop_camera()

    def start_camera(self):
        if self.worker:
            return
        self.eng.cfg.camera_source = self.src.currentText().strip()
        self.worker = RegisterWorker(self.eng.cfg, parse_source(self.src.currentText()))
        self.worker.frame.connect(self.on_frame)
        self.worker.captured.connect(self.on_captured)
        self.worker.finished_capture.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.view.clear()
        self.view.placeholder = "Camera is off"
        self.set_capturing(False)

    def restart_camera(self):
        self.stop_camera()
        self.start_camera()

    def on_error(self, msg):
        self.view.placeholder, self.view.placeholder_sub = "Camera unavailable", msg
        self.view.update()
        self.worker = None

    def on_frame(self, frame, box, status, ok):
        self.view.guide = (ok, status)
        self.view.guide_box = box
        if box is None:
            h, w = frame.shape[:2]
            self.view.guide_box = (int(w * 0.36), int(h * 0.22), int(w * 0.28), int(h * 0.5))
        self.view.set_frame(frame)

    def set_capturing(self, on):
        self.go.setText("■  Stop capture" if on else "●  Start capture")
        self.go.setObjectName("danger" if on else "primary")
        self.go.setStyle(self.go.style())
        self.name.setEnabled(not on)
        self.count.setEnabled(not on)

    def toggle_capture(self):
        if not self.worker:
            self.start_camera()
        if self.worker.enroller or self.worker.pending:
            self.worker.cancel()
            self.set_capturing(False)
            self.status("Capture stopped")
            return
        from facerec.enroll import safe_name
        person = safe_name(self.name.text())
        if not person:
            QMessageBox.warning(self, "Name required", "Enter the person's name first.")
            return
        existing = os.path.join(abspath(self.eng.cfg.dataset_dir), person)
        if os.path.isdir(existing) and os.listdir(existing):
            r = QMessageBox.question(self, "Person exists",
                                     f"'{person}' already has samples.\n\nYes = add more samples\nNo = replace them",
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                return
            if r == QMessageBox.No:
                shutil.rmtree(existing)
        self.thumbs.clear()
        self.progress.setValue(0)
        self.progress_lbl.setText(f"0 / {self.count.value()}")
        self.worker.begin(person, self.count.value())
        self.set_capturing(True)
        self.status(f"Capturing samples for {person}…")

    def on_captured(self, chip, n, total):
        self.progress.setValue(int(100 * n / total))
        self.progress_lbl.setText(f"{n} / {total}")
        it = QListWidgetItem(QIcon(chip_pixmap(chip, 56)), "")
        self.thumbs.insertItem(0, it)

    def on_done(self, person, n):
        self.set_capturing(False)
        self.status(f"Captured {n} samples for {person}")
        if self.auto_train.isChecked():
            self.request_train.emit()
        else:
            QMessageBox.information(self, "Registration complete",
                                    f"Captured {n} samples for {person}.\nGo to Train to update the model.")


class TrainWorker(QThread):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, eng: Engine):
        super().__init__()
        self.eng = eng

    def run(self):
        from facerec.enroll import build_gallery
        try:
            t = time.monotonic()
            self.log.emit(f"Dataset: {abspath(self.eng.cfg.dataset_dir)}")
            g = build_gallery(abspath(self.eng.cfg.dataset_dir), self.eng.embedder,
                              progress=lambda v, s: self.progress.emit(v, s), log=self.log.emit)
            path = abspath(self.eng.cfg.gallery_path)
            if os.path.exists(path):
                shutil.copy2(path, path + ".bak")
            g.save(path)
            self.progress.emit(100, "Done")
            self.done.emit(True, f"Model saved: {len(g.people)} people, {len(g.names)} embeddings "
                                 f"in {time.monotonic() - t:.1f}s")
        except Exception as e:
            self.done.emit(False, str(e))


class TrainPage(Page):
    trained = pyqtSignal()

    def __init__(self, eng: Engine, status):
        super().__init__("Train Model", "Builds ArcFace embeddings for everyone in the dataset folder")
        self.eng, self.status = eng, status
        self.worker = None
        self.open_btn = QPushButton("Open dataset folder")
        self.open_btn.clicked.connect(lambda: QDesktopServicesOpen(abspath(self.eng.cfg.dataset_dir)))
        self.actions.addWidget(self.open_btn)
        self.train_btn = QPushButton("⟳  Train model")
        self.train_btn.setObjectName("primary")
        self.train_btn.clicked.connect(self.train)
        self.actions.addWidget(self.train_btn)

        top = QHBoxLayout()
        top.setSpacing(12)
        self.c_people = MetricCard("People in dataset")
        self.c_images = MetricCard("Images")
        self.c_model = MetricCard("People in model")
        self.c_state = MetricCard("Model status")
        for c in (self.c_people, self.c_images, self.c_model, self.c_state):
            top.addWidget(c)
        self.root.addLayout(top)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "PERSON", "IMAGES", "IN MODEL", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 64)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (2, 3):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(4, 100)
        self.table.verticalHeader().setDefaultSectionSize(58)
        self.root.addWidget(self.table, 3)

        self.progress = QProgressBar()
        self.root.addWidget(self.progress)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumHeight(170)
        self.root.addWidget(self.logbox, 1)

    def on_show(self):
        self.refresh()

    def refresh(self):
        from facerec.enroll import dataset_summary
        ds = dataset_summary(abspath(self.eng.cfg.dataset_dir))
        counts = self.eng.gallery.counts()
        names = sorted(set(ds) | set(counts))
        self.table.setRowCount(0)
        for r, n in enumerate(names):
            self.table.insertRow(r)
            thumb = QLabel()
            thumb.setAlignment(Qt.AlignCenter)
            img = cv2.imread(ds[n][len(ds[n]) // 2]) if n in ds else None
            thumb.setPixmap(chip_pixmap(cv2.resize(img, (112, 112)), 46) if img is not None else avatar(n, 40))
            self.table.setCellWidget(r, 0, thumb)
            self.table.setItem(r, 1, QTableWidgetItem(n))
            self.table.setItem(r, 2, QTableWidgetItem(str(len(ds.get(n, [])))))
            inm = counts.get(n, 0)
            it = QTableWidgetItem(f"{inm} embeddings" if inm else "not trained")
            it.setForeground(QColor(C["green"] if inm else C["amber"]))
            self.table.setItem(r, 3, it)
            if n in ds:
                b = QPushButton("Delete")
                b.setStyleSheet(f"padding: 5px 10px; margin: 10px 8px; color: {C['red']};")
                b.clicked.connect(lambda _, name=n: self.delete(name))
                self.table.setCellWidget(r, 4, b)
        self.c_people.set(str(len(ds)))
        self.c_images.set(str(sum(len(v) for v in ds.values())))
        self.c_model.set(str(len(self.eng.gallery)))
        stale = set(ds) != set(counts) or self.eng.gallery.legacy
        if self.eng.gallery.legacy:
            self.c_state.set("Legacy", C["amber"])
        elif not counts:
            self.c_state.set("Empty", C["amber"])
        else:
            self.c_state.set("Needs training" if stale else "Up to date", C["amber"] if stale else C["green"])
        self.train_btn.setEnabled(self.eng.embedder is not None and self.worker is None)

    def delete(self, name):
        if QMessageBox.question(self, "Delete person",
                                f"Delete all samples of '{name}'? Retrain afterwards to remove them from the model.") \
                == QMessageBox.Yes:
            shutil.rmtree(os.path.join(abspath(self.eng.cfg.dataset_dir), name), ignore_errors=True)
            self.refresh()

    def train(self):
        if self.eng.embedder is None:
            QMessageBox.warning(self, "Model not ready", self.eng.model_msg)
            return
        self.logbox.clear()
        self.progress.setValue(0)
        self.train_btn.setEnabled(False)
        self.worker = TrainWorker(self.eng)
        self.worker.progress.connect(lambda v, s: (self.progress.setValue(v), self.status(s)))
        self.worker.log.connect(self.logbox.appendPlainText)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def on_done(self, ok, msg):
        self.worker = None
        self.logbox.appendPlainText(("✔ " if ok else "✖ ") + msg)
        self.status(msg)
        if ok:
            self.eng.reload_gallery()
            self.trained.emit()
        else:
            QMessageBox.critical(self, "Training failed", msg)
        self.refresh()


def QDesktopServicesOpen(path):
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices
    os.makedirs(path, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class AttendancePage(Page):
    def __init__(self, eng: Engine, status):
        super().__init__("Attendance", "One check-in per person per day, with match confidence")
        self.eng, self.status = eng, status
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search name…")
        self.search.setMinimumWidth(200)
        self.search.textChanged.connect(self.apply)
        self.actions.addWidget(self.search)
        self.all_dates = QCheckBox("All dates")
        self.all_dates.setChecked(False)
        self.all_dates.toggled.connect(self.apply)
        self.actions.addWidget(self.all_dates)
        self.date = QDateEdit(QDate.currentDate())
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("dd MMM yyyy")
        self.date.dateChanged.connect(self.apply)
        self.actions.addWidget(self.date)
        rb = QPushButton("Refresh")
        rb.clicked.connect(self.reload)
        self.actions.addWidget(rb)
        eb = QPushButton("Export CSV")
        eb.setObjectName("primary")
        eb.clicked.connect(self.export)
        self.actions.addWidget(eb)

        top = QHBoxLayout()
        top.setSpacing(12)
        self.c_shown = MetricCard("Records shown")
        self.c_people = MetricCard("Unique people")
        self.c_today = MetricCard("Present today")
        self.c_days = MetricCard("Days logged")
        for c in (self.c_shown, self.c_people, self.c_today, self.c_days):
            top.addWidget(c)
        self.root.addLayout(top)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["NAME", "DATE", "TIME", "CONFIDENCE"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.root.addWidget(self.table, 1)
        self.rows = []

    def on_show(self):
        self.reload()

    def reload(self):
        self.rows = self.eng.attendance.rows()
        self.apply()

    def apply(self):
        q = self.search.text().strip().lower()
        day = None if self.all_dates.isChecked() else self.date.date().toString("yyyy-MM-dd")
        self.date.setEnabled(day is not None)
        rows = [r for r in self.rows if (not q or q in r[0].lower()) and (day is None or r[1].startswith(day))]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for i, (n, ts, conf) in enumerate(rows):
            name_item = QTableWidgetItem(QIcon(avatar(n, 26)), "  " + n)
            self.table.setItem(i, 0, name_item)
            self.table.setItem(i, 1, QTableWidgetItem(ts[:10]))
            self.table.setItem(i, 2, QTableWidgetItem(ts[11:19]))
            ci = QTableWidgetItem(conf or "–")
            try:
                v = float(conf)
                ci.setForeground(QColor(C["green"] if v >= 0.5 else C["amber"]))
            except ValueError:
                ci.setForeground(QColor(C["muted"]))
            self.table.setItem(i, 3, ci)
        self.table.setSortingEnabled(True)
        self.table.sortItems(1, Qt.DescendingOrder)
        self.c_shown.set(str(len(rows)))
        self.c_people.set(str(len({r[0] for r in rows})))
        self.c_today.set(str(len(self.eng.attendance.present_today())), C["green"])
        self.c_days.set(str(len({r[1][:10] for r in self.rows})))

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export attendance",
                                              os.path.join(BASE_DIR, f"attendance_{time.strftime('%Y%m%d')}.csv"),
                                              "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Name", "Date", "Time", "Confidence"])
            for r in range(self.table.rowCount()):
                w.writerow([self.table.item(r, 0).text().strip(), self.table.item(r, 1).text(),
                            self.table.item(r, 2).text(), self.table.item(r, 3).text()])
        self.status(f"Exported {self.table.rowCount()} rows to {path}")


class SettingsPage(Page):
    saved = pyqtSignal(bool)        # True when the model must be reloaded

    def __init__(self, eng: Engine, status):
        super().__init__("Settings", "Model, camera, detection and decision parameters")
        self.eng, self.status = eng, status
        sb = QPushButton("Save settings")
        sb.setObjectName("primary")
        sb.clicked.connect(self.save)
        self.actions.addWidget(sb)
        rb = QPushButton("Reset to defaults")
        rb.clicked.connect(self.reset)
        self.actions.insertWidget(0, rb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setObjectName("transparent")
        scroll.viewport().setStyleSheet("background: transparent;")
        grid = QGridLayout(inner)
        grid.setSpacing(16)
        grid.setContentsMargins(0, 0, 8, 0)
        scroll.setWidget(inner)
        self.root.addWidget(scroll, 1)

        def group(title, r, c):
            f, lay = card()
            lay.addWidget(label(title, "section"))
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignLeft)
            form.setHorizontalSpacing(18)
            form.setVerticalSpacing(10)
            lay.addLayout(form)
            lay.addStretch(1)
            grid.addWidget(f, r, c)
            return form

        def path_row(form, title, filt=None, is_dir=False, save=False):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            e = QLineEdit()
            b = QPushButton("Browse")
            b.setFixedWidth(78)

            def pick():
                start = abspath(e.text()) or BASE_DIR
                if is_dir:
                    p = QFileDialog.getExistingDirectory(self, title, start)
                elif save:
                    p, _ = QFileDialog.getSaveFileName(self, title, start, filt)
                else:
                    p, _ = QFileDialog.getOpenFileName(self, title, start, filt)
                if p:
                    e.setText(relpath(p))
            b.clicked.connect(pick)
            h.addWidget(e, 1)
            h.addWidget(b)
            form.addRow(title, w)
            return e

        f = group("MODEL", 0, 0)
        self.model = path_row(f, "ArcFace model", "OpenVINO / ONNX (*.xml *.onnx)")
        self.gallery = path_row(f, "Face database", "Pickle (*.pkl)", save=True)
        self.device = QComboBox()
        self.device.addItems(["CPU", "GPU", "AUTO", "NPU"])
        f.addRow("Inference device", self.device)
        self.requests = QSpinBox()
        self.requests.setRange(1, 8)
        self.requests.setToolTip("How many faces ArcFace embeds concurrently")
        f.addRow("Parallel requests", self.requests)

        f = group("DATA", 0, 1)
        self.dataset = path_row(f, "Dataset folder", is_dir=True)
        self.att = path_row(f, "Attendance CSV", "CSV (*.csv)", save=True)
        self.camera = QLineEdit()
        f.addRow("Default camera", self.camera)
        self.cam_w = QSpinBox()
        self.cam_w.setRange(320, 3840)
        self.cam_h = QSpinBox()
        self.cam_h.setRange(240, 2160)
        res = QWidget()
        rl = QHBoxLayout(res)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.cam_w)
        rl.addWidget(label("×", "muted"))
        rl.addWidget(self.cam_h)
        f.addRow("Resolution", res)

        f = group("HAAR DETECTION", 1, 0)
        self.det_w = QSpinBox()
        self.det_w.setRange(240, 1280)
        self.det_w.setSingleStep(40)
        self.det_w.setToolTip("Frame is downscaled to this width before Haar. Lower = faster, higher = smaller faces")
        f.addRow("Detection width", self.det_w)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(1.03, 1.5)
        self.scale.setSingleStep(0.01)
        f.addRow("Scale factor", self.scale)
        self.neigh = QSpinBox()
        self.neigh.setRange(2, 12)
        self.neigh.setToolTip("Higher = fewer false detections")
        f.addRow("Min neighbours", self.neigh)
        self.minface = QDoubleSpinBox()
        self.minface.setRange(0.02, 0.4)
        self.minface.setSingleStep(0.01)
        f.addRow("Min face size (of width)", self.minface)

        f = group("RECOGNITION", 1, 1)
        thr = QWidget()
        tl = QHBoxLayout(thr)
        tl.setContentsMargins(0, 0, 0, 0)
        self.thr = QSlider(Qt.Horizontal)
        self.thr.setRange(20, 80)
        self.thr_lbl = QLabel()
        self.thr_lbl.setFixedWidth(40)
        self.thr.valueChanged.connect(lambda v: self.thr_lbl.setText(f"{v / 100:.2f}"))
        tl.addWidget(self.thr, 1)
        tl.addWidget(self.thr_lbl)
        f.addRow("Match threshold", thr)
        hint = label("Higher = stricter (fewer false matches, more 'Unknown')", "muted", wrap=True)
        hint.setStyleSheet(f"color:{C['muted']}; font-size: 11px;")
        f.addRow("", hint)
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 0.3)
        self.margin.setSingleStep(0.01)
        self.margin.setToolTip("Best person must beat the runner-up by at least this much")
        f.addRow("Identity margin", self.margin)
        self.votes = QSpinBox()
        self.votes.setRange(1, 8)
        self.votes.setToolTip("Consistent frames required before a name is confirmed and attendance is logged")
        f.addRow("Votes to confirm", self.votes)
        self.reverify = QDoubleSpinBox()
        self.reverify.setRange(0.2, 10)
        self.reverify.setSuffix(" s")
        f.addRow("Re-verify every", self.reverify)
        self.flip = QCheckBox("Mirror test-time augmentation (2× ArcFace cost)")
        f.addRow("", self.flip)
        grid.setRowStretch(2, 1)
        self.load()

    def load(self):
        c = self.eng.cfg
        self.model.setText(c.arcface_model)
        self.gallery.setText(c.gallery_path)
        self.device.setCurrentText(c.device)
        self.requests.setValue(c.infer_requests)
        self.dataset.setText(c.dataset_dir)
        self.att.setText(c.attendance_csv)
        self.camera.setText(str(c.camera_source))
        self.cam_w.setValue(c.camera_width)
        self.cam_h.setValue(c.camera_height)
        self.det_w.setValue(c.detect_width)
        self.scale.setValue(c.scale_factor)
        self.neigh.setValue(c.min_neighbors)
        self.minface.setValue(c.min_face_frac)
        self.thr.setValue(int(round(c.threshold * 100)))
        self.thr_lbl.setText(f"{c.threshold:.2f}")
        self.margin.setValue(c.margin)
        self.votes.setValue(c.votes_to_confirm)
        self.reverify.setValue(c.reverify_seconds)
        self.flip.setChecked(c.flip_tta)

    def reset(self):
        keep = {k: getattr(self.eng.cfg, k) for k in ("arcface_model", "gallery_path", "dataset_dir", "attendance_csv")}
        self.eng.cfg = Config(**keep)
        self.load()

    def save(self):
        c = self.eng.cfg
        reload_model = (self.model.text() != c.arcface_model or self.device.currentText() != c.device
                        or self.requests.value() != c.infer_requests)
        c.arcface_model, c.gallery_path = self.model.text().strip(), self.gallery.text().strip()
        c.device, c.infer_requests = self.device.currentText(), self.requests.value()
        c.dataset_dir, c.attendance_csv = self.dataset.text().strip(), self.att.text().strip()
        c.camera_source = self.camera.text().strip() or "0"
        c.camera_width, c.camera_height = self.cam_w.value(), self.cam_h.value()
        c.detect_width, c.scale_factor = self.det_w.value(), self.scale.value()
        c.min_neighbors, c.min_face_frac = self.neigh.value(), self.minface.value()
        c.threshold, c.margin = self.thr.value() / 100, self.margin.value()
        c.votes_to_confirm, c.reverify_seconds = self.votes.value(), self.reverify.value()
        c.flip_tta = self.flip.isChecked()
        try:
            c.save()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings: {e}")
            return
        self.status("Settings saved")
        self.saved.emit(reload_model)


# --------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceAttend — Facial Attendance System")
        self.setWindowIcon(app_icon())
        self.resize(1400, 860)
        self.setMinimumSize(1100, 700)
        self.eng = Engine()

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(232)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(16, 22, 16, 18)
        sl.setSpacing(6)
        brand = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(app_icon().pixmap(38, 38))
        brand.addWidget(ic)
        bt = QVBoxLayout()
        bt.setSpacing(0)
        bt.addWidget(label("FaceAttend", "brand"))
        bt.addWidget(label("HAAR × ARCFACE", "brandSub"))
        brand.addLayout(bt)
        brand.addStretch(1)
        sl.addLayout(brand)
        sl.addSpacing(22)

        self.stack = QStackedWidget()
        st = self.set_status
        self.live = LivePage(self.eng, st)
        self.reg = RegisterPage(self.eng, st)
        self.train = TrainPage(self.eng, st)
        self.att = AttendancePage(self.eng, st)
        self.settings = SettingsPage(self.eng, st)
        self.pages = [self.live, self.reg, self.train, self.att, self.settings]
        names = ["◉   Live", "＋  Register", "⟳   Train", "☰   Attendance", "⚙   Settings"]
        self.nav = QButtonGroup(self)
        for i, (p, n) in enumerate(zip(self.pages, names)):
            self.stack.addWidget(p)
            b = QPushButton(n)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.nav.addButton(b, i)
            sl.addWidget(b)
        self.nav.button(0).setChecked(True)
        self.nav.idClicked.connect(self.go)
        sl.addStretch(1)

        ec, el = card(margins=12, spacing=4)
        el.addWidget(label("ENGINE", "section"))
        self.engine_lbl = label("Loading model…", wrap=True)
        self.engine_lbl.setStyleSheet(f"color: {C['amber']}; font-weight: 600;")
        el.addWidget(self.engine_lbl)
        self.gallery_lbl = label("", "muted", wrap=True)
        el.addWidget(self.gallery_lbl)
        sl.addWidget(ec)

        h.addWidget(side)
        h.addWidget(self.stack, 1)
        self.statusBar().showMessage("Ready")
        self.current = 0

        self.reg.request_train.connect(lambda: (self.go(2), self.nav.button(2).setChecked(True), self.train.train()))
        self.train.trained.connect(self.on_trained)
        self.settings.saved.connect(self.on_settings)
        self.update_gallery_label()
        self.load_model()

    def set_status(self, msg):
        self.statusBar().showMessage(msg, 8000)

    def update_gallery_label(self):
        g = self.eng.gallery
        txt = f"{len(g)} people · {len(g.names)} embeddings"
        if g.legacy:
            txt += "\n⚠ legacy model — retrain"
        self.gallery_lbl.setText(txt)

    def load_model(self):
        self.eng.embedder = None
        self.eng.model_msg = "Loading ArcFace model…"
        self.engine_lbl.setText("Loading ArcFace…")
        self.engine_lbl.setStyleSheet(f"color: {C['amber']}; font-weight: 600;")
        self.loader = ModelLoader(self.eng.cfg)
        self.loader.done.connect(self.on_model)
        self.loader.start()

    def on_model(self, emb, msg):
        self.eng.embedder = emb
        self.eng.model_msg = msg
        if emb is None:
            self.engine_lbl.setText("Model failed to load")
            self.engine_lbl.setStyleSheet(f"color: {C['red']}; font-weight: 600;")
            self.engine_lbl.setToolTip(msg)
            self.set_status("ArcFace model could not be loaded - see Settings")
            QMessageBox.warning(self, "ArcFace model", msg)
        else:
            self.engine_lbl.setText(f"● ArcFace ready · {self.eng.cfg.device}")
            self.engine_lbl.setStyleSheet(f"color: {C['green']}; font-weight: 600;")
            self.set_status(msg)
        self.live.update_banner()
        self.train.train_btn.setEnabled(emb is not None)

    def on_trained(self):
        self.update_gallery_label()
        self.live.update_banner()
        if self.live.pipe:                     # hot-swap the new gallery
            self.live.stop()
            self.live.start()

    def on_settings(self, reload_model):
        was_live = self.live.pipe is not None
        self.live.stop()
        self.eng.reload_gallery()
        self.eng.reload_attendance()
        self.update_gallery_label()
        self.live.refresh_present()
        if reload_model:
            self.load_model()
        elif was_live:
            self.live.start()

    def go(self, idx):
        if idx == self.current:
            return
        # the camera can only be owned by one page at a time
        if idx == 1 and self.live.pipe:
            self.live.stop()
            self.set_status("Live recognition paused while registering")
        self.pages[self.current].on_hide()
        self.current = idx
        self.stack.setCurrentIndex(idx)
        self.pages[idx].on_show()

    def closeEvent(self, e):
        self.live.stop()
        self.reg.stop_camera()
        if self.train.worker:
            self.train.worker.wait(10000)
        e.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    app.setWindowIcon(app_icon())
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
