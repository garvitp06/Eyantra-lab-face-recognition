"""Command-line entry point (the GUI is frontend.py).

    python main.py run                       # live recognition in an OpenCV window
    python main.py run --source rtsp://...   # IP camera
    python main.py register "Alice"          # capture samples for a person
    python main.py train                     # rebuild model/model.pkl from dataset/
    python main.py bench --source video.mp4  # print latency numbers, no window
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from facerec import AttendanceLog, Config, RecognitionPipeline
from facerec.config import abspath


def draw(frame, tracks):
    for t in tracks:
        x, y, w, h = t["box"]
        if t["label"]:
            color = (80, 200, 120) if t["confirmed"] else (0, 190, 255)
            text = f'{t["label"]} {t["score"]:.2f}' + ("" if t["confirmed"] else " ?")
        elif t["unknown"]:
            color, text = (80, 80, 240), f'Unknown {t["score"]:.2f}'
        else:
            color, text = (200, 200, 200), "..."
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
        cv2.putText(frame, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def cmd_run(cfg, args, show=True):
    pipe = RecognitionPipeline(cfg, attendance=AttendanceLog(abspath(cfg.attendance_csv)))
    if pipe.gallery.legacy:
        print("[warn] model.pkl was built by the old pipeline - run `python main.py train` for full accuracy.")
    print(f"Loaded {len(pipe.gallery)} people: {', '.join(pipe.gallery.people) or '-'}")
    src = args.source if args.source is not None else cfg.camera()
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    pipe.start(src)
    last_fid, t_end = 0, time.monotonic() + (args.seconds or 1e9)
    try:
        while time.monotonic() < t_end and pipe.running and not pipe.finished:
            while pipe.events:
                e = pipe.events.popleft()
                print(f"[attendance] {e['name']} ({e['score']:.2f}) " + ("logged " + e["time"] if e["new"] else "already present"))
            frame, tracks, fid = pipe.snapshot()
            if show:
                if frame is not None and fid != last_fid:
                    last_fid = fid
                    m = pipe.metrics()
                    out = draw(frame.copy(), tracks)
                    cv2.putText(out, f'{m["proc_fps"]:.0f} fps | haar {m["detect_ms"]:.1f} ms | arcface {m["recog_ms"]:.1f} ms/face',
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Face Recognition  (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
    finally:
        m = pipe.metrics()
        pipe.stop()
        if show:
            cv2.destroyAllWindows()
    if pipe.error:
        print(pipe.error)
    return m


def cmd_bench(cfg, args):
    m = cmd_run(cfg, args, show=False)
    print("\n--- pipeline metrics (moving averages) ---")
    for k in ("camera_fps", "proc_fps", "detect_ms", "recog_ms", "latency_ms"):
        print(f"{k:12s} {m[k]:8.2f}")


def cmd_register(cfg, args):
    from facerec.detector import HaarDetector
    from facerec.enroll import Enroller
    det = HaarDetector(cfg.detect_width, cfg.scale_factor, cfg.min_neighbors, cfg.min_face_frac)
    enr = Enroller(det, abspath(cfg.dataset_dir), args.name, args.count or cfg.samples_per_person)
    src = cfg.camera() if args.source is None else (int(args.source) if args.source.isdigit() else args.source)
    cap = cv2.VideoCapture(src)
    while not enr.done:
        ok, frame = cap.read()
        if not ok:
            break
        box, status, good, _ = enr.process(frame)
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 120) if good else (0, 160, 255), 2)
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Register (q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {enr.count} samples to {enr.dir}. Now run: python main.py train")


def cmd_train(cfg, args):
    from facerec.embedder import ArcFaceEmbedder
    from facerec.enroll import build_gallery
    emb = ArcFaceEmbedder(cfg.resolve_model(), cfg.device, cfg.infer_requests)
    g = build_gallery(abspath(cfg.dataset_dir), emb)
    g.save(abspath(cfg.gallery_path))
    print(f"Saved {abspath(cfg.gallery_path)}: {g.counts()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--source", help="camera index, RTSP URL or video file")
        p.add_argument("--seconds", type=float, help="stop after N seconds")
    p = sub.add_parser("register")
    p.add_argument("name")
    p.add_argument("--count", type=int)
    p.add_argument("--source")
    sub.add_parser("train")
    for p in sub.choices.values():
        p.add_argument("--device", help="CPU / GPU / AUTO")
        p.add_argument("--threshold", type=float)
    args = ap.parse_args(argv)
    cfg = Config.load()
    if args.device:
        cfg.device = args.device
    if args.threshold:
        cfg.threshold = args.threshold
    {"run": cmd_run, "bench": cmd_bench, "register": cmd_register, "train": cmd_train}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
