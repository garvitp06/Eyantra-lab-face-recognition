# FaceAttend: face recognition attendance (Haar × ArcFace)

Real-time face recognition attendance system. A **Haar cascade** detects faces and
**ArcFace-ResNet100** (OpenVINO) identifies them. The two run **in parallel pipeline
threads**, so detection never waits for recognition.

```
camera thread ──newest frame──▶ Haar thread (detect + track + align) ──face chips──▶ ArcFace thread
     (drops stale frames)              │                                         (async OpenVINO requests)
                                       └───────── snapshot ◀──── identity votes ────────┘
                                                      │
                                               PyQt5 UI / CLI
```

## Why it's fast
* The capture thread keeps **only the newest frame**, so lag never accumulates (important for RTSP).
* Haar runs on a **480 px downscaled, equalised** frame, about 10× cheaper than full-res.
* ArcFace runs **only when a track needs an identity**: a few times when a face appears, then a
  re-check every 1.5 s. It does not run on every frame. Several faces are embedded concurrently
  with OpenVINO async requests.
* The OpenVINO compiled-model cache makes the second start-up much faster.

## Why it's accurate
* Faces are **aligned to the ArcFace 112×112 template** (Haar eye detection + similarity warp)
  and fed as **RGB**. The old code fed unaligned BGR crops, which costs a lot of accuracy.
* Matching uses the person's centroid plus their top-3 samples, with a **margin over the
  runner-up** to avoid look-alike confusion.
* **Temporal voting**: a name is confirmed, and attendance is logged, only after 3 consistent
  recognitions on the same tracked face. This removes flicker and one-off false matches.
* Registration saves only **sharp, single-face, pose-varied** samples, and training adds
  mirror augmentation and removes outliers.

## Setup
```bash
pip install -r requirements.txt
python scripts/get_model.py        # downloads arcfaceresnet100-8 and converts to OpenVINO IR (~130 MB)
python frontend.py                 # desktop app
```
> OpenCV **5.x removed Haar cascades**, so keep `opencv-python<5` (pinned in requirements).

## Using the app
1. **Register**: type a name and press *Start capture*. Slowly turn your head while about 40 samples
   are taken automatically. Training runs automatically when capture finishes.
2. **Live**: press *Start*. Green = confirmed (and logged once per day), amber = still verifying,
   red = unknown.
3. **Attendance**: search, filter by day and export to CSV.
4. **Settings**: device (CPU/GPU/AUTO), match threshold, Haar parameters, camera/RTSP source.

## CLI
```bash
python main.py run                         # OpenCV window
python main.py run --source rtsp://...     # IP camera
python main.py register "Alice"            # capture samples
python main.py train                       # rebuild model/model.pkl from dataset/
python main.py bench --source clip.mp4     # latency numbers
```

## Layout
```
frontend.py          PyQt5 desktop app
main.py              CLI
facerec/
  pipeline.py        threaded capture → Haar → ArcFace pipeline
  detector.py        Haar face + eye detection
  align.py           ArcFace template alignment
  embedder.py        OpenVINO ArcFace with async request pool
  gallery.py         enrolled embeddings + matching
  tracker.py         IoU tracker + identity voting
  enroll.py          registration capture + training
  attendance.py      once-per-day CSV log
  cascades/          bundled Haar XMLs
scripts/get_model.py download/convert ArcFace
scripts/build_exe.py PyInstaller bundle
dataset/<name>/      registered face samples (112×112 aligned)
model/model.pkl      trained face database
attendance/attendance.csv
```

**Upgrading from the old version:** the existing `model/model.pkl` was built with the old
(BGR, unaligned) preprocessing. It still loads, and the app reproduces that preprocessing for
it, but you should re-register everyone and train again for full accuracy.
