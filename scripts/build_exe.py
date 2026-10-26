"""Bundle the desktop app into a standalone folder with PyInstaller.

    pip install pyinstaller
    python scripts/build_exe.py            # -> dist/FaceAttend/
    python scripts/build_exe.py --onefile  # single exe (slower start-up)
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEP = ";" if os.name == "nt" else ":"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onefile", action="store_true")
    ap.add_argument("--console", action="store_true")
    a = ap.parse_args()
    data = [("facerec/cascades", "facerec/cascades"), ("openvino_model", "openvino_model"), ("model", "model")]
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--name", "FaceAttend",
           "--onefile" if a.onefile else "--onedir", "--collect-all", "openvino"]
    if not a.console:
        cmd.append("--windowed")
    for src, dst in data:
        if os.path.exists(os.path.join(ROOT, src)):
            cmd += ["--add-data", f"{os.path.join(ROOT, src)}{SEP}{dst}"]
    cmd.append(os.path.join(ROOT, "frontend.py"))
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
