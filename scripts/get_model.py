"""Download ArcFace-ResNet100 (ONNX model zoo, arcfaceresnet100-8) and convert it to
OpenVINO IR (FP16) in openvino_model/.

    python scripts/get_model.py                 # download + convert
    python scripts/get_model.py --onnx my.onnx  # convert a file you already have
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "openvino_model")
URLS = [
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
]


def download(dst: str) -> None:
    for url in URLS:
        try:
            print(f"Downloading {url}")
            def hook(n, bs, total):
                if total > 0:
                    sys.stdout.write(f"\r  {min(100, n * bs * 100 // total):3d}%  ({total / 1e6:.0f} MB)")
                    sys.stdout.flush()
            urllib.request.urlretrieve(url, dst, hook)
            print()
            if os.path.getsize(dst) > 50_000_000:     # LFS pointer files are tiny
                return
        except Exception as e:
            print(f"  failed: {e}")
    raise SystemExit("Could not download the model. Download arcfaceresnet100-8.onnx manually and "
                     "run: python scripts/get_model.py --onnx path/to/arcfaceresnet100-8.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", help="existing arcfaceresnet100-8.onnx")
    ap.add_argument("--fp32", action="store_true", help="keep FP32 weights (default FP16, half the size)")
    a = ap.parse_args()

    import openvino as ov
    os.makedirs(OUT_DIR, exist_ok=True)
    onnx = a.onnx or os.path.join(OUT_DIR, "arcfaceresnet100-8.onnx")
    if not os.path.exists(onnx):
        download(onnx)
    print("Converting to OpenVINO IR ...")
    model = ov.convert_model(onnx)
    xml = os.path.join(OUT_DIR, "arcfaceresnet100-8.xml")
    ov.save_model(model, xml, compress_to_fp16=not a.fp32)
    print(f"Saved {xml}")
    if not a.onnx and os.path.exists(xml):
        os.remove(onnx)   # IR is all we need


if __name__ == "__main__":
    main()
