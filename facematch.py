"""Compare faces between two photos and report a similarity percentage (command-line tool).

The admin app calls the same logic over HTTP via app.py; this script is for
ad-hoc checks and threshold calibration.

Usage:
    python facematch.py                      # compares the first two images in ./input
    python facematch.py a.jpg b.jpg
    python facematch.py a.jpg b.jpg --threshold 0.35
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from core import Bands, FaceMatcher, ImageDecodeError, decode_bytes

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"}


def to_percent(sim: float) -> float:
    return max(0.0, sim) * 100


def annotate(img: np.ndarray, faces, labels, colours) -> np.ndarray:
    out = img.copy()
    for face, label, colour in zip(faces, labels, colours):
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(out, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)
    return out


def side_by_side(a: np.ndarray, b: np.ndarray, height: int = 800) -> np.ndarray:
    def fit(im):
        return cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height))
    return np.hstack([fit(a), np.full((height, 20, 3), 255, np.uint8), fit(b)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, help="reference image then comparison image")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="cosine similarity at/above which faces are considered the same person (default 0.35)")
    parser.add_argument("--output", type=Path, default=Path("output/result.jpg"))
    args = parser.parse_args()

    images = args.images
    if not images:
        images = sorted(p for p in Path("input").iterdir() if p.suffix.lower() in IMAGE_EXTS)[:2]
    if len(images) != 2:
        parser.error("need exactly two images (or put two images in ./input)")

    matcher = FaceMatcher(Bands.from_env())
    detected = []
    for path in images:
        try:
            detected.append(matcher.detect(decode_bytes(path.read_bytes())))
        except ImageDecodeError as exc:
            print(f"Cannot read {path}: {exc}")
            return 1
        if not detected[-1].faces:
            print(f"No face found in {path}")
            return 1
    a, b = detected

    # Reference = the largest face in the first image.
    ref = matcher.reference_face(a)
    m = matcher.match(ref, b)
    sims, best = m.similarities, m.face_index

    print(f"\nReference : {images[0].name}  ({len(a.faces)} face(s), using largest)")
    print(f"Compared  : {images[1].name}  ({len(b.faces)} face(s), left to right)\n")
    for i, sim in enumerate(sims):
        verdict = "MATCH" if sim >= args.threshold else "no match"
        marker = "  <- best" if i == best else ""
        print(f"  Face {i + 1}: {to_percent(sim):5.1f}%  (cosine {sim:+.3f})  {verdict}{marker}")

    best_sim = sims[best]
    same = best_sim >= args.threshold
    print(f"\nResult: {'SAME PERSON' if same else 'DIFFERENT PERSON'} - "
          f"best similarity {to_percent(best_sim):.1f}% (threshold {to_percent(args.threshold):.0f}%), "
          f"band '{matcher.bands.classify(best_sim)}'")
    for note in a.warnings + m.warnings:
        print(f"  note: {note}")

    green, red, blue = (0, 200, 0), (0, 0, 220), (220, 120, 0)
    left = annotate(a.image, [ref], ["Reference"], [blue])
    right = annotate(b.image, b.faces,
                     [f"{i + 1}: {to_percent(s):.1f}%" for i, s in enumerate(sims)],
                     [green if s >= args.threshold else red for s in sims])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), side_by_side(left, right))
    print(f"Annotated comparison saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
