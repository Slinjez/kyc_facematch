"""Face matching core shared by the HTTP service (app.py) and the CLI (facematch.py).

Uses InsightFace (ArcFace, "buffalo_l" model pack) running locally via ONNX Runtime.
Images are decoded in memory and never written to disk.
"""

import io
import os
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

warnings.filterwarnings("ignore", category=FutureWarning)  # noisy skimage deprecation inside insightface

from insightface.app import FaceAnalysis  # noqa: E402  (after the warnings filter)

MODEL_PACK = "buffalo_l"
MODEL_NAME = f"insightface/{MODEL_PACK}"

MAX_SIDE = 1600  # downscale large phone photos; detection doesn't need 8MP
PDF_DPI = 200
PDF_MAX_PAGES = 3  # look for a face on the first few pages of a scanned document
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]

MIN_FACE_PX = 60  # narrower than this and the embedding is unreliable
MIN_DET_SCORE = 0.6


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Bands:
    """Cosine-similarity cut-offs. Calibrate on our own selfie/ID pairs before go-live."""

    strong: float = 0.45
    possible: float = 0.35
    weak: float = 0.25

    @classmethod
    def from_env(cls) -> "Bands":
        return cls(
            strong=_env_float("FACE_BAND_STRONG", cls.strong),
            possible=_env_float("FACE_BAND_POSSIBLE", cls.possible),
            weak=_env_float("FACE_BAND_WEAK", cls.weak),
        )

    def classify(self, sim: float) -> str:
        if sim >= self.strong:
            return "strong"
        if sim >= self.possible:
            return "possible"
        if sim >= self.weak:
            return "weak"
        return "no_match"


class ImageDecodeError(ValueError):
    pass


@dataclass
class DetectedImage:
    image: np.ndarray  # BGR, as the faces were found (possibly rotated)
    faces: list  # insightface Face objects, left to right
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReferenceMatch:
    face_found: bool
    similarity: float | None  # raw cosine similarity of the best face
    face_index: int | None
    similarities: list[float]
    warnings: list[str]


def decode_bytes(data: bytes, content_type: str | None = None) -> list[np.ndarray]:
    """Decode an image or PDF into BGR pages, honouring EXIF orientation and capping resolution."""
    if not data:
        raise ImageDecodeError("empty file")
    if data[:5] == b"%PDF-" or (content_type or "").lower() == "application/pdf":
        pages = _render_pdf(data)
    else:
        try:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageDecodeError("not a readable image") from exc
        pages = [img]
    out = []
    for page in pages:
        page.thumbnail((MAX_SIDE, MAX_SIDE))
        out.append(cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2BGR))
    return out


def _render_pdf(data: bytes) -> list[Image.Image]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise ImageDecodeError("not a readable PDF") from exc
    try:
        if len(pdf) == 0:
            raise ImageDecodeError("PDF has no pages")
        return [
            pdf[i].render(scale=PDF_DPI / 72).to_pil().convert("RGB")
            for i in range(min(len(pdf), PDF_MAX_PAGES))
        ]
    finally:
        pdf.close()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def face_area(face) -> float:
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


class FaceMatcher:
    def __init__(self, bands: Bands | None = None, det_size: int = 640):
        self.bands = bands or Bands()
        self.model = MODEL_NAME
        self.app = FaceAnalysis(name=MODEL_PACK, allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def detect(self, pages: list[np.ndarray]) -> DetectedImage:
        """Detect faces, trying each page and retrying with 90/180 degree rotations if none are found upright."""
        for page_no, page in enumerate(pages):
            for rot in ROTATIONS:
                candidate = page if rot is None else cv2.rotate(page, rot)
                faces = self.app.get(candidate)
                if faces:
                    notes = []
                    if page_no:
                        notes.append(f"Face found on page {page_no + 1} of the document.")
                    if rot is not None:
                        notes.append("Image had to be rotated to find the face.")
                    return DetectedImage(candidate, sorted(faces, key=lambda f: f.bbox[0]), notes)
        return DetectedImage(pages[0], [], [])

    def detect_bytes(self, data: bytes, content_type: str | None = None) -> DetectedImage:
        return self.detect(decode_bytes(data, content_type))

    @staticmethod
    def quality_warnings(face, what: str) -> list[str]:
        notes = []
        if face.bbox[2] - face.bbox[0] < MIN_FACE_PX:
            notes.append(f"The {what} face is small or low resolution; the score is less reliable.")
        if float(face.det_score) < MIN_DET_SCORE:
            notes.append(f"The {what} face is unclear (blurred, covered or at an angle); the score is less reliable.")
        return notes

    def reference_face(self, detected: DetectedImage):
        """The probe face: the largest face in the selfie."""
        return max(detected.faces, key=face_area)

    def match(self, probe_face, reference: DetectedImage) -> ReferenceMatch:
        """Compare the probe face against every face in a reference image; the best one counts."""
        notes = list(reference.warnings)
        if not reference.faces:
            return ReferenceMatch(False, None, None, [], notes + ["No face found in this image."])
        sims = [cosine(probe_face.normed_embedding, f.normed_embedding) for f in reference.faces]
        best = int(np.argmax(sims))
        if len(reference.faces) > 1:
            notes.append(f"{len(reference.faces)} faces found in this image; the closest one was used.")
        notes += self.quality_warnings(reference.faces[best], "document")
        return ReferenceMatch(True, sims[best], best, sims, notes)


def to_score(sim: float) -> float:
    """Cosine similarity as a 0..1 score (negative similarity means no resemblance at all)."""
    return round(max(0.0, min(1.0, sim)), 4)
