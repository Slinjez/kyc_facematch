"""Face-match HTTP service for the KYC admin (kyc_v02/src/services/faceServiceClient.ts).

    POST /v1/verify   header x-api-key
        multipart/form-data:
            selfie          one file
            references      one file per reference, in order
            reference_ids   one field per reference, same order
        200 {
            model: str,
            selfie: { face_found: bool, warnings: [str] },
            results: [{ reference_id, face_found, score: float|null, band, warnings: [str] }]
        }
        non-2xx { detail: str }

    GET /health       no auth; 200 once the model is loaded

Bands: strong | possible | weak | no_match, plus no_face when a face is missing.
Images are processed in memory and never stored; kyc_v02 saves the results.

Run:  .venv\\Scripts\\python app.py      (reads .env next to this file)
"""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

BASE_DIR = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    """Minimal .env reader; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(BASE_DIR / ".env")

from core import Bands, FaceMatcher, ImageDecodeError, to_score  # noqa: E402  (after .env is loaded)

API_KEY = os.environ.get("FACE_SERVICE_API_KEY", "").strip()
HOST = os.environ.get("FACE_SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FACE_SERVICE_PORT", "8100"))
MAX_FILE_BYTES = int(os.environ.get("FACE_SERVICE_MAX_FILE_MB", "15")) * 1024 * 1024
MAX_REFERENCES = int(os.environ.get("FACE_SERVICE_MAX_REFERENCES", "10"))
BUSY_WAIT_SECONDS = float(os.environ.get("FACE_SERVICE_BUSY_WAIT_SECONDS", "45"))

log = logging.getLogger("face-service")

# One comparison at a time: the model is CPU-bound and shares one ONNX session.
_inference_lock = threading.Lock()
_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not API_KEY:
        raise RuntimeError("FACE_SERVICE_API_KEY is not set (put it in facematch/.env)")
    _state["matcher"] = FaceMatcher(Bands.from_env())
    log.info("face-service ready: %s %s", _state["matcher"].model, _state["matcher"].bands)
    yield
    _state.clear()


app = FastAPI(title="KYC face-service", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid api key")


def read_upload(upload: UploadFile, what: str) -> bytes:
    data = upload.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"{what} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(status_code=400, detail=f"{what} is empty")
    return data


@app.get("/health")
def health():
    matcher: FaceMatcher | None = _state.get("matcher")
    if matcher is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": matcher.model}


@app.post("/v1/verify", dependencies=[Depends(require_api_key)])
def verify(
    selfie: UploadFile = File(...),
    references: list[UploadFile] = File(default=[]),
    reference_ids: list[str] = Form(default=[]),
):
    if not references:
        raise HTTPException(status_code=422, detail="at least one reference image is required")
    if len(references) != len(reference_ids):
        raise HTTPException(status_code=422, detail="references and reference_ids must be the same length")
    if len(references) > MAX_REFERENCES:
        raise HTTPException(status_code=422, detail=f"at most {MAX_REFERENCES} references per request")
    if len(set(reference_ids)) != len(reference_ids):
        raise HTTPException(status_code=422, detail="reference_ids must be unique")

    selfie_bytes = read_upload(selfie, "selfie")
    reference_bytes = [read_upload(r, f"reference {i + 1}") for i, r in enumerate(references)]

    if not _inference_lock.acquire(timeout=BUSY_WAIT_SECONDS):
        raise HTTPException(status_code=503, detail="busy")
    try:
        return _compare(_state["matcher"], selfie, selfie_bytes, references, reference_ids, reference_bytes)
    finally:
        _inference_lock.release()


def _compare(matcher: FaceMatcher, selfie, selfie_bytes, references, reference_ids, reference_bytes) -> dict:
    try:
        selfie_detected = matcher.detect_bytes(selfie_bytes, selfie.content_type)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"selfie is {exc}")

    selfie_warnings = list(selfie_detected.warnings)
    if not selfie_detected.faces:
        return {
            "model": matcher.model,
            "selfie": {"face_found": False, "warnings": selfie_warnings + ["No face found in the selfie."]},
            "results": [
                {"reference_id": rid, "face_found": False, "score": None, "band": "no_face",
                 "warnings": ["Not compared: no face found in the selfie."]}
                for rid in reference_ids
            ],
        }

    probe = matcher.reference_face(selfie_detected)
    if len(selfie_detected.faces) > 1:
        selfie_warnings.append(f"{len(selfie_detected.faces)} faces found in the selfie; the largest one was used.")
    selfie_warnings += matcher.quality_warnings(probe, "selfie")

    results = []
    for rid, upload, data in zip(reference_ids, references, reference_bytes):
        try:
            detected = matcher.detect_bytes(data, upload.content_type)
        except ImageDecodeError as exc:
            results.append({"reference_id": rid, "face_found": False, "score": None, "band": "no_result",
                            "warnings": [f"Could not read this file ({exc})."]})
            continue
        m = matcher.match(probe, detected)
        results.append({
            "reference_id": rid,
            "face_found": m.face_found,
            "score": None if m.similarity is None else to_score(m.similarity),
            "band": "no_face" if m.similarity is None else matcher.bands.classify(m.similarity),
            "warnings": m.warnings,
        })

    return {"model": matcher.model, "selfie": {"face_found": True, "warnings": selfie_warnings}, "results": results}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
