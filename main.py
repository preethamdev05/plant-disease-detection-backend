"""
Plant Disease Detection FastAPI Backend
Production-grade, Cloud Run ready, deterministic inference.

Contract:
- Accept JPEG/PNG images via multipart/form-data
- Resize to 256x256 RGB
- Input dtype: uint8 (preserve [0,255] range)
- Model contains internal Rescaling(1.0) - NO external normalization
- Load TensorFlow model from GCS with compile=False
- Three output heads: crop (softmax), disease (softmax), is_diseased (sigmoid)
- Inference logic: Strict waterfall (NON_CROP -> HEALTH_GATE -> DISEASE)
- CORS: Allowed origins read from env var (comma-separated)
- Health threshold: 0.5 (is_diseased sigmoid output)
"""

import json
import logging
import os
from io import BytesIO
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from google.cloud import storage
from PIL import Image

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

MODEL_GCS_PATH = "gs://plant-model-global/model/model_40f4b8e9.keras"
METADATA_FILENAME = "inference_metadata.json"
HEALTH_THRESHOLD = 0.5
TARGET_IMAGE_SIZE = (256, 256)
ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".png"}

# CORS configuration: Read from environment
ALLOWED_ORIGINS_STR = os.getenv(
    'ALLOWED_ORIGINS',
    'https://preethamdev05.github.io,https://plantdoc-pro-812118174928.us-west1.run.app'
)
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(',')]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"CORS Allowed Origins: {ALLOWED_ORIGINS}")

# =============================================================================
# GLOBAL STATE: Model, Metadata, Readiness
# =============================================================================

_model: Optional[tf.keras.Model] = None
_crop_map: Optional[dict] = None
_disease_map: Optional[dict] = None
_ready: bool = False


def _load_metadata_from_git() -> dict:
    """
    Load inference_metadata.json from Git-tracked location.
    Expected format:
    {
        "crop_map": {"0": "Apple", "1": "Blueberry", ...},
        "disease_map": {"0": "apple_scab", "1": "apple_cedar_rust", ...},
        "config_hash": "...",
        "timestamp": "..."
    }
    """
    # In production on Cloud Run, metadata is in the container (Git-committed).
    # For local development, check current directory first.
    possible_paths = [
        METADATA_FILENAME,
        f"/app/{METADATA_FILENAME}",
        os.path.join(os.path.dirname(__file__), METADATA_FILENAME),
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            logger.info(f"Loading metadata from {path}")
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Metadata file '{METADATA_FILENAME}' not found in any known location. "
        f"Checked: {possible_paths}"
    )


def _download_model_from_gcs() -> bytes:
    """Download model file from Google Cloud Storage (GCS)."""
    logger.info(f"Downloading model from {MODEL_GCS_PATH}")

    # Parse GCS path: gs://bucket-name/path/to/file
    if not MODEL_GCS_PATH.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {MODEL_GCS_PATH}")

    path_parts = MODEL_GCS_PATH[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_path = path_parts[1] if len(path_parts) > 1 else ""

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        model_bytes = blob.download_as_bytes()
        logger.info(f"Downloaded {len(model_bytes)} bytes from GCS")
        return model_bytes
    except Exception as e:
        raise RuntimeError(f"GCS download failed: {e}") from e


def _load_model_from_gcs() -> tf.keras.Model:
    """Load TensorFlow model from GCS. Must use compile=False."""
    logger.info("Loading TensorFlow model...")
    model_bytes = _download_model_from_gcs()

    # Save to temporary location
    temp_path = "/tmp/model.keras"
    with open(temp_path, "wb") as f:
        f.write(model_bytes)

    # Load with compile=False (critical - training artifacts not needed)
    model = tf.keras.models.load_model(temp_path, compile=False)
    logger.info(f"Model loaded successfully. Architecture: {model.name}")
    return model


def _startup_load_model():
    """
    Blocking startup: Load model and metadata.
    Fail fast if either fails - no lazy loading.
    Process crashes if this fails (monotonic guarantee).
    """
    global _model, _crop_map, _disease_map, _ready

    logger.info("=" * 80)
    logger.info("STARTUP: Loading model and metadata")
    logger.info("=" * 80)

    try:
        # Load metadata first (fast, Git-tracked)
        metadata = _load_metadata_from_git()
        logger.info("Metadata loaded successfully")

        # Convert string keys to integers (JSON limitation)
        # crop_map: {"0": "Apple", "1": "Blueberry", ...} -> {0: "Apple", 1: "Blueberry", ...}
        _crop_map = {int(k): v for k, v in metadata["crop_map"].items()}
        _disease_map = {int(k): v for k, v in metadata["disease_map"].items()}

        logger.info(f"Crop classes: {len(_crop_map)}")
        logger.info(f"Disease classes: {len(_disease_map)}")

        # Load model from GCS (slow)
        _model = _load_model_from_gcs()

        # Mark ready
        _ready = True
        logger.info("=" * 80)
        logger.info("STARTUP: READY - /health returns 200")
        logger.info("=" * 80)

    except Exception as e:
        logger.critical(f"STARTUP FAILED: {e}")
        logger.critical("Process will exit with non-zero status")
        raise


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Plant Disease Detection Backend",
    description="Production-grade inference API for plant disease detection",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Startup event: Load model and metadata before accepting requests."""
    _startup_load_model()


@app.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint.
    Returns 200 ONLY if model and metadata are fully loaded.
    Never returns 503 after startup (monotonic guarantee).
    """
    if not _ready or _model is None or _crop_map is None or _disease_map is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or metadata not ready"
        )
    return {"status": "ready"}


def _sanitize_label(label: str) -> str:
    """
    Sanitize disease/crop labels for display.
    - Replace underscores with spaces
    - Apply manual fixes (e.g., "haunglongbing" -> "Huanglongbing")
    - Capitalize each word
    """
    # Replace underscores with spaces
    label = label.replace("_", " ")

    # Manual fixes for common mis-formatted diseases
    fixes = {
        "spider mites two-spotted spider mite": "Two-Spotted Spider Mite",
        "haunglongbing": "Huanglongbing",
        "early blight": "Early Blight",
        "late blight": "Late Blight",
        "powdery mildew": "Powdery Mildew",
        "septoria leaf spot": "Septoria Leaf Spot",
        "brown spot": "Brown Spot",
        "bacterial spot": "Bacterial Spot",
    }

    label_lower = label.lower().strip()
    if label_lower in fixes:
        return fixes[label_lower]

    # Default: capitalize each word
    return " ".join(word.capitalize() for word in label.split())


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode image from file bytes (JPEG or PNG only).
    Returns array of shape (256, 256, 3) with dtype uint8.
    """
    try:
        img = Image.open(BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}") from e

    # Validate format
    if img.format not in ("JPEG", "PNG"):
        raise ValueError(f"Unsupported image format: {img.format}. Only JPEG and PNG allowed.")

    # Convert to RGB (handle RGBA, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to target size
    img = img.resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Convert to numpy array (uint8, [0, 255])
    arr = np.array(img, dtype=np.uint8)

    assert arr.shape == (256, 256, 3), f"Expected shape (256, 256, 3), got {arr.shape}"
    assert arr.dtype == np.uint8, f"Expected dtype uint8, got {arr.dtype}"

    return arr


def _run_inference(image_array: np.ndarray) -> dict:
    """
    Run inference with strict waterfall logic.

    Inputs:
        image_array: (256, 256, 3), dtype uint8

    Returns:
        {
            "crop": {"label": str, "confidence": float},
            "health": {"status": "non_crop"|"healthy"|"diseased", "probability": float},
            "disease": null | {"label": str, "confidence": float}
        }
    """
    assert _model is not None, "Model not loaded"
    assert _crop_map is not None, "Crop map not loaded"
    assert _disease_map is not None, "Disease map not loaded"

    # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
    batch = np.expand_dims(image_array, axis=0)

    # Run inference (model expects uint8 [0, 255])
    predictions = _model(batch, training=False)

    # Extract outputs
    crop_probs = predictions["crop"].numpy()[0]
    disease_probs = predictions["disease"].numpy()[0]
    isdiseased_raw = float(predictions["is_diseased"].numpy()[0, 0])

    # Get argmax indices
    crop_idx = int(np.argmax(crop_probs))
    disease_idx = int(np.argmax(disease_probs))

    crop_label = _crop_map[crop_idx]
    disease_label_raw = _disease_map[disease_idx]
    crop_confidence = float(crop_probs[crop_idx])
    disease_confidence = float(disease_probs[disease_idx])

    # =====STRICT WATERFALL LOGIC=====

    # STEP 1: NON_CROP (OOD)
    if crop_label == "NON_CROP":
        return {
            "crop": {"label": "NON_CROP", "confidence": crop_confidence},
            "health": {"status": "non_crop", "probability": 0.0},
            "disease": None,
        }

    # STEP 2: HEALTH GATE (sole authority)
    is_diseased = isdiseased_raw >= HEALTH_THRESHOLD
    if not is_diseased:
        health_status = "healthy"
        health_prob = 1.0 - isdiseased_raw
    else:
        health_status = "diseased"
        health_prob = isdiseased_raw

    if not is_diseased:
        return {
            "crop": {"label": _sanitize_label(crop_label), "confidence": crop_confidence},
            "health": {"status": health_status, "probability": health_prob},
            "disease": None,
        }

    # STEP 3: DISEASE IDENTIFICATION
    if disease_label_raw == "healthy":
        disease_label = "Unspecified Disease"
    else:
        disease_label = _sanitize_label(disease_label_raw)

    return {
        "crop": {"label": _sanitize_label(crop_label), "confidence": crop_confidence},
        "health": {"status": health_status, "probability": health_prob},
        "disease": {"label": disease_label, "confidence": disease_confidence},
    }


@app.post("/predict", status_code=200)
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint.

    POST /predict with multipart/form-data
    Form field: file (JPEG or PNG image)

    Returns:
    {
        "crop": {"label": str, "confidence": float},
        "health": {"status": "non_crop"|"healthy"|"diseased", "probability": float},
        "disease": null | {"label": str, "confidence": float}
    }
    """
    if not _ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )

    # Validate filename extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File extension {ext} not allowed. Only .jpeg, .jpg, .png supported."
            )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise ValueError("Empty file")

        image_array = _decode_image(file_bytes)
        result = _run_inference(image_array)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
