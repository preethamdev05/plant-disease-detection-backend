# Plant Disease Detection Backend

**Production-grade FastAPI backend for multi-task plant disease detection.**

Deterministic inference. Cloud Run ready. Zero downtime. CORS integration with GitHub Pages frontend.

## Overview

This backend provides a production-grade REST API for plant disease detection using a multi-task CNN trained on TensorFlow/Keras.

### Architecture

- **Framework**: FastAPI + Uvicorn
- **Model**: EfficientNetB1 multi-task CNN (3 output heads)
- **Deployment**: Google Cloud Run
- **Model Storage**: Google Cloud Storage (GCS)
- **Metadata**: Git-tracked `inference_metadata.json`

### Model Outputs

Three independent heads trained as separate tasks:

1. **crop** (softmax, 15 classes): Crop species detection, including NON_CROP for out-of-domain detection
2. **disease** (softmax, 21 classes): Disease identification (trained on crop-only samples)
3. **is_diseased** (sigmoid, binary): Health binary gate (0=healthy, 1=diseased)

## Critical Design Principles

### Input Contract

- **Accept**: JPEG/PNG images via `multipart/form-data`
- **Resize**: 256×256 RGB
- **dtype**: uint8 (preserve [0, 255] range)
- **Forbidden**: External normalization (dividing by 255.0)
- **Reason**: Model contains internal `Rescaling(1.0)` layer expecting raw [0, 255] values

### Inference Waterfall Logic

**Strict, deterministic, no heuristics:**

```
STEP 1: NON_CROP (Out-of-Domain Detection)
IF crop.label == "NON_CROP"
    → status = "non_crop"
    → probability = 0.0
    → disease = null
    RETURN

STEP 2: HEALTH GATE (Sole Authority)
IF is_diseased < 0.5
    → status = "healthy"
    → probability = is_diseased
    → disease = null
    RETURN

STEP 3: DISEASE IDENTIFICATION
IF is_diseased >= 0.5
    → status = "diseased"
    → probability = is_diseased
    → disease = disease_label (or "Unspecified Disease" if model predicts "healthy")
    RETURN
```

### Readiness & Monotonicity

- **Startup**: Blocks until model + metadata fully load
- **/health**: Returns 200 ONLY when ready; never 503 after startup
- **/predict**: Can never return 503 if `/health` returned 200 (monotonic guarantee)
- **Failure mode**: Process crashes (fail-fast) if model or metadata loading fails

### CORS

**Exactly one origin allowed:**

```
https://preethamdev05.github.io
```

No wildcards. Strict same-origin policy.

## Deployment: Google Cloud Run

### Build & Deploy

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build image
docker build -t gcr.io/YOUR_PROJECT_ID/plant-disease-backend .

# 3. Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/plant-disease-backend

# 4. Deploy to Cloud Run
gcloud run deploy plant-disease-backend \
  --image gcr.io/YOUR_PROJECT_ID/plant-disease-backend \
  --region us-central1 \
  --platform managed \
  --memory 4Gi \
  --timeout 3600 \
  --set-env-vars "PORT=8080"
```

### Environment Variables

- `PORT` (default: 8080): HTTP server port. Cloud Run sets this automatically.

### GCS Permissions

Service account must have:

- `storage.buckets.get`
- `storage.objects.get` (on model bucket)

```yaml
Role: roles/storage.objectViewer
Resource: gs://plant-model-global/
```

## API Endpoints

### GET /health

**Health check.** Returns 200 if model is fully loaded; 503 otherwise.

```bash
curl https://YOUR_BACKEND/health
```

**Response (200 OK):**

```json
{"status": "ready"}
```

**Response (503 Service Unavailable):**

```json
{"detail": "Model not ready"}
```

### POST /predict

**Predict disease from image.**

```bash
curl -X POST https://YOUR_BACKEND/predict \
  -F "file=@image.jpg"
```

**Response (200 OK):**

```json
{
  "crop": {
    "label": "Tomato",
    "confidence": 0.9834
  },
  "health": {
    "status": "diseased",
    "probability": 0.8721
  },
  "disease": {
    "label": "Tomato Mosaic Virus",
    "confidence": 0.7645
  }
}
```

**Response (400 Bad Request):**

```json
{"detail": "File extension .gif not allowed. Only .jpeg, .jpg, .png supported."}
```

**Response (503 Service Unavailable):**

```json
{"detail": "Model not ready"}
```

## Output Schema

**Invariant contract:**

```json
{
  "crop": {
    "label": "string (sanitized, space-separated)",
    "confidence": "float [0.0, 1.0]"
  },
  "health": {
    "status": "'non_crop' | 'healthy' | 'diseased'",
    "probability": "float [0.0, 1.0] (raw is_diseased sigmoid output)"
  },
  "disease": "null | { 'label': 'string', 'confidence': 'float [0.0, 1.0]' }"
}
```

**Semantic guarantee:**

- `health.probability` always represents the model's probability of disease (raw `is_diseased` output)
- `health.status` reflects the binary decision (healthy vs diseased)
- `disease` is `null` when `health.status` is "healthy" or "non_crop"
- `disease.label` is "Unspecified Disease" if binary gate says diseased but disease head predicts "healthy"

## Metadata Handling

**File**: `inference_metadata.json` (Git-tracked)

**Schema:**

```json
{
  "crop_map": { "0": "Apple", "1": "Blueberry", ..., "15": "NON_CROP" },
  "disease_map": { "0": "apple_scab", "1": "apple_black_rot", ... },
  "config_hash": "3238348c",
  "timestamp": "2026-01-26T07:28:50Z"
}
```

**Critical**: JSON uses string keys (limitation). Backend converts to integers:

```python
CROP_CLASSES = {int(k): v for k, v in metadata["crop_map"].items()}
DISEASE_CLASSES = {int(k): v for k, v in metadata["disease_map"].items()}
```

Failure to do this causes `KeyError` at runtime.

## Model Loading

**Sequence:**

1. Load `inference_metadata.json` from Git (fast, must be in container)
2. Download model from GCS: `gs://plant-model-global/model/model_40f4b8e9.keras` (slow)
3. Load with `tf.keras.models.load_model(path, compile=False)` (critical)
4. Set `_ready = True`
5. Accept requests

**Failure modes:**

- If metadata missing → Process crashes (no lazy loading)
- If GCS unreachable → Process crashes (no fallback)
- If model corrupted → Process crashes (no silent fallback)
- If load successful → Never regress (monotonic)

## Image Processing

1. Decode JPEG/PNG only (reject others)
2. Convert to RGB (handle RGBA, grayscale)
3. Resize to 256×256 (BILINEAR interpolation)
4. Convert to numpy `uint8` array
5. Validate shape `(256, 256, 3)` and dtype
6. Add batch dimension: `(1, 256, 256, 3)`
7. Pass to model (expects `[0, 255]`)

## Label Sanitization

**Applied before returning JSON:**

- Replace `_` with spaces: `apple_scab` → `apple scab`
- Apply manual fixes:
  - `haunglongbing` → `Huanglongbing`
  - `tomato_mosaic_virus` → `Tomato Mosaic Virus`
- Capitalize each word: `apple scab` → `Apple Scab`

## Testing

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires GCS access and inference_metadata.json)
python main.py

# Test health endpoint
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict -F "file=@test_image.jpg"

# View Swagger docs
open http://localhost:8080/docs
```

### Validation Checks

- [ ] Model loads without errors
- [ ] Metadata keys are converted to integers
- [ ] `/health` returns 200 before accepting requests
- [ ] `/predict` works with JPEG and PNG
- [ ] CORS allows only `https://preethamdev05.github.io`
- [ ] Output schema matches contract exactly
- [ ] Waterfall logic produces correct status/disease combinations
- [ ] Image resizing preserves RGB and uint8

## Frontend Integration

**Repository**: [Agri-AI](https://github.com/preethamdev05/Agri-AI)
**Hosted**: https://preethamdev05.github.io

Frontend fetches from this backend via CORS:

```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('https://YOUR_BACKEND/predict', {
  method: 'POST',
  body: formData,
  mode: 'cors'
});

const result = await response.json();
console.log(result);
```

## Production Checklist

- [ ] Model downloaded and verified from GCS
- [ ] Metadata committed to Git and included in container
- [ ] CORS origin exactly matches frontend URL
- [ ] Health check endpoint working
- [ ] Prediction endpoint working with valid images
- [ ] Invalid image uploads rejected gracefully
- [ ] Waterfall logic produces deterministic output
- [ ] No silent failures or lazy initialization
- [ ] Container image builds and runs
- [ ] Cloud Run deployment configured with correct memory/timeout
- [ ] Service account has GCS read permissions
- [ ] Logs are visible and informative

## License

MIT

## Contact

[@preethamdev05](https://github.com/preethamdev05)
