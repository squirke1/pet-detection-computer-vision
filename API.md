# REST API Documentation

Complete guide for the Pet Detection REST API.

---

## 🚀 Quick Start

### Start the API Server

```bash
# Option 1: Direct Python execution
python -m uvicorn api.main:app --reload

# Option 2: Docker
docker-compose up

# Option 3: Production with Gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

The API will be available at `http://localhost:8000`

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Web UI**: http://localhost:8000

---

## 📡 API Endpoints

### 1. POST `/detect` - Run Detection

Upload an image and get pet detection results.

#### Request

**Content-Type**: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Image file (JPG, PNG, WebP) |
| conf | float | No | Confidence threshold (0.0-1.0) |

#### Response (200 OK)

```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "dog",
      "confidence": 0.92,
      "bbox": [145.2, 234.1, 456.8, 678.3]
    }
  ],
  "num_detections": 1,
  "inference_time_ms": 45.3,
  "image_shape": [640, 480, 3],
  "model_name": "yolov8n-pets"
}
```

#### Example (cURL)

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/image.jpg" \
  -F "conf=0.25"
```

#### Example (Python)

```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect', files=files)
    result = response.json()
    print(f"Found {result['num_detections']} pets")
```

#### Example (JavaScript)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log('Detections:', data.num_detections));
```

---

### 2. GET `/health` - Health Check

Check API health and model status.

#### Response (200 OK)

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "model_loaded": true,
  "model_path": "models/yolov8n-pets.pt"
}
```

#### Example

```bash
curl http://localhost:8000/health
```

---

### 3. GET `/models` - List Models

Get information about available models.

#### Response (200 OK)

```json
{
  "success": true,
  "current_model": "yolov8n-pets",
  "available_models": [
    {
      "name": "yolov8n-pets",
      "path": "models/yolov8n-pets.pt",
      "size_mb": 6.23,
      "type": "YOLOv8"
    }
  ]
}
```

#### Example

```bash
curl http://localhost:8000/models
```

---

## 🐳 Docker Deployment

### Basic Deployment (CPU)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### GPU Deployment

Requires NVIDIA Docker runtime.

```bash
# Start with GPU support
docker-compose --profile gpu up -d api-gpu

# The GPU service runs on port 8001
curl http://localhost:8001/health
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | models/yolov8n-pets.pt | Path to model weights |
| CONF_THRESHOLD | 0.25 | Default confidence threshold |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| ./models | /app/models | Model weights (read-only) |
| ./outputs | /app/outputs | Detection outputs |
| ./uploads | /app/uploads | Temporary uploads |

---

## 🔧 Configuration

### Model Selection

Place model files in the `models/` directory:

```bash
models/
├── yolov8n-pets.pt      # Nano (fastest)
├── yolov8s-pets.pt      # Small
├── yolov8m-pets.pt      # Medium
└── yolov8l-pets.pt      # Large (most accurate)
```

Set the `MODEL_PATH` environment variable:

```bash
export MODEL_PATH=models/yolov8m-pets.pt
python -m uvicorn api.main:app
```

### Confidence Threshold

Default: 0.25 (25%)

- **Lower** (0.1-0.2): More detections, more false positives
- **Medium** (0.25-0.5): Balanced
- **Higher** (0.5-0.9): Fewer but more confident detections

---

## 📊 Performance Tips

### 1. **Batch Processing**

Process multiple images efficiently:

```python
import concurrent.futures
import requests

def detect_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        return requests.post('http://localhost:8000/detect', files=files).json()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(detect_image, image_paths))
```

### 2. **Connection Pooling**

Reuse connections for better performance:

```python
import requests

session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=10))

# Use session for all requests
result = session.post('http://localhost:8000/detect', files=files)
```

### 3. **Model Size Selection**

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| YOLOv8n | ~6 MB | ~2ms | Good |
| YOLOv8s | ~22 MB | ~3ms | Better |
| YOLOv8m | ~52 MB | ~6ms | Great |
| YOLOv8l | ~87 MB | ~10ms | Best |

Choose based on your latency/accuracy requirements.

### 4. **Hardware Acceleration**

- **CPU**: Works on any machine
- **GPU**: 5-10x faster (use Docker GPU profile)
- **Batch Size**: Increase for GPU inference (handled automatically)

---

## 🔒 Security Considerations

### Production Deployment

1. **Authentication**: Add API key or JWT authentication
2. **Rate Limiting**: Implement request rate limits
3. **CORS**: Configure specific allowed origins
4. **File Size Limits**: Restrict upload sizes
5. **Input Validation**: Validate image formats and sizes

Example with API key:

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/detect", dependencies=[Depends(verify_api_key)])
async def detect(file: UploadFile = File(...)):
    # ... detection logic
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Not Found

```
Error: Model not loaded. Please check server logs.
```

**Solution**: Ensure model file exists at the specified `MODEL_PATH`.

```bash
# Check if model exists
ls -lh models/yolov8n-pets.pt

# Download if missing
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n-pets.pt
```

#### 2. Port Already in Use

```
Error: [Errno 48] Address already in use
```

**Solution**: Change the port or stop existing service.

```bash
# Use different port
uvicorn api.main:app --port 8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

#### 3. Out of Memory (Docker)

```
Error: CUDA out of memory
```

**Solution**: Increase Docker memory limit or use smaller model.

```bash
# Edit docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### 4. Slow Inference

**Possible causes**:
- Large images → Resize before upload
- CPU-only inference → Use GPU
- Large model → Use smaller variant (yolov8n)

---

## 📈 Monitoring

### Health Check Endpoint

```bash
# Check status periodically
watch -n 5 curl -s http://localhost:8000/health | jq
```

### Logging

```bash
# View Docker logs
docker-compose logs -f --tail=100

# Save logs to file
docker-compose logs > api_logs.txt
```

### Metrics

Track key metrics:
- **Requests per second**
- **Average inference time**
- **Error rate**
- **Memory usage**

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **GitHub Repository**: https://github.com/squirke1/pet-detection-computer-vision
- **YOLOv8 Documentation**: https://docs.ultralytics.com
- **FastAPI Documentation**: https://fastapi.tiangolo.com

---

## 🤝 Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section above

---

**Last Updated**: February 8, 2026  
**API Version**: 2.1.0
