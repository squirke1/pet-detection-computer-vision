"""
FastAPI Application for Pet Detection

Provides REST API endpoints for YOLOv8 pet detection inference.

Endpoints:
    GET  /              - Web UI
    POST /detect        - Run inference on uploaded image
    GET  /health        - Health check
    GET  /models        - List available models
    POST /models/{name} - Switch active model
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api import __version__
from api.models import (
    DetectionResponse,
    HealthResponse,
    ErrorResponse,
    ModelsResponse,
    ModelInfo
)
from api.inference import InferenceEngine


# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n-pets.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
MODELS_DIR = Path("models")

# Initialize FastAPI app
app = FastAPI(
    title="Pet Detection API",
    description="YOLOv8-based API for detecting dogs and cats in images",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine (loaded on startup)
engine: Optional[InferenceEngine] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global engine
    
    print(f"🚀 Starting Pet Detection API v{__version__}")
    print(f"📦 Loading model: {MODEL_PATH}")
    
    try:
        engine = InferenceEngine(MODEL_PATH, conf_threshold=CONF_THRESHOLD)
        engine.load_model()
        print(f"✅ Model loaded successfully")
        print(f"📊 Model info: {engine.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"⚠️  API will start but inference will fail until model is loaded")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the web UI."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pet Detection API</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1rem;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        
        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8ebff;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .upload-icon {
            font-size: 3rem;
            margin-bottom: 10px;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        #preview {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 10px;
            display: none;
        }
        
        #canvas {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 10px;
            display: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        #results {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 10px;
            display: none;
        }
        
        .detection {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .detection-class {
            font-weight: bold;
            color: #667eea;
            text-transform: uppercase;
        }
        
        .confidence {
            color: #764ba2;
            font-weight: bold;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-box {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .links {
            margin-top: 30px;
            text-align: center;
        }
        
        .links a {
            color: #667eea;
            text-decoration: none;
            margin: 0 15px;
            font-weight: 500;
        }
        
        .links a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐾 Pet Detection API</h1>
        <p class="subtitle">Upload an image to detect dogs and cats using YOLOv8</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📷</div>
            <h3>Drop an image here or click to upload</h3>
            <p style="color: #999; margin-top: 10px;">Supports JPG, PNG, WebP</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <button id="detectBtn" disabled>Detect Pets</button>
        
        <div class="loader" id="loader"></div>
        
        <img id="preview" />
        <canvas id="canvas"></canvas>
        
        <div id="results"></div>
        
        <div class="links">
            <a href="/docs" target="_blank">📚 API Documentation</a>
            <a href="/health" target="_blank">💚 Health Check</a>
            <a href="https://github.com/squirke1/pet-detection-computer-vision" target="_blank">🔗 GitHub</a>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const detectBtn = document.getElementById('detectBtn');
        const preview = document.getElementById('preview');
        const canvas = document.getElementById('canvas');
        const results = document.getElementById('results');
        const loader = document.getElementById('loader');
        let selectedFile = null;
        
        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // File selection
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file || !file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }
            
            selectedFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            detectBtn.disabled = false;
            results.style.display = 'none';
            canvas.style.display = 'none';
        }
        
        // Detect button
        detectBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            detectBtn.disabled = true;
            loader.style.display = 'block';
            results.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    results.innerHTML = `<p style="color: red;">❌ Error: ${data.error}</p>`;
                    results.style.display = 'block';
                }
            } catch (error) {
                results.innerHTML = `<p style="color: red;">❌ Request failed: ${error.message}</p>`;
                results.style.display = 'block';
            } finally {
                loader.style.display = 'none';
                detectBtn.disabled = false;
            }
        });
        
        function drawBoundingBoxes(detections) {
            const ctx = canvas.getContext('2d');
            const img = preview;
            
            // Wait for image to load if not ready
            if (!img.complete) {
                img.onload = () => drawBoundingBoxes(detections);
                return;
            }
            
            // Hide preview, show canvas
            preview.style.display = 'none';
            
            // Set canvas size to match image
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            
            // Draw image
            ctx.drawImage(img, 0, 0);
            
            // Draw bounding boxes
            detections.forEach((det) => {
                const [x1, y1, x2, y2] = det.bbox;
                const width = x2 - x1;
                const height = y2 - y1;
                
                // Box color based on class
                const color = det.class_name === 'dog' ? '#3b82f6' : '#10b981';
                
                // Draw box
                ctx.strokeStyle = color;
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, width, height);
                
                // Draw label background
                const label = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
                ctx.font = 'bold 18px Arial';
                const textWidth = ctx.measureText(label).width;
                
                ctx.fillStyle = color;
                ctx.fillRect(x1, y1 - 30, textWidth + 12, 30);
                
                // Draw label text
                ctx.fillStyle = 'white';
                ctx.fillText(label, x1 + 6, y1 - 8);
            });
            
            canvas.style.display = 'block';
        }
        
        function displayResults(data) {
            const { detections, num_detections, inference_time_ms, model_name } = data;
            
            // Draw boxes on image
            if (num_detections > 0) {
                drawBoundingBoxes(detections);
            }
            
            let html = '<h3 style="margin-bottom: 15px;">Detection Results</h3>';
            
            // Stats
            html += '<div class="stats">';
            html += `<div class="stat-box"><div class="stat-value">${num_detections}</div><div class="stat-label">Detections</div></div>`;
            html += `<div class="stat-box"><div class="stat-value">${inference_time_ms.toFixed(1)}ms</div><div class="stat-label">Inference Time</div></div>`;
            html += '</div>';
            
            // Detections
            if (num_detections > 0) {
                html += '<h4 style="margin: 20px 0 10px 0;">Detected Pets:</h4>';
                detections.forEach((det, idx) => {
                    html += `
                        <div class="detection">
                            <span class="detection-class">${det.class_name}</span>
                            <span class="confidence">${(det.confidence * 100).toFixed(1)}%</span>
                            <p style="color: #666; margin-top: 5px; font-size: 0.9rem;">
                                BBox: [${det.bbox.map(v => v.toFixed(0)).join(', ')}]
                            </p>
                        </div>
                    `;
                });
            } else {
                html += '<p style="color: #999; margin-top: 20px;">No pets detected in this image.</p>';
            }
            
            results.innerHTML = html;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        200: {"description": "Successful detection"},
        400: {"model": ErrorResponse, "description": "Invalid image"},
        500: {"model": ErrorResponse, "description": "Inference error"}
    }
)
async def detect(
    file: UploadFile = File(..., description="Image file to process"),
    conf: Optional[float] = Query(None, ge=0.0, le=1.0, description="Confidence threshold override")
):
    """
    Run pet detection on uploaded image.
    
    Upload an image file and receive detection results with bounding boxes,
    class labels, and confidence scores.
    """
    if engine is None or not engine.is_loaded():
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Run inference
        detections, inference_time, image_shape = engine.predict_from_bytes(
            image_bytes,
            conf_threshold=conf
        )
        
        return DetectionResponse(
            success=True,
            detections=detections,
            num_detections=len(detections),
            inference_time_ms=inference_time,
            image_shape=list(image_shape),
            model_name=engine.model_path.stem
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns service status and model information.
    """
    model_loaded = engine is not None and engine.is_loaded()
    model_path = str(engine.model_path) if engine else None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=__version__,
        model_loaded=model_loaded,
        model_path=model_path
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models.
    
    Returns information about the current model and all available models
    in the models directory.
    """
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get current model
    current_model = engine.model_path.stem if engine and engine.is_loaded() else "none"
    
    # Find all .pt files in models directory
    model_files = list(MODELS_DIR.glob("*.pt"))
    
    available_models = []
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        available_models.append(
            ModelInfo(
                name=model_file.stem,
                path=str(model_file),
                size_mb=round(size_mb, 2),
                type="YOLOv8"
            )
        )
    
    return ModelsResponse(
        success=True,
        current_model=current_model,
        available_models=available_models
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
