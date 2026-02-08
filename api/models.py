"""
Pydantic models for API request/response schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single object detection."""
    
    class_id: int = Field(..., description="Class ID (0=dog, 1=cat)")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: List[float] = Field(..., min_length=4, max_length=4, 
                             description="Bounding box [x1, y1, x2, y2]")


class DetectionResponse(BaseModel):
    """Response for detection endpoint."""
    
    success: bool = Field(..., description="Whether inference succeeded")
    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    num_detections: int = Field(..., description="Total number of detections")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    image_shape: List[int] = Field(..., description="Image shape [height, width, channels]")
    model_name: str = Field(..., description="Model name used for inference")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: Optional[str] = Field(None, description="Path to loaded model")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class ModelInfo(BaseModel):
    """Model information."""
    
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Model file path")
    size_mb: float = Field(..., description="Model file size in MB")
    type: str = Field(..., description="Model type")


class ModelsResponse(BaseModel):
    """Response for models list endpoint."""
    
    success: bool = Field(..., description="Whether request succeeded")
    current_model: str = Field(..., description="Currently loaded model")
    available_models: List[ModelInfo] = Field(..., description="Available models")
