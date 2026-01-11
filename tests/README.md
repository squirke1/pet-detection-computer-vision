# Testing Suite

This directory contains comprehensive tests for the pet detection computer vision project.

## Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest configuration
├── unit/                          # Unit tests for individual functions
│   └── test_utils.py             # Tests for utility functions
└── integration/                   # Integration tests for complete workflows
    └── test_inference.py         # Tests for inference scripts
```

## Running Tests

### Prerequisites

1. Install testing dependencies:
```bash
pip install pytest pytest-cov pytest-mock
```

2. Install project dependencies:
```bash
pip install -r requirements.txt
```

3. **Important**: For integration tests to run successfully, you need a valid YOLOv8 model:
   - Download a pre-trained YOLOv8 model or train your own
   - Place it at `models/yolov8n-pets.pt`
   - The model file should not be empty (current file is 0 bytes)

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run only unit tests (work without model)
pytest tests/unit/ -v

# Run only integration tests (require valid model)
pytest tests/integration/ -v
```

### Run Specific Test Classes

```bash
# Test specific utility functions
pytest tests/unit/test_utils.py::TestLoadImage -v

# Test inference scripts
pytest tests/integration/test_inference.py::TestInferOnImageScript -v
```

### Generate Coverage Report

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html
```

## Test Categories

### Unit Tests (`tests/unit/`)

Tests individual functions in isolation. These tests:
- Don't require the YOLOv8 model
- Use synthetic test data
- Run quickly
- Test edge cases and error handling

**Coverage:**
- Image loading and saving
- Drawing detection boxes
- File operations
- Edge detection features (Canny, Sobel, Laplacian)
- Keypoint detection (SIFT, ORB)
- Contour detection
- Enhanced visualization functions

### Integration Tests (`tests/integration/`)

Tests complete workflows and script execution. These tests:
- **Require a valid YOLOv8 model file** (currently not available)
- Test end-to-end inference pipelines
- Run actual inference scripts as subprocesses
- Verify output files are created correctly

**Coverage:**
- Single image inference (`infer_on_image.py`)
- Batch folder inference (`infer_on_folder.py`)
- Enhanced inference with CV features (`infer_enhanced.py`)
- Complete detection workflows

## Current Status

✅ **Unit Tests**: 28/28 passing
- All utility functions tested
- Edge cases covered
- No external dependencies required

⚠️ **Integration Tests**: Currently skipped/failing
- Reason: Model file `models/yolov8n-pets.pt` is empty (0 bytes)
- To fix: Replace with a valid YOLOv8 model file
- Once fixed, integration tests will validate complete inference workflows

## Test Fixtures

Common test fixtures defined in `conftest.py`:

- `sample_image`: Simple 100x100 RGB test image
- `sample_image_with_edges`: Image with clear edges and patterns
- `sample_gray_image`: Grayscale test image
- `sample_detections`: Mock detection data (boxes, classes, confidences)
- `temp_image_dir`: Temporary directory with test images
- `temp_output_dir`: Temporary output directory

## Adding New Tests

### Unit Test Template

```python
def test_my_function(sample_image):
    """Test description."""
    result = my_function(sample_image)
    
    assert result is not None
    assert isinstance(result, expected_type)
    # Add more assertions
```

### Integration Test Template

```python
@pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="YOLOv8 model not available"
)
def test_my_workflow(test_image_with_pet, tmp_path):
    """Test description."""
    # Setup
    output_path = tmp_path / "output.jpg"
    
    # Execute
    result = run_inference(test_image_with_pet, output_path)
    
    # Verify
    assert result.success
    assert output_path.exists()
```

## CI/CD Integration

Tests are automatically run in the GitHub Actions CI pipeline:
- Configured in `.github/workflows/gitflow-ci.yml`
- Runs on pushes and pull requests
- Tests are part of the quality gate before merging

## Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "Model file not found" or "Ran out of input"
The model file is missing or corrupted. To fix:
1. Download YOLOv8n: `wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt`
2. Place in `models/` directory
3. Or train your own model on pet dataset

### Integration tests are skipped
This is expected if the model file is not available. Unit tests will still run successfully.
