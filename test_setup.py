"""
Test script to verify basic setup and dependencies.
Run this after installing requirements.txt
"""

import sys

def test_imports():
    """Test that core packages are installed."""
    print("Testing basic imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV not found: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib not found: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow (PIL)")
    except ImportError as e:
        print(f"✗ Pillow not found: {e}")
        return False
    
    return True


def test_opencv_basic():
    """Test basic OpenCV functionality."""
    import cv2
    import numpy as np
    
    print("\nTesting OpenCV basic operations...")
    
    # Create a test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 0, 0]  # Blue square
    
    # Test color conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Test resize
    resized = cv2.resize(img, (50, 50))
    
    print(f"✓ Created test image: {img.shape}")
    print(f"✓ Grayscale conversion: {gray.shape}")
    print(f"✓ Resize operation: {resized.shape}")
    
    return True


def main():
    print("="*50)
    print("Pet Detection - Environment Setup Test")
    print("="*50 + "\n")
    
    if not test_imports():
        print("\n❌ Import test failed. Please install requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    if not test_opencv_basic():
        print("\n❌ OpenCV functionality test failed")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✅ All tests passed! Environment is ready.")
    print("="*50)


if __name__ == "__main__":
    main()
