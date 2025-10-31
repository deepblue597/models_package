# Quick Fix Guide: Standardizing Your Strategy Pattern

## Problem: Inconsistent `predict()` Signatures

Your two models have different signatures, making them non-interchangeable.

---

## Fix #1: Update `RiverSegmentationModel.predict()`

### Current Code (WRONG):

```python
# river_segmentation_model.py, line ~138
def predict(self, X: dict, image_data: bytes) -> dict:
    #image_link = X[image_link_key]

    # Get image data from MinIO
    #image_bytes = self.MinIOconnector.get_object(image_link)

    # Decode the image bytes
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # ... rest of code
```

### Fixed Code (CORRECT):

```python
# river_segmentation_model.py, line ~138
def predict(self, image_data: bytes) -> Dict[str, Any]:
    """
    Make prediction on river image.

    Args:
        image_data: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        Dictionary with prediction results
    """
    # Decode the image bytes
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image")

    # ... rest of code (keep as is)

    # At the end, return result (already correct)
    return {
        "timestamp": datetime.now(),
        "model_name": self.model_name,
        "original_image": original_encoded,
        "encoded_mask": encoded_mask,
        "encoded_confidence": encoded_confidence,
        "overlay_image": overlay_output_image,
        "water_coverage": water_percentage,
        "avg_confidence": avg_confidence,
        "overflow_flag": overflow_flag,
    }
```

**Changes:**

1. Remove `X: dict` parameter (only keep `image_data`)
2. Change type hint: `Dict[str, Any]` instead of `dict`
3. Add docstring
4. Remove commented MinIO code

---

## Fix #2: Update `abstract_model.py`

### Add at the top:

```python
from typing import Any, Dict, Union

# Type aliases for clarity
ImageData = Union[bytes, Dict[str, bytes]]
PredictionResult = Dict[str, Any]
```

### Update the abstract method:

```python
@abstractmethod
def predict(self, image_data: ImageData) -> PredictionResult:
    """
    Make a prediction on input image data.

    IMPORTANT: All concrete implementations MUST have the same signature!

    Args:
        image_data: Input image data. Can be:
            - bytes: Raw image bytes (for single-image models)
            - Dict[str, bytes]: Multiple images (e.g., {"s1": bytes, "s2": bytes})

    Returns:
        Dictionary containing prediction results with at least:
            - "model_name": str - Name of the model used
            - Strategy-specific results (masks, statistics, etc.)

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If prediction fails
    """
    pass
```

---

## Fix #3: Update `__init__.py`

Add the Context class to exports:

```python
"""
Models package - A collection of segmentation models for water and river detection.
"""

from .abstract_model import ModelSegmentation
from .water_segmentation_model import WaterSegmentationModel
from .river_segmentation_model import RiverSegmentationModel
from .segmentation_context import SegmentationContext  # ← ADD THIS LINE

__all__ = [
    "ModelSegmentation",
    "WaterSegmentationModel",
    "RiverSegmentationModel",
    "SegmentationContext",  # ← ADD THIS LINE
]

__version__ = "0.1.0"
```

---

## Fix #4: Verify `WaterSegmentationModel.predict()` is correct

Your current signature is correct:

```python
def predict(self, image_data: Dict[str, bytes]) -> Dict[str, Any]:
    # ✅ This is correct! Keep as is.
```

This is fine because `Dict[str, bytes]` is compatible with the `ImageData` type alias.

---

## Testing Your Fixes

### Test 1: Both models should work with Context

```python
from models_package.models import (
    WaterSegmentationModel,
    RiverSegmentationModel,
    SegmentationContext
)

# Test water model
water_model = WaterSegmentationModel(model_path="water.pth")
water_context = SegmentationContext(water_model)
water_context.load_model()

water_data = {
    "s1": b"...",
    "s2": b"...",
    "slope": b"..."
}
water_result = water_context.predict(water_data)
print(f"Water: {water_result['model_name']}")

# Test river model
river_model = RiverSegmentationModel(model_path="river.pth")
river_context = SegmentationContext(river_model)
river_context.load_model()

river_data = b"...jpeg bytes..."
river_result = river_context.predict(river_data)
print(f"River: {river_result['model_name']}")

# ✅ Both should work without errors!
```

### Test 2: Strategy switching should work

```python
context = SegmentationContext(water_model)
context.load_model()

# Use water model
result1 = context.predict(water_data)

# Switch to river model
context.set_strategy(river_model)
context.load_model()

# Use river model
result2 = context.predict(river_data)

# ✅ Should work seamlessly!
```

---

## Summary of Changes

| File                          | Line(s) | Change                                         |
| ----------------------------- | ------- | ---------------------------------------------- |
| `abstract_model.py`           | ~1-10   | Add type imports: `Union`, `Dict`, `Any`       |
| `abstract_model.py`           | ~112    | Update `predict()` signature with type aliases |
| `river_segmentation_model.py` | ~138    | Remove `X: dict` parameter from `predict()`    |
| `river_segmentation_model.py` | ~138    | Change return type to `Dict[str, Any]`         |
| `__init__.py`                 | ~7, ~13 | Import and export `SegmentationContext`        |

---

## Verification Checklist

- [ ] Both models have same `predict()` signature
- [ ] `abstract_model.py` uses type aliases
- [ ] `SegmentationContext` is exported in `__init__.py`
- [ ] Tests pass with both models
- [ ] Strategy switching works
- [ ] No import errors

---

## Need Help?

If you encounter any issues:

1. Check import statements are correct
2. Verify file paths match your project structure
3. Run tests to ensure compatibility
4. Review the full examples in `examples/usage_examples.py`

---

**After these fixes, your Strategy pattern will be production-ready! 🚀**
