"""
Example usage of the Strategy Pattern with Segmentation Models.

This file demonstrates:
1. How to use models directly (without Context)
2. How to use models with Context (recommended)
3. How to switch strategies at runtime
4. How to add hooks for extensibility
5. Best practices for integrating with external services
"""

from models_package.models import WaterSegmentationModel, RiverSegmentationModel
from models_package.models.segmentation_context import SegmentationContext
from typing import Dict, Any


# ============================================================================
# Example 1: Using strategies directly (without Context)
# ============================================================================
def example_direct_usage():
    """
    Using models directly without the Context class.

    This works but lacks:
    - Pre/post processing hooks
    - Timing metrics
    - Validation
    - Strategy switching
    """
    print("\n" + "=" * 70)
    print("Example 1: Direct Strategy Usage (No Context)")
    print("=" * 70)

    # Create and load water segmentation model
    water_model = WaterSegmentationModel(
        model_path="models/water_model.pth",
        model_indx=0,  # RGBNIRVVVHSLOPE
    )
    water_model.load_model()

    # Prepare input data (mock data for example)
    image_data = {
        "s1": b"...sentinel1 bytes...",
        "s2": b"...sentinel2 bytes...",
        "slope": b"...slope bytes...",
    }

    # Make prediction
    result = water_model.predict(image_data)

    print(f"Model: {result['model_name']}")
    print(f"Water coverage: {result['water_coverage_stats']['water_percentage']:.2f}%")


# ============================================================================
# Example 2: Using Context (Recommended Approach)
# ============================================================================
def example_context_usage():
    """
    Using models with the Context class.

    Benefits:
    - Automatic validation
    - Timing metrics
    - Pre/post processing hooks
    - Easy strategy switching
    """
    print("\n" + "=" * 70)
    print("Example 2: Context Usage (Recommended)")
    print("=" * 70)

    # Create a strategy
    water_model = WaterSegmentationModel(
        model_path="models/water_model.pth", model_indx=0
    )

    # Wrap it in a context
    context = SegmentationContext(
        strategy=water_model, enable_timing=True, enable_validation=True
    )

    # Load the model
    context.load_model()

    # Prepare input
    image_data = {
        "s1": b"...sentinel1 bytes...",
        "s2": b"...sentinel2 bytes...",
        "slope": b"...slope bytes...",
    }

    # Make prediction (automatically timed and validated)
    result = context.predict(image_data)

    print(f"Prediction time: {result.elapsed_time} s")

    # Get metrics
    metrics = context.get_metrics()
    print(f"Total predictions: {metrics['prediction_count']}")
    print(f"Average time: {metrics.get('average_time_seconds', 'N/A')}s")


# ============================================================================
# Example 3: Strategy Switching at Runtime
# ============================================================================
def example_strategy_switching():
    """
    Switching between different segmentation strategies at runtime.

    This is the key benefit of the Strategy pattern!
    """
    print("\n" + "=" * 70)
    print("Example 3: Runtime Strategy Switching")
    print("=" * 70)

    # Start with water segmentation
    water_strategy = WaterSegmentationModel(model_path="models/water_model.pth")
    context = SegmentationContext(water_strategy, enable_timing=True)
    context.load_model()

    # Process water image
    water_data = {"s1": b"...", "s2": b"...", "slope": b"..."}
    water_result = context.predict(water_data)
    print(f"Water model result: {water_result['model_name']}")

    # Switch to river segmentation
    river_strategy = RiverSegmentationModel(model_path="models/river_model.pth")
    context.set_strategy(river_strategy)
    context.load_model()

    # Process river image (using same context!)
    river_data = b"...river image bytes..."
    river_result = context.predict(river_data)
    print(f"River model result: {river_result['model_name']}")

    # Both strategies used through the same interface!


# ============================================================================
# Example 4: Adding Hooks for Extensibility
# ============================================================================
def example_with_hooks():
    """
    Using hooks to add functionality before/after prediction.

    This is how you can integrate with external services (MinIO, TimescaleDB, etc.)
    without coupling them to the model logic.
    """
    print("\n" + "=" * 70)
    print("Example 4: Pre/Post Processing Hooks")
    print("=" * 70)

    # Create context
    model = WaterSegmentationModel(model_path="models/water_model.pth")
    context = SegmentationContext(model)
    context.load_model()

    # Add pre-prediction hook (e.g., logging, validation)
    def log_input_sizes(data: Dict[str, Any]) -> Dict[str, Any]:
        """Log the size of input data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, bytes):
                    print(f"  Input '{key}': {len(value)} bytes")
        return data

    # Add post-prediction hook (e.g., save to database)
    def save_to_database(result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate saving result to database."""
        print(f"  Saving result to database: {result.get('model_name')}")
        # In real code: db.insert(result)
        return result

    # Register hooks
    context.add_pre_predict_hook(log_input_sizes)
    context.add_post_predict_hook(save_to_database)

    # Make prediction (hooks will run automatically)
    image_data = {
        "s1": b"sentinel1_data" * 100,
        "s2": b"sentinel2_data" * 100,
        "slope": b"slope_data" * 100,
    }

    print("\nMaking prediction with hooks:")
    result = context.predict(image_data)
    print("Prediction complete!\n")


# ============================================================================
# Example 5: Integrating with External Services (Best Practice)
# ============================================================================
def example_external_services():
    """
    Best practice for integrating with external services like MinIO, TimescaleDB.

    Key principle: Use hooks/decorators, NOT direct coupling in model code.
    """
    print("\n" + "=" * 70)
    print("Example 5: External Service Integration (Best Practice)")
    print("=" * 70)

    # Mock external service connectors
    class MinIOConnector:
        def save_image(self, name: str, data: bytes):
            print(f"    [MinIO] Saved '{name}' ({len(data)} bytes)")

    class TimescaleDBConnector:
        def insert_stats(self, stats: dict):
            print(f"    [TimescaleDB] Inserted stats: {stats}")

    # Create connectors (dependency injection!)
    minio = MinIOConnector()
    timescale = TimescaleDBConnector()

    # Create hook that uses external services
    def save_to_external_services(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook that saves results to external services.

        Note: Services are passed via closure, not hardcoded in model!
        """
        # Save prediction image to MinIO
        if "prediction_result_tiff" in result:
            minio.save_image(
                name=f"predictions/{result['model_name']}.tiff",
                data=result["prediction_result_tiff"],
            )

        # Save statistics to TimescaleDB
        if "water_coverage_stats" in result:
            timescale.insert_stats(result["water_coverage_stats"])

        return result

    # Set up context with the hook
    model = WaterSegmentationModel(model_path="models/water_model.pth")
    context = SegmentationContext(model)
    context.load_model()
    context.add_post_predict_hook(save_to_external_services)

    # Make prediction (external services used via hook)
    image_data = {"s1": b"...", "s2": b"...", "slope": b"..."}

    print("\nMaking prediction with external service integration:")
    result = context.predict(image_data)
    print("\n✓ Prediction saved to external services via hooks!\n")


# ============================================================================
# Example 6: Model Selection Based on Input Type
# ============================================================================
def example_dynamic_strategy_selection():
    """
    Dynamically selecting which strategy to use based on input characteristics.

    This demonstrates the flexibility of the Strategy pattern.
    """
    print("\n" + "=" * 70)
    print("Example 6: Dynamic Strategy Selection")
    print("=" * 70)

    def select_strategy(image_data: Any) -> SegmentationContext:
        """
        Select appropriate segmentation strategy based on input.

        This is a simple example - in production, you might analyze
        image metadata, file type, etc.
        """
        if isinstance(image_data, dict) and "slope" in image_data:
            # Multi-band data with slope → use water model
            print("Detected multi-band data → Using WaterSegmentationModel")
            strategy = WaterSegmentationModel(model_path="models/water_model.pth")
        else:
            # Single image → use river model
            print("Detected single-band data → Using RiverSegmentationModel")
            strategy = RiverSegmentationModel(model_path="models/river_model.pth")

        context = SegmentationContext(strategy, enable_timing=True)
        context.load_model()
        return context

    # Example 1: Multi-band data
    multi_band_data = {"s1": b"...", "s2": b"...", "slope": b"..."}
    context1 = select_strategy(multi_band_data)
    result1 = context1.predict(multi_band_data)
    print(f"Result: {result1['model_name']}\n")

    # Example 2: Single-band data
    single_band_data = b"...single image..."
    context2 = select_strategy(single_band_data)
    result2 = context2.predict(single_band_data)
    print(f"Result: {result2['model_name']}\n")


# ============================================================================
# Example 7: Testing Strategies
# ============================================================================
def example_testing():
    """
    How to test strategies effectively.

    The Strategy pattern makes testing much easier!
    """
    print("\n" + "=" * 70)
    print("Example 7: Testing Strategies")
    print("=" * 70)

    # Mock strategy for testing
    class MockSegmentationModel:
        def __init__(self):
            self.model_name = "MockModel"
            self.device = "cpu"
            self.model_path = "mock.pth"

        def load_model(self):
            print("  [Mock] Model loaded")

        def predict(self, image_data):
            print("  [Mock] Making prediction")
            return {
                "model_name": self.model_name,
                "prediction": "mock_result",
                "confidence": 0.95,
            }

    # Use mock in tests
    mock_model = MockSegmentationModel()
    context = SegmentationContext(mock_model)
    context.load_model()

    # Test prediction
    result = context.predict({"test": b"data"})

    assert result["model_name"] == "MockModel"
    assert result["confidence"] == 0.95
    print("\n✓ All tests passed!\n")


# ============================================================================
# Run all examples
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STRATEGY PATTERN: SEGMENTATION MODELS - USAGE EXAMPLES")
    print("=" * 70)

    try:
        # Note: These examples use mock data and won't actually run
        # without real model files. They demonstrate the patterns.

        print("\n[INFO] These are demonstration examples")
        print("[INFO] To run with real models, provide actual model paths\n")

        # Uncomment to run examples:
        # example_direct_usage()
        # example_context_usage()
        # example_strategy_switching()
        example_with_hooks()
        example_external_services()
        example_dynamic_strategy_selection()
        example_testing()

        print("\n" + "=" * 70)
        print("✓ All examples completed!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}\n")
