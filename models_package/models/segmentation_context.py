"""
Context class for managing segmentation model strategies.

This class provides a unified interface for working with different segmentation models,
handling pre/post-processing, validation, and common workflows.
"""

from typing import Dict, Any, Callable

from models_package.models.response_models import PredictionResult
from models_package.models.water_segmentation_model import WaterSegmentationModel
from .abstract_model import ModelSegmentation
import time


class SegmentationContext:
    """
    Context class that manages segmentation strategies.

    This class:
    - Encapsulates strategy selection and switching
    - Provides pre/post-processing hooks
    - Handles common workflows (load, predict, analyze)
    - Enables logging, metrics, and validation

    Usage:
        # Create context with a strategy
        context = SegmentationContext(WaterSegmentationModel())

        # Load model
        context.load_model()

        # Make predictions with automatic pre/post processing
        result = context.predict(image_data)

        # Switch strategies at runtime
        context.set_strategy(RiverSegmentationModel())
    """

    def __init__(
        self,
        strategy: ModelSegmentation,
        enable_timing: bool = False,
        enable_validation: bool = True,
    ):
        """
        Initialize the context with a segmentation strategy.

        Args:
            strategy: The segmentation model strategy to use
            enable_timing: Whether to track prediction timing
            enable_validation: Whether to validate inputs/outputs
        """
        self._strategy = strategy
        self.enable_timing = enable_timing
        self.enable_validation = enable_validation
        self._is_loaded = False
        self._prediction_count = 0
        self._total_time = 0.0

        # Hooks for extensibility
        self.pre_predict_hooks: list[Callable] = []
        self.post_predict_hooks: list[Callable] = []

    def set_strategy(self, strategy: ModelSegmentation) -> None:
        """
        Switch to a different segmentation strategy.

        Args:
            strategy: New strategy to use
        """
        print(
            f"Switching strategy from {self._strategy.model_name} to {strategy.model_name}"
        )
        self._strategy = strategy
        self._is_loaded = False  # New strategy needs to be loaded

    def get_strategy(self) -> ModelSegmentation:
        """Get the current strategy."""
        return self._strategy

    def load_model(self) -> None:
        """
        Load the model for the current strategy.
        Adds error handling and validation.
        """
        if self._is_loaded:
            print(f"Model {self._strategy.model_name} is already loaded")
            return

        try:
            print(f"Loading model: {self._strategy.model_name}")
            self._strategy.load_model()
            self._is_loaded = True
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def add_pre_predict_hook(
        self, hook: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """
        Add a hook to run before prediction.

        Hook signature: func(input_data: Dict) -> Dict

        Example:
            def log_input(data):
                print(f"Processing image of size {len(data.get('s2', b''))} bytes")
                return data
            context.add_pre_predict_hook(log_input)
        """
        self.pre_predict_hooks.append(hook)

    def add_post_predict_hook(
        self, hook: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """
        Add a hook to run after prediction.

        Hook signature: func(result: Dict) -> Dict

        Example:
            def save_to_db(result):
                # Save result to database
                return result
            context.add_post_predict_hook(save_to_db)
        """
        self.post_predict_hooks.append(hook)

    def _validate_input(self, image_data: Any) -> None:
        """Validate input data before prediction."""
        if not self.enable_validation:
            return

        if image_data is None:
            raise ValueError("image_data cannot be None")

        # Strategy-specific validation could be added here
        if isinstance(image_data, dict):
            if not image_data:
                raise ValueError("image_data dictionary is empty")
            elif isinstance(self._strategy , WaterSegmentationModel) and len(image_data) != 3  :
                raise ValueError("image_data dictionary must contain exactly 3 images for WaterSegmentationModel")
        elif isinstance(image_data, bytes):
            if len(image_data) == 0:
                raise ValueError("image_data bytes are empty")

    def _validate_output(self, result: PredictionResult) -> None:
        """Validate prediction output."""
        if not self.enable_validation:
            return

        if result is None:
            raise ValueError("Prediction returned None")

        if not isinstance(result, dict):
            raise ValueError(f"Prediction must return dict, got {type(result)}")

        # Check for model_name in result
        if "model_name" not in result:
            print("⚠ Warning: result missing 'model_name' field")

    def predict(self, image_data: Any) -> PredictionResult:
        """
        Make a prediction using the current strategy.

        This method:
        1. Validates the model is loaded
        2. Validates input data
        3. Runs pre-prediction hooks
        4. Executes the strategy's predict method
        5. Runs post-prediction hooks
        6. Validates output
        7. Tracks timing metrics if enabled

        Args:
            image_data: Input data for prediction (format depends on strategy)

        Returns:
            Dictionary containing prediction results
        """
        # Ensure model is loaded
        if not self._is_loaded:
            raise RuntimeError(
                f"Model {self._strategy.model_name} not loaded. Call load_model() first."
            )

        # Validate input
        self._validate_input(image_data)

        # Run pre-prediction hooks
        processed_input = image_data
        for hook in self.pre_predict_hooks:
            try:
                processed_input = hook(processed_input)
            except Exception as e:
                print(f"⚠ Warning: Pre-predict hook failed: {e}")

        # Make prediction with timing
        start_time = time.time() if self.enable_timing else None

        try:
            result = self._strategy.predict(processed_input)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise

        if self.enable_timing and start_time is not None:
            elapsed = time.time() - start_time
            self._total_time += elapsed
            self._prediction_count += 1
            if result is not None:
                result.elapsed_time = round(elapsed, 3)
                print(f"⏱ Prediction took {elapsed:.3f}s")

        # Validate output
        if result is not None:
            self._validate_output(result)

        # Run post-prediction hooks
        processed_result = result
        for hook in self.post_predict_hooks:
            try:
                processed_result = hook(processed_result)
            except Exception as e:
                print(f"⚠ Warning: Post-predict hook failed: {e}")
        if processed_result is None:
            raise ValueError("Post-predict hooks resulted in None output")
        return processed_result

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this context.

        Returns:
            Dictionary with metrics like prediction count, average time, etc.
        """
        metrics = {
            "model_name": self._strategy.model_name,
            "prediction_count": self._prediction_count,
            "is_loaded": self._is_loaded,
        }

        if self.enable_timing and self._prediction_count > 0:
            metrics.update(
                {
                    "total_time_seconds": round(self._total_time, 3),
                    "average_time_seconds": round(
                        self._total_time / self._prediction_count, 3
                    ),
                }
            )

        return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._prediction_count = 0
        self._total_time = 0.0

    def __str__(self) -> str:
        return (
            f"SegmentationContext("
            f"strategy={self._strategy.model_name}, "
            f"loaded={self._is_loaded}, "
            f"predictions={self._prediction_count})"
        )

    def __repr__(self) -> str:
        return self.__str__()
