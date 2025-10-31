"""
Improved abstract base class for segmentation models with consistent interface.

This demonstrates best practices for the Strategy pattern implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import torch
from .response_models import PredictionResult

# Type alias for better readability
ImageData = Union[bytes, Dict[str, bytes]]
Prediction  = PredictionResult | None


class ModelSegmentation(ABC):
    """
    Abstract base class for segmentation models following the Strategy pattern.

    This class defines the common interface that all segmentation strategies must implement.
    It handles device management and provides abstract methods for model creation,
    loading, and prediction.

    Design Pattern: Strategy Pattern
    - This is the Strategy interface
    - Concrete strategies: WaterSegmentationModel, RiverSegmentationModel
    - Context: SegmentationContext (optional but recommended)

    Attributes:
        model_path (str): Path to the model weights file
        device (torch.device): Device to run the model on (CPU or GPU)
        model_name (str): Name/identifier for this model

    Example:
        # Don't instantiate this directly - use concrete implementations
        model = WaterSegmentationModel(model_path="weights.pth")
        model.load_model()
        result = model.predict(image_data)
    """

    def __init__(self, model_path: str, model_name: str):
        """
        Initialize the segmentation model.

        Args:
            model_path: Path to the model weights file
            model_name: Human-readable name for this model
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Initialized {model_name} on device: {self.device}")

    @abstractmethod
    def _create_model(self) -> Union[torch.nn.Module, list]:
        """
        Create and return the model architecture.

        This method should instantiate the neural network architecture
        but NOT load weights (that's done in load_model()).

        Returns:
            torch.nn.Module or list: The model architecture(s)

        Example implementation:
            def _create_model(self):
                return smp.Unet(
                    encoder_name="efficientnet-b0",
                    in_channels=3,
                    classes=1
                )
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the trained model weights.

        This method should:
        1. Create the model architecture (if not already done)
        2. Load weights from self.model_path
        3. Move model to self.device
        4. Set model to evaluation mode
        5. Store the model as an instance attribute

        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If model loading fails

        Example implementation:
            def load_model(self):
                self.model = self._create_model()
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.to(self.device)
                self.model.eval()
        """
        pass

    @abstractmethod
    def predict(self, image_data: ImageData) -> Prediction:
        """
        Make a prediction on the input image data.

        IMPORTANT: All concrete implementations must have the same signature!
        This ensures strategies are interchangeable.

        Args:
            image_data: Input image data. Can be:
                - bytes: Raw image bytes (for simple single-image models)
                - Dict[str, bytes]: Multiple images (e.g., {"s1": bytes, "s2": bytes})

        Returns:
            Dictionary containing prediction results with at least:
                - "model_name": str - Name of the model used
                - Strategy-specific results (masks, statistics, etc.)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If prediction fails

        Example implementation:
            def predict(self, image_data: ImageData) -> PredictionResult:
                # Preprocess
                tensor = self.preprocess(image_data)

                # Predict
                with torch.no_grad():
                    output = self.model(tensor)

                # Post-process
                return {
                    "model_name": self.model_name,
                    "prediction_mask": output.cpu().numpy(),
                    "confidence": float(output.mean())
                }
        """
        pass

    def __str__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.model_name}, "
            f"path={self.model_path}, "
            f"device={self.device})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return self.__str__()


# Example of how to create a consistent strategy:
# class ExampleSegmentationModel(ModelSegmentation):
#     """
#     Example implementation showing the correct pattern.

#     This demonstrates:
#     - Consistent predict() signature
#     - Proper model lifecycle (create → load → predict)
#     - Clear return types
#     """

#     def __init__(self, model_path: str = "model.pth"):
#         super().__init__(model_path=model_path, model_name="ExampleSegmentationModel")
#         self.model = None

#     def _create_model(self) -> torch.nn.Module:
#         """Create a simple example model."""
#         import segmentation_models_pytorch as smp

#         return smp.Unet(
#             encoder_name="resnet34",
#             encoder_weights="imagenet",
#             in_channels=3,
#             classes=1,
#         )

#     def load_model(self) -> None:
#         """Load the model weights."""
#         self.model = self._create_model()
#         # In real implementation, load weights here
#         self.model.to(self.device)
#         self.model.eval()

#     def predict(self, image_data: ImageData) -> PredictionResult:
#         """
#         Make prediction with consistent interface.

#         Note: Takes ImageData, returns PredictionResult
#         This signature is the same across ALL strategies.
#         """
#         if self.model is None:
#             raise RuntimeError("Model not loaded. Call load_model() first.")

#         # Example implementation
#         return {
#             "model_name": self.model_name,
#             "status": "success",
#             "prediction": "example_result",
#         }
