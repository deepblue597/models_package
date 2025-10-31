"""
Models package - A collection of segmentation models for water and river detection.
"""

from .abstract_model import ModelSegmentation
from .water_segmentation_model import WaterSegmentationModel
from .river_segmentation_model import RiverSegmentationModel
from .segmentation_context import SegmentationContext
from .response_models import PredictionResult

__all__ = [
    "ModelSegmentation",
    "WaterSegmentationModel",
    "RiverSegmentationModel",
    "SegmentationContext",
    "PredictionResult"
]

__version__ = "1.0.0"
