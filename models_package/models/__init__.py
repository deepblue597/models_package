"""
Models package - A collection of segmentation models for water and river detection.
"""

from .abstract_model import ModelSegmentation
from .water_segmentation_model import WaterSegmentationModel
from .river_segmentation_model import RiverSegmentationModel

__all__ = [
    "ModelSegmentation",
    "WaterSegmentationModel", 
    "RiverSegmentationModel",
]

__version__ = "0.1.0"