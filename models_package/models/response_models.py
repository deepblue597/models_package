from typing import Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel

class PredictionResult(BaseModel):
    """Standard result structure for all segmentation models."""
    # Required fields (all strategies must return these)
    model_name: str
    start_time: datetime
    prediction_mask: bytes
    elapsed_time: Optional[float]
    # Common optional fields
    statistics: Dict[str, Any]
    #confidence: float

    # Strategy-specific fields go here
    extras: Optional[Dict[str, Any]]
    

class PredictionOutput(PredictionResult):
    """Extended prediction output with additional metadata."""
    # You can add more fields here if needed for extended outputs
    elapsed_time: Optional[float]  # Time taken for prediction in seconds