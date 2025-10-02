from pydantic import BaseModel


class ImageStatistics(BaseModel):
    image_dimensions: tuple[int, int]
    total_pixels: int
    water_pixels: int
    land_pixels: int
    water_percentage: float
    land_percentage: float
    pixel_size_meters: float
    water_area_m2: float
    water_area_km2: float
    water_area_hectares: float
    total_area_km2: float
