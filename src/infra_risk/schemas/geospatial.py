"""
Geospatial & Construction Progress Engine Schemas
Sentinel-2 imagery processing, spectral indices, site progress tracking
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class SatelliteProvider(str, Enum):
    SENTINEL_2 = "sentinel-2"
    LANDSAT_8 = "landsat-8"
    PLANET = "planet"
    MAXAR = "maxar"


class SpectralBand(str, Enum):
    """Sentinel-2 bands for surface reflectance"""
    B1 = "B1"  # Coastal aerosol
    B2 = "B2"  # Blue
    B3 = "B3"  # Green
    B4 = "B4"  # Red
    B5 = "B5"  # Red edge 1
    B6 = "B6"  # Red edge 2
    B7 = "B7"  # Red edge 3
    B8 = "B8"  # NIR
    B8A = "B8A"  # Narrow NIR
    B9 = "B9"  # Water vapor
    B10 = "B10"  # SWIR - Cirrus
    B11 = "B11"  # SWIR 1
    B12 = "B12"  # SWIR 2


class SatelliteImage(BaseModel):
    """Raw satellite image metadata"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    image_id: str = Field(..., description="Unique image identifier")
    provider: SatelliteProvider = SatelliteProvider.SENTINEL_2
    acquisition_date: datetime
    cloud_cover_pct: float = Field(ge=0, le=100)
    geometry: Dict[str, Any] = Field(..., description="GeoJSON geometry")
    bands: Dict[SpectralBand, np.ndarray] = Field(default_factory=dict)
    crs: str = "EPSG:4326"
    resolution_m: float = 10.0
    processing_level: str = "L2A"


class SpectralIndices(BaseModel):
    """Calculated spectral indices from satellite imagery"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ndvi: np.ndarray = Field(..., description="Normalized Difference Vegetation Index")
    ndbi: np.ndarray = Field(..., description="Normalized Difference Built-up Index")
    ndwi: Optional[np.ndarray] = Field(None, description="Normalized Difference Water Index")
    mndwi: Optional[np.ndarray] = Field(None, description="Modified Normalized Difference Water Index")
    bsi: Optional[np.ndarray] = Field(None, description="Bare Soil Index")
    nbr: Optional[np.ndarray] = Field(None, description="Normalized Burn Ratio")
    
    # Statistics
    ndvi_mean: float
    ndvi_std: float
    ndbi_mean: float
    ndbi_std: float
    calculation_date: datetime = Field(default_factory=datetime.utcnow)


class ConstructionPhase(str, Enum):
    SITE_PREPARATION = "site_preparation"
    EARTHWORK = "earthwork"
    FOUNDATION = "foundation"
    SUPERSTRUCTURE = "superstructure"
    ENVELOPE = "envelope"
    FIT_OUT = "fit_out"
    COMMISSIONING = "commissioning"
    OPERATIONAL = "operational"


class SiteProgress(BaseModel):
    """Site progress estimation from Siamese ResNet-50"""
    project_id: str
    timestamp: datetime
    phase: ConstructionPhase
    progress_pct: float = Field(ge=0, le=100, description="Overall progress percentage")
    phase_progress: Dict[ConstructionPhase, float] = Field(default_factory=dict)
    confidence_interval: tuple[float, float] = Field(description="(lower, upper) 95% CI")
    mape: float = Field(ge=0, description="Mean Absolute Percentage Error vs ground truth")
    spectral_indices: SpectralIndices
    reference_image_id: str
    current_image_id: str
    model_version: str = "siamese-resnet50-v1.0"


class AnomalyType(str, Enum):
    SITE_ABANDONMENT = "site_abandonment"
    SCOPE_CHANGE = "scope_change"
    DELAY = "delay"
    ACCELERATION = "acceleration"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    QUALITY_ISSUE = "quality_issue"


class ProgressAnomaly(BaseModel):
    """Detected anomaly in construction progress"""
    anomaly_id: str
    project_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: float = Field(ge=0, le=1, description="Anomaly severity score")
    description: str
    affected_area_pct: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    detected_by: str = "siamese-resnet50-anomaly-detector"
    evidence_images: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)