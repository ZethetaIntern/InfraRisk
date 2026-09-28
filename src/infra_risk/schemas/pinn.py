"""
Physics-Informed Structural Degradation (PINN) Schemas
AASHTO, Paris' Law, carbonation, corrosion models embedded in neural loss functions
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class AssetType(str, Enum):
    PAVEMENT = "pavement"
    BRIDGE_DECK = "bridge_deck"
    BRIDGE_STEEL = "bridge_steel"
    CONCRETE_STRUCTURE = "concrete_structure"
    STEEL_STRUCTURE = "steel_structure"
    PIPELINE = "pipeline"
    TUNNEL = "tunnel"


class DegradationMechanism(str, Enum):
    AASHTO_PAVEMENT = "aashto_pavement"
    PARIS_LAW_FATIGUE = "paris_law_fatigue"
    CONCRETE_CARBONATION = "concrete_carbonation"
    CHLORIDE_INDUSTION = "chloride_induction"
    ATMOSPHERIC_CORROSION = "atmospheric_corrosion"
    FREEZE_THAW = "freeze_thaw"
    ASR = "alkali_silica_reaction"


class PhysicsParameters(BaseModel):
    """Physics-based degradation parameters"""
    # Paris' Law parameters
    paris_C: Optional[float] = Field(None, description="Paris' Law coefficient")
    paris_m: Optional[float] = Field(None, description="Paris' Law exponent")
    delta_K_threshold: Optional[float] = Field(None, description="Stress intensity threshold")
    
    # Carbonation parameters
    carbonation_K: Optional[float] = Field(None, description="Carbonation coefficient (mm/sqrt(year))")
    carbonation_co2_concentration: Optional[float] = Field(None, description="CO2 concentration (%)")
    
    # Corrosion parameters
    corrosion_rate: Optional[float] = Field(None, description="Corrosion rate (mm/year)")
    chloride_diffusion: Optional[float] = Field(None, description="Chloride diffusion coefficient")
    chloride_surface: Optional[float] = Field(None, description="Surface chloride concentration")
    
    # AASHTO pavement
    aashto_initial_psi: Optional[float] = Field(None, description="Initial Present Serviceability Index")
    aashto_terminal_psi: Optional[float] = Field(None, description="Terminal PSI")
    aashto_traffic_factor: Optional[float] = Field(None, description="Traffic growth factor")
    
    # Climate factors
    temperature_avg: Optional[float] = None
    humidity_avg: Optional[float] = None
    precipitation: Optional[float] = None
    freeze_thaw_cycles: Optional[int] = None
    
    # IPCC scenario
    climate_scenario: str = "SSP2-4.5"


class DegradationModel(BaseModel):
    """PINN degradation model configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    asset_id: str
    asset_type: AssetType
    mechanisms: List[DegradationMechanism]
    physics_params: PhysicsParameters
    geometry: Dict[str, float] = Field(default_factory=dict)  # dimensions
    material_properties: Dict[str, float] = Field(default_factory=dict)
    inspection_history: List[Dict[str, Any]] = Field(default_factory=list)
    maintenance_history: List[Dict[str, Any]] = Field(default_factory=list)
    model_version: str = "pinn-degradation-v1.0"


class DegradationState(BaseModel):
    """Current degradation state"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamp: datetime
    asset_id: str
    # Pavement
    psi: Optional[float] = None
    rut_depth_mm: Optional[float] = None
    cracking_pct: Optional[float] = None
    iri: Optional[float] = None  # International Roughness Index
    
    # Concrete
    carbonation_depth_mm: Optional[float] = None
    chloride_content_pct: Optional[float] = None
    corrosion_potential_mv: Optional[float] = None
    crack_width_mm: Optional[float] = None
    
    # Steel
    section_loss_pct: Optional[float] = None
    fatigue_crack_length_mm: Optional[float] = None
    coating_condition: Optional[float] = None
    
    # General
    condition_rating: Optional[int] = Field(None, ge=1, le=5)  # 1=excellent, 5=failed
    reliability_index: Optional[float] = None


class CARULOutput(BaseModel):
    """Climate-Adjusted Remaining Useful Life output"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    asset_id: str
    assessment_date: datetime
    # Base RUL (no climate adjustment)
    base_rul_years: float
    # Climate-adjusted RUL per scenario
    carul_by_scenario: Dict[str, float] = Field(description="RUL per IPCC scenario")
    # Probabilistic RUL
    rul_p10: float
    rul_p50: float
    rul_p90: float
    # Degradation trajectories
    degradation_trajectories: Dict[str, Dict[str, np.ndarray]] = Field(default_factory=dict)
    # Key milestones
    intervention_threshold_year: Optional[float] = None
    major_rehab_year: Optional[float] = None
    replacement_year: Optional[float] = None
    # Sensitivity
    sensitivity_analysis: Dict[str, float] = Field(default_factory=dict)
    model_version: str = "pinn-carul-v1.0"


class PINNLossComponents(BaseModel):
    """Components of PINN loss function"""
    data_loss: float
    physics_loss: float
    boundary_loss: float
    initial_condition_loss: float
    total_loss: float
    weights: Dict[str, float] = Field(default_factory=dict)


class PINNTrainingConfig(BaseModel):
    """PINN training configuration"""
    epochs: int = 5000
    learning_rate: float = 1e-3
    lr_scheduler: str = "cosine"
    physics_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 0.5
    network_architecture: List[int] = Field(default_factory=lambda: [64, 64, 64, 64, 1])
    activation: str = "tanh"
    optimizer: str = "adam"
    early_stopping_patience: int = 500