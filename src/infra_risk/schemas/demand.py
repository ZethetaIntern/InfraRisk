"""
Demand Forecasting Module Schemas
Temporal Fusion Transformer for multi-horizon probabilistic demand forecasting
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class SectorType(str, Enum):
    TOLL_ROAD = "toll_road"
    ELECTRICITY = "electricity"
    PORT = "port"
    AIRPORT = "airport"
    RAIL = "rail"
    WATER = "water"
    TELECOM = "telecom"
    GAS = "gas"


class DemandMetric(str, Enum):
    """Sector-specific demand metrics"""
    ADT = "adt"  # Average Daily Traffic (toll roads)
    DISPATCH_MWH = "dispatch_mwh"  # Electricity dispatch
    TEU = "teu"  # Twenty-foot Equivalent Units (ports)
    PASSENGERS = "passengers"  # Airport passenger throughput
    TONNAGE = "tonnage"  # Rail/port tonnage
    VOLUME_M3 = "volume_m3"  # Water volume


class TimeHorizon(str, Enum):
    SHORT = "short"   # 1-4 quarters
    MEDIUM = "medium"  # 1-3 years
    LONG = "long"     # 3-10 years


class QuantileForecast(BaseModel):
    """Probabilistic forecast with quantiles"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    p10: np.ndarray = Field(..., description="10th percentile (P10)")
    p50: np.ndarray = Field(..., description="50th percentile / Median (P50)")
    p90: np.ndarray = Field(..., description="90th percentile (P90)")
    timestamps: List[datetime] = Field(..., description="Forecast timestamps")
    horizon: TimeHorizon


class StaticCovariates(BaseModel):
    """Static project covariates for TFT"""
    project_id: str
    sector: SectorType
    country: str
    region: str
    capacity: float = Field(..., description="Nameplate capacity")
    vintage_year: int
    concession_years: int
    gdp_per_capita: float
    population_catchment: float
    gdp_growth_rate: float
    urbanization_rate: float
    institutional_quality: float = Field(ge=0, le=100)
    latitude: float
    longitude: float


class KnownFutureInputs(BaseModel):
    """Known future events/inputs for TFT"""
    timestamps: List[datetime]
    scheduled_expansions: List[Dict[str, Any]] = Field(default_factory=list)
    policy_changes: List[Dict[str, Any]] = Field(default_factory=list)
    macro_projections: Dict[str, List[float]] = Field(default_factory=dict)
    climate_scenario: str = "SSP2-4.5"


class HistoricalVariables(BaseModel):
    """Historical time-varying variables for TFT"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamps: List[datetime]
    demand_history: np.ndarray
    gdp_history: np.ndarray
    inflation_history: np.ndarray
    interest_rate_history: np.ndarray
    exchange_rate_history: np.ndarray
    commodity_price_history: np.ndarray
    weather_variables: Dict[str, np.ndarray] = Field(default_factory=dict)


class SectorDemandInput(BaseModel):
    """Complete input for sector demand forecasting"""
    static_covariates: StaticCovariates
    known_future: KnownFutureInputs
    historical: HistoricalVariables
    forecast_horizon_quarters: int = 40  # 10 years quarterly


class DemandForecast(BaseModel):
    """Complete demand forecast output"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    project_id: str
    sector: SectorType
    metric: DemandMetric
    forecast_date: datetime = Field(default_factory=datetime.utcnow)
    quantiles: QuantileForecast
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    model_version: str = "tft-v1.0"
    validation_metrics: Dict[str, float] = Field(default_factory=dict)


class MacroProfile(BaseModel):
    """Country-level macroeconomic profile"""
    country_code: str
    country_name: str
    year: int
    gdp_usd: float
    gdp_growth: float
    inflation: float
    interest_rate: float
    exchange_rate_usd: float
    current_account_pct_gdp: float
    debt_to_gdp: float
    sovereign_rating: str
    institutional_quality: float
    political_stability: float
    rule_of_law: float
    control_of_corruption: float
    infrastructure_quality: float
    energy_consumption_per_capita: float
    co2_emissions_per_capita: float
    population: int
    urbanization_rate: float