"""
Core Project Schema
Unified project model integrating all multi-modal data
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class ProjectSector(str, Enum):
    TRANSPORT = "transport"
    ENERGY = "energy"
    WATER = "water"
    TELECOM = "telecom"
    SOCIAL = "social"
    INDUSTRIAL = "industrial"


class ProjectSubSector(str, Enum):
    # Transport
    TOLL_ROAD = "toll_road"
    RAIL = "rail"
    PORT = "port"
    AIRPORT = "airport"
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    
    # Energy
    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    THERMAL = "thermal"
    TRANSMISSION = "transmission"
    DISTRIBUTION = "distribution"
    BATTERY_STORAGE = "battery_storage"
    GREEN_HYDROGEN = "green_hydrogen"
    
    # Water
    WATER_TREATMENT = "water_treatment"
    WASTEWATER = "wastewater"
    DESALINATION = "desalination"
    IRRIGATION = "irrigation"
    
    # Telecom
    FIBER = "fiber"
    TOWER = "tower"
    DATA_CENTER = "data_center"
    
    # Social
    HOSPITAL = "hospital"
    SCHOOL = "school"
    HOUSING = "housing"
    PRISON = "prison"
    
    # Industrial
    LOGISTICS = "logistics"
    MANUFACTURING = "manufacturing"


class ProjectStage(str, Enum):
    CONCEPT = "concept"
    PRE_FEASIBILITY = "pre_feasibility"
    FEASIBILITY = "feasibility"
    PROCUREMENT = "procurement"
    CONSTRUCTION = "construction"
    RAMP_UP = "ramp_up"
    OPERATIONAL = "operational"
    REFINANCING = "refinancing"
    EXTENSION = "extension"
    HANDOVER = "handover"


class ContractType(str, Enum):
    PPP = "ppp"
    BOT = "bot"
    BOOT = "boot"
    DBFOM = "dbfom"
    DBOM = "dbom"
    EPC = "epc"
    PPA = "ppa"
    MERCHANT = "merchant"
    REGULATED = "regulated"
    CONCESSION = "concession"


class Project(BaseModel):
    """Core project entity integrating all risk dimensions"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Identity
    project_id: str
    name: str
    description: Optional[str] = None
    
    # Classification
    sector: ProjectSector
    sub_sector: ProjectSubSector
    stage: ProjectStage
    contract_type: ContractType
    
    # Geography
    country: str
    country_code: str
    region: str
    city: Optional[str] = None
    latitude: float
    longitude: float
    
    # Timeline
    financial_close_date: Optional[datetime] = None
    construction_start: Optional[datetime] = None
    cod_target: Optional[datetime] = None  # Commercial Operations Date
    cod_actual: Optional[datetime] = None
    concession_end: Optional[datetime] = None
    concession_years: int
    
    # Financial
    total_capex: float
    capex_currency: str = "USD"
    debt_amount: float
    equity_amount: float
    debt_currency: str = "USD"
    equity_currency: str = "USD"
    senior_debt_pct: float
    mezzanine_pct: float = 0
    equity_irr_target: float
    
    # Operations
    capacity: float
    capacity_unit: str
    offtaker: Optional[str] = None
    offtake_currency: str = "USD"
    tariff_escalation: Optional[str] = None
    
    # Sponsors & Lenders
    sponsors: List[str] = Field(default_factory=list)
    lenders: List[str] = Field(default_factory=list)
    epc_contractor: Optional[str] = None
    om_contractor: Optional[str] = None
    
    # Risk assessments (populated by engines)
    geospatial_progress: Optional[Any] = None  # SiteProgress
    demand_forecast: Optional[Any] = None  # DemandForecast
    contagion_index: Optional[Any] = None  # ContagionIndex
    carul: Optional[Any] = None  # CARULOutput
    contract_risk: Optional[Any] = None  # ContractRiskScore
    credit_score: Optional[Any] = None  # CreditScore
    coverage_ratios: Optional[Any] = None  # CoverageRatios
    
    # ESG
    e_score: Optional[float] = None
    s_score: Optional[float] = None
    g_score: Optional[float] = None
    esg_certifications: List[str] = Field(default_factory=list)
    carbon_intensity: Optional[float] = None
    
    # Metadata
    data_sources: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1


class ProjectPortfolio(BaseModel):
    """Portfolio of projects"""
    portfolio_id: str
    name: str
    description: Optional[str] = None
    projects: List[Project] = Field(default_factory=list)
    total_capex: float = 0
    total_debt: float = 0
    total_equity: float = 0
    sector_allocation: Dict[ProjectSector, float] = Field(default_factory=dict)
    geographic_allocation: Dict[str, float] = Field(default_factory=dict)
    stage_allocation: Dict[ProjectStage, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataSourceConfig(BaseModel):
    """Configuration for external data sources"""
    # Satellite
    gee_project_id: Optional[str] = None
    gee_credentials_path: Optional[str] = None
    sentinel2_collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    landsat8_collection: str = "LANDSAT/LC08/C02/T1_L2"
    
    # Macro
    imf_api_key: Optional[str] = None
    world_bank_api_key: Optional[str] = None
    oecd_api_key: Optional[str] = None
    bis_api_key: Optional[str] = None
    
    # Financial
    bloomberg_api_key: Optional[str] = None
    refinitiv_api_key: Optional[str] = None
    s_and_p_api_key: Optional[str] = None
    moodys_api_key: Optional[str] = None
    fitch_api_key: Optional[str] = None
    
    # Climate
    ipcc_scenarios: List[str] = Field(default_factory=lambda: ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"])
    climate_data_source: str = "CMIP6"
    
    # Legal
    legal_db_api_key: Optional[str] = None


class MLflowConfig(BaseModel):
    """MLflow tracking configuration"""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "infra-risk"
    artifact_location: Optional[str] = None
    registry_uri: Optional[str] = None


class AppConfig(BaseModel):
    """Main application configuration"""
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501
    
    # Data
    data_dir: str = "./data"
    models_dir: str = "./models"
    outputs_dir: str = "./outputs"
    
    # Sub-configs
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    
    # Model versions
    geospatial_model_version: str = "siamese-resnet50-v1.0"
    demand_model_version: str = "tft-v1.0"
    gnn_model_version: str = "gnn-contagion-v1.0"
    pinn_model_version: str = "pinn-carul-v1.0"
    legal_model_version: str = "legal-bert-v1.0"
    credit_model_version: str = "credit-ensemble-v1.0"
    simulation_model_version: str = "rl-opponent-v1.0"