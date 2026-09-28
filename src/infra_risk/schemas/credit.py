"""
Credit Scoring Ensemble Schemas
Stacking ensemble: XGBoost + GNN features + TFT outputs -> PD, LGD, EL
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class CreditModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    GNN = "gnn"
    TFT = "tft"
    STACKING = "stacking"
    LOGISTIC = "logistic"


class CreditFeatures(BaseModel):
    """Consolidated features for credit scoring"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    project_id: str
    # Financial features
    dscr_current: Optional[float] = None
    dscr_forecast: Optional[np.ndarray] = None
    llcr: Optional[float] = None
    plcr: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_service_coverage: Optional[float] = None
    cash_sweep_pct: Optional[float] = None
    dsrca_balance: Optional[float] = None
    mra_balance: Optional[float] = None
    
    # Demand features (from TFT)
    demand_p50: Optional[np.ndarray] = None
    demand_volatility: Optional[float] = None
    demand_growth_rate: Optional[float] = None
    
    # GNN features
    contagion_index: Optional[float] = None
    centrality_score: Optional[float] = None
    systemic_importance: Optional[float] = None
    
    # Geospatial features
    construction_progress: Optional[float] = None
    progress_volatility: Optional[float] = None
    anomaly_count: Optional[int] = None
    
    # PINN features
    carul_years: Optional[float] = None
    degradation_rate: Optional[float] = None
    
    # Legal features
    contract_risk_score: Optional[float] = None
    covenant_strength: Optional[float] = None
    force_majeure_score: Optional[float] = None
    
    # Macro features
    sovereign_rating: Optional[str] = None
    gdp_growth: Optional[float] = None
    inflation: Optional[float] = None
    interest_rate: Optional[float] = None
    
    # Project features
    sector: Optional[str] = None
    country: Optional[str] = None
    stage: Optional[str] = None
    vintage: Optional[int] = None
    concession_life: Optional[int] = None
    
    # ESG features
    e_score: Optional[float] = None
    s_score: Optional[float] = None
    g_score: Optional[float] = None
    
    feature_vector: Optional[np.ndarray] = None
    feature_names: List[str] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class PDPrediction(BaseModel):
    """Probability of Default prediction"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    project_id: str
    pd_1yr: float = Field(ge=0, le=1)
    pd_5yr: float = Field(ge=0, le=1)
    pd_10yr: float = Field(ge=0, le=1)
    pd_lifetime: float = Field(ge=0, le=1)
    pd_term_structure: Dict[int, float] = Field(default_factory=dict)
    confidence_interval: tuple[float, float]
    model_contributions: Dict[CreditModelType, float] = Field(default_factory=dict)
    shap_values: Optional[np.ndarray] = None
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "credit-ensemble-v1.0"


class LGDPrediction(BaseModel):
    """Loss Given Default prediction"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    project_id: str
    lgd_point: float = Field(ge=0, le=1)
    lgd_distribution: Optional[np.ndarray] = None
    lgd_p10: float = Field(ge=0, le=1)
    lgd_p50: float = Field(ge=0, le=1)
    lgd_p90: float = Field(ge=0, le=1)
    recovery_rate: float = Field(ge=0, le=1)
    recovery_time_years: float
    collateral_coverage: float = Field(ge=0, le=1)
    seniority: str = "senior"
    jurisdiction_recovery: Optional[float] = None
    shap_values: Optional[np.ndarray] = None
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "credit-ensemble-v1.0"


class CreditScore(BaseModel):
    """Complete credit assessment"""
    project_id: str
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Core predictions
    pd: PDPrediction
    lgd: LGDPrediction
    el: ELDistribution
    
    # Risk rating
    internal_rating: str = Field(description="AAA to D")
    implied_spread_bps: float
    
    # Explainability
    shap_global_importance: Dict[str, float] = Field(default_factory=dict)
    shap_local_values: Dict[str, float] = Field(default_factory=dict)
    counterfactuals: List[Dict[str, Any]] = Field(default_factory=list)
    key_drivers: List[str] = Field(default_factory=list)
    mitigation_actions: List[str] = Field(default_factory=list)
    
    # Model metadata
    ensemble_weights: Dict[CreditModelType, float] = Field(default_factory=dict)
    validation_auroc: float = Field(ge=0, le=1)
    validation_auprc: float = Field(ge=0, le=1)
    calibration_score: float
    model_version: str = "credit-ensemble-v1.0"


class EnsembleConfig(BaseModel):
    """Stacking ensemble configuration"""
    base_models: List[CreditModelType] = Field(
        default_factory=lambda: [CreditModelType.XGBOOST, CreditModelType.GNN, CreditModelType.TFT]
    )
    meta_learner: str = "logistic"  # logistic, xgboost, neural_net
    cv_folds: int = 5
    use_probas: bool = True
    passthrough_features: bool = True
    
    # XGBoost params
    xgb_params: Dict[str, Any] = Field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc"
    })
    
    # LightGBM params
    lgbm_params: Dict[str, Any] = Field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "metric": "auc"
    })


class ModelPerformance(BaseModel):
    """Model validation performance"""
    model_type: CreditModelType
    auroc: float
    auprc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    calibration_slope: float
    calibration_intercept: float
    hosmer_lemeshow_p: float
    ks_statistic: float
    gini: float
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)