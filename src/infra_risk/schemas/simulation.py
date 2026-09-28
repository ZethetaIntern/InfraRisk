"""
InfraRisk Lab Simulation Engine Schemas
Gamified simulation with time engine, decision engine, event engine, AI opponent
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class GameMode(str, Enum):
    SINGLE_DEAL = "single_deal"       # Tutorial
    PORTFOLIO_MANAGER = "portfolio_manager"  # Standard 10-15 deals
    CRISIS_MANAGER = "crisis_manager"  # Workout/Restructuring
    DEAL_STRUCTURER = "deal_structurer"  # Specialist Greenfield


class SimulationPhase(str, Enum):
    SETUP = "setup"
    QUARTERLY_DECISION = "quarterly_decision"
    EVENT_RESOLUTION = "event_resolution"
    SCORING = "scoring"
    COMPLETE = "complete"


class TimeEngineConfig(BaseModel):
    """Time engine configuration"""
    concession_years: int = 25
    real_time_minutes: int = 90
    time_step: str = "quarterly"  # quarterly, semi_annual, annual
    steps_per_year: int = 4
    total_steps: int = 100  # 25 years * 4 quarters
    minutes_per_step: float = 0.9  # 90 min / 100 steps


class DecisionType(str, Enum):
    CHANGE_ORDER = "change_order"
    TARIFF_ADJUSTMENT = "tariff_adjustment"
    COVENANT_WAIVER = "covenant_waiver"
    CAPEX_APPROVAL = "capex_approval"
    REFINANCING = "refinancing"
    HEDGING = "hedging"
    DIVIDEND_POLICY = "dividend_policy"
    ESG_INVESTMENT = "esg_investment"
    CONTRACT_RENEGOTIATION = "contract_renegotiation"
    ASSET_SALE = "asset_sale"


class Decision(BaseModel):
    """Player decision"""
    decision_id: str
    player_id: str
    project_id: Optional[str] = None
    decision_type: DecisionType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    quarter: int
    cost: float = 0
    expected_impact: Dict[str, float] = Field(default_factory=dict)
    is_executed: bool = False


class EventType(str, Enum):
    RATE_SHOCK = "rate_shock"
    SOVEREIGN_DOWNGRADE = "sovereign_downgrade"
    PANDEMIC = "pandemic"
    CLIMATE_DAMAGE = "climate_damage"
    POLITICAL_UNREST = "political_unrest"
    REGULATORY_CHANGE = "regulatory_change"
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    COMMODITY_SHOCK = "commodity_shock"
    CURRENCY_CRISIS = "currency_crisis"
    COUNTERPARTY_DEFAULT = "counterparty_default"
    FORCE_MAJEURE = "force_majeure"
    ESG_INCIDENT = "esg_incident"


class EventSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class StochasticEvent(BaseModel):
    """Stochastic event from event engine"""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    description: str
    probability: float = Field(ge=0, le=1)
    impact_parameters: Dict[str, float] = Field(default_factory=dict)
    affected_projects: List[str] = Field(default_factory=list)
    affected_portfolio: bool = False
    trigger_quarter: Optional[int] = None
    duration_quarters: int = 1
    is_triggered: bool = False
    resolved: bool = False


class EventEngineConfig(BaseModel):
    """Event engine configuration"""
    base_event_rate: float = 0.15  # events per quarter
    rate_shock_magnitude_bps: int = 300
    sovereign_downgrade_notches: int = 2
    climate_damage_pct_range: tuple[float, float] = (5, 40)
    pandemic_duration_quarters: tuple[int, int] = (4, 12)
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    scenario_weights: Dict[str, float] = Field(default_factory=dict)


class AIAction(BaseModel):
    """AI opponent action"""
    action_id: str
    action_type: DecisionType
    project_id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    confidence: float = Field(ge=0, le=1)
    expected_score_impact: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AIOpponentConfig(BaseModel):
    """AI opponent configuration"""
    model_version: str = "rl-opponent-v1.0"
    risk_aversion: float = 0.5
    return_target: float = 0.15
    max_leverage: float = 0.80
    min_dscr: float = 1.30
    diversification_target: float = 0.70
    esg_weight: float = 0.10
    lookback_quarters: int = 8
    exploration_rate: float = 0.1


class ScoringCategory(str, Enum):
    CREDIT_RISK = "credit_risk_management"
    FINANCIAL_STRUCTURING = "financial_structuring"
    PORTFOLIO_DIVERSIFICATION = "portfolio_diversification"
    CRISIS_RESPONSE = "crisis_response"
    MARKET_TIMING = "market_timing"
    ESG_INTEGRATION = "esg_integration"


class ScoringWeights(BaseModel):
    """1000-point scoring system weights"""
    credit_risk_management: int = 300
    financial_structuring: int = 200
    portfolio_diversification: int = 150
    crisis_response: int = 150
    market_timing: int = 100
    esg_integration: int = 100
    
    def total(self) -> int:
        return sum([
            self.credit_risk_management,
            self.financial_structuring,
            self.portfolio_diversification,
            self.crisis_response,
            self.market_timing,
            self.esg_integration
        ])


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown"""
    credit_risk_management: float = 0
    financial_structuring: float = 0
    portfolio_diversification: float = 0
    crisis_response: float = 0
    market_timing: float = 0
    esg_integration: float = 0
    total: float = 0
    percentile: Optional[float] = None


class SimulationState(BaseModel):
    """Complete simulation state"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    simulation_id: str
    game_mode: GameMode
    phase: SimulationPhase = SimulationPhase.SETUP
    current_quarter: int = 0
    total_quarters: int = 100
    
    # Players
    player_id: str
    ai_opponent_id: str
    
    # Portfolio
    portfolio_projects: List[str] = Field(default_factory=list)
    available_projects: List[str] = Field(default_factory=list)
    
    # Financial state
    cash_balance: float = 0
    total_debt: float = 0
    equity_value: float = 0
    
    # Scores
    player_score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    ai_score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    
    # History
    decisions: List[Decision] = Field(default_factory=list)
    events: List[StochasticEvent] = Field(default_factory=list)
    ai_actions: List[AIAction] = Field(default_factory=list)
    quarterly_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Config
    time_config: TimeEngineConfig = Field(default_factory=TimeEngineConfig)
    event_config: EventEngineConfig = Field(default_factory=EventEngineConfig)
    ai_config: AIOpponentConfig = Field(default_factory=AIOpponentConfig)
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SimulationResult(BaseModel):
    """Final simulation results"""
    simulation_id: str
    game_mode: GameMode
    player_id: str
    final_score: ScoreBreakdown
    ai_final_score: ScoreBreakdown
    winner: str  # "player" or "ai"
    total_quarters_played: int
    
    # Performance metrics
    portfolio_irr: float
    portfolio_moic: float
    max_drawdown: float
    avg_dscr: float
    min_dscr: float
    default_count: int
    restructuring_count: int
    
    # Decision quality
    decision_quality_score: float
    timing_score: float
    risk_management_score: float
    
    completed_at: datetime = Field(default_factory=datetime.utcnow)