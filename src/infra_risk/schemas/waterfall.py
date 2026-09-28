"""
Cash Flow Waterfall & Coverage Ratios Schemas
Priority of payments modeling, DSCR/LLCR/PLCR calculation
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class WaterfallTierType(str, Enum):
    GROSS_REVENUE = "gross_revenue"
    OPERATING_EXPENSES = "operating_expenses"
    TAXES = "taxes"
    SENIOR_DEBT_SERVICE = "senior_debt_service"
    DSRA_FUNDING = "dsra_funding"
    MRA_FUNDING = "mra_funding"
    CASH_SWEEP = "cash_sweep"
    SUBORDINATED_DEBT = "subordinated_debt"
    EQUITY_DISTRIBUTION = "equity_distribution"


class WaterfallTier(BaseModel):
    """Individual tier in the cash flow waterfall"""
    tier_id: str
    tier_type: WaterfallTierType
    priority: int = Field(..., description="Lower number = higher priority")
    name: str
    description: str
    
    # Calculation parameters
    calculation_method: str  # fixed, percentage, formula, min/max
    amount: Optional[float] = None
    percentage: Optional[float] = None
    formula: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    
    # Reserve account specifics
    target_balance: Optional[float] = None
    funding_priority: Optional[int] = None
    
    # Sweep specifics
    sweep_trigger_dscr: Optional[float] = None
    sweep_percentage: Optional[float] = None
    
    is_active: bool = True


class CashFlowWaterfall(BaseModel):
    """Complete cash flow waterfall model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    project_id: str
    period_start: datetime
    period_end: datetime
    currency: str = "USD"
    
    # Inputs
    gross_revenue: float
    operating_expenses: float
    taxes: float
    
    # Debt service
    senior_interest: float
    senior_principal: float
    subordinated_interest: float = 0
    subordinated_principal: float = 0
    
    # Reserve accounts
    dsra_opening: float = 0
    dsra_target: float = 0
    mra_opening: float = 0
    mra_target: float = 0
    
    # Waterfall tiers
    tiers: List[WaterfallTier] = Field(default_factory=list)
    
    # Results per tier
    tier_results: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Summary
    cfads: float = Field(description="Cash Flow Available for Debt Service")
    dscr: Optional[float] = None
    llcr: Optional[float] = None
    plcr: Optional[float] = None
    
    # Equity
    equity_irr: Optional[float] = None
    equity_npv: Optional[float] = None
    equity_cashflows: Optional[np.ndarray] = None
    
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class CoverageRatios(BaseModel):
    """Debt coverage ratios"""
    project_id: str
    calculation_date: datetime
    
    # DSCR
    cfads: float
    interest: float
    scheduled_principal: float
    dscr: float = Field(description="CFADS / (Interest + Scheduled Principal)")
    dscr_forecast: Optional[Dict[int, float]] = None  # year -> DSCR
    
    # LLCR
    npv_cfads_loan_life: float
    outstanding_debt: float
    llcr: float = Field(description="NPV(CFADS_loan_life) / Outstanding Debt")
    loan_life_years: float
    discount_rate: float
    
    # PLCR
    npv_cfads_concession_life: float
    plcr: float = Field(description="NPV(CFADS_concession_life) / Outstanding Debt")
    concession_life_years: float
    
    # Additional ratios
    debt_to_equity: Optional[float] = None
    debt_to_capital: Optional[float] = None
    interest_coverage: Optional[float] = None
    cash_flow_to_debt: Optional[float] = None
    
    # Sensitivities
    dscr_sensitivity: Dict[str, float] = Field(default_factory=dict)
    llcr_sensitivity: Dict[str, float] = Field(default_factory=dict)
    plcr_sensitivity: Dict[str, float] = Field(default_factory=dict)


class DebtFacility(BaseModel):
    """Individual debt facility"""
    facility_id: str
    name: str
    facility_type: str  # senior, mezzanine, subordinated, bond
    principal: float
    interest_rate: float
    is_floating: bool = False
    margin_bps: Optional[float] = None
    benchmark: Optional[str] = None  # SOFR, EURIBOR, etc.
    tenor_years: int
    grace_period_years: int = 0
    repayment_profile: str = "sculpted"  # sculpted, annuity, bullet, custom
    custom_schedule: Optional[List[Dict[str, float]]] = None
    
    # Fees
    upfront_fee_bps: float = 0
    commitment_fee_bps: float = 0
    agency_fee_annual: float = 0
    
    # Covenants
    min_dscr: float = 1.30
    min_llcr: float = 1.20
    max_leverage: Optional[float] = None
    
    # Reserve accounts
    dsra_required: bool = True
    dsra_size_months: int = 6
    mra_required: bool = True
    mra_annual_funding: Optional[float] = None


class DebtStructure(BaseModel):
    """Complete debt structure"""
    project_id: str
    facilities: List[DebtFacility]
    total_debt: float
    weighted_avg_cost: float
    weighted_avg_life: float
    senior_debt_pct: float
    mezzanine_pct: float
    subordinated_pct: float
    hedge_ratio: float = 0


class SculptingParams(BaseModel):
    """Debt sculpting parameters"""
    target_dscr: float = 1.30
    min_dscr: float = 1.20
    max_dscr: float = 2.00
    sculpting_method: str = "proportional"  # proportional, flat, custom
    tail_percentage: float = 0.10  # % of principal in tail


class HedgingStrategy(BaseModel):
    """Interest rate hedging strategy"""
    strategy_id: str
    hedge_type: str  # swap, cap, collar, swaption
    notional: float
    fixed_rate: Optional[float] = None
    strike_rate: Optional[float] = None
    cap_rate: Optional[float] = None
    floor_rate: Optional[float] = None
    start_date: datetime
    end_date: datetime
    cost_bps: float = 0