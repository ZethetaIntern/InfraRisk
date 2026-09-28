"""
Cash Flow Waterfall Engine
Models priority of payments and calculates DSCR, LLCR, PLCR
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import numpy_financial as npf

from src.infra_risk.schemas.waterfall import (
    CashFlowWaterfall,
    WaterfallTier,
    WaterfallTierType,
    CoverageRatios,
    DebtFacility,
    DebtStructure,
    SculptingParams,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class WaterfallEngine:
    """Cash flow waterfall and coverage ratio calculator"""
    
    def __init__(self):
        logger.info("WaterfallEngine initialized")
    
    def calculate_waterfall(
        self,
        project_id: str,
        period_start: datetime,
        period_end: datetime,
        gross_revenue: float,
        operating_expenses: float,
        taxes: float,
        debt_facilities: List[DebtFacility],
        dsra_opening: float = 0,
        dsra_target: float = 0,
        mra_opening: float = 0,
        mra_target: float = 0,
        tiers: Optional[List[WaterfallTier]] = None,
    ) -> CashFlowWaterfall:
        """Calculate complete cash flow waterfall"""
        
        # Default tiers if not provided
        if tiers is None:
            tiers = self._get_default_tiers(debt_facilities, dsra_target, mra_target)
        
        # Calculate debt service
        senior_interest = sum(
            f.principal * f.interest_rate * (1 if not f.is_floating else 1)  # Simplified
            for f in debt_facilities if f.facility_type == "senior"
        )
        senior_principal = sum(
            f.principal / max(1, f.tenor_years - f.grace_period_years)
            for f in debt_facilities if f.facility_type == "senior"
        )
        subordinated_interest = sum(
            f.principal * f.interest_rate
            for f in debt_facilities if f.facility_type in ["mezzanine", "subordinated"]
        )
        subordinated_principal = sum(
            f.principal / max(1, f.tenor_years - f.grace_period_years)
            for f in debt_facilities if f.facility_type in ["mezzanine", "subordinated"]
        )
        
        # CFADS
        cfads = gross_revenue - operating_expenses - taxes
        
        # Run waterfall
        tier_results = {}
        available_cash = cfads
        
        for tier in sorted(tiers, key=lambda t: t.priority):
            if not tier.is_active:
                continue
            
            result = self._calculate_tier(tier, available_cash, {
                "cfads": cfads,
                "senior_interest": senior_interest,
                "senior_principal": senior_principal,
                "subordinated_interest": subordinated_interest,
                "subordinated_principal": subordinated_principal,
                "dsra_opening": dsra_opening,
                "dsra_target": dsra_target,
                "mra_opening": mra_opening,
                "mra_target": mra_target,
            })
            
            tier_results[tier.tier_id] = result
            available_cash -= result.get("amount_paid", 0)
        
        # Calculate coverage ratios
        dscr = cfads / (senior_interest + senior_principal) if (senior_interest + senior_principal) > 0 else 0
        
        return CashFlowWaterfall(
            project_id=project_id,
            period_start=period_start,
            period_end=period_end,
            gross_revenue=gross_revenue,
            operating_expenses=operating_expenses,
            taxes=taxes,
            senior_interest=senior_interest,
            senior_principal=senior_principal,
            subordinated_interest=subordinated_interest,
            subordinated_principal=subordinated_principal,
            dsra_opening=dsra_opening,
            dsra_target=dsra_target,
            mra_opening=mra_opening,
            mra_target=mra_target,
            tiers=tiers,
            tier_results=tier_results,
            cfads=cfads,
            dscr=dscr,
        )
    
    def _get_default_tiers(
        self,
        debt_facilities: List[DebtFacility],
        dsra_target: float,
        mra_target: float,
    ) -> List[WaterfallTier]:
        """Get default waterfall tiers"""
        return [
            WaterfallTier(
                tier_id="1_revenue",
                tier_type=WaterfallTierType.GROSS_REVENUE,
                priority=1,
                name="Gross Revenue",
                description="Total project revenue",
                calculation_method="fixed",
                amount=0,  # Input
            ),
            WaterfallTier(
                tier_id="2_opex",
                tier_type=WaterfallTierType.OPERATING_EXPENSES,
                priority=2,
                name="Operating Expenses",
                description="O&M costs",
                calculation_method="fixed",
                amount=0,  # Input
            ),
            WaterfallTier(
                tier_id="3_taxes",
                tier_type=WaterfallTierType.TAXES,
                priority=3,
                name="Taxes",
                description="Corporate income taxes",
                calculation_method="fixed",
                amount=0,  # Input
            ),
            WaterfallTier(
                tier_id="4_senior_debt",
                tier_type=WaterfallTierType.SENIOR_DEBT_SERVICE,
                priority=4,
                name="Senior Debt Service",
                description="Senior interest and principal",
                calculation_method="formula",
                formula="senior_interest + senior_principal",
            ),
            WaterfallTier(
                tier_id="5_dsra",
                tier_type=WaterfallTierType.DSRA_FUNDING,
                priority=5,
                name="DSRA Funding",
                description="Debt Service Reserve Account",
                calculation_method="formula",
                formula="max(0, dsra_target - dsra_opening)",
                target_balance=dsra_target,
            ),
            WaterfallTier(
                tier_id="6_mra",
                tier_type=WaterfallTierType.MRA_FUNDING,
                priority=6,
                name="MRA Funding",
                description="Maintenance Reserve Account",
                calculation_method="formula",
                formula="max(0, mra_target - mra_opening)",
                target_balance=mra_target,
            ),
            WaterfallTier(
                tier_id="7_sweep",
                tier_type=WaterfallTierType.CASH_SWEEP,
                priority=7,
                name="Cash Sweep",
                description="Excess cash sweep to debt prepayment",
                calculation_method="formula",
                formula="available_cash * sweep_percentage",
                sweep_trigger_dscr=1.3,
                sweep_percentage=0.5,
            ),
            WaterfallTier(
                tier_id="8_sub_debt",
                tier_type=WaterfallTierType.SUBORDINATED_DEBT,
                priority=8,
                name="Subordinated Debt Service",
                description="Mezzanine/subordinated interest and principal",
                calculation_method="formula",
                formula="subordinated_interest + subordinated_principal",
            ),
            WaterfallTier(
                tier_id="9_equity",
                tier_type=WaterfallTierType.EQUITY_DISTRIBUTION,
                priority=9,
                name="Equity Distribution",
                description="Distributions to equity holders",
                calculation_method="formula",
                formula="available_cash",
            ),
        ]
    
    def _calculate_tier(
        self,
        tier: WaterfallTier,
        available_cash: float,
        context: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate payment for a single tier"""
        result = {"amount_due": 0, "amount_paid": 0, "shortfall": 0}
        
        if tier.calculation_method == "fixed" and tier.amount is not None:
            result["amount_due"] = tier.amount
        elif tier.calculation_method == "formula" and tier.formula:
            # Evaluate formula with context
            try:
                result["amount_due"] = eval(tier.formula, {"__builtins__": {}}, context)
            except:
                result["amount_due"] = 0
        elif tier.calculation_method == "percentage" and tier.percentage is not None:
            result["amount_due"] = available_cash * tier.percentage
        
        # Apply min/max
        if tier.min_amount is not None:
            result["amount_due"] = max(result["amount_due"], tier.min_amount)
        if tier.max_amount is not None:
            result["amount_due"] = min(result["amount_due"], tier.max_amount)
        
        # Pay from available cash
        result["amount_paid"] = min(result["amount_due"], max(0, available_cash))
        result["shortfall"] = max(0, result["amount_due"] - result["amount_paid"])
        
        return result
    
    def calculate_coverage_ratios(
        self,
        project_id: str,
        cfads_forecast: np.ndarray,
        debt_facilities: List[DebtFacility],
        discount_rate: float = 0.08,
        loan_life_years: float = 20,
        concession_life_years: float = 25,
    ) -> CoverageRatios:
        """Calculate DSCR, LLCR, PLCR"""
        
        # Current period debt service
        senior_interest = sum(
            f.principal * f.interest_rate
            for f in debt_facilities if f.facility_type == "senior"
        )
        senior_principal = sum(
            self._calculate_principal_payment(f)
            for f in debt_facilities if f.facility_type == "senior"
        )
        
        # DSCR
        cfads_current = cfads_forecast[0] if len(cfads_forecast) > 0 else 0
        total_debt_service = senior_interest + senior_principal
        dscr = cfads_current / total_debt_service if total_debt_service > 0 else 0
        
        # DSCR forecast
        dscr_forecast = {}
        for i, cfads in enumerate(cfads_forecast):
            year = i // 4 + 1
            ds = self._get_debt_service_year(debt_facilities, year)
            dscr_forecast[year] = cfads / ds if ds > 0 else 0
        
        # LLCR
        npv_cfads_loan = sum(
            cfads_forecast[i] / (1 + discount_rate) ** (i / 4)
            for i in range(min(len(cfads_forecast), int(loan_life_years * 4)))
        )
        outstanding_debt = sum(f.principal for f in debt_facilities)
        llcr = npv_cfads_loan / outstanding_debt if outstanding_debt > 0 else 0
        
        # PLCR
        npv_cfads_concession = sum(
            cfads_forecast[i] / (1 + discount_rate) ** (i / 4)
            for i in range(len(cfads_forecast))
        )
        plcr = npv_cfads_concession / outstanding_debt if outstanding_debt > 0 else 0
        
        return CoverageRatios(
            project_id=project_id,
            calculation_date=datetime.utcnow(),
            cfads=cfads_current,
            interest=senior_interest,
            scheduled_principal=senior_principal,
            dscr=dscr,
            dscr_forecast=dscr_forecast,
            npv_cfads_loan_life=npv_cfads_loan,
            outstanding_debt=outstanding_debt,
            llcr=llcr,
            loan_life_years=loan_life_years,
            discount_rate=discount_rate,
            npv_cfads_concession_life=npv_cfads_concession,
            plcr=plcr,
            concession_life_years=concession_life_years,
        )
    
    def _calculate_principal_payment(self, facility: DebtFacility) -> float:
        """Calculate annual principal payment"""
        if facility.repayment_profile == "annuity":
            return -npf.ppmt(facility.interest_rate, 1, facility.tenor_years, -facility.principal)
        elif facility.repayment_profile == "sculpted":
            return facility.principal / max(1, facility.tenor_years - facility.grace_period_years)
        elif facility.repayment_profile == "bullet":
            return 0 if facility.tenor_years > 1 else facility.principal
        else:
            return facility.principal / max(1, facility.tenor_years)
    
    def _get_debt_service_year(self, facilities: List[DebtFacility], year: int) -> float:
        """Get debt service for a specific year"""
        total = 0
        for f in facilities:
            if year <= f.grace_period_years:
                total += f.principal * f.interest_rate
            elif year <= f.tenor_years:
                total += f.principal * f.interest_rate + self._calculate_principal_payment(f)
        return total
    
    def project_cashflows(
        self,
        project_id: str,
        revenue_forecast: np.ndarray,
        opex_forecast: np.ndarray,
        tax_rate: float,
        debt_facilities: List[DebtFacility],
        dsra_target: float,
        mra_target: float,
        initial_dsra: float = 0,
        initial_mra: float = 0,
    ) -> Dict[str, np.ndarray]:
        """Project cashflows over concession life"""
        
        n_periods = len(revenue_forecast)
        results = {
            "cfads": np.zeros(n_periods),
            "dscr": np.zeros(n_periods),
            "dsra_balance": np.zeros(n_periods),
            "mra_balance": np.zeros(n_periods),
            "equity_cashflow": np.zeros(n_periods),
            "debt_balance": np.zeros(n_periods),
        }
        
        dsra_balance = initial_dsra
        mra_balance = initial_mra
        debt_balances = {f.facility_id: f.principal for f in debt_facilities}
        
        for i in range(n_periods):
            # Revenue and opex
            revenue = revenue_forecast[i]
            opex = opex_forecast[i]
            taxes = max(0, (revenue - opex) * tax_rate)
            
            # Debt service
            senior_interest = sum(
                debt_balances.get(f.facility_id, 0) * f.interest_rate
                for f in debt_facilities if f.facility_type == "senior"
            )
            senior_principal = sum(
                self._calculate_principal_payment(f)
                for f in debt_facilities if f.facility_type == "senior"
            )
            
            # CFADS
            cfads = revenue - opex - taxes
            
            # Waterfall
            waterfall = self.calculate_waterfall(
                project_id=project_id,
                period_start=datetime.utcnow(),  # Placeholder
                period_end=datetime.utcnow(),
                gross_revenue=revenue,
                operating_expenses=opex,
                taxes=taxes,
                debt_facilities=debt_facilities,
                dsra_opening=dsra_balance,
                dsra_target=dsra_target,
                mra_opening=mra_balance,
                mra_target=mra_target,
            )
            
            # Equity cashflow
            equity_cf = waterfall.tier_results.get("9_equity", {}).get("amount_paid", 0)
            
            # Update balances
            dsra_balance = waterfall.tier_results.get("5_dsra", {}).get("amount_paid", 0) + dsra_balance
            mra_balance = waterfall.tier_results.get("6_mra", {}).get("amount_paid", 0) + mra_balance
            
            for f in debt_facilities:
                if f.facility_type == "senior":
                    debt_balances[f.facility_id] = max(0, debt_balances.get(f.facility_id, 0) - senior_principal)
            
            # Store
            results["cfads"][i] = cfads
            results["dscr"][i] = waterfall.dscr
            results["dsra_balance"][i] = dsra_balance
            results["mra_balance"][i] = mra_balance
            results["equity_cashflow"][i] = equity_cf
            results["debt_balance"][i] = sum(debt_balances.values())
        
        return results
    
    def calculate_equity_irr(self, equity_cashflows: np.ndarray) -> float:
        """Calculate equity IRR"""
        try:
            return float(npf.irr(equity_cashflows))
        except:
            return 0.0
    
    def calculate_equity_npv(self, equity_cashflows: np.ndarray, discount_rate: float) -> float:
        """Calculate equity NPV"""
        return float(npf.npv(discount_rate, equity_cashflows))