"""
Event Engine - Stochastic events for simulation
Rate shocks, climate damage, sovereign downgrades, pandemics, etc.
"""
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import random

from src.infra_risk.schemas.simulation import (
    StochasticEvent,
    EventType,
    EventSeverity,
    EventEngineConfig,
    SimulationState,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class EventEngine:
    """Generates and applies stochastic events"""
    
    def __init__(self, config: Optional[EventEngineConfig] = None):
        self.config = config or EventEngineConfig()
        
        # Event templates with probabilities and impacts
        self.event_templates = {
            EventType.RATE_SHOCK: {
                "base_prob": 0.08,
                "severities": [EventSeverity.MEDIUM, EventSeverity.HIGH, EventSeverity.EXTREME],
                "impact": {
                    "interest_rate_delta": (0.5, 3.0),  # percentage points
                    "dscr_impact": (-0.3, -0.1),
                    "cash_impact": (-0.1, -0.05),
                },
            },
            EventType.SOVEREIGN_DOWNGRADE: {
                "base_prob": 0.05,
                "severities": [EventSeverity.MEDIUM, EventSeverity.HIGH],
                "impact": {
                    "spread_widening": (50, 300),  # bps
                    "equity_impact": (-0.15, -0.05),
                    "refinancing_risk": 0.3,
                },
            },
            EventType.PANDEMIC: {
                "base_prob": 0.02,
                "severities": [EventSeverity.HIGH, EventSeverity.EXTREME],
                "impact": {
                    "revenue_drop": (0.3, 0.7),
                    "duration_quarters": (4, 12),
                    "opex_increase": (0.1, 0.3),
                },
            },
            EventType.CLIMATE_DAMAGE: {
                "base_prob": 0.10,
                "severities": [EventSeverity.LOW, EventSeverity.MEDIUM, EventSeverity.HIGH],
                "impact": {
                    "capex_damage": (0.05, 0.40),
                    "downtime_quarters": (1, 8),
                    "insurance_recovery": (0.3, 0.8),
                },
            },
            EventType.POLITICAL_UNREST: {
                "base_prob": 0.06,
                "severities": [EventSeverity.MEDIUM, EventSeverity.HIGH],
                "impact": {
                    "construction_delay": (1, 6),  # quarters
                    "revenue_impact": (-0.2, -0.05),
                    "security_cost_increase": (0.05, 0.15),
                },
            },
            EventType.REGULATORY_CHANGE: {
                "base_prob": 0.07,
                "severities": [EventSeverity.LOW, EventSeverity.MEDIUM],
                "impact": {
                    "tariff_change": (-0.15, 0.10),
                    "compliance_cost": (0.01, 0.05),
                    "permit_delay": (0, 4),
                },
            },
            EventType.TECHNOLOGY_DISRUPTION: {
                "base_prob": 0.04,
                "severities": [EventSeverity.LOW, EventSeverity.MEDIUM],
                "impact": {
                    "demand_shift": (-0.2, 0.1),
                    "stranded_asset_risk": 0.2,
                    "capex_need": (0.05, 0.2),
                },
            },
            EventType.COMMODITY_SHOCK: {
                "base_prob": 0.08,
                "severities": [EventSeverity.MEDIUM, EventSeverity.HIGH],
                "impact": {
                    "fuel_cost_change": (-0.3, 0.5),
                    "revenue_impact": (-0.1, 0.2),
                    "hedge_effectiveness": (0.5, 0.9),
                },
            },
            EventType.CURRENCY_CRISIS: {
                "base_prob": 0.05,
                "severities": [EventSeverity.HIGH, EventSeverity.EXTREME],
                "impact": {
                    "fx_devaluation": (0.2, 0.6),
                    "debt_service_impact": (0.3, 1.0),  # multiplier
                    "revenue_impact": (-0.4, -0.1),
                },
            },
            EventType.COUNTERPARTY_DEFAULT: {
                "base_prob": 0.04,
                "severities": [EventSeverity.MEDIUM, EventSeverity.HIGH],
                "impact": {
                    "revenue_loss": (0.1, 0.5),
                    "replacement_cost": (0.05, 0.2),
                    "legal_costs": (0.01, 0.03),
                },
            },
            EventType.FORCE_MAJEURE: {
                "base_prob": 0.03,
                "severities": [EventSeverity.HIGH, EventSeverity.EXTREME],
                "impact": {
                    "construction_delay": (2, 12),
                    "cost_overrun": (0.1, 0.5),
                    "termination_risk": 0.1,
                },
            },
            EventType.ESG_INCIDENT: {
                "base_prob": 0.05,
                "severities": [EventSeverity.LOW, EventSeverity.MEDIUM],
                "impact": {
                    "reputation_damage": (0.05, 0.2),
                    "financing_cost_increase": (0.005, 0.02),
                    "remediation_cost": (0.01, 0.1),
                },
            },
        }
    
    def generate_events(
        self,
        total_quarters: int,
        project_ids: List[str],
    ) -> List[StochasticEvent]:
        """Generate stochastic events for entire simulation"""
        events = []
        
        for quarter in range(1, total_quarters + 1):
            # Determine number of events this quarter (Poisson)
            n_events = np.random.poisson(self.config.base_event_rate)
            
            for _ in range(n_events):
                # Select event type based on weights
                event_type = self._sample_event_type()
                template = self.event_templates[event_type]
                
                # Select severity
                severity = np.random.choice(template["severities"])
                
                # Generate impact parameters
                impact_params = {}
                for key, (low, high) in template["impact"].items():
                    if isinstance(low, float) and isinstance(high, float):
                        impact_params[key] = np.random.uniform(low, high)
                    else:
                        impact_params[key] = low
                
                # Determine affected projects
                n_affected = min(len(project_ids), max(1, int(len(project_ids) * np.random.uniform(0.1, 0.5))))
                affected = np.random.choice(project_ids, size=n_affected, replace=False).tolist()
                
                # Duration
                duration = impact_params.get("duration_quarters", 
                    np.random.randint(1, 5))
                
                event = StochasticEvent(
                    event_id=f"evt_{event_type.value}_{quarter}_{len(events)}",
                    event_type=event_type,
                    severity=severity,
                    description=self._generate_description(event_type, severity, impact_params),
                    probability=template["base_prob"],
                    impact_parameters=impact_params,
                    affected_projects=affected,
                    affected_portfolio=len(affected) > len(project_ids) / 2,
                    trigger_quarter=quarter,
                    duration_quarters=duration,
                )
                events.append(event)
        
        # Sort by trigger quarter
        events.sort(key=lambda e: e.trigger_quarter)
        
        logger.info("Generated events", count=len(events), quarters=total_quarters)
        return events
    
    def _sample_event_type(self) -> EventType:
        """Sample event type based on probabilities"""
        types = list(self.event_templates.keys())
        probs = [self.event_templates[t]["base_prob"] for t in types]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(types, p=probs)
    
    def _generate_description(
        self,
        event_type: EventType,
        severity: EventSeverity,
        impact: Dict[str, float],
    ) -> str:
        """Generate human-readable event description"""
        descriptions = {
            EventType.RATE_SHOCK: f"Interest rate shock: +{impact.get('interest_rate_delta', 1.0):.1f}% ({severity.value})",
            EventType.SOVEREIGN_DOWNGRADE: f"Sovereign downgrade: {impact.get('spread_widening', 100):.0f} bps spread widening",
            EventType.PANDEMIC: f"Pandemic outbreak: {impact.get('duration_quarters', 8)} quarters duration",
            EventType.CLIMATE_DAMAGE: f"Climate damage: {impact.get('capex_damage', 0.1)*100:.0f}% asset damage",
            EventType.POLITICAL_UNREST: f"Political unrest: {impact.get('construction_delay', 3)} quarter delay",
            EventType.REGULATORY_CHANGE: f"Regulatory change: tariff {impact.get('tariff_change', 0)*100:+.0f}%",
            EventType.TECHNOLOGY_DISRUPTION: f"Technology disruption: demand shift {impact.get('demand_shift', 0)*100:+.0f}%",
            EventType.COMMODITY_SHOCK: f"Commodity shock: fuel cost {impact.get('fuel_cost_change', 0)*100:+.0f}%",
            EventType.CURRENCY_CRISIS: f"Currency crisis: {impact.get('fx_devaluation', 0.3)*100:.0f}% devaluation",
            EventType.COUNTERPARTY_DEFAULT: f"Counterparty default: {impact.get('revenue_loss', 0.2)*100:.0f}% revenue loss",
            EventType.FORCE_MAJEURE: f"Force majeure: {impact.get('construction_delay', 6)} quarter delay",
            EventType.ESG_INCIDENT: f"ESG incident: reputation impact {impact.get('reputation_damage', 0.1)*100:.0f}%",
        }
        return descriptions.get(event_type, f"{event_type.value} event ({severity.value})")
    
    def apply_event(
        self,
        event: StochasticEvent,
        state: SimulationState,
    ) -> Dict[str, Any]:
        """Apply event impact to simulation state"""
        impact = {}
        params = event.impact_parameters
        
        if event.event_type == EventType.RATE_SHOCK:
            # Increase debt service for floating rate debt
            rate_delta = params.get("interest_rate_delta", 1.0) / 100
            additional_cost = state.total_debt * rate_delta / 4  # Quarterly
            state.cash_balance -= additional_cost
            impact["additional_debt_service"] = additional_cost
            impact["dscr_impact"] = params.get("dscr_impact", -0.2)
            
        elif event.event_type == EventType.SOVEREIGN_DOWNGRADE:
            # Increase spreads, reduce equity value
            spread_widening = params.get("spread_widening", 100) / 10000
            state.equity_value *= (1 + params.get("equity_impact", -0.1))
            impact["spread_widening_bps"] = params.get("spread_widening", 100)
            impact["equity_change"] = params.get("equity_impact", -0.1)
            
        elif event.event_type == EventType.PANDEMIC:
            # Revenue drop for duration
            revenue_drop = params.get("revenue_drop", 0.5)
            opex_increase = params.get("opex_increase", 0.2)
            quarterly_revenue = state.quarterly_results[-1].get("revenue", 0) if state.quarterly_results else 0
            revenue_loss = quarterly_revenue * revenue_drop
            opex_increase_amt = state.quarterly_results[-1].get("opex", 0) * opex_increase if state.quarterly_results else 0
            state.cash_balance -= (revenue_loss + opex_increase_amt)
            impact["revenue_loss"] = revenue_loss
            impact["opex_increase"] = opex_increase_amt
            impact["duration"] = params.get("duration_quarters", 8)
            
        elif event.event_type == EventType.CLIMATE_DAMAGE:
            # Capex damage, insurance recovery
            damage_pct = params.get("capex_damage", 0.15)
            insurance_recovery = params.get("insurance_recovery", 0.6)
            
            # Estimate asset value
            asset_value = state.total_debt + state.equity_value
            damage = asset_value * damage_pct
            recovery = damage * insurance_recovery
            net_loss = damage - recovery
            
            state.cash_balance -= net_loss
            state.equity_value -= net_loss
            
            impact["damage"] = damage
            impact["insurance_recovery"] = recovery
            impact["net_loss"] = net_loss
            impact["downtime_quarters"] = params.get("downtime_quarters", 2)
            
        elif event.event_type == EventType.POLITICAL_UNREST:
            delay = params.get("construction_delay", 3)
            revenue_impact = params.get("revenue_impact", -0.1)
            security_cost = params.get("security_cost_increase", 0.1)
            
            impact["construction_delay_quarters"] = delay
            impact["revenue_impact"] = revenue_impact
            impact["security_cost_increase"] = security_cost
            
        elif event.event_type == EventType.REGULATORY_CHANGE:
            tariff_change = params.get("tariff_change", 0)
            compliance_cost = params.get("compliance_cost", 0.02)
            permit_delay = params.get("permit_delay", 0)
            
            impact["tariff_change"] = tariff_change
            impact["compliance_cost"] = compliance_cost
            impact["permit_delay"] = permit_delay
            
        elif event.event_type == EventType.TECHNOLOGY_DISRUPTION:
            demand_shift = params.get("demand_shift", 0)
            capex_need = params.get("capex_need", 0.1)
            
            impact["demand_shift"] = demand_shift
            impact["capex_need"] = capex_need
            
        elif event.event_type == EventType.COMMODITY_SHOCK:
            fuel_change = params.get("fuel_cost_change", 0)
            revenue_impact = params.get("revenue_impact", 0)
            hedge_eff = params.get("hedge_effectiveness", 0.7)
            
            # Net impact after hedging
            net_fuel_impact = fuel_change * (1 - hedge_eff)
            
            impact["fuel_cost_change"] = fuel_change
            impact["revenue_impact"] = revenue_impact
            impact["net_impact"] = net_fuel_impact + revenue_impact
            
        elif event.event_type == EventType.CURRENCY_CRISIS:
            deval = params.get("fx_devaluation", 0.3)
            debt_mult = params.get("debt_service_impact", 0.5)
            revenue_impact = params.get("revenue_impact", -0.2)
            
            # Increase foreign currency debt service
            fc_debt = state.total_debt * 0.5  # Assume 50% FC
            additional_ds = fc_debt * deval * debt_mult / 4
            state.cash_balance -= additional_ds
            
            impact["devaluation"] = deval
            impact["additional_debt_service"] = additional_ds
            impact["revenue_impact"] = revenue_impact
            
        elif event.event_type == EventType.COUNTERPARTY_DEFAULT:
            revenue_loss = params.get("revenue_loss", 0.2)
            replacement = params.get("replacement_cost", 0.1)
            legal = params.get("legal_costs", 0.02)
            
            quarterly_revenue = state.quarterly_results[-1].get("revenue", 0) if state.quarterly_results else 0
            loss = quarterly_revenue * revenue_loss
            
            state.cash_balance -= (loss + replacement + legal)
            impact["revenue_loss"] = loss
            impact["replacement_cost"] = replacement
            impact["legal_costs"] = legal
            
        elif event.event_type == EventType.FORCE_MAJEURE:
            delay = params.get("construction_delay", 6)
            overrun = params.get("cost_overrun", 0.2)
            term_risk = params.get("termination_risk", 0.1)
            
            impact["construction_delay"] = delay
            impact["cost_overrun"] = overrun
            impact["termination_risk"] = term_risk
            
        elif event.event_type == EventType.ESG_INCIDENT:
            rep_damage = params.get("reputation_damage", 0.1)
            fin_cost = params.get("financing_cost_increase", 0.01)
            remediation = params.get("remediation_cost", 0.05)
            
            state.equity_value *= (1 - rep_damage)
            state.cash_balance -= remediation
            
            impact["reputation_damage"] = rep_damage
            impact["financing_cost_increase"] = fin_cost
            impact["remediation_cost"] = remediation
        
        return impact
    
    def get_active_events(self, state: SimulationState, quarter: int) -> List[StochasticEvent]:
        """Get events active in a given quarter"""
        active = []
        for event in state.events:
            if event.trigger_quarter <= quarter < event.trigger_quarter + event.duration_quarters:
                active.append(event)
        return active