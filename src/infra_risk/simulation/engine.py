"""
InfraRisk Lab - Simulation Engine
90-minute time-compressed simulation with quarterly steps
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import random

from src.infra_risk.schemas.simulation import (
    SimulationState,
    GameMode,
    SimulationPhase,
    Decision,
    DecisionType,
    StochasticEvent,
    EventType,
    EventSeverity,
    AIAction,
    ScoreBreakdown,
    ScoringWeights,
    SimulationResult,
    TimeEngineConfig,
    EventEngineConfig,
    AIOpponentConfig,
)
from src.infra_risk.schemas.project import Project
from src.infra_risk.utils.logging import get_logger
from src.infra_risk.simulation.event_engine import EventEngine
from src.infra_risk.simulation.ai_opponent import AIOpponent
from src.infra_risk.simulation.scoring import ScoringSystem
from src.infra_risk.simulation.game_modes import GameModeManager

logger = get_logger(__name__)


class SimulationEngine:
    """Main simulation engine for InfraRisk Lab"""
    
    def __init__(
        self,
        time_config: Optional[TimeEngineConfig] = None,
        event_config: Optional[EventEngineConfig] = None,
        ai_config: Optional[AIOpponentConfig] = None,
        scoring_weights: Optional[ScoringWeights] = None,
    ):
        self.time_config = time_config or TimeEngineConfig()
        self.event_config = event_config or EventEngineConfig()
        self.ai_config = ai_config or AIOpponentConfig()
        self.scoring_weights = scoring_weights or ScoringWeights()
        
        # Components
        self.event_engine = EventEngine(self.event_config)
        self.ai_opponent = AIOpponent(self.ai_config)
        self.scoring = ScoringSystem(self.scoring_weights)
        self.game_mode_manager = GameModeManager()
        
        # State
        self.state: Optional[SimulationState] = None
        
        logger.info("SimulationEngine initialized")
    
    def start_simulation(
        self,
        game_mode: GameMode,
        player_id: str,
        projects: List[Project],
        initial_cash: float = 100_000_000,
    ) -> SimulationState:
        """Start a new simulation"""
        logger.info("Starting simulation", mode=game_mode.value, player=player_id)
        
        # Initialize state
        self.state = SimulationState(
            simulation_id=f"sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            game_mode=game_mode,
            player_id=player_id,
            ai_opponent_id="ai_opponent_1",
            portfolio_projects=[p.project_id for p in projects],
            available_projects=[p.project_id for p in projects],
            cash_balance=initial_cash,
            total_debt=0,
            equity_value=initial_cash,
            time_config=self.time_config,
            event_config=self.event_config,
            ai_config=self.ai_config,
            scoring_weights=self.scoring_weights,
        )
        
        # Generate events for the simulation
        self.state.events = self.event_engine.generate_events(
            self.state.total_quarters,
            self.state.portfolio_projects,
        )
        
        # Initialize AI opponent
        self.ai_opponent.initialize(self.state)
        
        # Advance to first quarter
        self._advance_quarter()
        
        return self.state
    
    def _advance_quarter(self) -> None:
        """Advance simulation by one quarter"""
        if self.state is None:
            return
        
        self.state.current_quarter += 1
        self.state.phase = SimulationPhase.QUARTERLY_DECISION
        
        # Process quarterly cashflows for portfolio
        self._process_quarterly_cashflows()
        
        # Check for events this quarter
        self._process_events()
        
        # AI opponent takes action
        ai_action = self.ai_opponent.act(self.state)
        if ai_action:
            self.state.ai_actions.append(ai_action)
            self._execute_ai_action(ai_action)
        
        self.state.updated_at = datetime.utcnow()
    
    def _process_quarterly_cashflows(self) -> None:
        """Process quarterly cashflows for all portfolio projects"""
        if self.state is None:
            return
        
        # Simplified cashflow processing
        # In practice, would call waterfall engine for each project
        quarterly_revenue = 0
        quarterly_debt_service = 0
        
        for project_id in self.state.portfolio_projects:
            # Simulate project cashflows
            revenue = np.random.normal(5_000_000, 1_000_000)
            opex = np.random.normal(2_000_000, 500_000)
            debt_service = np.random.normal(1_500_000, 300_000)
            
            quarterly_revenue += max(0, revenue)
            quarterly_debt_service += max(0, debt_service)
            
            self.state.total_debt += max(0, debt_service * 4)  # Rough debt estimate
        
        cfads = quarterly_revenue - quarterly_debt_service
        self.state.cash_balance += cfads
        self.state.equity_value = self.state.cash_balance
        
        # Calculate quarterly DSCR
        dscr = quarterly_revenue / quarterly_debt_service if quarterly_debt_service > 0 else 0
        
        # Store quarterly results
        self.state.quarterly_results.append({
            "quarter": self.state.current_quarter,
            "revenue": quarterly_revenue,
            "debt_service": quarterly_debt_service,
            "cfads": cfads,
            "dscr": dscr,
            "cash_balance": self.state.cash_balance,
            "equity_value": self.state.equity_value,
        })
    
    def _process_events(self) -> None:
        """Process stochastic events for current quarter"""
        if self.state is None:
            return
        
        for event in self.state.events:
            if event.trigger_quarter == self.state.current_quarter and not event.is_triggered:
                event.is_triggered = True
                self.state.phase = SimulationPhase.EVENT_RESOLUTION
                
                # Apply event impact
                impact = self.event_engine.apply_event(event, self.state)
                
                # Score crisis response
                self.scoring.score_crisis_response(self.state, event, impact)
                
                event.resolved = True
    
    def _execute_ai_action(self, action: AIAction) -> None:
        """Execute AI opponent action"""
        # Apply action impact
        if action.action_type == DecisionType.CHANGE_ORDER:
            # AI requests change order
            pass
        elif action.action_type == DecisionType.REFINANCING:
            # AI refinances debt
            pass
        elif action.action_type == DecisionType.HEDGING:
            # AI adjusts hedging
            pass
    
    def make_decision(self, decision: Decision) -> Dict[str, Any]:
        """Process player decision"""
        if self.state is None:
            return {"error": "No active simulation"}
        
        # Validate decision
        if decision.quarter != self.state.current_quarter:
            return {"error": f"Decision for wrong quarter: {decision.quarter} vs {self.state.current_quarter}"}
        
        # Execute decision
        result = self._execute_decision(decision)
        
        # Record decision
        decision.is_executed = True
        self.state.decisions.append(decision)
        
        # Score decision
        self.scoring.score_decision(self.state, decision, result)
        
        return result
    
    def _execute_decision(self, decision: Decision) -> Dict[str, Any]:
        """Execute a specific decision"""
        result = {"success": True, "impact": {}}
        
        if decision.decision_type == DecisionType.CHANGE_ORDER:
            cost = decision.parameters.get("cost", 0)
            self.state.cash_balance -= cost
            result["impact"]["cash_change"] = -cost
            
        elif decision.decision_type == DecisionType.TARIFF_ADJUSTMENT:
            pct = decision.parameters.get("percentage", 0)
            result["impact"]["revenue_change_pct"] = pct
            
        elif decision.decision_type == DecisionType.COVENANT_WAIVER:
            result["impact"]["waiver_granted"] = True
            
        elif decision.decision_type == DecisionType.CAPEX_APPROVAL:
            cost = decision.parameters.get("cost", 0)
            self.state.cash_balance -= cost
            result["impact"]["cash_change"] = -cost
            
        elif decision.decision_type == DecisionType.REFINANCING:
            new_rate = decision.parameters.get("new_rate", 0)
            result["impact"]["interest_savings"] = new_rate
            
        elif decision.decision_type == DecisionType.HEDGING:
            hedge_ratio = decision.parameters.get("hedge_ratio", 0)
            result["impact"]["hedge_ratio"] = hedge_ratio
            
        elif decision.decision_type == DecisionType.DIVIDEND_POLICY:
            payout = decision.parameters.get("payout_pct", 0)
            dividend = self.state.cash_balance * payout
            self.state.cash_balance -= dividend
            result["impact"]["dividend_paid"] = dividend
            
        elif decision.decision_type == DecisionType.ESG_INVESTMENT:
            cost = decision.parameters.get("cost", 0)
            self.state.cash_balance -= cost
            result["impact"]["esg_improvement"] = True
        
        return result
    
    def next_quarter(self) -> SimulationState:
        """Advance to next quarter"""
        if self.state is None:
            raise ValueError("No active simulation")
        
        if self.state.current_quarter >= self.state.total_quarters:
            return self.end_simulation()
        
        self._advance_quarter()
        return self.state
    
    def end_simulation(self) -> SimulationResult:
        """End simulation and calculate final results"""
        if self.state is None:
            raise ValueError("No active simulation")
        
        logger.info("Ending simulation", simulation_id=self.state.simulation_id)
        
        self.state.phase = SimulationPhase.SCORING
        
        # Final scoring
        final_score = self.scoring.calculate_final_score(self.state)
        ai_score = self.scoring.calculate_ai_score(self.state)
        
        # Portfolio metrics
        portfolio_irr = self._calculate_portfolio_irr()
        portfolio_moic = self._calculate_moic()
        max_drawdown = self._calculate_max_drawdown()
        
        # DSCR stats
        dscr_values = [q["dscr"] for q in self.state.quarterly_results]
        
        result = SimulationResult(
            simulation_id=self.state.simulation_id,
            game_mode=self.state.game_mode,
            player_id=self.state.player_id,
            final_score=final_score,
            ai_final_score=ai_score,
            winner="player" if final_score.total > ai_score.total else "ai",
            total_quarters_played=self.state.current_quarter,
            portfolio_irr=portfolio_irr,
            portfolio_moic=portfolio_moic,
            max_drawdown=max_drawdown,
            avg_dscr=float(np.mean(dscr_values)) if dscr_values else 0,
            min_dscr=float(np.min(dscr_values)) if dscr_values else 0,
            default_count=0,  # Would track defaults
            restructuring_count=0,
            decision_quality_score=final_score.financial_structuring / self.scoring_weights.financial_structuring * 100,
            timing_score=final_score.market_timing / self.scoring_weights.market_timing * 100,
            risk_management_score=final_score.credit_risk_management / self.scoring_weights.credit_risk_management * 100,
        )
        
        self.state.phase = SimulationPhase.COMPLETE
        return result
    
    def _calculate_portfolio_irr(self) -> float:
        """Calculate portfolio IRR"""
        if not self.state.quarterly_results:
            return 0.0
        
        cashflows = [-self.state.quarterly_results[0].get("equity_value", 100_000_000)]
        for q in self.state.quarterly_results:
            cashflows.append(q.get("cfads", 0))
        
        try:
            import numpy_financial as npf
            return float(npf.irr(cashflows)) * 4  # Annualize quarterly
        except:
            return 0.0
    
    def _calculate_moic(self) -> float:
        """Calculate MOIC"""
        if self.state.equity_value <= 0:
            return 0.0
        return self.state.equity_value / 100_000_000  # Assuming 100M initial equity
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.state.quarterly_results:
            return 0.0
        
        equity_values = [q["equity_value"] for q in self.state.quarterly_results]
        peak = equity_values[0]
        max_dd = 0
        
        for val in equity_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return float(max_dd)
    
    def get_state(self) -> Optional[SimulationState]:
        """Get current simulation state"""
        return self.state