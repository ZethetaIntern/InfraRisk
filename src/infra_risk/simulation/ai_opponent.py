"""
AI Opponent - RL agent competing against human players
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import random

from src.infra_risk.schemas.simulation import (
    SimulationState,
    AIAction,
    DecisionType,
    AIOpponentConfig,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class AIOpponent:
    """AI opponent using rule-based strategy with RL elements"""
    
    def __init__(self, config: Optional[AIOpponentConfig] = None):
        self.config = config or AIOpponentConfig()
        self.strategy_weights = self._init_strategy_weights()
        logger.info("AIOpponent initialized")
    
    def _init_strategy_weights(self) -> Dict[str, float]:
        """Initialize strategy weights based on config"""
        return {
            "risk_aversion": self.config.risk_aversion,
            "return_target": self.config.return_target,
            "max_leverage": self.config.max_leverage,
            "min_dscr": self.config.min_dscr,
            "diversification": self.config.diversification_target,
            "esg_weight": self.config.esg_weight,
        }
    
    def initialize(self, state: SimulationState) -> None:
        """Initialize AI for a new simulation"""
        # Analyze portfolio and set strategy
        pass
    
    def act(self, state: SimulationState) -> Optional[AIAction]:
        """Generate AI action for current quarter"""
        if state.current_quarter % 4 != 0:  # Act annually
            return None
        
        # Evaluate possible actions
        actions = self._generate_candidate_actions(state)
        
        if not actions:
            return None
        
        # Score actions
        scored_actions = [(a, self._score_action(a, state)) for a in actions]
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        # Add exploration
        if random.random() < self.config.exploration_rate:
            chosen = random.choice(actions)
        else:
            chosen = scored_actions[0][0]
        
        # Create AI action
        action = AIAction(
            action_id=f"ai_{chosen['type'].value}_{state.current_quarter}",
            action_type=chosen["type"],
            project_id=chosen.get("project_id"),
            parameters=chosen.get("parameters", {}),
            rationale=chosen.get("rationale", ""),
            confidence=chosen.get("confidence", 0.7),
            expected_score_impact=chosen.get("score_impact", 0),
        )
        
        return action
    
    def _generate_candidate_actions(self, state: SimulationState) -> List[Dict[str, Any]]:
        """Generate candidate actions based on state"""
        actions = []
        
        # Refinancing if rates dropped
        if self._should_refinance(state):
            actions.append({
                "type": DecisionType.REFINANCING,
                "project_id": state.portfolio_projects[0] if state.portfolio_projects else None,
                "parameters": {"new_rate": 0.055},
                "rationale": "Refinance to lower rate",
                "confidence": 0.8,
                "score_impact": 10,
            })
        
        # Hedging adjustment
        if self._should_hedge(state):
            actions.append({
                "type": DecisionType.HEDGING,
                "project_id": state.portfolio_projects[0] if state.portfolio_projects else None,
                "parameters": {"hedge_ratio": 0.8},
                "rationale": "Increase hedge ratio due to rate outlook",
                "confidence": 0.7,
                "score_impact": 5,
            })
        
        # Capex approval for maintenance
        if self._should_approve_capex(state):
            actions.append({
                "type": DecisionType.CAPEX_APPROVAL,
                "project_id": state.portfolio_projects[0] if state.portfolio_projects else None,
                "parameters": {"cost": 2_000_000},
                "rationale": "Approve preventive maintenance",
                "confidence": 0.9,
                "score_impact": 8,
            })
        
        # Dividend policy
        if self._should_pay_dividend(state):
            actions.append({
                "type": DecisionType.DIVIDEND_POLICY,
                "parameters": {"payout_pct": 0.3},
                "rationale": "Distribute excess cash",
                "confidence": 0.6,
                "score_impact": 3,
            })
        
        # ESG investment
        if self._should_invest_esg(state):
            actions.append({
                "type": DecisionType.ESG_INVESTMENT,
                "project_id": state.portfolio_projects[0] if state.portfolio_projects else None,
                "parameters": {"cost": 1_000_000},
                "rationale": "Improve ESG score for better financing",
                "confidence": 0.7,
                "score_impact": 7,
            })
        
        return actions
    
    def _should_refinance(self, state: SimulationState) -> bool:
        """Determine if refinancing is attractive"""
        # Check if any project has high interest debt
        for project_id in state.portfolio_projects:
            # Simplified: check average DSCR
            if state.quarterly_results:
                last_dscr = state.quarterly_results[-1].get("dscr", 0)
                if last_dscr > 1.5 and state.cash_balance > 10_000_000:
                    return True
        return False
    
    def _should_hedge(self, state: SimulationState) -> bool:
        """Determine if hedging should be adjusted"""
        # Simple rule: hedge more if rates trending up
        return state.current_quarter > 10 and random.random() < 0.3
    
    def _should_approve_capex(self, state: SimulationState) -> bool:
        """Determine if maintenance capex should be approved"""
        # Approve if DSCR is healthy and cash available
        if state.quarterly_results:
            last_dscr = state.quarterly_results[-1].get("dscr", 0)
            return last_dscr > 1.3 and state.cash_balance > 5_000_000
        return False
    
    def _should_pay_dividend(self, state: SimulationState) -> bool:
        """Determine if dividend should be paid"""
        # Pay if cash exceeds 2x annual debt service
        if state.quarterly_results:
            annual_ds = state.quarterly_results[-1].get("debt_service", 0) * 4
            return state.cash_balance > annual_ds * 2
        return False
    
    def _should_invest_esg(self, state: SimulationState) -> bool:
        """Determine if ESG investment is warranted"""
        # Invest if ESG weight is significant and cash available
        return self.config.esg_weight > 0.05 and state.cash_balance > 20_000_000
    
    def _score_action(self, action: Dict, state: SimulationState) -> float:
        """Score an action based on expected impact"""
        base_score = action.get("score_impact", 0)
        confidence = action.get("confidence", 0.5)
        
        # Adjust for risk aversion
        risk_adj = 1 - self.config.risk_aversion * 0.5
        
        # Adjust for return target
        return_adj = min(1.0, base_score / (self.config.return_target * 100))
        
        return base_score * confidence * risk_adj * return_adj