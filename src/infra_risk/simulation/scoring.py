"""
Scoring System - 1000-point scoring for InfraRisk Lab
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from src.infra_risk.schemas.simulation import (
    SimulationState,
    ScoreBreakdown,
    ScoringWeights,
    Decision,
    StochasticEvent,
    SimulationResult,
    DecisionType,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class ScoringSystem:
    """1000-point scoring system for simulation"""
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()
        logger.info("ScoringSystem initialized", total=self.weights.total())
    
    def score_decision(
        self,
        state: SimulationState,
        decision: Decision,
        result: Dict[str, Any],
    ) -> None:
        """Score a player decision"""
        if state is None:
            return
        
        score = 0
        
        if decision.decision_type == DecisionType.CHANGE_ORDER:
            score = self._score_change_order(decision, result)
        elif decision.decision_type == DecisionType.TARIFF_ADJUSTMENT:
            score = self._score_tariff_adjustment(decision, result)
        elif decision.decision_type == DecisionType.COVENANT_WAIVER:
            score = self._score_covenant_waiver(decision, result)
        elif decision.decision_type == DecisionType.CAPEX_APPROVAL:
            score = self._score_capex_approval(decision, result)
        elif decision.decision_type == DecisionType.REFINANCING:
            score = self._score_refinancing(decision, result)
        elif decision.decision_type == DecisionType.HEDGING:
            score = self._score_hedging(decision, result)
        elif decision.decision_type == DecisionType.DIVIDEND_POLICY:
            score = self._score_dividend(decision, result)
        elif decision.decision_type == DecisionType.ESG_INVESTMENT:
            score = self._score_esg_investment(decision, result)
        
        # Add to financial structuring score
        state.player_score.financial_structuring += score
        state.player_score.total = sum([
            state.player_score.credit_risk_management,
            state.player_score.financial_structuring,
            state.player_score.portfolio_diversification,
            state.player_score.crisis_response,
            state.player_score.market_timing,
            state.player_score.esg_integration,
        ])
    
    def _score_change_order(self, decision: Decision, result: Dict) -> float:
        """Score change order decision"""
        cost = decision.parameters.get("cost", 0)
        # Penalize costly changes, reward necessary ones
        if "necessary" in decision.parameters.get("justification", "").lower():
            return 10
        return max(-20, -cost / 1_000_000)
    
    def _score_tariff_adjustment(self, decision: Decision, result: Dict) -> float:
        """Score tariff adjustment"""
        pct = decision.parameters.get("percentage", 0)
        # Reward inflation-indexed increases
        if abs(pct - 0.03) < 0.01:  # Near inflation
            return 15
        elif pct > 0:
            return 5
        return -10
    
    def _score_covenant_waiver(self, decision: Decision, result: Dict) -> float:
        """Score covenant waiver request"""
        # Waivers are negative but sometimes necessary
        return -5
    
    def _score_capex_approval(self, decision: Decision, result: Dict) -> float:
        """Score capex approval"""
        cost = decision.parameters.get("cost", 0)
        if decision.parameters.get("type") == "maintenance":
            return 10  # Preventive maintenance is good
        elif decision.parameters.get("type") == "expansion":
            return 5   # Growth capex
        return -cost / 1_000_000
    
    def _score_refinancing(self, decision: Decision, result: Dict) -> float:
        """Score refinancing"""
        savings = result.get("impact", {}).get("interest_savings", 0)
        return min(20, savings * 1000)  # bps saved * 10
    
    def _score_hedging(self, decision: Decision, result: Dict) -> float:
        """Score hedging decision"""
        ratio = decision.parameters.get("hedge_ratio", 0)
        if 0.6 <= ratio <= 0.9:
            return 10  # Optimal range
        elif ratio > 0:
            return 5
        return 0
    
    def _score_dividend(self, decision: Decision, result: Dict) -> float:
        """Score dividend policy"""
        payout = decision.parameters.get("payout_pct", 0)
        if 0.2 <= payout <= 0.5:
            return 8  # Prudent payout
        elif payout > 0.5:
            return -10  # Too aggressive
        return 0
    
    def _score_esg_investment(self, decision: Decision, result: Dict) -> float:
        """Score ESG investment"""
        cost = decision.parameters.get("cost", 0)
        return min(15, cost / 500_000)  # Reward ESG investment
    
    def score_crisis_response(
        self,
        state: SimulationState,
        event: StochasticEvent,
        impact: Dict[str, Any],
    ) -> None:
        """Score crisis response to an event"""
        if state is None:
            return
        
        # Base score for responding
        base_score = 20
        
        # Bonus for good mitigation
        if impact.get("net_loss", 0) < impact.get("damage", 0) * 0.5:
            base_score += 15  # Good insurance/recovery
        
        # Bonus for maintaining DSCR
        if state.quarterly_results:
            last_dscr = state.quarterly_results[-1].get("dscr", 0)
            if last_dscr > 1.2:
                base_score += 10
        
        state.player_score.crisis_response += base_score
        state.player_score.total = sum([
            state.player_score.credit_risk_management,
            state.player_score.financial_structuring,
            state.player_score.portfolio_diversification,
            state.player_score.crisis_response,
            state.player_score.market_timing,
            state.player_score.esg_integration,
        ])
    
    def score_quarterly(self, state: SimulationState) -> None:
        """Score quarterly performance"""
        if state is None or not state.quarterly_results:
            return
        
        last = state.quarterly_results[-1]
        
        # Credit risk management: DSCR maintenance
        dscr = last.get("dscr", 0)
        if dscr >= 1.5:
            state.player_score.credit_risk_management += 5
        elif dscr >= 1.3:
            state.player_score.credit_risk_management += 3
        elif dscr >= 1.0:
            state.player_score.credit_risk_management += 1
        else:
            state.player_score.credit_risk_management -= 10
        
        # Portfolio diversification: check sector exposure
        # Simplified
        state.player_score.portfolio_diversification += 2
        
        # Market timing: buying low, selling high
        # Would track entry/exit timing
        
        state.player_score.total = sum([
            state.player_score.credit_risk_management,
            state.player_score.financial_structuring,
            state.player_score.portfolio_diversification,
            state.player_score.crisis_response,
            state.player_score.market_timing,
            state.player_score.esg_integration,
        ])
    
    def calculate_final_score(self, state: SimulationState) -> ScoreBreakdown:
        """Calculate final score with all components"""
        if state is None:
            return ScoreBreakdown()
        
        # Credit Risk Management (300 pts)
        avg_dscr = np.mean([q.get("dscr", 0) for q in state.quarterly_results]) if state.quarterly_results else 0
        min_dscr = np.min([q.get("dscr", 0) for q in state.quarterly_results]) if state.quarterly_results else 0
        default_count = 0  # Would track
        
        credit_score = 0
        if avg_dscr >= 1.5:
            credit_score += 100
        elif avg_dscr >= 1.3:
            credit_score += 80
        elif avg_dscr >= 1.2:
            credit_score += 50
        elif avg_dscr >= 1.0:
            credit_score += 20
        
        if min_dscr >= 1.3:
            credit_score += 100
        elif min_dscr >= 1.2:
            credit_score += 70
        elif min_dscr >= 1.0:
            credit_score += 30
        
        credit_score += max(0, 100 - default_count * 50)
        
        # Financial Structuring (200 pts) - from decisions
        structuring_score = min(200, state.player_score.financial_structuring)
        
        # Portfolio Diversification (150 pts)
        div_score = min(150, state.player_score.portfolio_diversification)
        
        # Crisis Response (150 pts)
        crisis_score = min(150, state.player_score.crisis_response)
        
        # Market Timing (100 pts)
        timing_score = min(100, state.player_score.market_timing)
        
        # ESG Integration (100 pts)
        esg_score = min(100, state.player_score.esg_integration)
        
        # Normalize to weights
        total_weight = self.weights.total()
        
        return ScoreBreakdown(
            credit_risk_management=credit_score / 300 * self.weights.credit_risk_management,
            financial_structuring=structuring_score / 200 * self.weights.financial_structuring,
            portfolio_diversification=div_score / 150 * self.weights.portfolio_diversification,
            crisis_response=crisis_score / 150 * self.weights.crisis_response,
            market_timing=timing_score / 100 * self.weights.market_timing,
            esg_integration=esg_score / 100 * self.weights.esg_integration,
        )
    
    def calculate_ai_score(self, state: SimulationState) -> ScoreBreakdown:
        """Calculate AI opponent score"""
        # Simplified: AI gets base score plus some variance
        base = ScoreBreakdown(
            credit_risk_management=self.weights.credit_risk_management * 0.75,
            financial_structuring=self.weights.financial_structuring * 0.7,
            portfolio_diversification=self.weights.portfolio_diversification * 0.8,
            crisis_response=self.weights.crisis_response * 0.65,
            market_timing=self.weights.market_timing * 0.7,
            esg_integration=self.weights.esg_integration * 0.6,
        )
        
        # Add some randomness
        for attr in ['credit_risk_management', 'financial_structuring', 'portfolio_diversification',
                     'crisis_response', 'market_timing', 'esg_integration']:
            val = getattr(base, attr)
            setattr(base, attr, val * np.random.uniform(0.9, 1.1))
        
        base.total = sum([
            base.credit_risk_management,
            base.financial_structuring,
            base.portfolio_diversification,
            base.crisis_response,
            base.market_timing,
            base.esg_integration,
        ])
        
        return base
    
    def get_percentile(self, score: float, game_mode: str) -> float:
        """Get percentile rank for a score"""
        # Simulated percentiles based on game mode
        distributions = {
            "single_deal": {"mean": 600, "std": 100},
            "portfolio_manager": {"mean": 550, "std": 120},
            "crisis_manager": {"mean": 500, "std": 150},
            "deal_structurer": {"mean": 450, "std": 130},
        }
        
        dist = distributions.get(game_mode, distributions["portfolio_manager"])
        from scipy import stats
        return float(stats.norm.cdf(score, dist["mean"], dist["std"]) * 100)