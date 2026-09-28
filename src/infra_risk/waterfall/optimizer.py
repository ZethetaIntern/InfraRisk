"""
Debt Structure Optimizer using Reinforcement Learning
Optimizes leverage, sculpted amortization, hedging, sweep triggers
"""
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.infra_risk.schemas.waterfall import (
    DebtFacility,
    DebtStructure,
    SculptingParams,
    HedgingStrategy,
)
from src.infra_risk.schemas.project import Project
from src.infra_risk.waterfall.engine import WaterfallEngine
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class DebtOptimizationEnv(gym.Env):
    """RL Environment for debt structure optimization"""
    
    def __init__(
        self,
        project: Project,
        revenue_forecast: np.ndarray,
        opex_forecast: np.ndarray,
        tax_rate: float,
        target_dscr: float = 1.30,
        max_leverage: float = 0.80,
    ):
        super().__init__()
        
        self.project = project
        self.revenue_forecast = revenue_forecast
        self.opex_forecast = opex_forecast
        self.tax_rate = tax_rate
        self.target_dscr = target_dscr
        self.max_leverage = max_leverage
        
        self.waterfall = WaterfallEngine()
        
        # Action space: [leverage_ratio, tenor_years, grace_period, sculpting_factor, sweep_trigger, hedge_ratio]
        self.action_space = spaces.Box(
            low=np.array([0.5, 10, 0, 0.5, 1.0, 0.0]),
            high=np.array([max_leverage, 25, 5, 2.0, 2.0, 1.0]),
            dtype=np.float32,
        )
        
        # Observation space: [current_dscr, avg_dscr, min_dscr, leverage, equity_irr, period]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        
        self.current_step = 0
        self.max_steps = 1  # Single episode optimization
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.zeros(6, dtype=np.float32), {}
    
    def step(self, action):
        self.current_step += 1
        
        leverage, tenor, grace, sculpting, sweep_trigger, hedge_ratio = action
        
        # Create debt structure from action
        debt_facilities = self._create_debt_structure(
            leverage, tenor, grace, sculpting, sweep_trigger, hedge_ratio
        )
        
        # Project cashflows
        results = self.waterfall.project_cashflows(
            project_id=self.project.project_id,
            revenue_forecast=self.revenue_forecast,
            opex_forecast=self.opex_forecast,
            tax_rate=self.tax_rate,
            debt_facilities=debt_facilities,
            dsra_target=debt_facilities[0].principal * 0.5 if debt_facilities else 0,
            mra_target=self.project.total_capex * 0.02,
        )
        
        # Calculate reward
        reward = self._calculate_reward(results, debt_facilities)
        
        # Observation
        obs = np.array([
            results["dscr"][0],
            np.mean(results["dscr"]),
            np.min(results["dscr"]),
            leverage,
            self.waterfall.calculate_equity_irr(results["equity_cashflow"]),
            self.current_step,
        ], dtype=np.float32)
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _create_debt_structure(
        self,
        leverage: float,
        tenor: float,
        grace: float,
        sculpting: float,
        sweep_trigger: float,
        hedge_ratio: float,
    ) -> List[DebtFacility]:
        """Create debt facilities from action parameters"""
        total_debt = self.project.total_capex * leverage
        
        # Senior debt (70% of debt)
        senior_principal = total_debt * 0.7
        
        return [
            DebtFacility(
                facility_id="senior_1",
                name="Senior Debt",
                facility_type="senior",
                principal=senior_principal,
                interest_rate=0.06,  # Base rate
                is_floating=False,
                tenor_years=int(tenor),
                grace_period_years=int(grace),
                repayment_profile="sculpted" if sculpting > 1.0 else "annuity",
                min_dscr=sweep_trigger,
                dsra_required=True,
                dsra_size_months=6,
            ),
            DebtFacility(
                facility_id="mezz_1",
                name="Mezzanine",
                facility_type="mezzanine",
                principal=total_debt * 0.2,
                interest_rate=0.10,
                tenor_years=int(tenor * 0.8),
                grace_period_years=int(grace),
                repayment_profile="bullet",
            ),
            DebtFacility(
                facility_id="sub_1",
                name="Subordinated",
                facility_type="subordinated",
                principal=total_debt * 0.1,
                interest_rate=0.14,
                tenor_years=int(tenor * 0.6),
                repayment_profile="bullet",
            ),
        ]
    
    def _calculate_reward(
        self,
        results: Dict,
        debt_facilities: List[DebtFacility],
    ) -> float:
        """Calculate RL reward"""
        min_dscr = np.min(results["dscr"])
        avg_dscr = np.mean(results["dscr"])
        equity_irr = self.waterfall.calculate_equity_irr(results["equity_cashflow"])
        
        # Reward components
        dscr_penalty = max(0, self.target_dscr - min_dscr) * 100  # Heavy penalty for breaching target
        dscr_bonus = min(avg_dscr - self.target_dscr, 0.5) * 50  # Bonus for exceeding target
        irr_reward = max(equity_irr - 0.12, 0) * 100  # Reward for IRR > 12%
        leverage_penalty = max(0, sum(f.principal for f in debt_facilities) / self.project.total_capex - self.max_leverage) * 50
        
        reward = dscr_bonus + irr_reward - dscr_penalty - leverage_penalty
        
        return float(reward)


class DebtOptimizer:
    """Reinforcement Learning based debt structure optimizer"""
    
    def __init__(
        self,
        target_dscr: float = 1.30,
        max_leverage: float = 0.80,
        model_path: Optional[str] = None,
    ):
        self.target_dscr = target_dscr
        self.max_leverage = max_leverage
        self.model_path = model_path
        self.model = None
        self.waterfall = WaterfallEngine()
        
        if model_path:
            self.load_model(model_path)
        
        logger.info("DebtOptimizer initialized", target_dscr=target_dscr, max_leverage=max_leverage)
    
    def optimize(
        self,
        project: Project,
        revenue_forecast: np.ndarray,
        opex_forecast: np.ndarray,
        tax_rate: float,
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Optimize debt structure using RL"""
        logger.info("Optimizing debt structure", project_id=project.project_id)
        
        # Create environment
        env = DebtOptimizationEnv(
            project=project,
            revenue_forecast=revenue_forecast,
            opex_forecast=opex_forecast,
            tax_rate=tax_rate,
            target_dscr=self.target_dscr,
            max_leverage=self.max_leverage,
        )
        
        # Train or use existing model
        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
            )
            self.model.learn(total_timesteps=n_trials * 1000)
        
        # Evaluate best action
        best_action, best_reward = self._evaluate_actions(env, n_trials)
        
        # Create optimal debt structure
        optimal_debt = self._create_debt_structure_from_action(best_action)
        
        # Run final projection
        final_results = self.waterfall.project_cashflows(
            project_id=project.project_id,
            revenue_forecast=revenue_forecast,
            opex_forecast=opex_forecast,
            tax_rate=tax_rate,
            debt_facilities=optimal_debt,
            dsra_target=optimal_debt[0].principal * 0.5,
            mra_target=project.total_capex * 0.02,
        )
        
        return {
            "optimal_leverage": float(best_action[0]),
            "optimal_tenor": float(best_action[1]),
            "optimal_grace_period": float(best_action[2]),
            "optimal_sculpting": float(best_action[3]),
            "optimal_sweep_trigger": float(best_action[4]),
            "optimal_hedge_ratio": float(best_action[5]),
            "debt_structure": optimal_debt,
            "min_dscr": float(np.min(final_results["dscr"])),
            "avg_dscr": float(np.mean(final_results["dscr"])),
            "equity_irr": float(self.waterfall.calculate_equity_irr(final_results["equity_cashflow"])),
            "equity_npv": float(self.waterfall.calculate_equity_npv(final_results["equity_cashflow"], 0.08)),
            "cashflows": final_results,
            "reward": float(best_reward),
        }
    
    def _evaluate_actions(self, env: DebtOptimizationEnv, n_trials: int) -> Tuple[np.ndarray, float]:
        """Evaluate multiple actions and return best"""
        best_reward = -np.inf
        best_action = None
        
        for _ in range(n_trials):
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action, best_reward
    
    def _create_debt_structure_from_action(self, action: np.ndarray) -> List[DebtFacility]:
        """Create debt facilities from action"""
        leverage, tenor, grace, sculpting, sweep_trigger, hedge_ratio = action
        
        total_debt = self.project.total_capex * leverage if hasattr(self, 'project') else 100_000_000 * leverage
        
        return [
            DebtFacility(
                facility_id="senior_opt",
                name="Optimized Senior Debt",
                facility_type="senior",
                principal=total_debt * 0.7,
                interest_rate=0.06,
                tenor_years=int(tenor),
                grace_period_years=int(grace),
                repayment_profile="sculpted" if sculpting > 1.0 else "annuity",
                min_dscr=sweep_trigger,
                dsra_required=True,
                dsra_size_months=6,
            ),
        ]
    
    def optimize_sculpting(
        self,
        project: Project,
        revenue_forecast: np.ndarray,
        opex_forecast: np.ndarray,
        tax_rate: float,
        debt_facilities: List[DebtFacility],
        params: SculptingParams,
    ) -> Dict[str, Any]:
        """Optimize sculpted repayment profile"""
        logger.info("Optimizing debt sculpting", project_id=project.project_id)
        
        # Project CFADS
        cfads_forecast = revenue_forecast - opex_forecast - (revenue_forecast - opex_forecast) * tax_rate
        
        # Target debt service based on DSCR
        target_ds = cfads_forecast / params.target_dscr
        
        # Adjust for min/max DSCR
        target_ds = np.clip(target_ds, cfads_forecast / params.max_dscr, cfads_forecast / params.min_dscr)
        
        # Tail adjustment
        tail_amount = sum(f.principal for f in debt_facilities) * params.tail_percentage
        target_ds[-4:] += tail_amount / 4  # Last year quarterly
        
        return {
            "sculpted_debt_service": target_ds,
            "annual_principal": target_ds - np.array([f.principal * f.interest_rate for f in debt_facilities if f.facility_type == "senior"]).mean(),
            "tail_percentage": params.tail_percentage,
        }
    
    def optimize_hedging(
        self,
        debt_facilities: List[DebtFacility],
        rate_forecast: np.ndarray,
        hedge_budget_bps: float = 50,
    ) -> List[HedgingStrategy]:
        """Optimize interest rate hedging strategy"""
        logger.info("Optimizing hedging strategy")
        
        # Calculate floating rate exposure
        floating_debt = sum(f.principal for f in debt_facilities if f.is_floating)
        fixed_debt = sum(f.principal for f in debt_facilities if not f.is_floating)
        total_debt = floating_debt + fixed_debt
        
        # Target hedge ratio based on rate outlook
        rate_volatility = np.std(rate_forecast)
        rate_trend = rate_forecast[-1] - rate_forecast[0]
        
        # If rates expected to rise, increase hedge ratio
        base_hedge_ratio = 0.7
        if rate_trend > 0:
            target_hedge = min(0.95, base_hedge_ratio + 0.1)
        else:
            target_hedge = max(0.5, base_hedge_ratio - 0.1)
        
        # Create hedging strategies
        strategies = []
        
        notional_to_hedge = floating_debt * target_hedge - fixed_debt
        if notional_to_hedge > 0:
            strategies.append(HedgingStrategy(
                strategy_id="swap_1",
                hedge_type="swap",
                notional=notional_to_hedge,
                fixed_rate=float(np.mean(rate_forecast) + 0.005),  # Swap rate
                start_date=__import__('datetime').datetime.utcnow(),
                end_date=__import__('datetime').datetime.utcnow() + __import__('datetime').timedelta(days=365*10),
                cost_bps=15,
            ))
        
        return strategies
    
    def save_model(self, path: str) -> None:
        """Save RL model"""
        if self.model:
            self.model.save(path)
            logger.info("Optimizer model saved", path=path)
    
    def load_model(self, path: str) -> None:
        """Load RL model"""
        self.model = PPO.load(path)
        logger.info("Optimizer model loaded", path=path)