"""
Game Mode Manager - Different simulation modes
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from src.infra_risk.schemas.simulation import GameMode, SimulationState
from src.infra_risk.schemas.project import Project, ProjectStage, ProjectSector
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class GameModeManager:
    """Manages different game modes and their configurations"""
    
    MODE_CONFIGS = {
        GameMode.SINGLE_DEAL: {
            "name": "Single Deal Tutorial",
            "description": "Learn the basics with one project",
            "num_projects": 1,
            "project_stages": [ProjectStage.CONSTRUCTION],
            "sectors": [ProjectSector.TRANSPORT],
            "difficulty": "easy",
            "tutorial": True,
            "time_limit_minutes": 30,
        },
        GameMode.PORTFOLIO_MANAGER: {
            "name": "Portfolio Manager",
            "description": "Manage 10-15 deals across sectors",
            "num_projects": 12,
            "project_stages": [ProjectStage.CONSTRUCTION, ProjectStage.RAMP_UP, ProjectStage.OPERATIONAL],
            "sectors": [ProjectSector.TRANSPORT, ProjectSector.ENERGY, ProjectSector.WATER],
            "difficulty": "medium",
            "tutorial": False,
            "time_limit_minutes": 90,
        },
        GameMode.CRISIS_MANAGER: {
            "name": "Crisis Manager",
            "description": "Handle workouts and restructurings",
            "num_projects": 8,
            "project_stages": [ProjectStage.OPERATIONAL, ProjectStage.REFINANCING],
            "sectors": [ProjectSector.TRANSPORT, ProjectSector.ENERGY],
            "difficulty": "hard",
            "tutorial": False,
            "time_limit_minutes": 90,
            "crisis_intensity": "high",
        },
        GameMode.DEAL_STRUCTURER: {
            "name": "Deal Structurer",
            "description": "Structure greenfield projects from scratch",
            "num_projects": 3,
            "project_stages": [ProjectStage.CONCEPT, ProjectStage.PRE_FEASIBILITY, ProjectStage.FEASIBILITY],
            "sectors": [ProjectSector.ENERGY, ProjectSector.TRANSPORT],
            "difficulty": "expert",
            "tutorial": False,
            "time_limit_minutes": 90,
            "greenfield_only": True,
        },
    }
    
    def get_config(self, mode: GameMode) -> Dict[str, Any]:
        """Get configuration for a game mode"""
        return self.MODE_CONFIGS.get(mode, self.MODE_CONFIGS[GameMode.PORTFOLIO_MANAGER])
    
    def generate_projects_for_mode(
        self,
        mode: GameMode,
        available_projects: List[Project],
    ) -> List[Project]:
        """Select appropriate projects for game mode"""
        config = self.get_config(mode)
        
        # Filter by stage and sector
        filtered = [
            p for p in available_projects
            if p.stage in config["project_stages"] and p.sector in config["sectors"]
        ]
        
        # Randomly select
        num = min(config["num_projects"], len(filtered))
        if num == 0:
            # Fallback: create synthetic projects
            return self._create_synthetic_projects(mode, config["num_projects"])
        
        selected = np.random.choice(filtered, size=num, replace=False)
        return list(selected)
    
    def _create_synthetic_projects(self, mode: GameMode, count: int) -> List[Project]:
        """Create synthetic projects for game mode"""
        projects = []
        config = self.get_config(mode)
        
        for i in range(count):
            stage = np.random.choice(config["project_stages"])
            sector = np.random.choice(config["sectors"])
            
            project = Project(
                project_id=f"syn_{mode.value}_{i}",
                name=f"Synthetic {sector.value} Project {i+1}",
                sector=sector,
                sub_sector=sector,  # Simplified
                stage=stage,
                country="Synthetic Country",
                country_code="SYN",
                region="Synthetic Region",
                latitude=0,
                longitude=0,
                concession_years=25,
                total_capex=np.random.uniform(50_000_000, 500_000_000),
                debt_amount=0,
                equity_amount=0,
                capacity=100,
                capacity_unit="MW" if sector == ProjectSector.ENERGY else "km",
            )
            projects.append(project)
        
        return projects
    
    def get_initial_conditions(self, mode: GameMode) -> Dict[str, Any]:
        """Get initial conditions for game mode"""
        config = self.get_config(mode)
        
        base_conditions = {
            "initial_cash": 100_000_000,
            "initial_debt": 0,
            "available_actions": [],
            "event_frequency": 0.15,
            "crisis_probability": 0.05,
        }
        
        if mode == GameMode.SINGLE_DEAL:
            base_conditions.update({
                "initial_cash": 50_000_000,
                "available_actions": ["basic"],
                "tutorial_steps": [
                    "Review project details",
                    "Check cash flow waterfall",
                    "Make first quarter decision",
                    "Respond to event",
                    "Complete simulation",
                ],
            })
        elif mode == GameMode.PORTFOLIO_MANAGER:
            base_conditions.update({
                "initial_cash": 200_000_000,
                "available_actions": ["all"],
            })
        elif mode == GameMode.CRISIS_MANAGER:
            base_conditions.update({
                "initial_cash": 50_000_000,
                "initial_debt": 500_000_000,
                "available_actions": ["restructuring", "workout", "waiver", "asset_sale"],
                "crisis_probability": 0.3,
            })
        elif mode == GameMode.DEAL_STRUCTURER:
            base_conditions.update({
                "initial_cash": 20_000_000,  # Development budget
                "available_actions": ["structuring", "financing", "contracting", "permits"],
                "event_frequency": 0.1,
            })
        
        return base_conditions
    
    def get_victory_conditions(self, mode: GameMode) -> Dict[str, Any]:
        """Get victory conditions for game mode"""
        conditions = {
            GameMode.SINGLE_DEAL: {
                "min_dscr": 1.2,
                "min_irr": 0.10,
                "max_quarters": 40,
                "must_complete": True,
            },
            GameMode.PORTFOLIO_MANAGER: {
                "min_avg_dscr": 1.3,
                "min_portfolio_irr": 0.12,
                "max_defaults": 1,
                "min_diversification": 0.7,
            },
            GameMode.CRISIS_MANAGER: {
                "recovery_rate": 0.6,
                "min_cash_preserved": 0.5,
                "restructuring_success": 0.7,
                "time_to_resolve": 12,  # quarters
            },
            GameMode.DEAL_STRUCTURER: {
                "financial_close": True,
                "min_equity_irr": 0.15,
                "bankability_score": 0.8,
                "esg_score": 0.7,
            },
        }
        
        return conditions.get(mode, conditions[GameMode.PORTFOLIO_MANAGER])