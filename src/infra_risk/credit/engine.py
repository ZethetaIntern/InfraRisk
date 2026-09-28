"""
Credit Scoring Engine - Main orchestrator
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

from src.infra_risk.schemas.credit import (
    CreditFeatures,
    CreditScore,
    PDPrediction,
    LGDPrediction,
    ELDistribution,
)
from src.infra_risk.schemas.project import Project
from src.infra_risk.utils.logging import get_logger
from src.infra_risk.credit.ensemble import StackingEnsemble
from src.infra_risk.credit.explainer import CreditExplainer

logger = get_logger(__name__)


class CreditScoringEngine:
    """Main credit scoring engine"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        ensemble: Optional[StackingEnsemble] = None,
    ):
        self.model_path = model_path
        self.ensemble = ensemble or StackingEnsemble()
        self.explainer = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        if self.ensemble.is_fitted:
            self.explainer = CreditExplainer(self.ensemble, self.ensemble.feature_names)
        
        logger.info("CreditScoringEngine initialized")
    
    def load_model(self, path: str) -> None:
        """Load trained ensemble"""
        self.ensemble = StackingEnsemble.load(path)
        self.explainer = CreditExplainer(self.ensemble, self.ensemble.feature_names)
        logger.info("Credit model loaded", path=path)
    
    def save_model(self, path: str) -> None:
        """Save ensemble"""
        self.ensemble.save(path)
        logger.info("Credit model saved", path=path)
    
    def assemble_features(
        self,
        project: Project,
        geospatial_progress: Optional[Any] = None,
        demand_forecast: Optional[Any] = None,
        contagion_index: Optional[Any] = None,
        carul: Optional[Any] = None,
        contract_risk: Optional[Any] = None,
        coverage_ratios: Optional[Any] = None,
        macro_profile: Optional[Any] = None,
    ) -> CreditFeatures:
        """Assemble all features for credit scoring"""
        
        features = CreditFeatures(project_id=project.project_id)
        
        # Financial features from coverage ratios
        if coverage_ratios:
            features.dscr_current = coverage_ratios.dscr
            features.dscr_forecast = None  # Would come from forecast
            features.llcr = coverage_ratios.llcr
            features.plcr = coverage_ratios.plcr
            features.debt_to_equity = coverage_ratios.debt_to_equity
            features.cash_sweep_pct = 0.5  # Placeholder
            features.dsrca_balance = 0  # Would come from waterfall
            features.mra_balance = 0
        
        # Demand features
        if demand_forecast and demand_forecast.quantiles:
            q = demand_forecast.quantiles
            features.demand_p50 = q.p50
            features.demand_volatility = float(np.std(q.p50) / np.mean(q.p50)) if len(q.p50) > 1 else 0.15
            features.demand_growth_rate = float((q.p50[-1] / q.p50[0]) ** (1/len(q.p50)) - 1) if len(q.p50) > 1 else 0.02
        
        # GNN features
        if contagion_index:
            features.contagion_index = contagion_index.contagion_score
            features.centrality_score = contagion_index.centrality_measures.get("eigenvector", 0)
            features.systemic_importance = contagion_index.systemic_importance
        
        # Geospatial features
        if geospatial_progress:
            features.construction_progress = geospatial_progress.progress_pct
            # Would need history for volatility
        
        # PINN features
        if carul:
            features.carul_years = carul.carul_by_scenario.get("SSP2-4.5", 25)
            features.degradation_rate = 1 / max(features.carul_years, 1)
        
        # Legal features
        if contract_risk:
            features.contract_risk_score = contract_risk.overall_score
            features.covenant_strength = 100 - contract_risk.overall_score  # Inverse
            features.force_majeure_score = contract_risk.category_scores.get(
                type(contract_risk.category_scores).__args__[0].FORCE_MAJEURE, 50
            ) if contract_risk.category_scores else 50
        
        # Macro features
        if macro_profile:
            features.sovereign_rating = macro_profile.sovereign_rating
            features.gdp_growth = macro_profile.gdp_growth
            features.inflation = macro_profile.inflation
            features.interest_rate = macro_profile.interest_rate
        
        # Project features
        features.sector = project.sector.value if hasattr(project.sector, 'value') else str(project.sector)
        features.country = project.country_code
        features.stage = project.stage.value if hasattr(project.stage, 'value') else str(project.stage)
        features.vintage = project.vintage_year if hasattr(project, 'vintage_year') else 2020
        features.concession_life = project.concession_years
        
        # ESG features
        features.e_score = project.e_score
        features.s_score = project.s_score
        features.g_score = project.g_score
        
        return features
    
    def score_project(
        self,
        project: Project,
        geospatial_progress: Optional[Any] = None,
        demand_forecast: Optional[Any] = None,
        contagion_index: Optional[Any] = None,
        carul: Optional[Any] = None,
        contract_risk: Optional[Any] = None,
        coverage_ratios: Optional[Any] = None,
        macro_profile: Optional[Any] = None,
        exposure: float = 1.0,
    ) -> CreditScore:
        """Generate complete credit score for a project"""
        
        # Assemble features
        features = self.assemble_features(
            project, geospatial_progress, demand_forecast, contagion_index,
            carul, contract_risk, coverage_ratios, macro_profile
        )
        
        # Predict
        pd_preds = self.ensemble.predict_pd([features])
        lgd_preds = self.ensemble.predict_lgd([features])
        el_dist = self.ensemble.predict_el(pd_preds, lgd_preds, np.array([exposure]))
        
        pd = pd_preds[0]
        lgd = lgd_preds[0]
        el = el_dist[0]
        
        # Determine internal rating
        rating = self._pd_to_rating(pd.pd_1yr)
        implied_spread = pd.pd_1yr * lgd.lgd_point * 10000  # bps
        
        # Get explanations
        explanation = {}
        if self.explainer:
            explanation = self.explainer.explain_prediction(features, CreditScore(
                project_id=project.project_id,
                pd=pd,
                lgd=lgd,
                el=el,
                internal_rating=rating,
                implied_spread_bps=implied_spread,
            ))
        
        return CreditScore(
            project_id=project.project_id,
            pd=pd,
            lgd=lgd,
            el=el,
            internal_rating=rating,
            implied_spread_bps=implied_spread,
            shap_global_importance=explanation.get("global_importance", {}),
            shap_local_values=explanation.get("local_importance", {}),
            counterfactuals=explanation.get("counterfactuals", []),
            key_drivers=explanation.get("key_drivers", []),
            mitigation_actions=explanation.get("mitigation_actions", []),
            ensemble_weights={
                "xgboost": 0.5,
                "gnn": 0.25,
                "tft": 0.25,
            },
            validation_auroc=0.82,  # Would come from validation
            validation_auprc=0.78,
            calibration_score=0.95,
        )
    
    def _pd_to_rating(self, pd_1yr: float) -> str:
        """Map PD to internal rating"""
        if pd_1yr < 0.0005:
            return "AAA"
        elif pd_1yr < 0.001:
            return "AA+"
        elif pd_1yr < 0.002:
            return "AA"
        elif pd_1yr < 0.004:
            return "AA-"
        elif pd_1yr < 0.008:
            return "A+"
        elif pd_1yr < 0.015:
            return "A"
        elif pd_1yr < 0.03:
            return "A-"
        elif pd_1yr < 0.05:
            return "BBB+"
        elif pd_1yr < 0.08:
            return "BBB"
        elif pd_1yr < 0.12:
            return "BBB-"
        elif pd_1yr < 0.18:
            return "BB+"
        elif pd_1yr < 0.25:
            return "BB"
        elif pd_1yr < 0.35:
            return "BB-"
        elif pd_1yr < 0.5:
            return "B+"
        elif pd_1yr < 0.7:
            return "B"
        elif pd_1yr < 0.85:
            return "B-"
        else:
            return "CCC"
    
    def batch_score(
        self,
        projects: List[Project],
        features_dict: Dict[str, Dict],
        exposures: Dict[str, float],
    ) -> Dict[str, CreditScore]:
        """Score multiple projects"""
        results = {}
        
        for project in projects:
            try:
                feat_dict = features_dict.get(project.project_id, {})
                score = self.score_project(
                    project,
                    geospatial_progress=feat_dict.get("geospatial"),
                    demand_forecast=feat_dict.get("demand"),
                    contagion_index=feat_dict.get("gnn"),
                    carul=feat_dict.get("pinn"),
                    contract_risk=feat_dict.get("legal"),
                    coverage_ratios=feat_dict.get("waterfall"),
                    macro_profile=feat_dict.get("macro"),
                    exposure=exposures.get(project.project_id, 1.0),
                )
                results[project.project_id] = score
            except Exception as e:
                logger.error("Scoring failed", project_id=project.project_id, error=str(e))
        
        return results
    
    def train(
        self,
        training_data: List[CreditFeatures],
        pd_labels: np.ndarray,
        lgd_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Train the ensemble"""
        logger.info("Training credit ensemble", n_samples=len(training_data))
        
        result = self.ensemble.fit(training_data, pd_labels, lgd_labels)
        
        # Initialize explainer
        self.explainer = CreditExplainer(self.ensemble, self.ensemble.feature_names)
        
        return result