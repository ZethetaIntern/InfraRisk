"""
Credit Explainer - SHAP explanations and counterfactuals
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import shap

from src.infra_risk.schemas.credit import CreditFeatures, CreditScore
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class CreditExplainer:
    """SHAP-based explainability for credit scoring"""
    
    def __init__(self, ensemble, feature_names: List[str]):
        self.ensemble = ensemble
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP explainer"""
        try:
            if hasattr(self.ensemble, 'base_models') and 'xgboost' in self.ensemble.base_models:
                self.explainer = shap.TreeExplainer(self.ensemble.base_models['xgboost'])
            else:
                # Use KernelExplainer as fallback
                self.explainer = None
        except Exception as e:
            logger.warning("SHAP explainer initialization failed", error=str(e))
            self.explainer = None
    
    def explain_prediction(
        self,
        features: CreditFeatures,
        prediction: CreditScore,
    ) -> Dict[str, Any]:
        """Generate local explanation for a single prediction"""
        # Prepare feature vector
        X, _ = self.ensemble.prepare_features([features])
        
        explanation = {
            "project_id": features.project_id,
            "global_importance": self._get_global_importance(),
            "local_importance": {},
            "shap_values": None,
            "force_plot_data": None,
            "counterfactuals": [],
            "key_drivers": [],
            "mitigation_actions": [],
        }
        
        # Get SHAP values
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class for binary
            
            explanation["shap_values"] = shap_values[0].tolist() if len(shap_values) > 0 else None
            
            # Local feature importance
            if explanation["shap_values"]:
                local_imp = dict(zip(self.feature_names, explanation["shap_values"]))
                explanation["local_importance"] = dict(
                    sorted(local_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
                )
                
                # Force plot data
                explanation["force_plot_data"] = {
                    "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value),
                    "shap_values": explanation["shap_values"],
                    "features": X[0].tolist(),
                    "feature_names": self.feature_names,
                }
        
        # Key drivers (top positive SHAP)
        if explanation["local_importance"]:
            sorted_features = sorted(
                explanation["local_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            explanation["key_drivers"] = [f[0] for f in sorted_features[:10] if f[1] > 0]
        
        # Counterfactuals
        explanation["counterfactuals"] = self._generate_counterfactuals(features, prediction)
        
        # Mitigation actions
        explanation["mitigation_actions"] = self._generate_mitigation_actions(features, prediction)
        
        return explanation
    
    def _get_global_importance(self) -> Dict[str, float]:
        """Get global feature importance"""
        if self.explainer is not None and hasattr(self.explainer, 'shap_values'):
            try:
                # Use mean absolute SHAP values
                pass
            except:
                pass
        
        # Fallback: return empty, will be populated from model
        return {}
    
    def _generate_counterfactuals(
        self,
        features: CreditFeatures,
        prediction: CreditScore,
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        counterfactuals = []
        
        # Define actionable features and their target improvements
        actionable_features = {
            "dscr_current": {"target": 1.5, "action": "Increase cash flow or reduce debt service"},
            "llcr": {"target": 1.4, "action": "Extend debt tenor or improve cash flow visibility"},
            "contract_risk_score": {"target": 30, "action": "Renegotiate contract terms"},
            "contagion_index": {"target": 0.3, "action": "Diversify counterparty exposure"},
            "demand_volatility": {"target": 0.15, "action": "Secure long-term offtake agreements"},
            "carul_years": {"target": 20, "action": "Invest in preventive maintenance"},
        }
        
        for feature, info in actionable_features.items():
            current_value = getattr(features, feature, None)
            if current_value is not None and current_value < info["target"]:
                # Estimate impact on PD
                # Simplified: assume linear relationship
                improvement_pct = (info["target"] - current_value) / max(info["target"], 0.01)
                pd_reduction = min(0.5, improvement_pct * prediction.pd.pd_1yr * 0.5)
                
                counterfactuals.append({
                    "feature": feature,
                    "current_value": float(current_value),
                    "target_value": info["target"],
                    "action": info["action"],
                    "estimated_pd_reduction": float(pd_reduction),
                    "feasibility": "High" if improvement_pct < 0.5 else "Medium",
                })
        
        return counterfactuals[:5]  # Top 5
    
    def _generate_mitigation_actions(
        self,
        features: CreditFeatures,
        prediction: CreditScore,
    ) -> List[str]:
        """Generate specific mitigation recommendations"""
        actions = []
        
        # DSCR-related
        if features.dscr_current and features.dscr_current < 1.3:
            actions.append("Restructure debt service schedule to achieve minimum 1.30x DSCR")
            actions.append("Negotiate cash sweep holiday during ramp-up period")
        
        # Covenant-related
        if features.contract_risk_score and features.contract_risk_score > 60:
            actions.append("Renegotiate weak financial covenants (DSCR, LLCR thresholds)")
            actions.append("Add cure periods for covenant breaches")
        
        # Counterparty risk
        if features.contagion_index and features.contagion_index > 0.5:
            actions.append("Diversify offtaker/supplier base to reduce concentration risk")
            actions.append("Require parent guarantees from key counterparties")
        
        # Demand risk
        if features.demand_volatility and features.demand_volatility > 0.2:
            actions.append("Secure minimum revenue guarantees or availability payments")
            actions.append("Implement tariff indexation to inflation")
        
        # Construction risk
        if features.construction_progress and features.construction_progress < 50 and features.anomaly_count and features.anomaly_count > 2:
            actions.append("Engage independent engineer for construction monitoring")
            actions.append("Increase DSRA funding to cover potential delays")
        
        # Degradation risk
        if features.carul_years and features.carul_years < 10:
            actions.append("Accelerate major maintenance schedule")
            actions.append("Increase MRA funding for early-life rehabilitation")
        
        # ESG
        if features.e_score and features.e_score < 50:
            actions.append("Develop climate adaptation plan for physical assets")
            actions.append("Implement environmental monitoring program")
        
        return actions[:10]
    
    def explain_portfolio(
        self,
        features_list: List[CreditFeatures],
        predictions: List[CreditScore],
    ) -> Dict[str, Any]:
        """Generate portfolio-level explanations"""
        # Aggregate SHAP values
        all_shap = []
        for f, p in zip(features_list, predictions):
            exp = self.explain_prediction(f, p)
            if exp["shap_values"]:
                all_shap.append(exp["shap_values"])
        
        if all_shap:
            mean_shap = np.mean(all_shap, axis=0)
            global_imp = dict(zip(self.feature_names, mean_shap.tolist()))
            global_imp = dict(sorted(global_imp.items(), key=lambda x: abs(x[1]), reverse=True))
        else:
            global_imp = {}
        
        # Portfolio risk drivers
        portfolio_drivers = list(global_imp.keys())[:15]
        
        return {
            "global_feature_importance": global_imp,
            "portfolio_risk_drivers": portfolio_drivers,
            "n_projects": len(features_list),
            "avg_pd": float(np.mean([p.pd.pd_1yr for p in predictions])),
            "avg_lgd": float(np.mean([p.lgd.lgd_point for p in predictions])),
            "total_el": float(sum(p.el.el_mean for p in predictions)),
        }
    
    def generate_credit_committee_report(
        self,
        features: CreditFeatures,
        prediction: CreditScore,
    ) -> str:
        """Generate formatted report for credit committee"""
        explanation = self.explain_prediction(features, prediction)
        
        report = f"""
CREDIT COMMITTEE REPORT - {features.project_id}
{'='*60}

EXECUTIVE SUMMARY
-----------------
Internal Rating: {prediction.internal_rating}
PD (1yr): {prediction.pd.pd_1yr:.2%}
LGD: {prediction.lgd.lgd_point:.2%}
Expected Loss: ${prediction.el.el_mean:,.0f}
Implied Spread: {prediction.implied_spread_bps:.0f} bps

KEY RISK DRIVERS
----------------
"""
        for driver in explanation["key_drivers"][:5]:
            report += f"• {driver}\n"
        
        report += "\nMITIGATION ACTIONS\n------------------\n"
        for action in explanation["mitigation_actions"][:5]:
            report += f"• {action}\n"
        
        report += "\nCOUNTERFACTUAL SCENARIOS\n------------------------\n"
        for cf in explanation["counterfactuals"][:3]:
            report += f"• {cf['action']}: PD reduction ~{cf['estimated_pd_reduction']:.2%}\n"
        
        return report