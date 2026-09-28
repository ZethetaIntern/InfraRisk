"""
Credit Scoring Stacking Ensemble
Combines XGBoost, GNN features, TFT outputs for PD, LGD, EL prediction
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import shap

from src.infra_risk.schemas.credit import (
    CreditFeatures,
    PDPrediction,
    LGDPrediction,
    ELDistribution,
    CreditScore,
    CreditModelType,
    EnsembleConfig,
    ModelPerformance,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class StackingEnsemble:
    """Level-1 Stacking Ensemble for credit scoring"""
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.base_models = {}
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
        # Initialize base models
        self._init_base_models()
    
    def _init_base_models(self):
        """Initialize base models from config"""
        for model_type in self.config.base_models:
            if model_type == CreditModelType.XGBOOST:
                self.base_models["xgboost"] = xgb.XGBClassifier(
                    **self.config.xgb_params,
                    n_jobs=-1,
                    random_state=42,
                )
            elif model_type == CreditModelType.LIGHTGBM:
                self.base_models["lightgbm"] = lgb.LGBMClassifier(
                    **self.config.lgbm_params,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=-1,
                )
            # GNN and TFT are feature providers, not standalone models here
    
    def prepare_features(
        self,
        features_list: List[CreditFeatures],
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from CreditFeatures objects"""
        feature_dicts = []
        
        for f in features_list:
            d = {}
            # Add all non-None attributes
            for key, value in f.__dict__.items():
                if key not in ["feature_vector", "feature_names", "extracted_at"] and value is not None:
                    if isinstance(value, np.ndarray):
                        # Flatten arrays
                        if value.ndim == 1:
                            for i, v in enumerate(value):
                                d[f"{key}_{i}"] = float(v)
                        else:
                            d[key] = float(np.mean(value))
                    elif isinstance(value, (int, float, str)):
                        d[key] = value
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            d[f"{key}_{k}"] = float(v) if isinstance(v, (int, float)) else 0
            
            # Handle categorical variables
            if f.sector:
                d[f"sector_{f.sector}"] = 1
            if f.country:
                d[f"country_{f.country}"] = 1
            if f.stage:
                d[f"stage_{f.stage}"] = 1
            
            feature_dicts.append(d)
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_dicts).fillna(0)
        
        # Ensure consistent columns
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        else:
            self.feature_names = df.columns.tolist()
        
        return df.values, self.feature_names
    
    def fit(
        self,
        features_list: List[CreditFeatures],
        pd_labels: np.ndarray,
        lgd_labels: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train the stacking ensemble"""
        logger.info("Training stacking ensemble", n_samples=len(features_list))
        
        X, _ = self.prepare_features(features_list)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train base models with cross-validation for meta-features
        meta_features_pd = np.zeros((len(features_list), len(self.base_models)))
        meta_features_lgd = np.zeros((len(features_list), len(self.base_models)))
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"Training base model: {name}")
            
            # PD model
            cv_preds_pd = cross_val_predict(
                model, X_scaled, pd_labels, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            meta_features_pd[:, i] = cv_preds_pd
            
            # Fit on full data
            model.fit(X_scaled, pd_labels)
            
            # LGD model (regressor)
            lgd_model = xgb.XGBRegressor(
                **{k: v for k, v in self.config.xgb_params.items() if k != "objective"},
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=42,
            )
            
            cv_preds_lgd = cross_val_predict(
                lgd_model, X_scaled, lgd_labels, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                n_jobs=-1
            )
            meta_features_lgd[:, i] = cv_preds_lgd
            lgd_model.fit(X_scaled, lgd_labels)
            self.base_models[f"{name}_lgd"] = lgd_model
        
        # Train meta-learner for PD
        if self.config.meta_learner == "logistic":
            self.meta_learner_pd = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        elif self.config.meta_learner == "xgboost":
            self.meta_learner_pd = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        else:
            self.meta_learner_pd = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        
        # Calibrate meta-learner
        self.meta_learner_pd = CalibratedClassifierCV(self.meta_learner_pd, cv=3, method="isotonic")
        self.meta_learner_pd.fit(meta_features_pd, pd_labels)
        
        # Meta-learner for LGD
        self.meta_learner_lgd = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        self.meta_learner_lgd.fit(meta_features_lgd, lgd_labels)
        
        self.is_fitted = True
        
        # Compute SHAP values for base models
        self._compute_shap_values(X_scaled)
        
        return {"status": "trained", "n_features": X.shape[1]}
    
    def _compute_shap_values(self, X: np.ndarray):
        """Compute SHAP values for explainability"""
        try:
            # Use TreeExplainer for XGBoost
            if "xgboost" in self.base_models:
                self.shap_explainer = shap.TreeExplainer(self.base_models["xgboost"])
                self.shap_values = self.shap_explainer.shap_values(X)
            else:
                self.shap_explainer = None
        except Exception as e:
            logger.warning("SHAP computation failed", error=str(e))
            self.shap_explainer = None
    
    def predict_pd(
        self,
        features_list: List[CreditFeatures],
    ) -> List[PDPrediction]:
        """Predict Probability of Default"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X, _ = self.prepare_features(features_list)
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        meta_features = np.zeros((len(features_list), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if "_lgd" not in name:
                meta_features[:, i] = model.predict_proba(X_scaled)[:, 1]
        
        # Meta-learner prediction
        pd_probs = self.meta_learner_pd.predict_proba(meta_features)[:, 1]
        
        # Generate term structure (simplified)
        predictions = []
        for i, f in enumerate(features_list):
            pd_1yr = float(pd_probs[i])
            pd_5yr = 1 - (1 - pd_1yr) ** 5
            pd_10yr = 1 - (1 - pd_1yr) ** 10
            pd_lifetime = min(1.0, pd_10yr * 1.5)
            
            # SHAP values for this prediction
            shap_vals = None
            if self.shap_explainer is not None:
                shap_vals = self.shap_values[i] if i < len(self.shap_values) else None
            
            predictions.append(PDPrediction(
                project_id=f.project_id,
                pd_1yr=pd_1yr,
                pd_5yr=pd_5yr,
                pd_10yr=pd_10yr,
                pd_lifetime=pd_lifetime,
                pd_term_structure={1: pd_1yr, 5: pd_5yr, 10: pd_10yr},
                confidence_interval=(max(0, pd_1yr * 0.5), min(1, pd_1yr * 2)),
                model_contributions={CreditModelType.XGBOOST: 0.6, CreditModelType.GNN: 0.2, CreditModelType.TFT: 0.2},
                shap_values=shap_vals,
            ))
        
        return predictions
    
    def predict_lgd(
        self,
        features_list: List[CreditFeatures],
    ) -> List[LGDPrediction]:
        """Predict Loss Given Default"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X, _ = self.prepare_features(features_list)
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        meta_features = np.zeros((len(features_list), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if "_lgd" in name:
                meta_features[:, i // 2] = model.predict(X_scaled)
        
        # Meta-learner prediction
        lgd_preds = self.meta_learner_lgd.predict(meta_features)
        lgd_preds = np.clip(lgd_preds, 0, 1)
        
        predictions = []
        for i, f in enumerate(features_list):
            lgd_point = float(lgd_preds[i])
            
            # Generate distribution (beta distribution approximation)
            alpha = lgd_point * 10
            beta = (1 - lgd_point) * 10
            dist = np.random.beta(alpha, beta, 10000)
            
            predictions.append(LGDPrediction(
                project_id=f.project_id,
                lgd_point=lgd_point,
                lgd_distribution=dist,
                lgd_p10=float(np.percentile(dist, 10)),
                lgd_p50=float(np.percentile(dist, 50)),
                lgd_p90=float(np.percentile(dist, 90)),
                recovery_rate=1 - lgd_point,
                recovery_time_years=2.5,
                collateral_coverage=0.6,
            ))
        
        return predictions
    
    def predict_el(
        self,
        pd_preds: List[PDPrediction],
        lgd_preds: List[LGDPrediction],
        exposure: np.ndarray,
    ) -> List[ELDistribution]:
        """Compute Expected Loss distribution"""
        el_distributions = []
        
        for pd, lgd, exp in zip(pd_preds, lgd_preds, exposure):
            # Monte Carlo simulation
            n_sims = 10000
            pd_samples = np.random.beta(pd.pd_1yr * 100, (1 - pd.pd_1yr) * 100, n_sims)
            lgd_samples = lgd.lgd_distribution[:n_sims] if len(lgd.lgd_distribution) >= n_sims else np.random.beta(
                pd.pd_1yr * 10, (1 - pd.pd_1yr) * 10, n_sims
            )
            
            el_samples = pd_samples * lgd_samples * exp
            
            el_distributions.append(ELDistribution(
                el_mean=float(np.mean(el_samples)),
                el_std=float(np.std(el_samples)),
                el_p10=float(np.percentile(el_samples, 10)),
                el_p50=float(np.percentile(el_samples, 50)),
                el_p90=float(np.percentile(el_samples, 90)),
                el_p99=float(np.percentile(el_samples, 99)),
                ul=float(np.percentile(el_samples, 99.9) - np.mean(el_samples)),
            ))
        
        return el_distributions
    
    def save(self, path: str):
        """Save ensemble to disk"""
        joblib.dump({
            "base_models": self.base_models,
            "meta_learner_pd": self.meta_learner_pd,
            "meta_learner_lgd": self.meta_learner_lgd,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
            "is_fitted": self.is_fitted,
        }, path)
        logger.info("Ensemble saved", path=path)
    
    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        """Load ensemble from disk"""
        data = joblib.load(path)
        ensemble = cls(data["config"])
        ensemble.base_models = data["base_models"]
        ensemble.meta_learner_pd = data["meta_learner_pd"]
        ensemble.meta_learner_lgd = data["meta_learner_lgd"]
        ensemble.scaler = data["scaler"]
        ensemble.feature_names = data["feature_names"]
        ensemble.is_fitted = data["is_fitted"]
        logger.info("Ensemble loaded", path=path)
        return ensemble