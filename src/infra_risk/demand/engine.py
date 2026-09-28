"""
Demand Forecasting Engine - Main orchestrator for TFT-based demand forecasting
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.infra_risk.schemas.demand import (
    SectorDemandInput,
    DemandForecast,
    QuantileForecast,
    StaticCovariates,
    KnownFutureInputs,
    HistoricalVariables,
    SectorType,
    DemandMetric,
    TimeHorizon,
    MacroProfile,
)
from src.infra_risk.schemas.project import Project
from src.infra_risk.utils.logging import get_logger
from src.infra_risk.demand.tft_model import TFTModel, create_tft_dataset, TFTLightning

logger = get_logger(__name__)


class DemandForecastingEngine:
    """Main demand forecasting engine using Temporal Fusion Transformer"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model: Optional[TFTModel] = None
        self.dataset = None
        self.dataloader = None
        
        # Default feature configurations
        self.static_categoricals = ["sector", "country", "region"]
        self.static_reals = [
            "capacity", "vintage_year", "concession_years",
            "gdp_per_capita", "population_catchment", "gdp_growth_rate",
            "urbanization_rate", "institutional_quality", "latitude", "longitude"
        ]
        self.time_varying_known_categoricals = []
        self.time_varying_known_reals = ["time_idx", "quarter", "year"]
        self.time_varying_unknown_categoricals = []
        self.time_varying_unknown_reals = [
            "demand", "gdp", "inflation", "interest_rate",
            "exchange_rate", "commodity_price"
        ]
        self.target = "demand"
        self.group_ids = ["project_id"]
        self.time_idx = "time_idx"
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        logger.info("DemandForecastingEngine initialized", device=str(self.device))
    
    def load_model(self, path: str) -> None:
        """Load trained TFT model"""
        checkpoint = torch.load(path, map_location=self.device)
        # Recreate model from checkpoint config
        # TODO: Implement proper model loading
        logger.info("Model loaded", path=path)
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint"""
        if self.model:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "static_categoricals": self.static_categoricals,
                    "static_reals": self.static_reals,
                    "time_varying_known_categoricals": self.time_varying_known_categoricals,
                    "time_varying_known_reals": self.time_varying_known_reals,
                    "time_varying_unknown_categoricals": self.time_varying_unknown_categoricals,
                    "time_varying_unknown_reals": self.time_varying_unknown_reals,
                    "target": self.target,
                    "max_encoder_length": self.model.max_encoder_length,
                    "max_prediction_length": self.model.max_prediction_length,
                }
            }, path)
            logger.info("Model saved", path=path)
    
    def prepare_training_data(
        self,
        projects: List[Project],
        macro_profiles: List[MacroProfile],
        demand_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare training data from projects and macro data"""
        logger.info("Preparing training data", n_projects=len(projects))
        
        # Merge project static features with macro time series
        # This creates the long-format DataFrame needed for TFT
        records = []
        
        for project in projects:
            # Get macro data for project's country
            country_macro = [m for m in macro_profiles if m.country_code == project.country_code]
            
            # Create time series for each quarter
            # TODO: Implement full data preparation pipeline
            pass
        
        return pd.DataFrame(records)
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Train the TFT model"""
        logger.info("Training TFT model", epochs=epochs, batch_size=batch_size)
        
        # Create datasets
        train_dataset = create_tft_dataset(
            train_data,
            target=self.target,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_categoricals=self.time_varying_unknown_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            group_ids=self.group_ids,
            time_idx=self.time_idx,
        )
        
        val_dataset = create_tft_dataset(
            val_data,
            target=self.target,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_categoricals=self.time_varying_unknown_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            group_ids=self.group_ids,
            time_idx=self.time_idx,
        )
        
        # Create dataloaders
        train_loader = train_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=4
        )
        val_loader = val_dataset.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=4
        )
        
        # Initialize model from dataset
        self.model = TFTModel.from_dataset(
            train_dataset,
            hidden_size=256,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=64,
            output_size=3,  # P10, P50, P90
            learning_rate=1e-3,
        ).to(self.device)
        
        # Train with PyTorch Lightning
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        
        lightning_model = TFTLightning(self.model)
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            ],
            logger=pl.loggers.TensorBoardLogger("logs/", name="tft_demand"),
        )
        
        trainer.fit(lightning_model, train_loader, val_loader)
        
        return {
            "best_model_path": trainer.checkpoint_callback.best_model_path,
            "best_val_loss": trainer.checkpoint_callback.best_model_score.item(),
        }
    
    def forecast(
        self,
        input_data: SectorDemandInput,
        horizon_quarters: int = 40,
    ) -> DemandForecast:
        """Generate demand forecast for a project"""
        logger.info("Generating forecast", project_id=input_data.static_covariates.project_id)
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        # Prepare input for prediction
        # Convert SectorDemandInput to DataFrame format
        pred_data = self._prepare_prediction_input(input_data, horizon_quarters)
        
        # Create dataset
        pred_dataset = create_tft_dataset(
            pred_data,
            target=self.target,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_categoricals=self.time_varying_unknown_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            group_ids=self.group_ids,
            time_idx=self.time_idx,
            max_encoder_length=self.model.max_encoder_length,
            max_prediction_length=horizon_quarters,
        )
        
        pred_loader = pred_dataset.to_dataloader(
            train=False, batch_size=1, num_workers=0
        )
        
        # Generate predictions
        predictions, info = self.model.predict(pred_loader)
        
        # Create timestamps
        last_hist_date = input_data.historical.timestamps[-1]
        forecast_dates = pd.date_range(
            start=last_hist_date + pd.DateOffset(months=3),
            periods=horizon_quarters,
            freq="Q",
        ).tolist()
        
        # Create QuantileForecast
        quantiles = QuantileForecast(
            p10=predictions["p10"][0],
            p50=predictions["p50"][0],
            p90=predictions["p90"][0],
            timestamps=forecast_dates,
            horizon=TimeHorizon.LONG if horizon_quarters > 12 else TimeHorizon.MEDIUM,
        )
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        return DemandForecast(
            project_id=input_data.static_covariates.project_id,
            sector=input_data.static_covariates.sector,
            metric=self._get_metric_for_sector(input_data.static_covariates.sector),
            forecast_date=datetime.utcnow(),
            quantiles=quantiles,
            feature_importance=importance,
            attention_weights=info.get("attention"),
            model_version="tft-v1.0",
        )
    
    def _prepare_prediction_input(
        self,
        input_data: SectorDemandInput,
        horizon: int,
    ) -> pd.DataFrame:
        """Convert SectorDemandInput to prediction DataFrame"""
        static = input_data.static_covariates
        known = input_data.known_future
        hist = input_data.historical
        
        # Combine historical and future known inputs
        # This creates the encoder + decoder format for TFT
        records = []
        
        n_hist = len(hist.timestamps)
        
        for i in range(n_hist):
            records.append({
                "project_id": static.project_id,
                "time_idx": i,
                "quarter": hist.timestamps[i].quarter,
                "year": hist.timestamps[i].year,
                "demand": hist.demand_history[i],
                "gdp": hist.gdp_history[i],
                "inflation": hist.inflation_history[i],
                "interest_rate": hist.interest_rate_history[i],
                "exchange_rate": hist.exchange_rate_history[i],
                "commodity_price": hist.commodity_price_history[i],
                **{k: v[i] for k, v in hist.weather_variables.items()},
                # Static features repeated
                "sector": static.sector.value,
                "country": static.country,
                "region": static.region,
                "capacity": static.capacity,
                "vintage_year": static.vintage_year,
                "concession_years": static.concession_years,
                "gdp_per_capita": static.gdp_per_capita,
                "population_catchment": static.population_catchment,
                "gdp_growth_rate": static.gdp_growth_rate,
                "urbanization_rate": static.urbanization_rate,
                "institutional_quality": static.institutional_quality,
                "latitude": static.latitude,
                "longitude": static.longitude,
            })
        
        return pd.DataFrame(records)
    
    def _get_metric_for_sector(self, sector: SectorType) -> DemandMetric:
        """Map sector to demand metric"""
        mapping = {
            SectorType.TOLL_ROAD: DemandMetric.ADT,
            SectorType.ELECTRICITY: DemandMetric.DISPATCH_MWH,
            SectorType.PORT: DemandMetric.TEU,
            SectorType.AIRPORT: DemandMetric.PASSENGERS,
            SectorType.RAIL: DemandMetric.TONNAGE,
            SectorType.WATER: DemandMetric.VOLUME_M3,
        }
        return mapping.get(sector, DemandMetric.ADT)
    
    def batch_forecast(
        self,
        projects: List[SectorDemandInput],
        horizon_quarters: int = 40,
    ) -> List[DemandForecast]:
        """Generate forecasts for multiple projects"""
        forecasts = []
        for project_input in projects:
            try:
                forecast = self.forecast(project_input, horizon_quarters)
                forecasts.append(forecast)
            except Exception as e:
                logger.error("Forecast failed", project_id=project_input.static_covariates.project_id, error=str(e))
        return forecasts