"""
Temporal Fusion Transformer for Demand Forecasting
Multi-horizon probabilistic forecasting with quantile outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer as PytorchTFT
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer wrapper for infrastructure demand forecasting.
    Outputs P10, P50, P90 quantiles for multi-horizon forecasts.
    """
    
    def __init__(
        self,
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_known_categoricals: List[str],
        time_varying_known_reals: List[str],
        time_varying_unknown_categoricals: List[str],
        time_varying_unknown_reals: List[str],
        target: str,
        max_encoder_length: int = 40,
        max_prediction_length: int = 40,
        hidden_size: int = 256,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 64,
        output_size: int = 3,  # P10, P50, P90
        loss: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        log_interval: int = 10,
        reduce_on_plateau_patience: int = 4,
    ):
        super().__init__()
        
        self.target = target
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Use PyTorch Forecasting's TFT implementation
        self.tft = PytorchTFT.from_dataset(
            dataset=None,  # Will be set later
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=output_size,
            loss=loss or QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            learning_rate=learning_rate,
            log_interval=log_interval,
            reduce_on_plateau_patience=reduce_on_plateau_patience,
        )
        
        # Store feature names for interpretability
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass returning quantile predictions"""
        # x contains: encoder_cont, encoder_cat, decoder_cont, decoder_cat, etc.
        output = self.tft(x)
        
        # Output format: [batch_size, prediction_length, n_quantiles]
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output
        
        return {
            "predictions": predictions,
            "attention": getattr(self.tft, "last_attention", None),
        }
    
    def predict(
        self,
        dataloader,
        mode: str = "quantiles",
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, Dict]:
        """Generate predictions with uncertainty quantification"""
        self.eval()
        
        all_predictions = []
        all_attention = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                output = self(x)
                all_predictions.append(output["predictions"].cpu().numpy())
                if output["attention"] is not None:
                    all_attention.append(output["attention"].cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        attention = np.concatenate(all_attention, axis=0) if all_attention else None
        
        # Extract quantiles
        if mode == "quantiles":
            # predictions shape: [batch, pred_len, n_quantiles]
            p10 = predictions[:, :, 0]
            p50 = predictions[:, :, 1]
            p90 = predictions[:, :, 2]
            
            return {
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }, {"attention": attention}
        elif mode == "samples":
            # Sample from quantile distribution
            samples = self._sample_from_quantiles(predictions, n_samples)
            return samples, {"attention": attention}
        
        return predictions, {"attention": attention}
    
    def _sample_from_quantiles(self, quantiles: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample from quantile predictions using linear interpolation"""
        batch_size, pred_len, n_quantiles = quantiles.shape
        
        # Quantile levels
        q_levels = np.array([0.1, 0.5, 0.9])
        
        samples = np.zeros((batch_size, pred_len, n_samples))
        
        for b in range(batch_size):
            for t in range(pred_len):
                q_vals = quantiles[b, t, :]
                # Linear interpolation between quantiles
                u = np.random.uniform(0, 1, n_samples)
                samples[b, t, :] = np.interp(u, q_levels, q_vals)
        
        return samples
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get variable importance from TFT"""
        # Variable selection weights
        importance = {}
        
        if hasattr(self.tft, "variable_selection"):
            # Static variables
            for i, name in enumerate(self.static_categoricals + self.static_reals):
                if hasattr(self.tft.variable_selection, "static_weights"):
                    importance[f"static_{name}"] = float(
                        self.tft.variable_selection.static_weights[0, i].item()
                    )
            
            # Time-varying variables
            for i, name in enumerate(
                self.time_varying_known_categoricals + 
                self.time_varying_known_reals + 
                self.time_varying_unknown_categoricals + 
                self.time_varying_unknown_reals
            ):
                if hasattr(self.tft.variable_selection, "temporal_weights"):
                    importance[f"temporal_{name}"] = float(
                        self.tft.variable_selection.temporal_weights[0, i].mean().item()
                    )
        
        return importance


class TFTLightning(nn.Module):
    """PyTorch Lightning compatible wrapper for training"""
    
    def __init__(self, model: TFTModel):
        super().__init__()
        self.model = model
        self.criterion = model.tft.loss
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output["predictions"], y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output["predictions"], y)
        
        # Calculate additional metrics
        p50_pred = output["predictions"][:, :, 1]  # Median
        mae = F.l1_loss(p50_pred, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        
        return {"val_loss": loss, "val_mae": mae}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.model.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.model.predict(x)


def create_tft_dataset(
    data: pd.DataFrame,
    target: str,
    max_encoder_length: int = 40,
    max_prediction_length: int = 40,
    static_categoricals: List[str] = None,
    static_reals: List[str] = None,
    time_varying_known_categoricals: List[str] = None,
    time_varying_known_reals: List[str] = None,
    time_varying_unknown_categoricals: List[str] = None,
    time_varying_unknown_reals: List[str] = None,
    group_ids: List[str] = None,
    time_idx: str = "time_idx",
) -> TimeSeriesDataSet:
    """Create TimeSeriesDataSet for TFT training"""
    
    dataset = TimeSeriesDataSet(
        data,
        time_idx=time_idx,
        target=target,
        group_ids=group_ids or ["project_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals or [],
        static_reals=static_reals or [],
        time_varying_known_categoricals=time_varying_known_categoricals or [],
        time_varying_known_reals=time_varying_known_reals or [],
        time_varying_unknown_categoricals=time_varying_unknown_categoricals or [],
        time_varying_unknown_reals=time_varying_unknown_reals or [],
        target_normalizer=None,  # Use custom normalizer
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    return dataset