"""
Geospatial Engine - Main orchestrator for satellite imagery processing
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.infra_risk.schemas.geospatial import (
    SatelliteImage,
    SpectralIndices,
    SiteProgress,
    ProgressAnomaly,
    ConstructionPhase,
    AnomalyType,
)
from src.infra_risk.utils.logging import get_logger
from src.infra_risk.geospatial.siamese_resnet import SiameseResNet50
from src.infra_risk.geospatial.spectral_indices import SpectralIndexCalculator
from src.infra_risk.geospatial.anomaly_detector import AnomalyDetector

logger = get_logger(__name__)


class GeospatialEngine:
    """Main geospatial processing engine"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gee_project_id: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.model_path = model_path
        self.gee_project_id = gee_project_id
        
        # Initialize components
        self.siamese_model = SiameseResNet50(pretrained=True).to(self.device)
        self.spectral_calculator = SpectralIndexCalculator()
        self.anomaly_detector = AnomalyDetector()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.siamese_model.eval()
        logger.info("GeospatialEngine initialized", device=str(self.device))
    
    def load_model(self, path: str) -> None:
        """Load trained model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.siamese_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model loaded", path=path, epoch=checkpoint.get("epoch"))
    
    def save_model(self, path: str, epoch: int, optimizer_state: dict, loss: float) -> None:
        """Save model checkpoint"""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.siamese_model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "loss": loss,
        }, path)
        logger.info("Model saved", path=path)
    
    def fetch_sentinel2_imagery(
        self,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 20,
    ) -> List[SatelliteImage]:
        """Fetch Sentinel-2 imagery from Google Earth Engine"""
        logger.info("Fetching Sentinel-2 imagery", geometry=geometry, start=start_date, end=end_date)
        
        # TODO: Implement GEE integration
        # This would use earthengine-api to query and download imagery
        # For now, return placeholder
        return []
    
    def calculate_spectral_indices(self, image: SatelliteImage) -> SpectralIndices:
        """Calculate spectral indices from satellite image"""
        return self.spectral_calculator.compute_all(image)
    
    def estimate_progress(
        self,
        reference_image: SatelliteImage,
        current_image: SatelliteImage,
        project_id: str,
    ) -> SiteProgress:
        """Estimate construction progress using Siamese ResNet-50"""
        logger.info("Estimating progress", project_id=project_id)
        
        # Calculate spectral indices
        ref_indices = self.calculate_spectral_indices(reference_image)
        curr_indices = self.calculate_spectral_indices(current_image)
        
        # Prepare input tensors
        ref_tensor = self._prepare_tensor(ref_indices)
        curr_tensor = self._prepare_tensor(curr_indices)
        
        # Run inference
        with torch.no_grad():
            progress_output = self.siamese_model(ref_tensor, curr_tensor)
        
        # Parse output
        progress_pct = float(progress_output["progress"].cpu().numpy()[0])
        phase_logits = progress_output["phase"].cpu().numpy()[0]
        phase_idx = int(np.argmax(phase_logits))
        phase = list(ConstructionPhase)[phase_idx]
        
        # Confidence interval (using Monte Carlo dropout or ensemble)
        ci_lower, ci_upper = self._estimate_confidence(ref_tensor, curr_tensor)
        
        # Calculate MAPE if ground truth available
        mape = 0.0  # Placeholder
        
        return SiteProgress(
            project_id=project_id,
            timestamp=datetime.utcnow(),
            phase=phase,
            progress_pct=progress_pct,
            confidence_interval=(ci_lower, ci_upper),
            mape=mape,
            spectral_indices=curr_indices,
            reference_image_id=reference_image.image_id,
            current_image_id=current_image.image_id,
        )
    
    def detect_anomalies(
        self,
        progress_history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect anomalies in construction progress"""
        return self.anomaly_detector.detect(progress_history, project_id)
    
    def _prepare_tensor(self, indices: SpectralIndices) -> torch.Tensor:
        """Prepare spectral indices as model input tensor"""
        # Stack indices as channels: [NDVI, NDBI, NDWI, MNDWI, BSI, NBR]
        channels = []
        for idx_name in ["ndvi", "ndbi", "ndwi", "mndwi", "bsi", "nbr"]:
            idx_data = getattr(indices, idx_name)
            if idx_data is not None:
                channels.append(idx_data)
            else:
                # Create zero channel if index not available
                channels.append(np.zeros_like(indices.ndvi))
        
        stacked = np.stack(channels, axis=0)  # [C, H, W]
        tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(self.device)  # [1, C, H, W]
        return tensor
    
    def _estimate_confidence(
        self,
        ref_tensor: torch.Tensor,
        curr_tensor: torch.Tensor,
        n_samples: int = 50,
    ) -> Tuple[float, float]:
        """Estimate confidence interval using Monte Carlo dropout"""
        self.siamese_model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.siamese_model(ref_tensor, curr_tensor)
                predictions.append(float(output["progress"].cpu().numpy()[0]))
        
        self.siamese_model.eval()
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, 2.5)
        upper = np.percentile(predictions, 97.5)
        
        return float(lower), float(upper)
    
    def process_project_site(
        self,
        project_id: str,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Complete site processing pipeline"""
        logger.info("Processing project site", project_id=project_id)
        
        # Fetch imagery
        images = self.fetch_sentinel2_imagery(geometry, start_date, end_date)
        
        if len(images) < 2:
            logger.warning("Insufficient imagery", project_id=project_id, count=len(images))
            return {"error": "Insufficient imagery"}
        
        # Sort by date
        images.sort(key=lambda x: x.acquisition_date)
        
        # Process pairs
        progress_history = []
        for i in range(1, len(images)):
            progress = self.estimate_progress(images[i-1], images[i], project_id)
            progress_history.append(progress)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(progress_history, project_id)
        
        return {
            "project_id": project_id,
            "progress_history": progress_history,
            "anomalies": anomalies,
            "latest_progress": progress_history[-1] if progress_history else None,
        }