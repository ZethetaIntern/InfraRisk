"""
Siamese ResNet-50 for Construction Progress Estimation
Processes Sentinel-2 spectral indices to estimate site progress
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional
import torch.nn.functional as F


class SiameseResNet50(nn.Module):
    """
    Siamese ResNet-50 architecture for construction progress estimation.
    Takes two sets of spectral indices (reference and current) and outputs
    progress percentage and construction phase classification.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        in_channels: int = 6,  # NDVI, NDBI, NDWI, MNDWI, BSI, NBR
        progress_dim: int = 1,
        phase_classes: int = 8,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        # Load ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer for spectral indices input
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Remove final FC layer
        backbone_out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Siamese branch - shared backbone
        # Progress estimation head
        self.progress_head = nn.Sequential(
            nn.Linear(backbone_out_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, progress_dim),
            nn.Sigmoid(),  # Output 0-1, multiply by 100 for percentage
        )
        
        # Phase classification head
        self.phase_head = nn.Sequential(
            nn.Linear(backbone_out_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, phase_classes),
        )
        
        # Difference embedding for anomaly detection
        self.diff_encoder = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights for new layers"""
        for m in [self.progress_head, self.phase_head, self.diff_encoder]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone for single input"""
        return self.backbone(x)
    
    def forward(
        self,
        reference: torch.Tensor,
        current: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Siamese network.
        
        Args:
            reference: Reference spectral indices [B, C, H, W]
            current: Current spectral indices [B, C, H, W]
            
        Returns:
            Dictionary with progress, phase logits, and embeddings
        """
        # Extract features from both branches
        ref_features = self.forward_once(reference)  # [B, 2048]
        curr_features = self.forward_once(current)   # [B, 2048]
        
        # Concatenate features
        combined = torch.cat([ref_features, curr_features], dim=1)  # [B, 4096]
        
        # Progress estimation
        progress = self.progress_head(combined) * 100  # Scale to percentage
        
        # Phase classification
        phase_logits = self.phase_head(combined)
        
        # Difference embedding for anomaly detection
        diff_features = torch.abs(ref_features - curr_features)
        diff_embedding = self.diff_encoder(diff_features)
        
        return {
            "progress": progress,
            "phase": phase_logits,
            "ref_features": ref_features,
            "curr_features": curr_features,
            "diff_embedding": diff_embedding,
        }
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone embedding for single input"""
        return self.forward_once(x)


class SiameseResNet50Lightning(nn.Module):
    """Lightning-compatible wrapper for training"""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = SiameseResNet50(*args, **kwargs)
        self.criterion_progress = nn.MSELoss()
        self.criterion_phase = nn.CrossEntropyLoss()
        
    def forward(self, reference: torch.Tensor, current: torch.Tensor) -> Dict:
        return self.model(reference, current)
    
    def training_step(self, batch, batch_idx):
        reference, current, progress_target, phase_target = batch
        output = self(reference, current)
        
        loss_progress = self.criterion_progress(output["progress"], progress_target)
        loss_phase = self.criterion_phase(output["phase"], phase_target)
        
        loss = loss_progress + 0.5 * loss_phase
        
        return {
            "loss": loss,
            "loss_progress": loss_progress,
            "loss_phase": loss_phase,
        }
    
    def validation_step(self, batch, batch_idx):
        reference, current, progress_target, phase_target = batch
        output = self(reference, current)
        
        loss_progress = self.criterion_progress(output["progress"], progress_target)
        loss_phase = self.criterion_phase(output["phase"], phase_target)
        
        # Calculate MAPE
        mape = torch.mean(
            torch.abs((progress_target - output["progress"]) / progress_target.clamp(min=1))
        ) * 100
        
        return {
            "val_loss": loss_progress + 0.5 * loss_phase,
            "val_mape": mape,
            "val_loss_progress": loss_progress,
            "val_loss_phase": loss_phase,
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mape",
        }