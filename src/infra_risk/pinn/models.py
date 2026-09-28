"""
PINN Models for Structural Degradation
Physics-Informed Neural Networks embedding AASHTO, Paris' Law, Carbonation, Corrosion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np


class DegradationPINN(nn.Module):
    """
    Physics-Informed Neural Network for structural degradation.
    Embeds physical equations in the loss function.
    """
    
    def __init__(
        self,
        input_dim: int = 5,  # time, temperature, humidity, stress, etc.
        hidden_layers: List[int] = None,
        output_dim: int = 1,  # degradation state
        activation: str = "tanh",
        physics_weight: float = 1.0,
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        
        # Activation function
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "swish":
            self.act = nn.SiLU()
        else:
            self.act = nn.Tanh()
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.act)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better PINN training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def predict_degradation(
        self,
        time: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict degradation state given time and environmental conditions"""
        # Combine time and conditions
        x = torch.cat([time, conditions], dim=-1)
        return self.forward(x)


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss functions for different degradation mechanisms.
    """
    
    def __init__(self):
        super().__init__()
    
    def aashto_pavement_loss(
        self,
        model: DegradationPINN,
        t: torch.Tensor,
        traffic: torch.Tensor,
        psi_initial: float,
        psi_terminal: float,
        target_psi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        AASHTO pavement deterioration physics loss.
        d(PSI)/dt = -a * (traffic)^b * (PSI - psi_terminal)^c
        """
        # Predict PSI
        x = torch.cat([t, traffic], dim=-1)
        psi_pred = model(x)
        
        # Physics residual: d(PSI)/dt + a * traffic^b * (PSI - psi_terminal)^c = 0
        psi = psi_pred
        dpsi_dt = torch.autograd.grad(
            psi, t, grad_outputs=torch.ones_like(psi),
            create_graph=True, retain_graph=True
        )[0]
        
        # AASHTO parameters (learnable or fixed)
        a = 0.01
        b = 0.5
        c = 1.2
        
        residual = dpsi_dt + a * traffic**b * (psi - psi_terminal)**c
        
        physics_loss = torch.mean(residual**2)
        
        # Data loss if targets available
        data_loss = 0
        if target_psi is not None:
            data_loss = F.mse_loss(psi, target_psi)
        
        return physics_loss + data_loss
    
    def paris_law_loss(
        self,
        model: DegradationPINN,
        n_cycles: torch.Tensor,
        delta_k: torch.Tensor,
        C: float,
        m: float,
        target_crack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Paris' Law for fatigue crack growth.
        da/dN = C * (ΔK)^m
        """
        x = torch.cat([n_cycles, delta_k], dim=-1)
        crack_pred = model(x)
        
        # Physics residual: da/dN - C * (ΔK)^m = 0
        crack = crack_pred
        dcrack_dn = torch.autograd.grad(
            crack, n_cycles, grad_outputs=torch.ones_like(crack),
            create_graph=True, retain_graph=True
        )[0]
        
        residual = dcrack_dn - C * delta_k**m
        
        physics_loss = torch.mean(residual**2)
        
        data_loss = 0
        if target_crack is not None:
            data_loss = F.mse_loss(crack, target_crack)
        
        return physics_loss + data_loss
    
    def carbonation_loss(
        self,
        model: DegradationPINN,
        t: torch.Tensor,
        co2_concentration: torch.Tensor,
        K: float,
        target_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concrete carbonation model.
        x(t) = K * sqrt(t) * sqrt(CO2_concentration)
        """
        x = torch.cat([t, co2_concentration], dim=-1)
        depth_pred = model(x)
        
        # Analytical solution
        depth_analytical = K * torch.sqrt(t) * torch.sqrt(co2_concentration)
        
        # Physics loss: difference from analytical
        physics_loss = torch.mean((depth_pred - depth_analytical)**2)
        
        data_loss = 0
        if target_depth is not None:
            data_loss = F.mse_loss(depth_pred, target_depth)
        
        return physics_loss + data_loss
    
    def corrosion_loss(
        self,
        model: DegradationPINN,
        t: torch.Tensor,
        chloride: torch.Tensor,
        temperature: torch.Tensor,
        humidity: torch.Tensor,
        target_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Atmospheric corrosion model.
        Corrosion rate = A * exp(-Ea/RT) * (humidity)^b * (chloride)^c
        """
        x = torch.cat([t, chloride, temperature, humidity], dim=-1)
        corrosion_pred = model(x)
        
        # Simplified physics: corrosion rate follows Arrhenius + humidity + chloride
        R = 8.314  # J/mol·K
        Ea = 50000  # J/mol
        A = 1e-6
        b = 1.5
        c = 0.8
        
        # Convert temperature to Kelvin
        T_K = temperature + 273.15
        
        rate = A * torch.exp(-Ea / (R * T_K)) * humidity**b * chloride**c
        corrosion_analytical = rate * t
        
        physics_loss = torch.mean((corrosion_pred - corrosion_analytical)**2)
        
        data_loss = 0
        if target_loss is not None:
            data_loss = F.mse_loss(corrosion_pred, target_loss)
        
        return physics_loss + data_loss
    
    def combined_loss(
        self,
        model: DegradationPINN,
        data: Dict[str, torch.Tensor],
        mechanisms: List[str],
        params: Dict[str, Dict],
    ) -> Dict[str, torch.Tensor]:
        """Combine multiple physics losses"""
        losses = {}
        
        if "aashto" in mechanisms:
            losses["aashto"] = self.aashto_pavement_loss(
                model, data["t"], data["traffic"],
                params["aashto"]["psi_initial"],
                params["aashto"]["psi_terminal"],
                data.get("psi_target"),
            )
        
        if "paris" in mechanisms:
            losses["paris"] = self.paris_law_loss(
                model, data["n_cycles"], data["delta_k"],
                params["paris"]["C"], params["paris"]["m"],
                data.get("crack_target"),
            )
        
        if "carbonation" in mechanisms:
            losses["carbonation"] = self.carbonation_loss(
                model, data["t"], data["co2"],
                params["carbonation"]["K"],
                data.get("carbonation_target"),
            )
        
        if "corrosion" in mechanisms:
            losses["corrosion"] = self.corrosion_loss(
                model, data["t"], data["chloride"],
                data["temperature"], data["humidity"],
                data.get("corrosion_target"),
            )
        
        total_loss = sum(losses.values())
        losses["total"] = total_loss
        
        return losses


class MultiMechanismPINN(nn.Module):
    """PINN with multiple output heads for different degradation mechanisms"""
    
    def __init__(
        self,
        input_dim: int,
        mechanisms: List[str],
        shared_layers: List[int] = None,
        head_layers: List[int] = None,
    ):
        super().__init__()
        
        self.mechanisms = mechanisms
        
        if shared_layers is None:
            shared_layers = [128, 128, 128]
        if head_layers is None:
            head_layers = [64, 32]
        
        # Shared trunk
        shared = []
        prev = input_dim
        for h in shared_layers:
            shared.append(nn.Linear(prev, h))
            shared.append(nn.Tanh())
            prev = h
        self.trunk = nn.Sequential(*shared)
        
        # Mechanism-specific heads
        self.heads = nn.ModuleDict()
        for mech in mechanisms:
            head = []
            prev = shared_layers[-1]
            for h in head_layers:
                head.append(nn.Linear(prev, h))
                head.append(nn.Tanh())
                prev = h
            head.append(nn.Linear(prev, 1))
            self.heads[mech] = nn.Sequential(*head)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.trunk(x)
        outputs = {}
        for mech, head in self.heads.items():
            outputs[mech] = head(features)
        return outputs


def create_pinn_model(config: Dict) -> DegradationPINN:
    """Factory function to create PINN from config"""
    return DegradationPINN(
        input_dim=config.get("input_dim", 5),
        hidden_layers=config.get("hidden_layers", [64, 64, 64, 64]),
        output_dim=config.get("output_dim", 1),
        activation=config.get("activation", "tanh"),
        physics_weight=config.get("physics_weight", 1.0),
    )