"""
Analytical Degradation Models
AASHTO pavement, Paris' Law fatigue, concrete carbonation, atmospheric corrosion
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AASHTOParams:
    """AASHTO pavement deterioration parameters"""
    psi_initial: float = 4.2
    psi_terminal: float = 2.5
    a: float = 0.01  # deterioration rate coefficient
    b: float = 0.5   # traffic exponent
    c: float = 1.2   # PSI exponent
    traffic_growth: float = 0.03  # annual traffic growth rate


class AASHTOPavement:
    """AASHTO Pavement Deterioration Model"""
    
    def __init__(self, params: AASHTOParams = None):
        self.params = params or AASHTOParams()
    
    def psi_at_time(self, t: np.ndarray, initial_traffic: float) -> np.ndarray:
        """
        Compute PSI at time t using analytical solution.
        PSI(t) = psi_terminal + (psi_initial - psi_terminal) * exp(-a * integral(traffic^b) dt)
        """
        # Traffic grows exponentially: traffic(t) = initial_traffic * (1 + g)^t
        traffic_t = initial_traffic * (1 + self.params.traffic_growth) ** t
        
        # Integral of traffic^b: ∫ traffic(t)^b dt
        # ≈ ∑ traffic(t_i)^b * dt for discrete time
        dt = np.diff(t, prepend=0)
        integral = np.cumsum(traffic_t**self.params.b * dt)
        
        psi = self.params.psi_terminal + \
              (self.params.psi_initial - self.params.psi_terminal) * \
              np.exp(-self.params.a * integral)
        
        return np.maximum(psi, self.params.psi_terminal)
    
    def time_to_terminal(self, initial_traffic: float) -> float:
        """Estimate time to reach terminal PSI"""
        # Solve numerically
        t = 0
        traffic = initial_traffic
        psi = self.params.psi_initial
        dt = 0.1  # years
        
        while psi > self.params.psi_terminal and t < 100:
            traffic *= (1 + self.params.traffic_growth)
            dpsi = -self.params.a * traffic**self.params.b * (psi - self.params.psi_terminal)**self.params.c * dt
            psi += dpsi
            t += dt
        
        return t
    
    def rul(self, current_psi: float, current_traffic: float, current_age: float) -> float:
        """Remaining Useful Life from current state"""
        if current_psi <= self.params.psi_terminal:
            return 0.0
        
        # Simulate forward
        t = 0
        traffic = current_traffic
        psi = current_psi
        dt = 0.1
        
        while psi > self.params.psi_terminal and t < 50:
            traffic *= (1 + self.params.traffic_growth)
            dpsi = -self.params.a * traffic**self.params.b * (psi - self.params.psi_terminal)**self.params.c * dt
            psi += dpsi
            t += dt
        
        return t


@dataclass
class ParisLawParams:
    """Paris' Law parameters for fatigue crack growth"""
    C: float = 1e-12  # coefficient (m/cycle/(MPa√m)^m)
    m: float = 3.0    # exponent
    delta_k_th: float = 2.0  # threshold stress intensity (MPa√m)
    delta_k_c: float = 100.0  # critical stress intensity (MPa√m)
    initial_crack: float = 0.001  # initial crack length (m)
    geometry_factor: float = 1.12  # geometry correction factor


class ParisLawFatigue:
    """Paris' Law Fatigue Crack Growth Model"""
    
    def __init__(self, params: ParisLawParams = None):
        self.params = params or ParisLawParams()
    
    def crack_length(self, n_cycles: np.ndarray, delta_sigma: float) -> np.ndarray:
        """
        Compute crack length after n cycles.
        da/dN = C * (ΔK)^m
        ΔK = Y * Δσ * sqrt(π * a)
        """
        a = self.params.initial_crack
        crack_history = [a]
        
        for i in range(1, len(n_cycles)):
            dn = n_cycles[i] - n_cycles[i-1]
            
            # Stress intensity factor range
            delta_k = self.params.geometry_factor * delta_sigma * np.sqrt(np.pi * a)
            
            if delta_k > self.params.delta_k_th:
                # Paris' law
                da_dn = self.params.C * (delta_k**self.params.m)
                a += da_dn * dn
            
            # Check for fracture
            if delta_k >= self.params.delta_k_c:
                a = self.params.delta_k_c / (self.params.geometry_factor * delta_sigma * np.sqrt(np.pi))
                break
            
            crack_history.append(a)
        
        return np.array(crack_history)
    
    def cycles_to_failure(self, delta_sigma: float) -> float:
        """Estimate cycles to failure"""
        # Analytical integration for constant Δσ
        a0 = self.params.initial_crack
        ac = (self.params.delta_k_c / (self.params.geometry_factor * delta_sigma * np.sqrt(np.pi)))**2
        
        if self.params.m == 2:
            Nf = (1 / (self.params.C * (self.params.geometry_factor * delta_sigma * np.sqrt(np.pi))**2)) * \
                 np.log(ac / a0)
        else:
            Nf = (2 / (self.params.C * (2 - self.params.m) * 
                       (self.params.geometry_factor * delta_sigma * np.sqrt(np.pi))**self.params.m)) * \
                 (ac**((2-self.params.m)/2) - a0**((2-self.params.m)/2))
        
        return max(0, Nf)
    
    def rul(self, current_crack: float, delta_sigma: float, cycles_elapsed: float) -> float:
        """Remaining cycles to failure"""
        a0 = current_crack
        ac = (self.params.delta_k_c / (self.params.geometry_factor * delta_sigma * np.sqrt(np.pi)))**2
        
        if a0 >= ac:
            return 0.0
        
        if self.params.m == 2:
            Nf = (1 / (self.params.C * (self.params.geometry_factor * delta_sigma * np.sqrt(np.pi))**2)) * \
                 np.log(ac / a0)
        else:
            Nf = (2 / (self.params.C * (2 - self.params.m) * 
                       (self.params.geometry_factor * delta_sigma * np.sqrt(np.pi))**self.params.m)) * \
                 (ac**((2-self.params.m)/2) - a0**((2-self.params.m)/2))
        
        return max(0, Nf - cycles_elapsed)


@dataclass
class CarbonationParams:
    """Concrete carbonation parameters"""
    K: float = 5.0  # carbonation coefficient (mm/√year)
    co2_concentration: float = 0.04  # CO2 concentration (400 ppm = 0.04%)
    humidity_factor: float = 1.0  # humidity correction
    temperature_factor: float = 1.0  # temperature correction
    cover_depth: float = 50.0  # concrete cover (mm)


class ConcreteCarbonation:
    """Concrete Carbonation Model"""
    
    def __init__(self, params: CarbonationParams = None):
        self.params = params or CarbonationParams()
    
    def carbonation_depth(self, t: np.ndarray) -> np.ndarray:
        """
        Carbonation depth: x(t) = K * sqrt(t * CO2_concentration)
        Enhanced with humidity and temperature factors
        """
        effective_time = t * self.params.co2_concentration * \
                        self.params.humidity_factor * self.params.temperature_factor
        return self.params.K * np.sqrt(effective_time)
    
    def time_to_depassivation(self) -> float:
        """Time for carbonation front to reach reinforcement"""
        if self.params.K <= 0:
            return np.inf
        t = (self.params.cover_depth / (self.params.K * 
                np.sqrt(self.params.co2_concentration * 
                       self.params.humidity_factor * 
                       self.params.temperature_factor)))**2
        return t
    
    def rul(self, current_depth: float) -> float:
        """Remaining time until depassivation"""
        if current_depth >= self.params.cover_depth:
            return 0.0
        
        remaining_depth = self.params.cover_depth - current_depth
        t_remaining = (remaining_depth / (self.params.K * 
                        np.sqrt(self.params.co2_concentration * 
                               self.params.humidity_factor * 
                               self.params.temperature_factor)))**2
        return t_remaining


@dataclass
class CorrosionParams:
    """Atmospheric corrosion parameters"""
    A: float = 1e-6  # pre-exponential factor
    Ea: float = 50000  # activation energy (J/mol)
    b: float = 1.5  # humidity exponent
    c: float = 0.8  # chloride exponent
    R: float = 8.314  # gas constant


class AtmosphericCorrosion:
    """Atmospheric Corrosion Model"""
    
    def __init__(self, params: CorrosionParams = None):
        self.params = params or CorrosionParams()
    
    def corrosion_rate(
        self,
        temperature_c: float,
        humidity: float,  # 0-1
        chloride: float,  # kg/m³ or ppm
    ) -> float:
        """
        Corrosion rate using Arrhenius + humidity + chloride model
        rate = A * exp(-Ea/RT) * humidity^b * chloride^c (mm/year)
        """
        T_K = temperature_c + 273.15
        rate = self.params.A * np.exp(-self.params.Ea / (self.params.R * T_K)) * \
               humidity**self.params.b * chloride**self.params.c
        return rate
    
    def section_loss(self, t: np.ndarray, temperature_c: float, humidity: float, chloride: float) -> np.ndarray:
        """Cumulative section loss over time"""
        rate = self.corrosion_rate(temperature_c, humidity, chloride)
        return rate * t
    
    def rul(self, current_loss: float, allowable_loss: float, temperature_c: float, humidity: float, chloride: float) -> float:
        """Remaining life until allowable section loss"""
        rate = self.corrosion_rate(temperature_c, humidity, chloride)
        if rate <= 0:
            return np.inf
        remaining = allowable_loss - current_loss
        return max(0, remaining / rate)


class CombinedDegradationModel:
    """Combines multiple degradation mechanisms"""
    
    def __init__(self):
        self.models = {}
    
    def add_model(self, name: str, model):
        self.models[name] = model
    
    def predict_all(self, t: np.ndarray, conditions: Dict) -> Dict[str, np.ndarray]:
        """Predict all degradation states"""
        results = {}
        for name, model in self.models.items():
            if hasattr(model, 'psi_at_time'):
                results[name] = model.psi_at_time(t, conditions.get('traffic', 1e6))
            elif hasattr(model, 'crack_length'):
                results[name] = model.crack_length(t, conditions.get('delta_sigma', 100))
            elif hasattr(model, 'carbonation_depth'):
                results[name] = model.carbonation_depth(t)
            elif hasattr(model, 'section_loss'):
                results[name] = model.section_loss(t, 
                    conditions.get('temperature', 20),
                    conditions.get('humidity', 0.7),
                    conditions.get('chloride', 0.1))
        return results
    
    def compute_rul(self, current_state: Dict, conditions: Dict) -> Dict[str, float]:
        """Compute RUL for all mechanisms"""
        rul = {}
        for name, model in self.models.items():
            if hasattr(model, 'rul'):
                if name == 'aashto':
                    rul[name] = model.rul(current_state.get('psi', 4.0), 
                                         conditions.get('traffic', 1e6), 0)
                elif name == 'paris':
                    rul[name] = model.rul(current_state.get('crack', 0.001),
                                         conditions.get('delta_sigma', 100), 0)
                elif name == 'carbonation':
                    rul[name] = model.rul(current_state.get('carbonation', 0))
                elif name == 'corrosion':
                    rul[name] = model.rul(current_state.get('corrosion', 0), 10,
                                         conditions.get('temperature', 20),
                                         conditions.get('humidity', 0.7),
                                         conditions.get('chloride', 0.1))
        return rul