"""
InfraRisk AI - Core Data Schemas
Multi-modal data models for infrastructure risk assessment
"""
from .geospatial import *
from .demand import *
from .gnn import *
from .pinn import *
from .legal import *
from .credit import *
from .waterfall import *
from .simulation import *
from .project import *

__all__ = [
    # Geospatial
    "SatelliteImage",
    "SpectralIndices",
    "SiteProgress",
    "ProgressAnomaly",
    # Demand
    "DemandForecast",
    "QuantileForecast",
    "SectorDemandInput",
    # GNN
    "DependencyGraph",
    "GraphNode",
    "GraphEdge",
    "ContagionIndex",
    # PINN
    "DegradationModel",
    "CARULOutput",
    "PhysicsParameters",
    # Legal
    "ContractDocument",
    "ClauseClassification",
    "ContractRiskScore",
    # Credit
    "CreditFeatures",
    "PDPrediction",
    "LGDPrediction",
    "ELPrediction",
    "CreditScore",
    # Waterfall
    "CashFlowWaterfall",
    "WaterfallTier",
    "CoverageRatios",
    # Simulation
    "SimulationState",
    "GameMode",
    "AIAction",
    "SimulationResult",
    "SimulationPhase",
    "ScoringWeights",
    "ScoreBreakdown",
    "DecisionType",
    "Decision",
    "StochasticEvent",
    "EventType",
    "EventSeverity",
    # Project
    "Project",
    "ProjectSector",
    "ProjectStage",
    "MacroProfile",
]