"""
InfraRisk AI - GNN Package
Graph Neural Networks for systemic dependency mapping and contagion analysis
"""
from .engine import GNNEngine
from .models import DependencyGNN, GraphSAGEModel, GATModel
from .contagion import ContagionCalculator

__all__ = [
    "GNNEngine",
    "DependencyGNN",
    "GraphSAGEModel",
    "GATModel",
    "ContagionCalculator",
]