"""
Systemic Dependency Mapping (GNN) Schemas
Graph Neural Networks for infrastructure dependency and contagion analysis
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class EdgeType(str, Enum):
    PHYSICAL = "physical"
    SUPPLY_CHAIN = "supply_chain"
    FINANCIAL = "financial"
    CONTRACTUAL = "contractual"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"


class NodeType(str, Enum):
    PROJECT = "project"
    COMPANY = "company"
    GOVERNMENT = "government"
    FINANCIAL_INSTITUTION = "financial_institution"
    SUPPLIER = "supplier"
    OFFTAKER = "offtaker"
    REGULATOR = "regulator"
    INFRASTRUCTURE_ASSET = "infrastructure_asset"


class GraphNode(BaseModel):
    """Node in the dependency graph"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    node_id: str
    node_type: NodeType
    name: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    # Financial attributes
    revenue: Optional[float] = None
    debt: Optional[float] = None
    equity: Optional[float] = None
    credit_rating: Optional[str] = None
    # Operational attributes
    capacity: Optional[float] = None
    utilization: Optional[float] = None
    location: Optional[Dict[str, float]] = None  # lat, lon
    # Embeddings
    embedding: Optional[np.ndarray] = None


class GraphEdge(BaseModel):
    """Directed edge in the dependency graph"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0, description="Dependency strength")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    # Financial exposure
    exposure_amount: Optional[float] = None
    # Contractual terms
    contract_duration: Optional[int] = None  # years
    termination_provisions: Optional[str] = None
    # Physical parameters
    capacity_share: Optional[float] = None
    redundancy: Optional[float] = None


class DependencyGraph(BaseModel):
    """Complete dependency graph for a portfolio"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    graph_id: str
    name: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Computed properties
    adjacency_matrix: Optional[np.ndarray] = None
    node_features: Optional[np.ndarray] = None
    edge_features: Optional[np.ndarray] = None


class ContagionIndex(BaseModel):
    """Project-level contagion index from GNN"""
    project_id: str
    contagion_score: float = Field(ge=0, le=1, description="Systemic contagion risk")
    centrality_measures: Dict[str, float] = Field(default_factory=dict)
    # Specific risk channels
    physical_contagion: float = Field(ge=0, le=1)
    financial_contagion: float = Field(ge=0, le=1)
    supply_chain_contagion: float = Field(ge=0, le=1)
    contractual_contagion: float = Field(ge=0, le=1)
    # Portfolio context
    portfolio_concentration: float = Field(ge=0, le=1)
    systemic_importance: float = Field(ge=0, le=1)
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "gnn-contagion-v1.0"


class PortfolioRiskMetrics(BaseModel):
    """Portfolio-level concentration and systemic risk"""
    portfolio_id: str
    total_exposure: float
    hhi: float = Field(description="Herfindahl-Hirschman Index")
    sector_concentration: Dict[str, float]
    geographic_concentration: Dict[str, float]
    counterparty_concentration: Dict[str, float]
    avg_contagion: float
    max_contagion: float
    systemic_risk_score: float = Field(ge=0, le=1)
    var_95: float = Field(description="Value at Risk 95%")
    var_99: float = Field(description="Value at Risk 99%")
    expected_shortfall: float
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class GNNConfig(BaseModel):
    """GNN model configuration"""
    model_type: str = "GraphSAGE"  # GraphSAGE, GAT, GCN, GIN
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    aggregator: str = "mean"  # mean, max, lstm
    num_heads: int = 4  # for GAT
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20