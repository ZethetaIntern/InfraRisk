"""
GNN Engine - Main orchestrator for dependency mapping and contagion analysis
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import numpy as np

from src.infra_risk.schemas.gnn import DependencyGraph, ContagionIndex, PortfolioRiskMetrics, GNNConfig
from src.infra_risk.schemas.project import Project
from src.infra_risk.utils.logging import get_logger
from src.infra_risk.gnn.models import DependencyGNN, create_pyg_data
from src.infra_risk.gnn.contagion import ContagionCalculator

logger = get_logger(__name__)


class GNNEngine:
    """Main GNN engine for systemic dependency mapping"""
    
    def __init__(
        self,
        config: Optional[GNNConfig] = None,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config or GNNConfig()
        self.device = torch.device(device)
        self.model_path = model_path
        self.model: Optional[DependencyGNN] = None
        self.contagion_calculator = ContagionCalculator()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        logger.info("GNNEngine initialized", device=str(self.device))
    
    def load_model(self, path: str) -> None:
        """Load trained GNN model"""
        checkpoint = torch.load(path, map_location=self.device)
        # Recreate model from config
        # TODO: Implement proper loading
        logger.info("GNN model loaded", path=path)
    
    def save_model(self, path: str, epoch: int, optimizer_state: dict, loss: float) -> None:
        """Save model checkpoint"""
        if self.model:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer_state,
                "loss": loss,
                "config": self.config.dict(),
            }, path)
            logger.info("GNN model saved", path=path)
    
    def build_dependency_graph(
        self,
        projects: List[Project],
        counterparties: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> DependencyGraph:
        """Build dependency graph from project and counterparty data"""
        logger.info("Building dependency graph", n_projects=len(projects))
        
        nodes = []
        edges = []
        
        # Add project nodes
        for proj in projects:
            nodes.append({
                "node_id": proj.project_id,
                "node_type": "project",
                "name": proj.name,
                "attributes": {
                    "revenue": proj.debt_amount * 0.1,  # Placeholder
                    "debt": proj.debt_amount,
                    "equity": proj.equity_amount,
                    "sector": proj.sector.value,
                    "country": proj.country_code,
                    "stage": proj.stage.value,
                    "capacity": proj.capacity,
                }
            })
        
        # Add counterparty nodes
        for cp in counterparties:
            nodes.append({
                "node_id": cp["id"],
                "node_type": cp["type"],  # company, government, financial_institution, etc.
                "name": cp["name"],
                "attributes": cp.get("attributes", {}),
            })
        
        # Add edges from relationships
        for rel in relationships:
            edges.append({
                "edge_id": f"{rel['source']}_{rel['target']}_{rel['type']}",
                "source_id": rel["source"],
                "target_id": rel["target"],
                "edge_type": rel["type"],
                "weight": rel.get("weight", 1.0),
                "exposure_amount": rel.get("exposure"),
                "contract_duration": rel.get("duration"),
                "capacity_share": rel.get("capacity_share"),
            })
        
        return DependencyGraph(
            graph_id="portfolio_dependency_graph",
            name="Portfolio Dependency Graph",
            nodes=[GraphNode(**n) for n in nodes],
            edges=[GraphEdge(**e) for e in edges],
        )
    
    def train(
        self,
        graphs: List[DependencyGraph],
        labels: Optional[Dict[str, float]] = None,
        epochs: int = None,
    ) -> Dict[str, Any]:
        """Train GNN model on dependency graphs"""
        epochs = epochs or self.config.epochs
        logger.info("Training GNN", n_graphs=len(graphs), epochs=epochs)
        
        # Combine graphs into single large graph for training
        # TODO: Implement proper training loop
        
        return {"status": "training_not_implemented"}
    
    def calculate_contagion(
        self,
        graph: DependencyGraph,
        project_id: str,
    ) -> ContagionIndex:
        """Calculate contagion index for a project"""
        return self.contagion_calculator.calculate_contagion(graph, project_id)
    
    def calculate_portfolio_risk(
        self,
        graph: DependencyGraph,
    ) -> PortfolioRiskMetrics:
        """Calculate portfolio-level risk metrics"""
        return self.contagion_calculator.calculate_portfolio_risk(graph)
    
    def get_node_embeddings(self, graph: DependencyGraph) -> Dict[str, np.ndarray]:
        """Get GNN embeddings for all nodes"""
        if self.model is None:
            logger.warning("No model loaded, returning random embeddings")
            return {node.node_id: np.random.randn(64) for node in graph.nodes}
        
        # Convert to PyG and run inference
        # TODO: Implement inference
        return {}
    
    def find_systemic_nodes(
        self,
        graph: DependencyGraph,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find most systemically important nodes"""
        projects = [n for n in graph.nodes if n.node_type.value == "project"]
        
        scores = []
        for p in projects:
            ci = self.calculate_contagion(graph, p.node_id)
            scores.append({
                "project_id": p.node_id,
                "name": p.name,
                "contagion_score": ci.contagion_score,
                "systemic_importance": ci.systemic_importance,
                "centrality": ci.centrality_measures,
            })
        
        scores.sort(key=lambda x: x["systemic_importance"], reverse=True)
        return scores[:top_k]
    
    def simulate_shock(
        self,
        graph: DependencyGraph,
        shock_node: str,
        shock_magnitude: float = 0.5,
    ) -> Dict[str, Any]:
        """Simulate shock propagation through network"""
        logger.info("Simulating shock", node=shock_node, magnitude=shock_magnitude)
        
        # Get contagion scores
        ci = self.calculate_contagion(graph, shock_node)
        
        # Simple linear propagation model
        affected = {}
        for node in graph.nodes:
            if node.node_id == shock_node:
                affected[node.node_id] = shock_magnitude
            else:
                # Propagate through edges
                impact = self._propagate_shock(graph, shock_node, node.node_id, shock_magnitude)
                if impact > 0.01:
                    affected[node.node_id] = impact
        
        return {
            "shock_node": shock_node,
            "magnitude": shock_magnitude,
            "affected_nodes": affected,
            "total_impact": sum(affected.values()),
        }
    
    def _propagate_shock(
        self,
        graph: DependencyGraph,
        source: str,
        target: str,
        magnitude: float,
        max_hops: int = 3,
    ) -> float:
        """Calculate shock propagation from source to target"""
        # Simplified: use shortest path with attenuation
        # TODO: Implement proper network propagation model
        return magnitude * 0.5  # Placeholder