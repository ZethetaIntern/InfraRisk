"""
Contagion Calculator for Systemic Risk
Computes project-level contagion indices from GNN embeddings
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from src.infra_risk.schemas.gnn import ContagionIndex, PortfolioRiskMetrics, DependencyGraph, GraphNode, GraphEdge
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class ContagionCalculator:
    """Calculates contagion indices and portfolio risk metrics"""
    
    def __init__(self, gnn_model=None):
        self.gnn_model = gnn_model
    
    def calculate_contagion(
        self,
        graph: DependencyGraph,
        project_id: str,
    ) -> ContagionIndex:
        """Calculate contagion index for a single project"""
        
        if self.gnn_model is not None:
            return self._calculate_gnn_contagion(graph, project_id)
        else:
            return self._calculate_heuristic_contagion(graph, project_id)
    
    def _calculate_gnn_contagion(
        self,
        graph: DependencyGraph,
        project_id: str,
    ) -> ContagionIndex:
        """Calculate contagion using trained GNN"""
        # Convert graph to PyG format and run inference
        # TODO: Implement GNN inference
        pass
    
    def _calculate_heuristic_contagion(
        self,
        graph: DependencyGraph,
        project_id: str,
    ) -> ContagionIndex:
        """Calculate contagion using network centrality heuristics"""
        
        # Build adjacency matrix
        node_ids = [n.node_id for n in graph.nodes]
        n = len(node_ids)
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        adj = np.zeros((n, n))
        edge_weights = {}
        
        for edge in graph.edges:
            i, j = node_idx[edge.source_id], node_idx[edge.target_id]
            adj[i, j] = edge.weight
            edge_weights[(i, j)] = {
                "weight": edge.weight,
                "type": edge.edge_type.value,
                "exposure": edge.exposure_amount or 0,
            }
        
        # Find project index
        if project_id not in node_idx:
            raise ValueError(f"Project {project_id} not found in graph")
        
        proj_idx = node_idx[project_id]
        
        # Calculate centrality measures
        centrality = self._compute_centrality(adj, proj_idx)
        
        # Calculate type-specific contagion
        type_contagion = self._compute_type_contagion(adj, edge_weights, proj_idx, node_idx)
        
        # Portfolio concentration
        portfolio_concentration = self._compute_portfolio_concentration(graph, project_id)
        
        # Systemic importance (combination of centrality and size)
        project_node = graph.nodes[proj_idx]
        size_factor = min(1.0, (project_node.attributes.get("revenue", 100) / 1e9)) if project_node.attributes else 0.5
        systemic_importance = 0.5 * centrality["eigenvector"] + 0.3 * centrality["betweenness"] + 0.2 * size_factor
        
        # Overall contagion score
        contagion_score = (
            0.3 * type_contagion["physical"] +
            0.25 * type_contagion["financial"] +
            0.2 * type_contagion["supply_chain"] +
            0.15 * type_contagion["contractual"] +
            0.1 * systemic_importance
        )
        
        return ContagionIndex(
            project_id=project_id,
            contagion_score=float(contagion_score),
            centrality_measures={k: float(v) for k, v in centrality.items()},
            physical_contagion=float(type_contagion["physical"]),
            financial_contagion=float(type_contagion["financial"]),
            supply_chain_contagion=float(type_contagion["supply_chain"]),
            contractual_contagion=float(type_contagion["contractual"]),
            portfolio_concentration=float(portfolio_concentration),
            systemic_importance=float(systemic_importance),
        )
    
    def _compute_centrality(self, adj: np.ndarray, node_idx: int) -> Dict[str, float]:
        """Compute various centrality measures"""
        n = adj.shape[0]
        
        # Out-degree centrality
        out_degree = adj[node_idx, :].sum()
        in_degree = adj[:, node_idx].sum()
        
        # Eigenvector centrality (power iteration)
        eig_centrality = self._eigenvector_centrality(adj)[node_idx]
        
        # Betweenness centrality (approximate)
        betweenness = self._betweenness_centrality_approx(adj, node_idx)
        
        # PageRank
        pagerank = self._pagerank(adj)[node_idx]
        
        # Katz centrality
        katz = self._katz_centrality(adj)[node_idx]
        
        return {
            "out_degree": float(out_degree / max(1, n-1)),
            "in_degree": float(in_degree / max(1, n-1)),
            "eigenvector": float(eig_centrality),
            "betweenness": float(betweenness),
            "pagerank": float(pagerank),
            "katz": float(katz),
        }
    
    def _eigenvector_centrality(self, adj: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Compute eigenvector centrality using power iteration"""
        n = adj.shape[0]
        x = np.ones(n) / n
        
        for _ in range(max_iter):
            x_new = adj.T @ x
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x_new = x_new / norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x
    
    def _pagerank(self, adj: np.ndarray, alpha: float = 0.85, max_iter: int = 100) -> np.ndarray:
        """Compute PageRank"""
        n = adj.shape[0]
        
        # Normalize adjacency
        out_degrees = adj.sum(axis=1)
        out_degrees[out_degrees == 0] = 1
        P = adj / out_degrees[:, np.newaxis]
        
        # Personalization vector (uniform)
        v = np.ones(n) / n
        
        x = v.copy()
        for _ in range(max_iter):
            x = alpha * (P.T @ x) + (1 - alpha) * v
        
        return x
    
    def _katz_centrality(self, adj: np.ndarray, alpha: float = 0.1, beta: float = 1.0) -> np.ndarray:
        """Compute Katz centrality"""
        n = adj.shape[0]
        I = np.eye(n)
        try:
            centrality = np.linalg.inv(I - alpha * adj.T) @ (beta * np.ones(n))
        except np.linalg.LinAlgError:
            centrality = np.ones(n)
        return centrality
    
    def _betweenness_centrality_approx(self, adj: np.ndarray, node_idx: int, k: int = 100) -> float:
        """Approximate betweenness centrality using random sampling"""
        n = adj.shape[0]
        if n < 3:
            return 0.0
        
        betweenness = 0.0
        
        # Sample random source-target pairs
        for _ in range(min(k, n * (n-1) // 2)):
            s = np.random.randint(0, n)
            t = np.random.randint(0, n)
            if s == t or s == node_idx or t == node_idx:
                continue
            
            # Check if node_idx lies on shortest path
            # Simplified: check if paths through node_idx are shorter
            try:
                # Direct path
                if adj[s, t] > 0:
                    direct = 1
                else:
                    direct = np.inf
                
                # Path through node_idx
                if adj[s, node_idx] > 0 and adj[node_idx, t] > 0:
                    through = 2
                else:
                    through = np.inf
                
                if through < direct:
                    betweenness += 1
            except:
                pass
        
        return betweenness / max(1, k)
    
    def _compute_type_contagion(
        self,
        adj: np.ndarray,
        edge_weights: Dict,
        proj_idx: int,
        node_idx: Dict,
    ) -> Dict[str, float]:
        """Compute contagion by dependency type"""
        type_scores = {
            "physical": 0.0,
            "financial": 0.0,
            "supply_chain": 0.0,
            "contractual": 0.0,
        }
        
        # Outgoing edges (project -> others)
        for j in range(adj.shape[1]):
            if adj[proj_idx, j] > 0:
                ew = edge_weights.get((proj_idx, j), {})
                etype = ew.get("type", "physical")
                weight = ew.get("weight", 1.0)
                exposure = ew.get("exposure", 0)
                
                if etype in type_scores:
                    # Weight by exposure if available
                    factor = weight * (1 + min(1.0, exposure / 1e8)) if exposure else weight
                    type_scores[etype] += factor
        
        # Incoming edges (others -> project)
        for i in range(adj.shape[0]):
            if adj[i, proj_idx] > 0:
                ew = edge_weights.get((i, proj_idx), {})
                etype = ew.get("type", "physical")
                weight = ew.get("weight", 1.0)
                exposure = ew.get("exposure", 0)
                
                if etype in type_scores:
                    factor = weight * (1 + min(1.0, exposure / 1e8)) if exposure else weight
                    type_scores[etype] += factor
        
        # Normalize
        for k in type_scores:
            type_scores[k] = min(1.0, type_scores[k] / 10.0)
        
        return type_scores
    
    def _compute_portfolio_concentration(self, graph: DependencyGraph, project_id: str) -> float:
        """Compute portfolio concentration (HHI)"""
        # Get all project nodes
        projects = [n for n in graph.nodes if n.node_type.value == "project"]
        
        if len(projects) < 2:
            return 0.0
        
        # Calculate exposure shares
        exposures = []
        for p in projects:
            exp = p.attributes.get("revenue", 0) or p.attributes.get("capex", 100)
            exposures.append(exp)
        
        total = sum(exposures)
        if total == 0:
            return 1.0 / len(projects)
        
        shares = [e / total for e in exposures]
        hhi = sum(s**2 for s in shares)
        
        return hhi
    
    def calculate_portfolio_risk(
        self,
        graph: DependencyGraph,
    ) -> PortfolioRiskMetrics:
        """Calculate portfolio-level risk metrics"""
        
        projects = [n for n in graph.nodes if n.node_type.value == "project"]
        
        if not projects:
            return PortfolioRiskMetrics(
                portfolio_id=graph.graph_id,
                total_exposure=0,
                hhi=0,
                sector_concentration={},
                geographic_concentration={},
                counterparty_concentration={},
                avg_contagion=0,
                max_contagion=0,
                systemic_risk_score=0,
                var_95=0,
                var_99=0,
                expected_shortfall=0,
            )
        
        # Calculate contagion for all projects
        contagion_scores = []
        for p in projects:
            ci = self.calculate_contagion(graph, p.node_id)
            contagion_scores.append(ci.contagion_score)
        
        # Total exposure
        total_exposure = sum(p.attributes.get("revenue", 0) or p.attributes.get("capex", 0) for p in projects)
        
        # Sector concentration
        sector_exp = {}
        for p in projects:
            sector = p.attributes.get("sector", "unknown")
            exp = p.attributes.get("revenue", 0) or p.attributes.get("capex", 0)
            sector_exp[sector] = sector_exp.get(sector, 0) + exp
        
        sector_conc = {k: v/total_exposure for k, v in sector_exp.items()}
        
        # Geographic concentration
        geo_exp = {}
        for p in projects:
            country = p.attributes.get("country", "unknown")
            exp = p.attributes.get("revenue", 0) or p.attributes.get("capex", 0)
            geo_exp[country] = geo_exp.get(country, 0) + exp
        
        geo_conc = {k: v/total_exposure for k, v in geo_exp.items()}
        
        # Counterparty concentration
        cp_exp = {}
        for edge in graph.edges:
            if edge.edge_type.value == "financial" and edge.exposure_amount:
                cp = edge.target_id if edge.source_id in [p.node_id for p in projects] else edge.source_id
                cp_exp[cp] = cp_exp.get(cp, 0) + edge.exposure_amount
        
        cp_conc = {k: v/sum(cp_exp.values()) for k, v in cp_exp.items()} if cp_exp else {}
        
        # Risk metrics
        avg_contagion = np.mean(contagion_scores)
        max_contagion = np.max(contagion_scores)
        systemic_risk = max_contagion * avg_contagion
        
        # VaR approximation (simplified)
        losses = np.array(contagion_scores) * total_exposure / len(projects)
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)
        es = losses[losses >= var_95].mean() if len(losses[losses >= var_95]) > 0 else var_95
        
        return PortfolioRiskMetrics(
            portfolio_id=graph.graph_id,
            total_exposure=total_exposure,
            hhi=self._compute_portfolio_concentration(graph, projects[0].node_id) if projects else 0,
            sector_concentration=sector_conc,
            geographic_concentration=geo_conc,
            counterparty_concentration=cp_conc,
            avg_contagion=float(avg_contagion),
            max_contagion=float(max_contagion),
            systemic_risk_score=float(systemic_risk),
            var_95=float(var_95),
            var_99=float(var_99),
            expected_shortfall=float(es),
        )