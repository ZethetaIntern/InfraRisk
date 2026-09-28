"""
GNN Models for Dependency Mapping
GraphSAGE, GAT, and custom Dependency GNN architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Dict, Any
import numpy as np


class GraphSAGEModel(nn.Module):
    """GraphSAGE for dependency graph embedding"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        aggregator: str = "mean",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))
        
        self.projection = nn.Linear(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer without activation
        x = self.convs[-1](x, edge_index)
        x = self.projection(x)
        
        return x
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings"""
        return self.forward(data.x, data.edge_index)


class GATModel(nn.Module):
    """Graph Attention Network for dependency graph"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout))
        
        self.projection = nn.Linear(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        x = self.projection(x)
        
        return x


class DependencyGNN(nn.Module):
    """
    Custom GNN for infrastructure dependency mapping.
    Combines node embeddings with edge-type specific message passing.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 3,
        num_edge_types: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_edge_types = num_edge_types
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Edge-type specific message passing
        self.edge_convs = nn.ModuleList()
        for _ in range(num_edge_types):
            self.edge_convs.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        
        # Edge type embeddings
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        
        # Contagion prediction head
        self.contagion_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [N, node_feature_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_feature_dim]
            edge_type: Edge type indices [E]
            
        Returns:
            Dictionary with embeddings and contagion scores
        """
        # Encode nodes
        h = self.node_encoder(x)  # [N, hidden_dim]
        
        # Message passing layers
        for layer_idx in range(self.num_layers):
            h_new = h.clone()
            
            # Process each edge type separately
            for et in range(self.num_edge_types):
                # Find edges of this type
                mask = edge_type == et
                if mask.sum() == 0:
                    continue
                
                et_edge_index = edge_index[:, mask]
                et_edge_attr = edge_attr[mask]
                
                # Source and target node features
                src, dst = et_edge_index
                src_feat = h[src]
                dst_feat = h[dst]
                
                # Edge type embedding
                et_emb = self.edge_type_emb(torch.tensor([et], device=x.device)).squeeze(0)
                
                # Compute messages
                msg_input = torch.cat([src_feat, dst_feat, et_edge_attr], dim=-1)
                messages = self.edge_convs[et](msg_input)
                
                # Aggregate messages (mean aggregation)
                h_new[dst] = h_new[dst] + messages.mean(dim=0)
            
            # Normalize and activate
            h = self.layer_norms[layer_idx](h_new)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output embeddings
        embeddings = self.output_proj(h)
        
        # Contagion scores
        contagion = self.contagion_head(embeddings).squeeze(-1)
        
        return {
            "embeddings": embeddings,
            "contagion": contagion,
            "node_features": h,
        }
    
    def compute_contagion_index(
        self,
        data: Data,
        project_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contagion index for project nodes"""
        output = self.forward(data.x, data.edge_index, data.edge_attr, data.edge_type)
        return output["contagion"][project_mask]


class GNNLightning(nn.Module):
    """Lightning wrapper for GNN training"""
    
    def __init__(self, model: DependencyGNN, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCELoss()
    
    def forward(self, data: Data) -> Dict:
        return self.model(
            data.x, data.edge_index, data.edge_attr, data.edge_type
        )
    
    def training_step(self, batch, batch_idx):
        output = self(batch)
        
        # Contagion prediction loss
        if hasattr(batch, "contagion_labels"):
            loss = self.criterion(output["contagion"], batch.contagion_labels)
        else:
            # Self-supervised: predict edge existence
            loss = self._link_prediction_loss(batch, output["embeddings"])
        
        self.log("train_loss", loss)
        return loss
    
    def _link_prediction_loss(self, data: Data, embeddings: torch.Tensor) -> torch.Tensor:
        """Self-supervised link prediction loss"""
        # Positive edges
        pos_edge_index = data.edge_index
        pos_src, pos_dst = pos_edge_index
        pos_score = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=-1).sigmoid()
        
        # Negative edges (random sampling)
        num_nodes = embeddings.size(0)
        neg_src = torch.randint(0, num_nodes, (pos_edge_index.size(1),), device=embeddings.device)
        neg_dst = torch.randint(0, num_nodes, (pos_edge_index.size(1),), device=embeddings.device)
        neg_score = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=-1).sigmoid()
        
        # Binary cross entropy
        pos_loss = -torch.log(pos_score + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_score + 1e-8).mean()
        
        return pos_loss + neg_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def create_pyg_data(
    nodes: List[Dict],
    edges: List[Dict],
    node_features: List[str],
    edge_features: List[str],
) -> Data:
    """Create PyTorch Geometric Data object from node/edge lists"""
    
    # Map node IDs to indices
    node_id_to_idx = {node["node_id"]: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # Node features
    x = torch.zeros(num_nodes, len(node_features))
    for i, node in enumerate(nodes):
        for j, feat in enumerate(node_features):
            x[i, j] = node.get(feat, 0)
    
    # Edge index and features
    edge_list = []
    edge_attr_list = []
    edge_type_list = []
    
    edge_type_map = {
        "physical": 0,
        "supply_chain": 1,
        "financial": 2,
        "contractual": 3,
        "regulatory": 4,
        "operational": 5,
    }
    
    for edge in edges:
        src = node_id_to_idx[edge["source_id"]]
        dst = node_id_to_idx[edge["target_id"]]
        edge_list.append([src, dst])
        
        # Edge features
        feat_vec = [edge.get(f, 0) for f in edge_features]
        edge_attr_list.append(feat_vec)
        
        # Edge type
        edge_type_list.append(edge_type_map.get(edge.get("edge_type", "physical"), 0))
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)