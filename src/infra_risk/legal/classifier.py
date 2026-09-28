"""
Legal-BERT Classifier for Clause Classification
Classifies contract clauses into 12 risk categories
"""
import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.infra_risk.schemas.legal import RiskCategory, ClauseType

from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class LegalBERTClassifier(nn.Module):
    """Fine-tuned Legal-BERT for clause risk classification"""
    
    RISK_CATEGORIES = [c.value for c in RiskCategory]
    CLAUSE_TYPES = [c.value for c in ClauseType]
    
    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        num_risk_categories: int = 12,
        num_clause_types: int = 15,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.max_length = max_length
        self.num_risk_categories = num_risk_categories
        self.num_clause_types = num_clause_types
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Base model
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_risk_categories,
            problem_type="multi_label_classification",
        ).to(self.device)
        
        # Additional heads
        hidden_size = self.bert.config.hidden_size
        
        self.clause_type_head = nn.Linear(hidden_size, num_clause_types)
        self.risk_score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        self.severity_head = nn.Linear(hidden_size, 4)  # Critical, High, Medium, Low
        
        logger.info("LegalBERTClassifier initialized", model=model_name, device=str(device))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get [CLS] token representation
        cls_hidden = outputs.hidden_states[-1][:, 0, :]
        
        # Risk category logits
        risk_logits = outputs.logits
        
        # Clause type
        clause_logits = self.clause_type_head(cls_hidden)
        
        # Risk score (0-100)
        risk_score = self.risk_score_head(cls_hidden) * 100
        
        # Severity
        severity_logits = self.severity_head(cls_hidden)
        
        return {
            "risk_logits": risk_logits,
            "clause_logits": clause_logits,
            "risk_score": risk_score,
            "severity_logits": severity_logits,
            "hidden_states": cls_hidden,
        }
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify a single clause text"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(encoding["input_ids"], encoding["attention_mask"])
        
        # Risk categories (multi-label)
        risk_probs = torch.sigmoid(outputs["risk_logits"]).cpu().numpy()[0]
        risk_categories = [
            self.RISK_CATEGORIES[i] 
            for i, p in enumerate(risk_probs) if p > 0.5
        ]
        
        # Clause type (single label)
        clause_probs = torch.softmax(outputs["clause_logits"], dim=-1).cpu().numpy()[0]
        clause_type_idx = clause_probs.argmax()
        clause_type = self.CLAUSE_TYPES[clause_type_idx]
        clause_confidence = float(clause_probs[clause_type_idx])
        
        # Risk score
        risk_score = float(outputs["risk_score"].cpu().numpy()[0][0])
        
        # Severity
        severity_probs = torch.softmax(outputs["severity_logits"], dim=-1).cpu().numpy()[0]
        severity_labels = ["Critical", "High", "Medium", "Low"]
        severity = severity_labels[severity_probs.argmax()]
        
        return {
            "risk_categories": risk_categories,
            "risk_probabilities": {self.RISK_CATEGORIES[i]: float(p) for i, p in enumerate(risk_probs)},
            "clause_type": clause_type,
            "clause_confidence": clause_confidence,
            "risk_score": risk_score,
            "severity": severity,
            "severity_probs": {severity_labels[i]: float(p) for i, p in enumerate(severity_probs)},
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple clauses"""
        results = []
        for text in texts:
            results.append(self.classify_text(text))
        return results


class LegalBERTTrainer:
    """Trainer for fine-tuning Legal-BERT"""
    
    def __init__(
        self,
        model: LegalBERTClassifier,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion_risk = nn.BCEWithLogitsLoss()
        self.criterion_clause = nn.CrossEntropyLoss()
        self.criterion_score = nn.MSELoss()
        self.criterion_severity = nn.CrossEntropyLoss()
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        
        # Risk category loss (multi-label)
        loss_risk = self.criterion_risk(outputs["risk_logits"], batch["risk_labels"])
        
        # Clause type loss
        loss_clause = self.criterion_clause(outputs["clause_logits"], batch["clause_labels"])
        
        # Risk score loss
        loss_score = self.criterion_score(outputs["risk_score"].squeeze(), batch["risk_scores"])
        
        # Severity loss
        loss_severity = self.criterion_severity(outputs["severity_logits"], batch["severity_labels"])
        
        # Combined loss
        total_loss = loss_risk + loss_clause + 0.1 * loss_score + 0.5 * loss_severity
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "loss_risk": loss_risk.item(),
            "loss_clause": loss_clause.item(),
            "loss_score": loss_score.item(),
            "loss_severity": loss_severity.item(),
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(batch["input_ids"], batch["attention_mask"])
            
            loss_risk = self.criterion_risk(outputs["risk_logits"], batch["risk_labels"])
            loss_clause = self.criterion_clause(outputs["clause_logits"], batch["clause_labels"])
            loss_score = self.criterion_score(outputs["risk_score"].squeeze(), batch["risk_scores"])
            loss_severity = self.criterion_severity(outputs["severity_logits"], batch["severity_labels"])
            
            total_loss = loss_risk + loss_clause + 0.1 * loss_score + 0.5 * loss_severity
            
            # Accuracy metrics
            risk_preds = (torch.sigmoid(outputs["risk_logits"]) > 0.5).float()
            risk_acc = (risk_preds == batch["risk_labels"]).float().mean().item()
            
            clause_preds = outputs["clause_logits"].argmax(dim=-1)
            clause_acc = (clause_preds == batch["clause_labels"]).float().mean().item()
            
            return {
                "val_loss": total_loss.item(),
                "val_risk_acc": risk_acc,
                "val_clause_acc": clause_acc,
            }