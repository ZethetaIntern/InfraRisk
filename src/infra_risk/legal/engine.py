"""
Legal NLP Engine - Main orchestrator for contract intelligence
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.infra_risk.schemas.legal import (
    ContractDocument,
    ContractRiskScore,
    ClauseClassification,
    RiskCategory,
)
from src.infra_risk.schemas.project import Project
from src.infra_risk.utils.logging import get_logger
from src.infra_risk.legal.parser import ContractParser
from src.infra_risk.legal.classifier import LegalBERTClassifier
from src.infra_risk.legal.extractor import ClauseExtractor, Clause, process_contract_clauses

logger = get_logger(__name__)


class LegalNLPEngine:
    """Main Legal NLP engine for contract intelligence"""
    
    def __init__(
        self,
        parser_model: str = "microsoft/layoutlmv3-base",
        classifier_model: str = "nlpaueb/legal-bert-base-uncased",
        device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu",
    ):
        self.device = device
        
        # Initialize components
        self.parser = ContractParser(model_name=parser_model, device=device)
        self.classifier = LegalBERTClassifier(model_name=classifier_model, device=device)
        self.extractor = ClauseExtractor()
        
        logger.info("LegalNLPEngine initialized", device=device)
    
    def analyze_contract(
        self,
        file_path: str,
        project_id: str,
    ) -> ContractRiskScore:
        """Complete contract analysis pipeline"""
        logger.info("Analyzing contract", file=file_path, project_id=project_id)
        
        # Parse PDF with LayoutLM
        parsed = self.parser.parse_pdf(file_path)
        
        # Create ContractDocument
        doc = ContractDocument(
            doc_id=f"{project_id}_{Path(file_path).stem}",
            project_id=project_id,
            file_path=file_path,
            pages=parsed["num_pages"],
            text_blocks=[{"page": p["page_num"], "text": p["text"]} for p in parsed["pages"]],
            tables=[{"page": p["page_num"], "tables": p["tables"]} for p in parsed["pages"]],
            key_value_pairs=self.parser.extract_key_value_pairs(parsed),
        )
        
        # Extract and classify clauses
        clauses = self._extract_clauses(parsed)
        classified_clauses = self._classify_clauses(clauses, doc.doc_id)
        
        # Extract structured information
        extracted = process_contract_clauses(classified_clauses, doc.doc_id)
        
        # Compute risk score
        risk_score = self._compute_risk_score(
            classified_clauses,
            extracted,
            doc.doc_id,
            project_id,
        )
        
        return risk_score
    
    def _extract_clauses(self, parsed: Dict) -> List[Clause]:
        """Extract individual clauses from parsed document"""
        clauses = []
        clause_id = 0
        
        for page in parsed["pages"]:
            text = page["text"]
            entities = page.get("entities", [])
            
            # Simple clause segmentation (by paragraphs/sections)
            # In practice, would use more sophisticated segmentation
            sections = self._segment_into_clauses(text, page["page_num"])
            
            for section in sections:
                clauses.append(Clause(
                    clause_id=f"clause_{clause_id}",
                    text=section["text"],
                    risk_category=RiskCategory.OTHER,  # Placeholder
                    clause_type="unknown",
                    page=section["page"],
                    bbox=section.get("bbox", [0, 0, 1000, 1000]),
                    confidence=1.0,
                ))
                clause_id += 1
        
        return clauses
    
    def _segment_into_clauses(self, text: str, page: int) -> List[Dict]:
        """Segment text into clauses"""
        # Simple segmentation by double newlines or section markers
        import re
        
        # Split by section patterns
        section_pattern = r'(?:^|\n)\s*(?:Section|Article|Clause|Schedule)\s+\d+[\.:]'
        sections = re.split(section_pattern, text)
        
        result = []
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:  # Minimum clause length
                result.append({
                    "text": section.strip(),
                    "page": page,
                    "bbox": [0, 0, 1000, 1000],  # Placeholder
                })
        
        return result if result else [{"text": text, "page": page, "bbox": [0, 0, 1000, 1000]}]
    
    def _classify_clauses(self, clauses: List[Clause], doc_id: str) -> List[ClauseClassification]:
        """Classify clauses using Legal-BERT"""
        classified = []
        
        for clause in clauses:
            # Get classification from Legal-BERT
            result = self.classifier.classify_text(clause.text)
            
            # Map to schema
            risk_cats = result.get("risk_categories", [])
            primary_risk = RiskCategory(risk_cats[0]) if risk_cats else RiskCategory.OTHER
            
            classified.append(ClauseClassification(
                clause_id=clause.clause_id,
                doc_id=doc_id,
                text=clause.text,
                page_number=clause.page,
                bbox=clause.bbox,
                risk_category=primary_risk,
                confidence=result.get("clause_confidence", 0.5),
                risk_score=result.get("risk_score", 0),
                severity=result.get("severity", "Low"),
            ))
        
        return classified
    
    def _compute_risk_score(
        self,
        clauses: List[ClauseClassification],
        extracted: Dict,
        doc_id: str,
        project_id: str,
    ) -> ContractRiskScore:
        """Compute aggregate contract risk score"""
        
        # Category scores
        category_scores = {}
        for cat in RiskCategory:
            cat_clauses = [c for c in clauses if c.risk_category == cat]
            if cat_clauses:
                category_scores[cat] = sum(c.risk_score for c in cat_clauses) / len(cat_clauses)
            else:
                category_scores[cat] = 0.0
        
        # Overall score (weighted average)
        weights = {
            RiskCategory.FORCE_MAJEURE: 0.15,
            RiskCategory.TERMINATION: 0.15,
            RiskCategory.COMPENSATION: 0.15,
            RiskCategory.TARIFF_ADJUSTMENT: 0.10,
            RiskCategory.CURRENCY: 0.10,
            RiskCategory.POLITICAL: 0.10,
            RiskCategory.ENVIRONMENTAL: 0.05,
            RiskCategory.TECHNICAL: 0.05,
            RiskCategory.FINANCIAL_COVENANT: 0.10,
            RiskCategory.PERFORMANCE: 0.05,
            RiskCategory.DISPUTE_RESOLUTION: 0.05,
            RiskCategory.GOVERNING_LAW: 0.05,
        }
        
        overall_score = sum(
            category_scores.get(cat, 0) * weight 
            for cat, weight in weights.items()
        )
        
        # Risk flags
        has_uncapped_termination = any(
            not c.cap_pct_of_investment 
            for c in extracted["termination_compensation"]
        )
        
        has_weak_force_majeure = len(extracted["force_majeure_triggers"]) == 0
        
        has_currency_mismatch = False  # Would check revenue vs debt currency
        
        has_political_risk_gaps = RiskCategory.POLITICAL in category_scores and category_scores[RiskCategory.POLITICAL] > 50
        
        has_weak_dscr_covenant = any(
            c.covenant_type == "DSCR" and c.threshold < 1.2
            for c in extracted["covenants"]
        )
        
        has_no_dispute_resolution = RiskCategory.DISPUTE_RESOLUTION not in category_scores or category_scores[RiskCategory.DISPUTE_RESOLUTION] == 0
        
        # Key risks and recommendations
        key_risks = []
        recommendations = []
        
        if has_uncapped_termination:
            key_risks.append("Uncapped termination compensation exposure")
            recommendations.append("Negotiate termination compensation caps")
        
        if has_weak_force_majeure:
            key_risks.append("Inadequate force majeure provisions")
            recommendations.append("Strengthen force majeure clauses with clear triggers")
        
        if has_weak_dscr_covenant:
            key_risks.append("Weak DSCR covenant threshold")
            recommendations.append("Increase minimum DSCR to 1.30x")
        
        return ContractRiskScore(
            doc_id=doc_id,
            project_id=project_id,
            overall_score=overall_score,
            category_scores=category_scores,
            has_uncapped_termination=has_uncapped_termination,
            has_weak_force_majeure=has_weak_force_majeure,
            has_currency_mismatch=has_currency_mismatch,
            has_political_risk_gaps=has_political_risk_gaps,
            has_weak_dscr_covenant=has_weak_dscr_covenant,
            has_no_dispute_resolution=has_no_dispute_resolution,
            covenants=extracted["covenants"],
            force_majeure_triggers=extracted["force_majeure_triggers"],
            termination_compensation=extracted["termination_compensation"],
            key_risks=key_risks,
            mitigation_recommendations=recommendations,
        )
    
    def batch_analyze(
        self,
        file_paths: List[str],
        project_ids: List[str],
    ) -> List[ContractRiskScore]:
        """Analyze multiple contracts"""
        results = []
        for file_path, project_id in zip(file_paths, project_ids):
            try:
                result = self.analyze_contract(file_path, project_id)
                results.append(result)
            except Exception as e:
                logger.error("Contract analysis failed", file=file_path, error=str(e))
        return results