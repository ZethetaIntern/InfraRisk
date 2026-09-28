"""
Contract Intelligence Pipeline (Legal NLP) Schemas
LayoutLM for PDF parsing, Legal-BERT for clause classification
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class DocumentType(str, Enum):
    CONCESSION_AGREEMENT = "concession_agreement"
    PPA = "power_purchase_agreement"
    EPC_CONTRACT = "epc_contract"
    OM_AGREEMENT = "om_agreement"
    FINANCING_AGREEMENT = "financing_agreement"
    SHAREHOLDERS_AGREEMENT = "shareholders_agreement"
    INTERCREDITOR_AGREEMENT = "intercreditor_agreement"
    DIRECT_AGREEMENT = "direct_agreement"
    ESCROW_AGREEMENT = "escrow_agreement"
    INSURANCE_POLICY = "insurance_policy"
    AMENDMENT = "amendment"
    SIDE_LETTER = "side_letter"


class RiskCategory(str, Enum):
    """12 risk categories for clause classification"""
    FORCE_MAJEURE = "force_majeure"
    TERMINATION = "termination"
    COMPENSATION = "compensation"
    TARIFF_ADJUSTMENT = "tariff_adjustment"
    CURRENCY = "currency"
    POLITICAL = "political"
    ENVIRONMENTAL = "environmental"
    TECHNICAL = "technical"
    FINANCIAL_COVENANT = "financial_covenant"
    PERFORMANCE = "performance"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GOVERNING_LAW = "governing_law"


class ClauseType(str, Enum):
    DEFINITION = "definition"
    OBLIGATION = "obligation"
    RIGHT = "right"
    CONDITION = "condition"
    WARRANTY = "warranty"
    INDEMNITY = "indemnity"
    LIMITATION = "limitation"
    TERMINATION = "termination"
    FORCE_MAJEURE = "force_majeure"
    DISPUTE = "dispute"
    GOVERNING_LAW = "governing_law"
    AMENDMENT = "amendment"
    ASSIGNMENT = "assignment"
    CONFIDENTIALITY = "confidentiality"
    OTHER = "other"


class ContractDocument(BaseModel):
    """Parsed contract document from LayoutLM"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    doc_id: str
    project_id: str
    document_type: DocumentType
    file_path: str
    pages: int
    # LayoutLM outputs
    text_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    key_value_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    # Metadata
    parties: List[str] = Field(default_factory=list)
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    governing_law: Optional[str] = None
    language: str = "en"
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    parser_version: str = "layoutlm-v3.0"


class ClauseClassification(BaseModel):
    """Legal-BERT clause classification output"""
    clause_id: str
    doc_id: str
    text: str
    page_number: int
    bbox: List[float] = Field(description="[x1, y1, x2, y2] normalized")
    # Classification
    risk_category: RiskCategory
    clause_type: ClauseType
    confidence: float = Field(ge=0, le=1)
    # Extracted entities
    parties_mentioned: List[str] = Field(default_factory=list)
    dates_mentioned: List[datetime] = Field(default_factory=list)
    monetary_amounts: List[Dict[str, Any]] = Field(default_factory=list)
    percentages: List[float] = Field(default_factory=list)
    # Risk scoring
    risk_score: float = Field(ge=0, le=100)
    severity: str = Field(description="Critical/High/Medium/Low")
    model_version: str = "legal-bert-v1.0"


class ExtractedCovenant(BaseModel):
    """Extracted financial covenant"""
    covenant_id: str
    clause_id: str
    covenant_type: str  # DSCR, LLCR, PLCR, D/E, etc.
    threshold: float
    measurement_frequency: str  # quarterly, semi-annual, annual
    cure_period_days: Optional[int] = None
    test_date: Optional[datetime] = None
    current_value: Optional[float] = None
    compliance_status: Optional[str] = None  # compliant, breach, cure_period


class ForceMajeureTrigger(BaseModel):
    """Extracted force majeure trigger"""
    trigger_id: str
    clause_id: str
    event_types: List[str]
    notification_period_days: Optional[int] = None
    mitigation_obligations: List[str] = Field(default_factory=list)
    termination_right_after_days: Optional[int] = None
    compensation_entitlement: Optional[str] = None


class TerminationCompensation(BaseModel):
    """Extracted termination compensation structure"""
    compensation_id: str
    clause_id: str
    termination_event: str  # default, force_majeure, convenience, etc.
    compensation_formula: str
    components: List[str] = Field(default_factory=list)  # senior_debt, equity_irr, breakage_costs
    cap_pct_of_investment: Optional[float] = None
    floor_pct_of_investment: Optional[float] = None
    currency: str = "USD"


class ContractRiskScore(BaseModel):
    """Aggregate contract risk score"""
    doc_id: str
    project_id: str
    overall_score: float = Field(ge=0, le=100)
    category_scores: Dict[RiskCategory, float] = Field(default_factory=dict)
    # Key risk flags
    has_uncapped_termination: bool = False
    has_weak_force_majeure: bool = False
    has_currency_mismatch: bool = False
    has_political_risk_gaps: bool = False
    has_weak_dscr_covenant: bool = False
    has_no_dispute_resolution: bool = False
    # Covenant analysis
    covenants: List[ExtractedCovenant] = Field(default_factory=list)
    force_majeure_triggers: List[ForceMajeureTrigger] = Field(default_factory=list)
    termination_compensation: List[TerminationCompensation] = Field(default_factory=list)
    # Summary
    key_risks: List[str] = Field(default_factory=list)
    mitigation_recommendations: List[str] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    analyzer_version: str = "contract-risk-v1.0"