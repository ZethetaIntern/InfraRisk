"""
Clause Extractor for Contract Intelligence
Extracts covenants, force majeure triggers, termination compensation
"""
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.infra_risk.schemas.legal import (
    ExtractedCovenant,
    ForceMajeureTrigger,
    TerminationCompensation,
    RiskCategory,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Clause:
    """Represents a contract clause"""
    clause_id: str
    text: str
    risk_category: RiskCategory
    clause_type: str
    page: int
    bbox: List[float]
    confidence: float


class ClauseExtractor:
    """Extracts structured information from classified clauses"""
    
    def __init__(self):
        # Regex patterns for extraction
        self.covenant_patterns = {
            "dscr": [
                r"DSCR\s*(?:of|not less than|shall be)\s*([\d.]+)",
                r"debt service coverage ratio\s*(?:of|not less than|shall be)\s*([\d.]+)",
            ],
            "llcr": [
                r"LLCR\s*(?:of|not less than|shall be)\s*([\d.]+)",
                r"loan life coverage ratio\s*(?:of|not less than|shall be)\s*([\d.]+)",
            ],
            "plcr": [
                r"PLCR\s*(?:of|not less than|shall be)\s*([\d.]+)",
                r"project life coverage ratio\s*(?:of|not less than|shall be)\s*([\d.]+)",
            ],
            "debt_to_equity": [
                r"debt.?to.?equity\s*(?:ratio\s*)?(?:of|not exceeding|shall not exceed)\s*([\d.:]+)",
                r"D/E\s*(?:ratio\s*)?(?:of|not exceeding|shall not exceed)\s*([\d.:]+)",
            ],
        }
        
        self.monetary_pattern = r"([\$\£\€]\s*[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|m|bn|k)?\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?))"
        self.percentage_pattern = r"(\d+(?:\.\d+)?\s*%)"
        self.date_pattern = r"(\d{1,2}[\s/-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s/-]\d{2,4}|\d{4}[\s/-]\d{1,2}[\s/-]\d{1,2})"
        self.party_pattern = r"(?:party|contractor|employer|client|owner|sponsor|lender|borrower)[\s:]+([A-Z][A-Za-z\s&.,]+(?:Ltd|Limited|Inc|Corporation|Corp|LLC|PLC)?)"
    
    def extract_covenants(self, clause: Clause, doc_id: str) -> List[ExtractedCovenant]:
        """Extract financial covenants from clause"""
        covenants = []
        text = clause.text
        
        for covenant_type, patterns in self.covenant_patterns.items():
            for pattern in self.covenant_patterns.get(covenant_type, []):
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    threshold = float(match.group(1))
                    
                    # Determine measurement frequency
                    freq = "quarterly"
                    if "annual" in text.lower() or "yearly" in text.lower():
                        freq = "annual"
                    elif "semi-annual" in text.lower() or "semi annual" in text.lower():
                        freq = "semi_annual"
                    
                    # Extract cure period
                    cure_period = self._extract_cure_period(text)
                    
                    covenant = ExtractedCovenant(
                        covenant_id=f"{doc_id}_{covenant_type}_{len(covenants)}",
                        clause_id=clause.clause_id,
                        covenant_type=covenant_type.upper(),
                        threshold=threshold,
                        measurement_frequency=freq,
                        cure_period_days=cure_period,
                    )
                    covenants.append(covenant)
        
        return covenants
    
    def extract_force_majeure(self, clause: Clause, doc_id: str) -> List[ForceMajeureTrigger]:
        """Extract force majeure triggers"""
        triggers = []
        text = clause.text.lower()
        
        if clause.risk_category != RiskCategory.FORCE_MAJEURE:
            return triggers
        
        # Common force majeure events
        event_keywords = [
            "earthquake", "flood", "hurricane", "typhoon", "storm",
            "war", "terrorism", "riot", "civil unrest", "rebellion",
            "pandemic", "epidemic", "quarantine", "health emergency",
            "strike", "lockout", "labor dispute",
            "government action", "expropriation", "nationalization",
            "law", "regulation", "permit", "license",
            "fire", "explosion", "accident",
            "act of god", "force majeure",
        ]
        
        found_events = [e for e in event_keywords if e in text]
        
        # Extract notification period
        notif_period = None
        notif_match = re.search(r"notif(?:y|ication)\s*(?:within|in|of)\s*(\d+)\s*(day|business day)", text)
        if notif_match:
            notif_period = int(notif_match.group(1))
            if "business" in notif_match.group(2):
                notif_period = int(notif_period * 1.4)  # Approximate business days
        
        # Extract termination right
        termination_days = None
        term_match = re.search(r"terminat(?:e|ion)\s*(?:after|if).*?(\d+)\s*(day|month)", text)
        if term_match:
            termination_days = int(term_match.group(1))
            if term_match.group(2) == "month":
                termination_days *= 30
        
        # Extract compensation
        compensation = None
        if "compensation" in text or "indemnif" in text:
            compensation = "Force majeure compensation per contract terms"
        
        trigger = ForceMajeureTrigger(
            trigger_id=f"{doc_id}_fm_{len(triggers)}",
            clause_id=clause.clause_id,
            event_types=found_events,
            notification_period_days=notif_period,
            mitigation_obligations=self._extract_mitigation_obligations(text),
            termination_right_after_days=termination_days,
            compensation_entitlement=compensation,
        )
        triggers.append(trigger)
        
        return triggers
    
    def extract_termination_compensation(
        self, 
        clause: Clause, 
        doc_id: str
    ) -> List[TerminationCompensation]:
        """Extract termination compensation structures"""
        compensations = []
        text = clause.text.lower()
        
        if clause.risk_category != RiskCategory.TERMINATION and clause.risk_category != RiskCategory.COMPENSATION:
            return compensations
        
        # Identify termination events
        termination_events = []
        if "default" in text:
            termination_events.append("default")
        if "force majeure" in text:
            termination_events.append("force_majeure")
        if "convenience" in text or "at will" in text:
            termination_events.append("convenience")
        if "expropriat" in text or "nationaliz" in text:
            termination_events.append("expropriation")
        
        for event in termination_events:
            # Extract compensation formula
            formula = self._extract_compensation_formula(text, event)
            
            # Extract components
            components = []
            if "senior debt" in text or "outstanding debt" in text:
                components.append("senior_debt")
            if "equity" in text or "invested capital" in text:
                components.append("equity_irr")
            if "breakage" in text or "swap" in text or "hedge" in text:
                components.append("breakage_costs")
            if "subcontractor" in text or "supplier" in text:
                components.append("subcontractor_claims")
            
            # Extract caps/floors
            cap = self._extract_percentage(text, ["cap", "maximum", "not exceeding", "ceiling"])
            floor = self._extract_percentage(text, ["floor", "minimum", "not less than", "at least"])
            
            compensation = TerminationCompensation(
                compensation_id=f"{doc_id}_term_{event}_{len(compensations)}",
                clause_id=clause.clause_id,
                termination_event=event,
                compensation_formula=formula,
                components=components,
                cap_pct_of_investment=cap,
                floor_pct_of_investment=floor,
            )
            compensations.append(compensation)
        
        return compensations
    
    def _extract_cure_period(self, text: str) -> Optional[int]:
        """Extract cure period in days"""
        patterns = [
            r"cure\s*(?:period|period of)\s*(\d+)\s*(day|business day)",
            r"remed(?:y|ied)\s*(?:within|in)\s*(\d+)\s*(day|business day)",
            r"(\d+)\s*(day|business day)\s*(?:cure|remedy)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                days = int(match.group(1))
                if "business" in match.group(2):
                    days = int(days * 1.4)
                return days
        return None
    
    def _extract_mitigation_obligations(self, text: str) -> List[str]:
        """Extract mitigation obligations"""
        obligations = []
        keywords = [
            "mitigate", "minimize", "reasonable steps", "best efforts",
            "promptly", "diligently", "commercially reasonable",
        ]
        
        for kw in keywords:
            if kw in text.lower():
                # Extract surrounding context
                idx = text.lower().find(kw)
                context = text[max(0, idx-50):idx+100].strip()
                obligations.append(context)
        
        return obligations[:5]  # Limit
    
    def _extract_compensation_formula(self, text: str, event: str) -> str:
        """Extract compensation formula description"""
        # Find relevant sentences
        sentences = re.split(r'[.!?]', text)
        relevant = []
        
        for sent in sentences:
            if event in sent.lower() or "compensation" in sent.lower() or "pay" in sent.lower():
                relevant.append(sent.strip())
        
        return " | ".join(relevant[:3]) if relevant else "Per contract terms"
    
    def _extract_percentage(self, text: str, keywords: List[str]) -> Optional[float]:
        """Extract percentage value near keywords"""
        for kw in keywords:
            pattern = rf"{kw}.*?(\d+(?:\.\d+)?)\s*%"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def extract_parties(self, text: str) -> List[str]:
        """Extract party names from text"""
        parties = []
        matches = re.finditer(self.party_pattern, text, re.IGNORECASE)
        for match in matches:
            party = match.group(1).strip()
            if len(party) > 3 and party not in parties:
                parties.append(party)
        return parties
    
    def extract_monetary_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts with context"""
        amounts = []
        matches = re.finditer(self.monetary_pattern, text)
        for match in matches:
            amounts.append({
                "amount": match.group(1),
                "context": text[max(0, match.start()-50):match.end()+50],
            })
        return amounts
    
    def extract_percentages(self, text: str) -> List[float]:
        """Extract all percentages"""
        percentages = []
        matches = re.finditer(self.percentage_pattern, text)
        for match in matches:
            try:
                val = float(match.group(1).replace("%", ""))
                percentages.append(val)
            except:
                pass
        return percentages
    
    def extract_dates(self, text: str) -> List[datetime]:
        """Extract dates from text"""
        dates = []
        matches = re.finditer(self.date_pattern, text)
        for match in matches:
            try:
                # Try multiple formats
                for fmt in ["%d %b %Y", "%d/%m/%Y", "%Y-%m-%d", "%d %B %Y"]:
                    try:
                        dates.append(datetime.strptime(match.group(1), fmt))
                        break
                    except:
                        continue
            except:
                pass
        return dates


def process_contract_clauses(
    clauses: List[Clause],
    doc_id: str,
) -> Dict[str, List]:
    """Process all clauses and extract structured information"""
    extractor = ClauseExtractor()
    
    all_covenants = []
    all_fm_triggers = []
    all_term_comp = []
    all_parties = set()
    
    for clause in clauses:
        all_covenants.extend(extractor.extract_covenants(clause, doc_id))
        all_fm_triggers.extend(extractor.extract_force_majeure(clause, doc_id))
        all_term_comp.extend(extractor.extract_termination_compensation(clause, doc_id))
        all_parties.update(extractor.extract_parties(clause.text))
    
    return {
        "covenants": all_covenants,
        "force_majeure_triggers": all_fm_triggers,
        "termination_compensation": all_term_comp,
        "parties": list(all_parties),
    }