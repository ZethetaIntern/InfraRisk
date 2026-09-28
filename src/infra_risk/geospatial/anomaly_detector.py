"""
Anomaly Detector for Construction Progress
Detects site abandonment, scope changes, delays, accelerations
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.infra_risk.schemas.geospatial import (
    SiteProgress,
    ProgressAnomaly,
    AnomalyType,
    ConstructionPhase,
)
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Detects anomalies in construction progress time series"""
    
    def __init__(
        self,
        abandonment_threshold: float = 0.02,  # progress per quarter
        scope_change_threshold: float = 0.15,  # sudden jump
        delay_threshold: float = 0.10,  # deviation from expected
        acceleration_threshold: float = 0.20,  # unexpected acceleration
        min_history_points: int = 4,
    ):
        self.abandonment_threshold = abandonment_threshold
        self.scope_change_threshold = scope_change_threshold
        self.delay_threshold = delay_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_history_points = min_history_points
    
    def detect(
        self,
        progress_history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect all anomaly types in progress history"""
        if len(progress_history) < self.min_history_points:
            logger.warning("Insufficient history for anomaly detection", project_id=project_id)
            return []
        
        anomalies = []
        
        # Sort by timestamp
        sorted_history = sorted(progress_history, key=lambda x: x.timestamp)
        
        # Detect each anomaly type
        anomalies.extend(self._detect_abandonment(sorted_history, project_id))
        anomalies.extend(self._detect_scope_change(sorted_history, project_id))
        anomalies.extend(self._detect_delay(sorted_history, project_id))
        anomalies.extend(self._detect_acceleration(sorted_history, project_id))
        anomalies.extend(self._detect_environmental_impact(sorted_history, project_id))
        anomalies.extend(self._detect_quality_issues(sorted_history, project_id))
        
        # Sort by severity
        anomalies.sort(key=lambda x: x.severity, reverse=True)
        
        logger.info("Anomaly detection complete", project_id=project_id, count=len(anomalies))
        return anomalies
    
    def _detect_abandonment(
        self,
        history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect site abandonment (progress stalls)"""
        anomalies = []
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            progress_diff = curr.progress_pct - prev.progress_pct
            time_diff = (curr.timestamp - prev.timestamp).days / 30  # months
            
            if time_diff > 0:
                monthly_progress = progress_diff / time_diff
                
                if monthly_progress < self.abandonment_threshold and curr.progress_pct > 10:
                    # Site appears abandoned
                    severity = min(1.0, (self.abandonment_threshold - monthly_progress) / self.abandonment_threshold)
                    
                    anomalies.append(ProgressAnomaly(
                        anomaly_id=f"{project_id}_abandonment_{i}",
                        project_id=project_id,
                        timestamp=curr.timestamp,
                        anomaly_type=AnomalyType.SITE_ABANDONMENT,
                        severity=severity,
                        description=f"Construction progress stalled at {curr.progress_pct:.1f}% "
                                   f"({monthly_progress:.2f}%/month)",
                        affected_area_pct=100 - curr.progress_pct,
                        confidence=0.7 + 0.3 * severity,
                        evidence_images=[curr.current_image_id],
                        recommended_actions=[
                            "Contact EPC contractor for schedule recovery plan",
                            "Review contract termination provisions",
                            "Assess DSRA adequacy for extended timeline",
                        ],
                    ))
        
        return anomalies
    
    def _detect_scope_change(
        self,
        history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect scope changes (sudden progress jumps or phase changes)"""
        anomalies = []
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            progress_diff = curr.progress_pct - prev.progress_pct
            time_diff = (curr.timestamp - prev.timestamp).days / 30
            
            if time_diff > 0:
                monthly_progress = progress_diff / time_diff
                
                # Unusually fast progress (potential scope reduction or acceleration)
                if monthly_progress > self.scope_change_threshold:
                    severity = min(1.0, monthly_progress / (self.scope_change_threshold * 2))
                    
                    anomalies.append(ProgressAnomaly(
                        anomaly_id=f"{project_id}_scope_change_{i}",
                        project_id=project_id,
                        timestamp=curr.timestamp,
                        anomaly_type=AnomalyType.SCOPE_CHANGE,
                        severity=severity,
                        description=f"Unusual progress acceleration: {monthly_progress:.1f}%/month "
                                   f"(phase change: {prev.phase.value} -> {curr.phase.value})",
                        affected_area_pct=abs(progress_diff),
                        confidence=0.6 + 0.4 * severity,
                        evidence_images=[prev.current_image_id, curr.current_image_id],
                        recommended_actions=[
                            "Verify scope change with EPC contractor",
                            "Review change order documentation",
                            "Assess impact on budget and timeline",
                            "Update financial model with revised scope",
                        ],
                    ))
        
        return anomalies
    
    def _detect_delay(
        self,
        history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect delays relative to expected progress curve"""
        anomalies = []
        
        # Simple expected progress: linear from 0 to 100 over construction period
        # In practice, this would use a planned S-curve
        if len(history) < 2:
            return anomalies
        
        # Estimate expected progress rate from early phases
        early_progress = [h for h in history if h.phase in [ConstructionPhase.SITE_PREPARATION, ConstructionPhase.EARTHWORK]]
        if len(early_progress) >= 2:
            early_rate = (early_progress[-1].progress_pct - early_progress[0].progress_pct) / \
                        ((early_progress[-1].timestamp - early_progress[0].timestamp).days / 30)
        else:
            early_rate = 5.0  # Default 5% per month
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            expected_progress = prev.progress_pct + early_rate * (curr.timestamp - prev.timestamp).days / 30
            actual_progress = curr.progress_pct
            
            if expected_progress > actual_progress:
                delay = expected_progress - actual_progress
                if delay > self.delay_threshold * 100:  # Convert to percentage points
                    severity = min(1.0, delay / (self.delay_threshold * 100 * 2))
                    
                    anomalies.append(ProgressAnomaly(
                        anomaly_id=f"{project_id}_delay_{i}",
                        project_id=project_id,
                        timestamp=curr.timestamp,
                        anomaly_type=AnomalyType.DELAY,
                        severity=severity,
                        description=f"Progress delayed by {delay:.1f}% vs expected "
                                   f"(expected: {expected_progress:.1f}%, actual: {actual_progress:.1f}%)",
                        affected_area_pct=delay,
                        confidence=0.7,
                        evidence_images=[curr.current_image_id],
                        recommended_actions=[
                            "Request detailed delay analysis from contractor",
                            "Assess liquidated damages applicability",
                            "Review schedule recovery options",
                            "Update cash flow forecast for delayed COD",
                        ],
                    ))
        
        return anomalies
    
    def _detect_acceleration(
        self,
        history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect unusual acceleration (potential quality issues)"""
        anomalies = []
        
        if len(history) < 3:
            return anomalies
        
        # Calculate rolling acceleration
        for i in range(2, len(history)):
            p1 = history[i-2].progress_pct
            p2 = history[i-1].progress_pct
            p3 = history[i].progress_pct
            
            t1 = history[i-2].timestamp
            t2 = history[i-1].timestamp
            t3 = history[i].timestamp
            
            # Approximate second derivative
            dt1 = (t2 - t1).days / 30
            dt2 = (t3 - t2).days / 30
            
            if dt1 > 0 and dt2 > 0:
                v1 = (p2 - p1) / dt1
                v2 = (p3 - p2) / dt2
                acceleration = (v2 - v1) / ((dt1 + dt2) / 2)
                
                if acceleration > self.acceleration_threshold * 100:
                    severity = min(1.0, acceleration / (self.acceleration_threshold * 100 * 2))
                    
                    anomalies.append(ProgressAnomaly(
                        anomaly_id=f"{project_id}_acceleration_{i}",
                        project_id=project_id,
                        timestamp=history[i].timestamp,
                        anomaly_type=AnomalyType.ACCELERATION,
                        severity=severity,
                        description=f"Unusual progress acceleration detected: {acceleration:.1f}%/month²",
                        affected_area_pct=min(100, abs(v2) * 2),
                        confidence=0.6,
                        evidence_images=[history[i].current_image_id],
                        recommended_actions=[
                            "Inspect work quality for rushed construction",
                            "Verify material deliveries match installation rate",
                            "Review QA/QC reports for the period",
                            "Consider independent engineer inspection",
                        ],
                    ))
        
        return anomalies
    
    def _detect_environmental_impact(
        self,
        history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect environmental impacts (vegetation loss, water changes)"""
        anomalies = []
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Check NDVI drops (vegetation loss)
            if prev.spectral_indices.ndvi is not None and curr.spectral_indices.ndvi is not None:
                ndvi_change = np.nanmean(curr.spectral_indices.ndvi) - np.nanmean(prev.spectral_indices.ndvi)
                
                if ndvi_change < -0.15:  # Significant vegetation loss
                    severity = min(1.0, abs(ndvi_change) / 0.5)
                    
                    anomalies.append(ProgressAnomaly(
                        anomaly_id=f"{project_id}_env_impact_{i}",
                        project_id=project_id,
                        timestamp=curr.timestamp,
                        anomaly_type=AnomalyType.ENVIRONMENTAL_IMPACT,
                        severity=severity,
                        description=f"Significant vegetation loss detected: NDVI change {ndvi_change:.3f}",
                        affected_area_pct=50,  # Estimate
                        confidence=0.8,
                        evidence_images=[prev.current_image_id, curr.current_image_id],
                        recommended_actions=[
                            "Verify environmental compliance",
                            "Check for unauthorized clearing",
                            "Review ESMP implementation",
                            "Engage environmental consultant if needed",
                        ],
                    ))
        
        return anomalies
    
    def _detect_quality_issues(
        self,
        history: List[SiteProgress],
        project_id: str,
    ) -> List[ProgressAnomaly]:
        """Detect potential quality issues from spectral signatures"""
        anomalies = []
        
        for curr in history:
            # Check for unusual spectral signatures
            if curr.spectral_indices.ndbi is not None:
                ndbi_std = np.nanstd(curr.spectral_indices.ndbi)
                
                # High NDBI variance might indicate patchy/poor construction
                if ndbi_std > 0.2:
                    severity = min(1.0, ndbi_std / 0.5)
                    
                    anomalies.append(ProgressAnomaly(
                        anomaly_id=f"{project_id}_quality_{curr.timestamp.strftime('%Y%m')}",
                        project_id=project_id,
                        timestamp=curr.timestamp,
                        anomaly_type=AnomalyType.QUALITY_ISSUE,
                        severity=severity,
                        description=f"High built-up index variance suggests potential quality issues: "
                                   f"NDBI std={ndbi_std:.3f}",
                        affected_area_pct=30,
                        confidence=0.5,
                        evidence_images=[curr.current_image_id],
                        recommended_actions=[
                            "Schedule independent quality inspection",
                            "Review concrete test results",
                            "Verify compaction test records",
                            "Check for rework indicators",
                        ],
                    ))
        
        return anomalies