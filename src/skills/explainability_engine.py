"""
Explainability Engine

Generates human-interpretable explanations for detection decisions.
"""

import logging
from typing import List, Dict, Optional
import numpy as np

from ..models import Finding, Severity, SkillResult, Verdict

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Generate explanations for detection results.
    
    Provides:
    - Natural language explanations
    - Attention maps for suspicious regions
    - Confidence breakdowns
    - Evidence summaries
    """
    
    def __init__(self):
        self.severity_icons = {
            Severity.LOW: "🟢",
            Severity.MEDIUM: "🟡", 
            Severity.HIGH: "🔴",
            Severity.CRITICAL: "⛔"
        }
        
    def generate_explanation(self, verdict: Verdict,
                            confidence: float,
                            findings: List[Finding],
                            skill_results: Dict[str, SkillResult]) -> str:
        """
        Generate human-readable explanation.
        
        Args:
            verdict: Detection verdict
            confidence: Confidence in verdict
            findings: List of detected findings
            skill_results: Results from each analysis skill
            
        Returns:
            Formatted explanation string
        """
        lines = []
        
        # Header with verdict
        verdict_icon = self._get_verdict_icon(verdict)
        lines.append(f"{verdict_icon} **Verdict: {verdict.value}**")
        lines.append(f"Confidence: {confidence:.1%}")
        lines.append("")
        
        # Summary based on verdict
        if verdict == Verdict.DEEPFAKE:
            lines.append("⚠️ This video shows signs of manipulation.")
        elif verdict == Verdict.REAL:
            lines.append("✓ This video appears to be authentic.")
        else:
            lines.append("❓ Analysis is inconclusive - additional review recommended.")
        
        lines.append("")
        
        # Key findings
        if findings:
            lines.append("**Key Findings:**")
            
            # Sort by severity
            sorted_findings = sorted(
                findings, 
                key=lambda f: list(Severity).index(f.severity),
                reverse=True
            )
            
            for finding in sorted_findings[:5]:  # Top 5 findings
                icon = self.severity_icons.get(finding.severity, "•")
                lines.append(f"  {icon} {finding.description} ({finding.confidence:.0%})")
            
            if len(findings) > 5:
                lines.append(f"  ... and {len(findings) - 5} more findings")
            
            lines.append("")
        
        # Skill breakdown
        lines.append("**Analysis Breakdown:**")
        
        for skill_name, result in skill_results.items():
            skill_display = skill_name.replace("_", " ").title()
            
            # Interpret score (higher = more suspicious)
            if result.score > 0.7:
                status = "🔴 High suspicion"
            elif result.score > 0.4:
                status = "🟡 Some concerns"
            else:
                status = "🟢 Normal"
            
            lines.append(f"  • {skill_display}: {status} ({result.score:.0%} anomaly)")
        
        lines.append("")
        
        # Recommendations
        lines.append("**Recommendations:**")
        
        if verdict == Verdict.DEEPFAKE:
            lines.append("  • Do not trust this video without additional verification")
            lines.append("  • Check original source if possible")
            lines.append("  • Look for corroborating evidence from other sources")
        elif verdict == Verdict.UNCERTAIN:
            lines.append("  • Request higher quality version if available")
            lines.append("  • Manually review flagged segments")
            lines.append("  • Consider additional forensic analysis")
        else:
            lines.append("  • No immediate concerns detected")
            lines.append("  • Standard verification procedures still recommended")
        
        return "\n".join(lines)
    
    def generate_attention_maps(self, frames: List[np.ndarray],
                                face_results: List,
                                findings: List[Finding]
                               ) -> Dict[int, np.ndarray]:
        """
        Generate attention maps highlighting suspicious regions.
        
        Returns dict mapping frame indices to attention maps.
        """
        attention_maps = {}
        
        # Find frames with findings
        frames_with_findings = set()
        for finding in findings:
            if finding.frame_range:
                for frame_idx in range(finding.frame_range[0], finding.frame_range[1] + 1):
                    frames_with_findings.add(frame_idx)
        
        # Generate attention maps for key frames
        sample_frames = list(frames_with_findings)[:10]  # Limit to 10
        
        if not sample_frames and len(frames) > 0:
            # Sample evenly if no specific frames
            sample_frames = list(np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int))
        
        for frame_idx in sample_frames:
            if frame_idx < len(frames):
                attention = self._generate_frame_attention(
                    frames[frame_idx],
                    face_results[frame_idx] if frame_idx < len(face_results) else None,
                    [f for f in findings if self._finding_in_frame(f, frame_idx)]
                )
                attention_maps[frame_idx] = attention
        
        return attention_maps
    
    def _generate_frame_attention(self, frame: np.ndarray,
                                  face_result,
                                  frame_findings: List[Finding]) -> np.ndarray:
        """Generate attention map for a single frame."""
        h, w = frame.shape[:2]
        attention = np.zeros((h, w), dtype=np.float32)
        
        # Highlight face region if detected
        if face_result and hasattr(face_result, 'faces') and face_result.faces:
            for face in face_result.faces:
                bbox = face.bbox
                x, y = bbox.x, bbox.y
                bw, bh = bbox.width, bbox.height
                
                # Create Gaussian attention around face
                y_coords, x_coords = np.ogrid[:h, :w]
                center_x = x + bw // 2
                center_y = y + bh // 2
                
                sigma_x = bw / 3
                sigma_y = bh / 3
                
                gaussian = np.exp(-((x_coords - center_x)**2 / (2 * sigma_x**2) +
                                   (y_coords - center_y)**2 / (2 * sigma_y**2)))
                
                attention = np.maximum(attention, gaussian * 0.5)
        
        # Highlight specific finding locations
        for finding in frame_findings:
            if finding.location:
                loc = finding.location
                
                # Intensity based on severity
                intensity = {
                    Severity.LOW: 0.3,
                    Severity.MEDIUM: 0.5,
                    Severity.HIGH: 0.8,
                    Severity.CRITICAL: 1.0
                }.get(finding.severity, 0.5)
                
                # Add attention at location
                x1 = max(0, loc.x)
                y1 = max(0, loc.y)
                x2 = min(w, loc.x + loc.width)
                y2 = min(h, loc.y + loc.height)
                
                attention[y1:y2, x1:x2] = np.maximum(
                    attention[y1:y2, x1:x2], 
                    intensity
                )
        
        return attention
    
    def _finding_in_frame(self, finding: Finding, frame_idx: int) -> bool:
        """Check if finding applies to specific frame."""
        if finding.frame_range:
            return finding.frame_range[0] <= frame_idx <= finding.frame_range[1]
        return True  # Findings without frame range apply everywhere
    
    def _get_verdict_icon(self, verdict: Verdict) -> str:
        """Get icon for verdict."""
        icons = {
            Verdict.REAL: "✅",
            Verdict.DEEPFAKE: "⚠️",
            Verdict.UNCERTAIN: "❓"
        }
        return icons.get(verdict, "•")
    
    def generate_report(self, result, output_format: str = "text") -> str:
        """
        Generate a complete analysis report.
        
        Args:
            result: DetectionResult object
            output_format: "text", "html", or "json"
            
        Returns:
            Formatted report string
        """
        if output_format == "json":
            import json
            return json.dumps(result.to_dict(), indent=2)
        
        elif output_format == "html":
            return self._generate_html_report(result)
        
        else:
            return self._generate_text_report(result)
    
    def _generate_text_report(self, result) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "DEEPFAKE DETECTION REPORT",
            "=" * 60,
            "",
            result.explanation,
            "",
            "-" * 60,
            "TECHNICAL DETAILS",
            "-" * 60,
            "",
            f"Processing time: {result.processing_time:.2f}s",
            f"Total frames analyzed: {result.total_frames}",
            f"Faces detected: {result.faces_detected}",
            f"Video duration: {result.video_duration:.1f}s",
            f"Frame rate: {result.fps:.1f} fps",
            "",
        ]
        
        if result.suspicious_segments:
            lines.append("SUSPICIOUS SEGMENTS:")
            for seg in result.suspicious_segments:
                lines.append(f"  • {seg.start:.1f}s - {seg.end:.1f}s "
                           f"(score: {seg.suspicion_score:.2f})")
            lines.append("")
        
        lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return "\n".join(lines)
    
    def _generate_html_report(self, result) -> str:
        """Generate HTML report."""
        verdict_color = {
            Verdict.REAL: "#4CAF50",
            Verdict.DEEPFAKE: "#F44336",
            Verdict.UNCERTAIN: "#FF9800"
        }.get(result.verdict, "#9E9E9E")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Deepfake Detection Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .verdict {{ font-size: 24px; color: {verdict_color}; font-weight: bold; margin-bottom: 10px; }}
        .confidence {{ color: #666; font-size: 18px; }}
        .section {{ margin-top: 30px; }}
        .section h3 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .finding {{ padding: 10px; margin: 5px 0; background: #f9f9f9; border-radius: 5px; }}
        .finding.high {{ border-left: 4px solid #F44336; }}
        .finding.medium {{ border-left: 4px solid #FF9800; }}
        .finding.low {{ border-left: 4px solid #4CAF50; }}
        .metric {{ display: inline-block; padding: 5px 15px; margin: 5px; background: #e3f2fd; border-radius: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="verdict">{result.verdict.value}</div>
        <div class="confidence">Confidence: {result.confidence:.1%}</div>
        
        <div class="section">
            <h3>Summary</h3>
            <pre>{result.explanation}</pre>
        </div>
        
        <div class="section">
            <h3>Metrics</h3>
            <span class="metric">Frames: {result.total_frames}</span>
            <span class="metric">Duration: {result.video_duration:.1f}s</span>
            <span class="metric">Faces: {result.faces_detected}</span>
            <span class="metric">Processing: {result.processing_time:.2f}s</span>
        </div>
        
        <div class="section">
            <h3>Findings ({len(result.findings)})</h3>
"""
        
        for finding in result.findings[:10]:
            severity_class = finding.severity.value.lower()
            html += f"""
            <div class="finding {severity_class}">
                <strong>{finding.category}</strong>: {finding.description}
                <br><small>Confidence: {finding.confidence:.0%}</small>
            </div>
"""
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

