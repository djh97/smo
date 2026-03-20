from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smo.formatting import build_opinion_panels_html, build_visit_summary_html, plain_text_to_html


class FormattingTests(unittest.TestCase):
    def test_summary_contains_expected_labels(self) -> None:
        html = build_visit_summary_html(
            patient_id="mso-1234abcd",
            timestamp="2026-03-20T10:00:00+00:00",
            age=6,
            weight=22,
            symptoms="Shortness of breath",
            spo2=89,
            heart_rate=115,
            history="Asthma",
        )
        self.assertIn("mso-1234abcd", html)
        self.assertIn("SpO2 Level", html)
        self.assertIn("summary-grid", html)

    def test_plain_text_to_html_formats_numbered_lines_and_bullets(self) -> None:
        html = plain_text_to_html(
            "1. Severity Classification: Severe asthma exacerbation.\n"
            "- **Immediate Management:**\n"
            "- Give oxygen"
        )
        self.assertIn("opinion-step", html)
        self.assertIn("<strong>Immediate Management:</strong>", html)
        self.assertIn("<ul class='opinion-list'>", html)

    def test_build_opinion_panels_html_splits_assessment_and_treatment(self) -> None:
        assessment_html, treatment_html = build_opinion_panels_html(
            "1. Condition trend: Not applicable - first recorded visit.\n"
            "2. Likely Diagnosis: Acute severe asthma exacerbation.\n"
            "3. Severity Classification: Severe.\n"
            "4. Recommended Treatment Plan:\n"
            "- **Immediate Management:**\n"
            "- Give oxygen\n"
            "- **Monitoring:**\n"
            "- Reassess frequently"
        )
        self.assertIn("Condition trend", assessment_html)
        self.assertIn("Likely Diagnosis", assessment_html)
        self.assertIn("Clinical Assessment", assessment_html)
        self.assertIn("treatment-groups", treatment_html)
        self.assertIn("Treatment Plan", treatment_html)
        self.assertIn("Immediate Management", treatment_html)
        self.assertIn("Monitoring", treatment_html)

    def test_build_opinion_panels_html_handles_plain_heading_style(self) -> None:
        _, treatment_html = build_opinion_panels_html(
            "1. Condition trend: Worsening.\n"
            "2. Likely Diagnosis: Severe asthma exacerbation.\n"
            "3. Severity Classification: Severe.\n"
            "4. Recommended Treatment Plan:\n"
            "- Immediate Actions:\n"
            "- Give oxygen\n"
            "- Monitoring:\n"
            "- Reassess every 15 minutes"
        )
        self.assertIn("treatment-groups", treatment_html)
        self.assertIn("Immediate Actions", treatment_html)
        self.assertIn("Monitoring", treatment_html)

    def test_build_opinion_panels_html_can_split_into_multiple_cards(self) -> None:
        assessment_html, treatment_html = build_opinion_panels_html(
            "1. Condition trend: "
            + "Worsening with persistent severe symptoms and declining oxygen saturation. " * 4
            + "\n2. Likely Diagnosis: Severe acute asthma exacerbation with possible severe bronchospasm.\n"
            + "3. Severity Classification: "
            + "Severe to life-threatening because of persistent hypoxemia, tachycardia, and respiratory distress. " * 3
            + "\n4. Recommended Treatment Plan:\n"
            + "- Immediate Actions:\n"
            + "- Start high-flow oxygen immediately and titrate to target saturation.\n"
            + "- Give repeated bronchodilator therapy every 20 minutes in the first hour.\n"
            + "- Add ipratropium bromide with each bronchodilator dose.\n"
            + "- Begin systemic corticosteroids promptly and continue for several days.\n"
            + "- Establish IV access and prepare escalation if there is no response.\n"
            + "\n- Monitoring:\n"
            + "- Reassess every 15 minutes and continuously monitor work of breathing.\n"
            + "- Monitor oxygen saturation, heart rate, and fatigue closely.\n"
            + "\n- Escalation:\n"
            + "- Transfer urgently if there is no improvement or if exhaustion develops.\n"
            + "- Prepare for higher-level respiratory support if the patient deteriorates.",
            selected_tools=["GPT-4o", "Claude Opus 4.6"],
        )
        self.assertIn("Tools used:", assessment_html)
        self.assertIn("Clinical Assessment (cont.)", assessment_html)
        self.assertIn("Treatment Plan (cont.)", treatment_html)


if __name__ == "__main__":
    unittest.main()
