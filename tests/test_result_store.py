from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smo.result_store import SavedResultStore
from smo.schemas import SavedCaseResult


class SavedResultStoreTests(unittest.TestCase):
    def test_latest_saved_result_can_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SavedResultStore(Path(temp_dir))
            first = SavedCaseResult(
                patient_id="p1",
                timestamp="2026-03-20T10:00:00+00:00",
                visit_type="New",
                age=6,
                weight=22,
                symptoms="Cough",
                spo2=89,
                heart_rate=115,
                history="Asthma",
                case_text="first",
                final_opinion="first opinion",
                selected_tools=["GPT-4o"],
            )
            second = SavedCaseResult(
                patient_id="p1",
                timestamp="2026-03-20T11:00:00+00:00",
                visit_type="Follow-up",
                age=6,
                weight=22,
                symptoms="Improved cough",
                spo2=94,
                heart_rate=100,
                history="Asthma",
                case_text="second",
                final_opinion="second opinion",
                selected_tools=["GPT-4o", "Gemini 2.5 Flash"],
            )
            store.save(first)
            store.save(second)

            latest = store.get_latest("p1")

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.case_text, "second")
            self.assertEqual(latest.visit_type, "Follow-up")

    def test_saved_results_can_be_listed_and_selected_by_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SavedResultStore(Path(temp_dir))
            first = SavedCaseResult(
                patient_id="p2",
                timestamp="2026-03-20T09:00:00+00:00",
                visit_type="New",
                age=5,
                weight=18,
                symptoms="Wheeze",
                spo2=92,
                heart_rate=110,
                history="Asthma",
                case_text="visit one",
                final_opinion="opinion one",
                selected_tools=["GPT-4o"],
            )
            second = SavedCaseResult(
                patient_id="p2",
                timestamp="2026-03-20T12:00:00+00:00",
                visit_type="Follow-up",
                age=5,
                weight=18,
                symptoms="Improving wheeze",
                spo2=95,
                heart_rate=98,
                history="Asthma",
                case_text="visit two",
                final_opinion="opinion two",
                selected_tools=["GPT-4o", "Claude Opus 4.6"],
            )
            store.save(first)
            store.save(second)

            visits = store.list_records("p2")
            selected = store.get_by_timestamp("p2", "2026-03-20T09:00:00+00:00")

            self.assertEqual(len(visits), 2)
            self.assertEqual(visits[0].timestamp, "2026-03-20T09:00:00+00:00")
            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertEqual(selected.case_text, "visit one")


if __name__ == "__main__":
    unittest.main()
