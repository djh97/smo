from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smo.patient_store import VisitStore


class VisitStoreTests(unittest.TestCase):
    def test_latest_visit_is_returned(self) -> None:
        store = VisitStore()
        store.record_visit("p1", "new", "first", "2026-03-20T10:00:00+00:00")
        store.record_visit("p1", "follow-up", "second", "2026-03-20T11:00:00+00:00")

        latest = store.get_latest_visit("p1")

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.case_text, "second")


if __name__ == "__main__":
    unittest.main()
