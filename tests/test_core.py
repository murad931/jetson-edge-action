import tempfile
import unittest
from pathlib import Path

from edge_anpr.core import CooldownFilter, Recognition, normalize_plate
from edge_anpr.storage import EventStore


class CoreTests(unittest.TestCase):
    def test_normalize_plate(self):
        self.assertEqual(normalize_plate(" ab-12 3 "), "AB123")

    def test_cooldown(self):
        gate = CooldownFilter(10)
        self.assertTrue(gate.allow("ABC123", now=100))
        self.assertFalse(gate.allow("ABC123", now=105))
        self.assertTrue(gate.allow("ABC123", now=111))

    def test_store(self):
        with tempfile.TemporaryDirectory() as folder:
            store = EventStore(str(Path(folder) / "events.db"))
            row_id = store.add(Recognition("ABC123", 0.9, 100.0))
            self.assertEqual(row_id, 1)
            store.close()


if __name__ == "__main__":
    unittest.main()

