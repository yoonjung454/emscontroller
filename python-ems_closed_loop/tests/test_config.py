import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AdaptiveModelConfig, AppConfig, load_config, save_config


class TestConfig(unittest.TestCase):
    def test_round_trip_preserves_nested_dataclass_types(self):
        """Regression test: config.py uses `from __future__ import annotations`,
        which turns dataclass field types into unevaluated strings. A naive
        `is_dataclass(field.type)` check silently fails on that and leaves
        nested config sections as plain dicts instead of real dataclass
        instances -- which crashes the first time any code does
        `config.adaptive_model.degree` instead of `config.adaptive_model["degree"]`."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = AppConfig()
            original.adaptive_model.degree = 2
            save_config(original, path)

            loaded = load_config(path)

            self.assertIsInstance(loaded.adaptive_model, AdaptiveModelConfig)
            self.assertEqual(loaded.adaptive_model.degree, 2)
            # This is exactly the line that used to raise AttributeError.
            self.assertIsInstance(loaded.adaptive_model.degree, int)

    def test_missing_file_creates_defaults(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "does_not_exist.json"
            cfg = load_config(path)
            self.assertTrue(path.exists())
            self.assertEqual(cfg.safety.safety_max_intensity, 0)

    def test_partial_file_falls_back_to_defaults_for_missing_fields(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"presets": {"light_percent": 15.0}}', encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.presets.light_percent, 15.0)
            self.assertEqual(cfg.presets.half_percent, 60.0)  # default, not present in file
            self.assertEqual(cfg.safety.safety_max_intensity, 0)  # default nested section


if __name__ == "__main__":
    unittest.main()
