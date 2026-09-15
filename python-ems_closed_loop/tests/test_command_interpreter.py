import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from command_interpreter import CommandInterpreter
from config import CommandPresetsConfig


class TestCommandInterpreter(unittest.TestCase):
    def setUp(self):
        self.presets = CommandPresetsConfig(light_percent=30.0, half_percent=60.0, strong_percent=90.0)
        self.interp = CommandInterpreter(self.presets)

    def test_light_preset(self):
        result = self.interp.interpret("손을 살짝 쥐어")
        self.assertEqual(result.action, "set_target")
        self.assertEqual(result.target_percent, 30.0)

    def test_half_preset(self):
        result = self.interp.interpret("반쯤 쥐어")
        self.assertEqual(result.action, "set_target")
        self.assertEqual(result.target_percent, 60.0)

    def test_strong_preset(self):
        result = self.interp.interpret("손을 꽉 쥐어")
        self.assertEqual(result.action, "set_target")
        self.assertEqual(result.target_percent, 90.0)

    def test_explicit_percent(self):
        result = self.interp.interpret("손을 45% 쥐어")
        self.assertEqual(result.action, "set_target")
        self.assertEqual(result.target_percent, 45.0)

    def test_explicit_percent_korean_unit(self):
        result = self.interp.interpret("손가락을 75퍼센트 구부려")
        self.assertEqual(result.action, "set_target")
        self.assertEqual(result.target_percent, 75.0)

    def test_percent_out_of_range_is_error(self):
        result = self.interp.interpret("150% 쥐어")
        self.assertEqual(result.action, "error")

    def test_stop_word(self):
        for word in ["정지", "멈춰", "손 펴"]:
            result = self.interp.interpret(word)
            self.assertEqual(result.action, "stop", msg=word)

    def test_quit_word(self):
        result = self.interp.interpret("종료")
        self.assertEqual(result.action, "quit")

    def test_ambiguous_grip_is_error(self):
        result = self.interp.interpret("쥐어")
        self.assertEqual(result.action, "error")

    def test_unknown_command_is_error(self):
        result = self.interp.interpret("오늘 날씨 어때")
        self.assertEqual(result.action, "error")

    def test_empty_command_is_error(self):
        result = self.interp.interpret("")
        self.assertEqual(result.action, "error")


if __name__ == "__main__":
    unittest.main()
