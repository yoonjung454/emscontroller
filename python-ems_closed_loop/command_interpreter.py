"""
command_interpreter.py
-----------------------
Turns a natural-language Korean command into a target finger-bend percentage
(0-100) or a control action (stop / quit / error), without any paid LLM API
or API key.

The class is deliberately structured so a real LLM backend can be dropped in
later without touching any caller: `CommandInterpreter.interpret()` is the
only method callers use, and it first asks `self._llm_backend` (None by
default) before falling back to the local rule-based parser. To wire in a
real LLM later, implement a callable with the signature
`(text: str) -> Optional[CommandResult]` and pass it as `llm_backend=...`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from config import CommandPresetsConfig

Action = str  # "set_target" | "stop" | "quit" | "error"


@dataclass
class CommandResult:
    action: Action
    target_percent: Optional[float] = None
    message: str = ""


# Keyword tables. Kept as module-level constants so they are easy to extend
# without touching the parsing logic.
_STOP_WORDS = ["정지", "멈춰", "멈춤", "스탑", "stop", "손 펴", "손펴"]
_QUIT_WORDS = ["종료", "끝내", "quit", "exit"]

# Order matters: checked top-to-bottom, first match wins. Keep more specific
# phrases above more general ones.
_PRESET_PHRASES: list[tuple[list[str], str]] = [
    (["살짝", "조금만", "약하게"], "light"),
    (["최대한", "꽉", "세게", "강하게"], "strong"),
    (["반쯤", "반만", "반 정도", "절반"], "half"),
]

_PERCENT_PATTERN = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|퍼센트|프로)")


class CommandInterpreter:
    def __init__(
        self,
        presets: CommandPresetsConfig,
        llm_backend: Optional[Callable[[str], Optional[CommandResult]]] = None,
    ):
        self.presets = presets
        self._llm_backend = llm_backend

    def interpret(self, text: str) -> CommandResult:
        text = (text or "").strip()
        if not text:
            return CommandResult("error", message="명령이 비어 있습니다.")

        # Future extension point: a real LLM/NLU backend can veto/replace the
        # rule-based result. If it returns None, we fall through to the
        # rule-based parser below (never silently do nothing).
        if self._llm_backend is not None:
            llm_result = self._llm_backend(text)
            if llm_result is not None:
                return llm_result

        return self._interpret_rule_based(text)

    # ------------------------------------------------------------------ #

    def _interpret_rule_based(self, text: str) -> CommandResult:
        lowered = text.lower()

        for word in _QUIT_WORDS:
            if word in lowered:
                return CommandResult("quit", message="프로그램을 종료합니다.")

        for word in _STOP_WORDS:
            if word in lowered:
                return CommandResult("stop", message="EMS 자극을 정지합니다.")

        # Explicit numeric percentage, e.g. "45% 쥐어", "75퍼센트 구부려", "40프로"
        match = _PERCENT_PATTERN.search(text)
        if match:
            value = float(match.group(1))
            if value < 0 or value > 100:
                return CommandResult(
                    "error",
                    message=f"목표 굽힘 값은 0~100% 사이여야 합니다 (입력값: {value}%).",
                )
            return CommandResult("set_target", target_percent=value, message=f"목표 굽힘 {value:.0f}% 설정")

        # Named presets ("살짝 쥐어", "반쯤 구부려", "꽉 쥐어", ...)
        for keywords, preset_name in _PRESET_PHRASES:
            if any(k in text for k in keywords):
                percent = {
                    "light": self.presets.light_percent,
                    "half": self.presets.half_percent,
                    "strong": self.presets.strong_percent,
                }[preset_name]
                return CommandResult(
                    "set_target", target_percent=percent, message=f"목표 굽힘 {percent:.0f}% 설정 ({preset_name})"
                )

        # A bare "쥐어"/"구부려" with no qualifier -> ask for clarification
        # instead of guessing a percentage.
        if "쥐어" in text or "구부려" in text or "쥐기" in text:
            return CommandResult(
                "error",
                message=(
                    "얼마나 쥘지 명확하지 않습니다. "
                    "'살짝/반쯤/꽉 쥐어' 또는 '45% 쥐어' 같이 정도를 포함해 말해주세요."
                ),
            )

        return CommandResult(
            "error",
            message=(
                "이해하지 못한 명령입니다. 예: '손을 살짝 쥐어', '반쯤 쥐어', '꽉 쥐어', "
                "'45% 쥐어', '정지', '종료'"
            ),
        )
