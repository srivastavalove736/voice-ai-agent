from typing import Any, Dict, List

from agent.schemas import PipelineResult


class SessionHistory:
    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []

    def add(self, result: PipelineResult) -> None:
        self._history.append(result.to_dict())

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history.copy()

    def clear(self) -> None:
        self._history = []
