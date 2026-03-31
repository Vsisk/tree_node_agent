from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any
import json


@dataclass(slots=True)
class TraceEvent:
    stage: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class PipelineTracer:
    """Lightweight tracer for standard logging output and structured debug capture."""

    def __init__(
        self,
        enabled: bool = False,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        self.enabled = enabled
        self.logger = logger or logging.getLogger("tree_insertion_rag.pipeline")
        self.level = level
        self.events: list[TraceEvent] = []

    def log(self, stage: str, message: str, data: dict[str, Any] | None = None) -> None:
        payload = data or {}
        event = TraceEvent(stage=stage, message=message, data=payload)
        self.events.append(event)
        if not self.enabled:
            return
        line = f"[{stage}] {message}"
        if payload:
            line = f"{line} | {json.dumps(payload, ensure_ascii=False, default=str)}"
        self.logger.log(self.level, line)

    def export(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": event.stage,
                "message": event.message,
                "data": event.data,
            }
            for event in self.events
        ]
