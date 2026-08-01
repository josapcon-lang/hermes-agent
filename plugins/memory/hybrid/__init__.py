"""Fail-open Holographic + Hindsight memory provider.

Holographic is the local, always-available operational store. Hindsight is
the asynchronous long-term analysis layer. A Hindsight outage must never
disable local recall or prevent the agent from starting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from ..holographic import HolographicMemoryProvider
from ..hindsight import HindsightMemoryProvider

logger = logging.getLogger(__name__)


class HybridMemoryProvider(MemoryProvider):
    """Compose the two bundled providers behind Hermes' one-provider API."""

    def __init__(self):
        self.holographic = HolographicMemoryProvider()
        self.hindsight = HindsightMemoryProvider()
        self._hindsight_active = False

    @property
    def name(self) -> str:
        return "hybrid"

    def is_available(self) -> bool:
        return self.holographic.is_available()

    def initialize(self, session_id: str, **kwargs) -> None:
        self.holographic.initialize(session_id, **kwargs)
        try:
            if self.hindsight.is_available():
                self.hindsight.initialize(session_id, **kwargs)
                self._hindsight_active = getattr(self.hindsight, "_mode", "") != "disabled"
        except Exception as exc:
            self._hindsight_active = False
            logger.warning("Hybrid memory: Hindsight unavailable; continuing locally: %s", exc)

    def _safe_hindsight(self, method: str, *args, default=None, **kwargs):
        if not self._hindsight_active:
            return default
        try:
            return getattr(self.hindsight, method)(*args, **kwargs)
        except Exception as exc:
            logger.warning("Hybrid memory: Hindsight %s failed; local memory remains active: %s", method, exc)
            return default

    def system_prompt_block(self) -> str:
        local = self.holographic.system_prompt_block()
        remote = self._safe_hindsight("system_prompt_block", default="")
        status = "active" if self._hindsight_active else "degraded (Holographic-only)"
        return "\n\n".join(x for x in (f"# Hybrid Memory\nHindsight: {status}.", local, remote) if x)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        local = self.holographic.prefetch(query, session_id=session_id)
        remote = self._safe_hindsight("prefetch", query, session_id=session_id, default="")
        return "\n\n".join(x for x in (local, remote) if x)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._safe_hindsight("queue_prefetch", query, session_id=session_id)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.holographic.sync_turn(user_content, assistant_content, session_id=session_id)
        self._safe_hindsight("sync_turn", user_content, assistant_content, session_id=session_id)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = list(self.holographic.get_tool_schemas())
        if self._hindsight_active:
            schemas.extend(self._safe_hindsight("get_tool_schemas", default=[]) or [])
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        local_names = {item["name"] for item in self.holographic.get_tool_schemas()}
        if tool_name in local_names:
            return self.holographic.handle_tool_call(tool_name, args, **kwargs)
        if self._hindsight_active:
            return self.hindsight.handle_tool_call(tool_name, args, **kwargs)
        return '{"error":"Hindsight is temporarily unavailable; Holographic memory remains active"}'

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self.holographic.on_turn_start(turn_number, message, **kwargs)
        self._safe_hindsight("on_turn_start", turn_number, message, **kwargs)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self.holographic.on_session_end(messages)
        self._safe_hindsight("on_session_end", messages)

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self.holographic.on_session_switch(new_session_id, **kwargs)
        self._safe_hindsight("on_session_switch", new_session_id, **kwargs)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        local = self.holographic.on_pre_compress(messages)
        remote = self._safe_hindsight("on_pre_compress", messages, default="")
        return "\n\n".join(x for x in (local, remote) if x)

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        self.holographic.on_memory_write(action, target, content)
        self._safe_hindsight("on_memory_write", action, target, content, metadata)

    def on_delegation(self, task: str, result: str, **kwargs) -> None:
        self.holographic.on_delegation(task, result, **kwargs)
        self._safe_hindsight("on_delegation", task, result, **kwargs)

    def backup_paths(self) -> List[str]:
        paths = self.holographic.backup_paths() + self.hindsight.backup_paths()
        return list(dict.fromkeys(paths))

    def shutdown(self) -> None:
        try:
            self._safe_hindsight("shutdown")
        finally:
            self.holographic.shutdown()


def register(ctx) -> None:
    ctx.register_memory_provider(HybridMemoryProvider())
