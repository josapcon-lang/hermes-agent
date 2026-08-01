from unittest.mock import MagicMock

from plugins.memory.hybrid import HybridMemoryProvider


def test_hindsight_failure_keeps_holographic_available():
    provider = HybridMemoryProvider()
    provider.holographic = MagicMock()
    provider.holographic.is_available.return_value = True
    provider.holographic.prefetch.return_value = "local"
    provider.hindsight = MagicMock()
    provider.hindsight.is_available.side_effect = RuntimeError("offline")

    provider.initialize("s1")

    assert provider._hindsight_active is False
    assert provider.prefetch("query") == "local"


def test_combines_recall_when_both_are_active():
    provider = HybridMemoryProvider()
    provider.holographic = MagicMock()
    provider.holographic.prefetch.return_value = "local"
    provider.hindsight = MagicMock()
    provider.hindsight.prefetch.return_value = "long-term"
    provider._hindsight_active = True

    assert provider.prefetch("query") == "local\n\nlong-term"
