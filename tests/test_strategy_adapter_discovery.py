from importlib.metadata import EntryPoint
from unittest.mock import MagicMock
import pytest

from finetuning_scheduler import fts_supporters
from finetuning_scheduler.fts import FinetuningScheduler
from finetuning_scheduler.fts_supporters import STRATEGY_ADAPTERS
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_discover_strategy_adapters(monkeypatch):
    # Create fake entrypoints that point to a plugin adapter using two common formats
    # Use a local fake adapter class for discovery tests to avoid importing external packages
    ep_colon = EntryPoint(
        name="fake_adapter_colon",
        value="tests.helpers.fake_adapter:FakeAdapter",
        group="finetuning_scheduler.strategy_adapters",
    )
    ep_dot = EntryPoint(
        name="fake_adapter_dot",
        value="tests.helpers.fake_adapter.FakeAdapter",
        group="finetuning_scheduler.strategy_adapters",
    )

    def fake_entry_points(group=None):
        if group == "finetuning_scheduler.strategy_adapters":
            return [ep_colon, ep_dot]
        return []

    # Override the entry_points function in the fts_supporters module.
    # Monkeypatch the importlib.metadata.entry_points function so our fake entrypoints are used by discovery
    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    # call discovery explicitly (no reload, since entry_points function is patched)
    # call the function again - it will load entrypoints and register
    fts_supporters._discover_strategy_adapters()
    assert "fake_adapter_colon" in fts_supporters.STRATEGY_ADAPTERS
    assert "fake_adapter_dot" in fts_supporters.STRATEGY_ADAPTERS


def test_resolve_strategy_adapter_by_qualname():
    adapter_map = {"single_device": "tests.helpers.fake_adapter:FakeAdapter"}
    fts = FinetuningScheduler()
    cls = fts._resolve_strategy_adapter("single_device", adapter_map)
    assert cls.__name__ == "FakeAdapter"


def test_resolve_strategy_adapter_by_dot_form():
    adapter_map = {"single_device": "tests.helpers.fake_adapter.FakeAdapter"}
    fts = FinetuningScheduler()
    cls = fts._resolve_strategy_adapter("single_device", adapter_map)
    assert cls.__name__ == "FakeAdapter"


def test_resolve_strategy_adapter_by_plugin_name():
    """Test importing strategy adapter using discovered plugin entry point name."""
    # register alias in STRATEGY_ADAPTERS to simulate a discovered plugin
    STRATEGY_ADAPTERS["fakeplugin"] = STRATEGY_ADAPTERS.get("fsdp")
    adapter_map = {"single_device": "fakeplugin"}
    fts = FinetuningScheduler()
    cls = fts._resolve_strategy_adapter("single_device", adapter_map)
    assert cls in STRATEGY_ADAPTERS.values()


def test_discover_strategy_adapters_ep_load_failure(monkeypatch):
    """Test that ep.load() failure triggers fallback import and succeeds."""
    # Create a mock entry point that will fail on ep.load() but succeed with fallback
    ep = MagicMock()
    ep.name = "fallback_adapter"
    ep.value = "tests.helpers.fake_adapter:FakeAdapter"
    ep.load.side_effect = ImportError("Simulated ep.load() failure")

    def fake_entry_points(group=None):
        if group == "finetuning_scheduler.strategy_adapters":
            return [ep]
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

    # Should succeed via fallback import despite ep.load() failing
    fts_supporters._discover_strategy_adapters()
    assert "fallback_adapter" in fts_supporters.STRATEGY_ADAPTERS


def test_discover_strategy_adapters_both_imports_fail(monkeypatch):
    """Test that both ep.load() and fallback import failures are handled gracefully."""
    # Create a mock entry point with invalid value that will fail both import methods
    ep = MagicMock()
    ep.name = "broken_adapter"
    ep.value = "nonexistent.module:NonexistentClass"
    ep.load.side_effect = ImportError("Simulated ep.load() failure")

    def fake_entry_points(group=None):
        if group == "finetuning_scheduler.strategy_adapters":
            return [ep]
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

    # Should not raise, just log warnings and continue
    with pytest.warns(UserWarning, match="Failed to load strategy adapter entry point"):
        fts_supporters._discover_strategy_adapters()

    # Adapter should not be registered
    assert "broken_adapter" not in fts_supporters.STRATEGY_ADAPTERS


def test_discover_strategy_adapters_no_value_attribute(monkeypatch):
    """Test handling of entry point with missing value attribute."""
    # Create a mock entry point without a value attribute
    ep = MagicMock()
    ep.name = "no_value_adapter"
    ep.load.side_effect = AttributeError("No value attribute")
    # Simulate missing value attribute
    type(ep).value = property(lambda self: None)

    def fake_entry_points(group=None):
        if group == "finetuning_scheduler.strategy_adapters":
            return [ep]
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

    # Should handle gracefully and log warning
    with pytest.warns(UserWarning, match="Failed to load strategy adapter entry point"):
        fts_supporters._discover_strategy_adapters()

    assert "no_value_adapter" not in fts_supporters.STRATEGY_ADAPTERS


def test_resolve_strategy_adapter_invalid_format():
    """Test that invalid adapter name format raises MisconfigurationException."""
    # Test with a name that has no dots or colons (invalid format)
    adapter_map = {"single_device": "invalidname"}
    fts = FinetuningScheduler()

    with pytest.raises(
        MisconfigurationException,
        match=r"Invalid adapter name 'invalidname'.*Must be either a registered plugin name.*or a fully qualified",
    ):
        fts._resolve_strategy_adapter("single_device", adapter_map)


def test_resolve_strategy_adapter_import_error():
    """Test that import errors are properly handled and re-raised as MisconfigurationException."""
    adapter_map = {"single_device": "nonexistent.module:NonexistentClass"}
    fts = FinetuningScheduler()

    with pytest.raises(
        MisconfigurationException,
        match=r"Could not import the specified custom strategy adapter class.*nonexistent\.module:NonexistentClass",
    ):
        fts._resolve_strategy_adapter("single_device", adapter_map)


def test_resolve_strategy_adapter_missing_strategy_key():
    """Test that missing strategy key in adapter_map raises MisconfigurationException."""
    adapter_map = {"ddp": "some_adapter"}  # Missing "single_device" key
    fts = FinetuningScheduler()

    with pytest.raises(
        MisconfigurationException,
        match=r"Current strategy name \(single_device\) does not map to a custom strategy adapter",
    ):
        fts._resolve_strategy_adapter("single_device", adapter_map)
