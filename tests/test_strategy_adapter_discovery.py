from importlib.metadata import EntryPoint

from finetuning_scheduler import fts_supporters
from finetuning_scheduler.fts import FinetuningScheduler
from finetuning_scheduler.fts_supporters import STRATEGY_ADAPTERS


def test_discover_strategy_adapters(monkeypatch):
    # Create fake entrypoints that point to the Interpretune plugin adapter using two common formats
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


def test_import_strategy_adapter_by_qualname():
    adapter_map = {"single_device": "tests.helpers.fake_adapter:FakeAdapter"}
    fts = FinetuningScheduler()
    cls = fts._import_strategy_adapter("single_device", adapter_map)
    assert cls.__name__ == "FakeAdapter"


def test_import_strategy_adapter_by_dot_form():
    adapter_map = {"single_device": "tests.helpers.fake_adapter.FakeAdapter"}
    fts = FinetuningScheduler()
    cls = fts._import_strategy_adapter("single_device", adapter_map)
    assert cls.__name__ == "FakeAdapter"


def test_import_strategy_adapter_by_shortname_alias():
    # register alias in STRATEGY_ADAPTERS
    STRATEGY_ADAPTERS["fakealias"] = STRATEGY_ADAPTERS.get("fsdp")
    adapter_map = {"single_device": "fakealias"}
    fts = FinetuningScheduler()
    cls = fts._import_strategy_adapter("single_device", adapter_map)
    assert cls in STRATEGY_ADAPTERS.values()
