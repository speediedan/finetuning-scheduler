from finetuning_scheduler.strategy_adapters import StrategyAdapter


class FakeAdapter(StrategyAdapter):
    """A simple fake Strategy Adapter class for entry point discovery tests.

    Exists solely so entry-points pointing into the package can be imported in unit tests. It subclasses
    :class:`StrategyAdapter` because adapter resolution refuses targets that do not.
    """
    pass
