# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Venngram integration (#173): address books by store reference, pinned into checkpoints; Fine-Tuning Scheduler's
support record; and schedules exported as address books, checked by venngram's conformance check."""
import pytest
import torch
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback
from venngram import AddressBook
from venngram.address import parse as parse_address
from venngram.address import to_mapping
from venngram.conformance import check
from venngram.errors import PinMismatchError
from venngram.store import publish, Reference
from venngram.support import SupportRecord

from finetuning_scheduler import CanonicalView, FinetuningScheduler, TuningScheduler
from finetuning_scheduler.core.books import schedule_book
from finetuning_scheduler.core.compile import compile_schedule, referenced_sets, support_record, SUPPORT_RECORD_PATH
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.grad_transforms import Scale
from finetuning_scheduler.core.schedule import load_address_sets, same_documents
from tests.test_addresses import _lightning_module, ADDRESSED, adapter_roles, Blocks, SGD
from tests.test_legacy_adapter_contract import _trainer

TOP = {"schema_version": 1, "name": "lora", "sets": {"top": {"role": "adapter", "layers": -1}}}


@pytest.fixture(autouse=True)
def venngram_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VENNGRAM_CACHE", str(tmp_path / "venngram-cache"))


def publish_books(root, books):
    """Publish ``{path: book}`` to a local store at ``root``; return its location."""
    src = root.parent / f"{root.name}-src"
    for path, book in books.items():
        (src / path).parent.mkdir(parents=True, exist_ok=True)
        (src / path).write_text(yaml.safe_dump(book))
    publish("local", str(root), src, documents=[("address_book", p) for p in books])
    return f"local:{root}"


def scheduler(model, schedule):
    return TuningScheduler(model, schedule, optimizer=SGD, view=CanonicalView(model, roles=adapter_roles(model)))


def test_a_book_named_by_reference_is_fetched_pinned_and_recorded(tmp_path):
    location = publish_books(tmp_path / "pub", {"books/lora.yaml": TOP})
    model = Blocks()
    fts = scheduler(model, {"addresses": f"{location}:books/lora.yaml",
                            0: {"params": ["@top"], "max_transition_epoch": 1},
                            1: {"params": ["role:adapter - @top"]}})
    fts.setup()
    assert fts.compiled_schedule.phases[0].physical == ("blocks.3.adapter.weight",)
    (pinned,) = fts.state_dict()["address_references"]
    ref = Reference.parse(pinned)
    assert (ref.store, ref.path) == ("local", "books/lora.yaml") and ref.sha256 is not None


def test_references_mix_with_paths_and_imports_by_reference_are_recorded(tmp_path):
    location = publish_books(tmp_path / "pub", {"books/lora.yaml": TOP})
    (tmp_path / "local.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "imports": {"lora": f"{location}:books/lora.yaml"},
        "sets": {"rest": {"role": "adapter", "exclude": "@lora.top"}}}))
    other = publish_books(tmp_path / "other", {"mid.yaml": {"schema_version": 1, "sets": {"mid": {"layers": 1}}}})
    sets = load_address_sets(["local.yaml", f"{other}:mid.yaml"], tmp_path)
    assert {"rest", "lora.top", "mid"} <= set(sets)
    assert [Reference.parse(r).path for r in sets.references] == ["books/lora.yaml", "mid.yaml"]


def test_a_resume_refuses_books_that_changed_since_the_save(tmp_path):
    location = publish_books(tmp_path / "pub", {"books/lora.yaml": TOP})
    schedule = {"addresses": f"{location}:books/lora.yaml", 0: {"params": ["@top"], "max_transition_epoch": 1},
                1: {"params": ["role:adapter - @top"]}}
    model = Blocks()
    fts = scheduler(model, schedule)
    fts.setup()
    saved = fts.state_dict()
    again = scheduler(model, schedule)
    again.setup()
    again.load_state_dict(saved)  # the same documents
    publish_books(tmp_path / "pub", {"books/lora.yaml": {**TOP, "description": "republished, changed"}})
    changed = scheduler(model, schedule)
    changed.setup()
    with pytest.raises(FTSConfigurationError, match="Cite the saved references"):
        changed.load_state_dict(saved)
    # documents compare by content: the same document found at another revision is the same document
    (pinned,) = saved["address_references"]
    ref = Reference.parse(pinned)
    elsewhere = str(Reference("hub-model", "me/books", ref.path, "0" * 40, ref.sha256))
    assert same_documents([pinned], [pinned]) and not same_documents([pinned], [elsewhere])
    moved = Reference("hub-model", "me/books", ref.path, "1" * 40, ref.sha256)
    assert same_documents([elsewhere], [str(moved)])


class _SaveState(Callback):
    def on_train_end(self, trainer, pl_module):
        self.state = trainer.callbacks[0].state_dict()


def test_the_lightning_callback_records_book_references(tmp_path):
    seed_everything(42)
    location = publish_books(tmp_path / "pub", {"lora.yaml": TOP})
    fts = FinetuningScheduler(ft_schedule={"addresses": [f"{location}:lora.yaml"],
                                           0: {"params": ["@top"], "max_transition_epoch": 1},
                                           1: {"params": ["role:adapter - @top"], "max_transition_epoch": 2}},
                              epoch_transitions_only=True)
    saver = _SaveState()
    _trainer(tmp_path, [fts, saver], max_epochs=1).fit(_lightning_module())
    (pinned,) = saver.state["address_references"]
    assert Reference.parse(pinned).path == "lora.yaml"
    fts.load_state_dict(saver.state)  # the same documents
    with pytest.raises(FTSConfigurationError, match="Cite the saved references"):
        fts.load_state_dict({**saver.state, "address_references": [pinned.replace(pinned[-4:], "0000")]})


def test_the_support_record_is_shipped_as_data_and_refuses_what_is_not_tuned():
    record = support_record()
    assert record == SupportRecord.from_file(SUPPORT_RECORD_PATH) and record.name == "finetuning-scheduler"
    assert record.tuning is not None and record.tuning.targets == frozenset({"param"})
    book = {"sets": {"top": {"role": "adapter", "layers": -1}, "probe": {"point": "resid_post"},
                     "uses_probe": "@top | @probe"}}
    model = Blocks()
    view = CanonicalView(model, roles=adapter_roles(model))
    # a set the schedule does not use is not judged, so a book shared with a steering consumer still applies
    compiled = compile_schedule({0: {"params": ["@top"]}}, view, addresses=load_address_sets(book))
    assert compiled.phases[0].physical == ("blocks.3.adapter.weight",)
    with pytest.raises(FTSConfigurationError, match=r"(?s)does not tune:\n.*set `probe`: it selects a `site`"):
        compile_schedule({0: {"params": ["@uses_probe"]}}, view, addresses=load_address_sets(book))
    assert referenced_sets({0: {"params": ["@a - @b", "head.weight"]}, 1: {"params": ["x"], "promote": {"@c": "l"}}}) \
        == {"a", "b", "c"}


def test_a_schedule_exports_as_a_book_that_selects_what_it_scheduled(tmp_path):
    model = Blocks()
    # a listed name and a regex (matched as a prefix), beside ADDRESSED's addresses and named sets
    schedule = {**ADDRESSED, 3: {"params": ["blocks.0.proj.weight", r"blocks\.(0\.proj\.b|1\.proj)"],
                                 "max_transition_epoch": 4}}
    fts = scheduler(model, schedule)
    fts.setup()
    book = fts.export_book(tmp_path / "exported.yaml", description="four phases")
    assert list(book["sets"]) == ["phase_0", "phase_1", "phase_2", "phase_3"]
    assert book["requires"] == {"naming_scheme": "physical"} and book["description"] == "four phases"
    assert yaml.safe_load((tmp_path / "exported.yaml").read_text()) == book
    # `@top` and `@rest` are inlined, so the book stands alone
    assert "@" not in yaml.safe_dump(book["sets"])
    exported = load_address_sets(str(tmp_path / "exported.yaml"))
    replay = compile_schedule({d: {"params": [f"@phase_{d}"]} for d in range(4)},
                              CanonicalView(model, roles=adapter_roles(model)), addresses=exported)
    assert [cp.physical for cp in replay.phases] == [cp.physical for cp in fts.compiled_schedule.phases]
    with pytest.raises(FTSConfigurationError, match="Phase 0 has no `params`"):
        schedule_book({0: {}}, name="empty")


def test_a_book_exported_from_fetched_sets_stands_alone_and_cites_them(tmp_path):
    """Sets fetched by reference, one guarded by `expect` and one a union referencing it, are inlined with their
    modifiers, so the exported book selects what the schedule did, and it cites the books it derives from."""
    guarded = {"schema_version": 1, "name": "lora",
               "sets": {"top": {"role": "adapter", "layers": -1, "expect": 1},
                        "upper": "@top | (role:adapter & layers:2)"}}
    location = publish_books(tmp_path / "pub", {"books/lora.yaml": guarded})
    sets = load_address_sets(f"{location}:books/lora.yaml")
    schedule = {0: {"params": ["@upper"]}, 1: {"params": ["role:adapter - @upper"]}}
    book = schedule_book(schedule, name="blocks", addresses=sets, applies_to={"architectures": ["Blocks"]})
    assert book["applies_to"] == {"architectures": ["Blocks"]}
    assert book["derived_from"] == list(sets.references) and len(book["derived_from"]) == 1
    assert "@" not in yaml.safe_dump(book["sets"])
    assert {"role": "adapter", "layers": -1, "expect": 1} in book["sets"]["phase_0"]["any"]
    model = Blocks()
    view = CanonicalView(model, roles=adapter_roles(model))
    replay = compile_schedule({0: {"params": ["@phase_0"]}, 1: {"params": ["@phase_1"]}}, view,
                              addresses=load_address_sets(book))
    original = compile_schedule(schedule, view, addresses=sets)
    assert [cp.physical for cp in replay.phases] == [cp.physical for cp in original.phases]
    assert replay.phases[0].physical == ("blocks.2.adapter.weight", "blocks.3.adapter.weight")


@pytest.mark.parametrize("schedule, addresses, match", [
    ({0: {"params": ["@missing"]}}, None, "`@missing` names no set of the schedule's `addresses`"),
    # the book refuses cycles when it loads; an assembled mapping is checked as it is inlined
    ({0: {"params": ["@a"]}}, {"a": parse_address("@b"), "b": parse_address("@a")}, "form a cycle through `@a`"),
    ({0: {"params": [{"role": "adapter"}]}}, None, r"Phase 0's `params` entry \{'role': 'adapter'\} is not a string"),
], ids=["unknown_set", "cycle", "not_a_string"])
def test_a_schedule_that_cannot_export_as_a_book_is_refused(schedule, addresses, match):
    with pytest.raises(FTSConfigurationError, match=match):
        schedule_book(schedule, name="refused", addresses=addresses)


def test_exported_books_pass_venngram_conformance_against_the_support_record(tmp_path):
    model = Blocks()
    fts = scheduler(model, ADDRESSED)
    fts.setup()
    src = tmp_path / "src"
    src.mkdir()
    (src / "blocks.yaml").write_text(yaml.safe_dump(fts.export_book(name="blocks")))
    (src / "support.yaml").write_text(SUPPORT_RECORD_PATH.read_text())
    (src / "probe.yaml").write_text(yaml.safe_dump({"schema_version": 1, "sets": {"probe": {"point": "resid_post"}}}))
    publish("local", str(tmp_path / "pub"), src, documents=[("address_book", "blocks.yaml"),
                                                           ("address_book", "probe.yaml"),
                                                           ("support_record", "support.yaml")])
    report = check(f"local:{tmp_path / 'pub'}", supports=[support_record()])
    results = {d.path: d for d in report.documents}
    assert results["blocks.yaml"].ok and results["support.yaml"].ok
    assert not results["probe.yaml"].ok  # a site set is not something a schedule tunes
    assert AddressBook.from_file(src / "blocks.yaml").name == "blocks"


def test_a_book_reference_to_another_kind_is_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "support.yaml").write_text(SUPPORT_RECORD_PATH.read_text())
    publish("local", str(tmp_path / "pub"), src, documents=[("support_record", "support.yaml")])
    with pytest.raises(FTSConfigurationError, match="names a `support_record` document, not an address book"):
        load_address_sets(f"local:{tmp_path / 'pub'}:support.yaml")


def test_a_book_reference_that_does_not_fetch_is_refused_before_anything_changes(tmp_path):
    model = Blocks()
    before = {n: (p.detach().clone(), p.requires_grad) for n, p in model.named_parameters()}
    fts = scheduler(model, {"addresses": f"local:{tmp_path / 'nowhere'}:lora.yaml", 0: {"params": ["@top"]}})
    with pytest.raises(FTSConfigurationError, match="cannot be loaded"):
        fts.setup()
    assert all(torch.equal(before[n][0], p) and before[n][1] == p.requires_grad for n, p in model.named_parameters())


BASE = {"id": "org/base", "revision": "c5ebcd4"}
SAE = {"release": {"sae_lens": "res-jb", "id": "layer_3"}}
PINNED = {"schema_version": 8, "name": "pinned", "requires": {"base_model": BASE, "latent_models": {"sae": SAE}},
          "sets": {"top": {"role": "adapter", "layers": -1}, "probe": {"point": "resid_post"},
                   "rows": {"role": "adapter", "layers": -1, "units": [0, 2]}}}
LOADED = {"base_model": BASE, "latent_models": {"sae": SAE}}


class PinCheckingView(CanonicalView):
    """A view that knows what its model loaded, and records the books it is asked to check."""

    def __init__(self, model, loaded=LOADED):
        super().__init__(model, roles=adapter_roles(model))
        self.loaded, self.checked = loaded, []

    def check_book(self, book):
        self.checked.append(book.name)
        book.check_pins(**self.loaded)


@pytest.fixture
def imports_pinned(tmp_path):
    """A book that pins nothing itself and imports ``PINNED``."""
    (tmp_path / "pinned.yaml").write_text(yaml.safe_dump(PINNED))
    return {"schema_version": 1, "name": "importer", "imports": {"p": str(tmp_path / "pinned.yaml")},
            "sets": {"top": "@p.top"}}


def test_a_pinned_book_drives_a_schedule_through_a_view_that_checks_its_pins(imports_pinned):
    for book in (PINNED, imports_pinned):
        view = PinCheckingView(Blocks())
        compiled = compile_schedule({0: {"params": ["@top"]}}, view, addresses=load_address_sets(book))
        assert compiled.phases[0].physical == ("blocks.3.adapter.weight",)
        assert view.checked == [book["name"]]
    # the sets an `of: grad` entry uses are judged with the pins checked too
    compiled = compile_schedule({0: {"params": ["@rows & of:grad"]}}, PinCheckingView(Blocks()),
                                addresses=load_address_sets(PINNED))
    assert compiled.phases[0].masks == {"blocks.3.adapter.weight": (0, 2)}
    # a checked book is still held to everything else the support record declares
    with pytest.raises(FTSConfigurationError, match=r"(?s)does not tune:\n.*set `probe`: it selects a `site`"):
        compile_schedule({0: {"params": ["@probe"]}}, PinCheckingView(Blocks()), addresses=load_address_sets(PINNED))


@pytest.mark.parametrize("loaded, match", [
    ({**LOADED, "base_model": {**BASE, "revision": "0000000"}}, "`requires.base_model`: .*org/base@0000000"),
    ({**LOADED, "latent_models": {}}, "latent model `sae`, which is not loaded"),
    ({**LOADED, "latent_models": {"sae": {"release": {**SAE["release"], "id": "layer_4"}}}},
     r"`requires.latent_models.sae.release.id`: pinned `layer_3`, loaded `layer_4`"),
], ids=["revision", "not_loaded", "release"])
def test_a_book_whose_pins_differ_from_what_the_view_loaded_is_refused(loaded, match, imports_pinned):
    for book in (PINNED, imports_pinned):
        with pytest.raises(FTSConfigurationError, match=f"do not apply through this view: .*{match}") as err:
            compile_schedule({0: {"params": ["@top"]}}, PinCheckingView(Blocks(), loaded),
                             addresses=load_address_sets(book))
        assert isinstance(err.value.__cause__, PinMismatchError)


def test_a_pinned_book_is_refused_through_a_view_that_checks_no_pins(imports_pinned):
    model = Blocks()
    valued = CanonicalView(model, roles=adapter_roles(model))
    valued.check_book = True  # a member that cannot be called checks nothing
    for view in (CanonicalView(model, roles=adapter_roles(model)), valued):
        for book in (PINNED, imports_pinned):
            for entry in ("@top", "@top & of:grad"):
                with pytest.raises(FTSConfigurationError, match="(?s)`pins: false`.*CanonicalView defines no"):
                    compile_schedule({0: {"params": [entry]}}, view, addresses=load_address_sets(book))


def test_every_pinned_book_of_a_schedule_is_checked_and_no_other(tmp_path):
    other = {"schema_version": 8, "name": "other", "requires": {"base_model": {**BASE, "revision": "fffffff"}},
             "sets": {"mid": {"role": "adapter", "layers": 1}}}
    for name, book in {"top": TOP, "pinned": {**PINNED, "sets": {"p": "role:adapter"}}, "other": other}.items():
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(book))
    view = PinCheckingView(Blocks())
    compile_schedule({0: {"params": ["@top"]}}, view,
                     addresses=load_address_sets(["top.yaml", "pinned.yaml"], tmp_path))
    assert view.checked == ["pinned"]
    # the schedule's own phases use nothing of `other`, and its pins bind all the same
    with pytest.raises(FTSConfigurationError, match="org/base@fffffff"):
        compile_schedule({0: {"params": ["@top"]}}, view,
                         addresses=load_address_sets(["top.yaml", "pinned.yaml", "other.yaml"], tmp_path))


def test_both_interfaces_check_a_schedules_pins_through_the_view(tmp_path):
    schedule = {"addresses": PINNED, 0: {"params": ["@top"], "max_transition_epoch": 1},
                1: {"params": ["role:adapter - @top"], "max_transition_epoch": 2}}
    stale = {**LOADED, "base_model": {**BASE, "revision": "0000000"}}

    def core(loaded):
        model = Blocks()
        fts = TuningScheduler(model, schedule, optimizer=SGD, view=PinCheckingView(model, loaded))
        fts.setup()
        return fts.compiled_schedule

    def lightning(loaded):
        seed_everything(42)
        module = _lightning_module()
        module.tunable_view = PinCheckingView(module, loaded)
        fts = FinetuningScheduler(ft_schedule=schedule, epoch_transitions_only=True)
        _trainer(tmp_path, [fts], max_epochs=1).fit(module)
        return fts.strategy_adapter.compiled_schedule

    for run in (core, lightning):
        assert [p.rpartition("blocks.")[2] for p in run(LOADED).phases[0].physical] == ["3.adapter.weight"]
        with pytest.raises(FTSConfigurationError, match="org/base@0000000"):
            run(stale)


def test_a_book_exported_from_pinned_books_pins_the_same_models(tmp_path):
    model = Blocks()
    fts = TuningScheduler(model, {"addresses": PINNED, 0: {"params": ["@top"], "max_transition_epoch": 1},
                                  1: {"params": ["head.weight"]}}, optimizer=SGD, view=PinCheckingView(model))
    fts.setup()
    book = fts.export_book(tmp_path / "exported.yaml")
    assert book["schema_version"] == 8
    assert book["requires"] == {"naming_scheme": "physical", **PINNED["requires"]}
    exported = load_address_sets(str(tmp_path / "exported.yaml"))
    replay = {0: {"params": ["@phase_0"]}, 1: {"params": ["@phase_1"]}}
    back = compile_schedule(replay, PinCheckingView(model), addresses=exported)
    assert [cp.physical for cp in back.phases] == [cp.physical for cp in fts.compiled_schedule.phases]
    with pytest.raises(FTSConfigurationError, match="defines no `check_book`"):
        compile_schedule(replay, CanonicalView(model, roles=adapter_roles(model)), addresses=exported)
    # the latent models of several books are pinned together, and books that pin one model differently are refused
    tc = {"release": {"hub": "org/tc", "revision": "9ab1f0e"}}
    for requires, match in (({"latent_models": {"tc": tc}}, None),
                            ({"base_model": {**BASE, "revision": "1111111"}}, "pin `base_model` differently"),
                            ({"latent_models": {"sae": tc}}, "pin `latent_models.sae` differently")):
        (tmp_path / "pinned.yaml").write_text(yaml.safe_dump(PINNED))
        (tmp_path / "other.yaml").write_text(yaml.safe_dump(
            {"schema_version": 8, "requires": {"base_model": BASE, **requires}, "sets": {"a": "layers:0"}}))
        sets = load_address_sets(["pinned.yaml", "other.yaml"], tmp_path)
        if match is None:
            pins = schedule_book({0: {"params": ["@top | @a"]}}, name="two", addresses=sets)["requires"]
            assert pins == {"base_model": BASE, "latent_models": {"sae": SAE, "tc": tc}}
            continue
        with pytest.raises(FTSConfigurationError, match=match):
            schedule_book({0: {"params": ["@top | @a"]}}, name="two", addresses=sets)


@pytest.mark.parametrize("entry, schema", [
    ("role:adapter", 1), ("role:adapter & of:grad", 2), ("role:adapter & layers:0.5d:", 7),
    ({"role": "adapter", "units": [0], "axis": 1, "of": "grad"}, 6),
], ids=["plain", "of", "depth", "axis"])
def test_an_exported_book_declares_the_schema_its_sets_need_and_loads(entry, schema):
    addresses = load_address_sets({"sets": {"s": entry}})
    book = schedule_book({0: {"params": ["@s"]}}, name="exported", addresses=addresses)
    assert book["schema_version"] == schema
    loaded = AddressBook.from_mapping(book, source="exported").sets()
    assert to_mapping(loaded["phase_0"]) == to_mapping(addresses["s"])


def test_a_schedule_whose_book_would_not_load_is_refused():
    scaled = {0: {"params": ["role:adapter"], "grad_transforms": {"role:adapter": Scale(0.5)}}}
    with pytest.raises(FTSConfigurationError, match="The exported book does not load: .*revision"):
        schedule_book(scaled, name="refused", model={"id": "org/base"})  # a treatment's model states its revision
