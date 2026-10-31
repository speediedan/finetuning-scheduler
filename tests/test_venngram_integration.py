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
from venngram.conformance import check
from venngram.store import publish, Reference
from venngram.support import SupportRecord

from finetuning_scheduler import CanonicalView, FinetuningScheduler, TuningScheduler
from finetuning_scheduler.core.books import schedule_book
from finetuning_scheduler.core.compile import compile_schedule, referenced_sets, support_record, SUPPORT_RECORD_PATH
from finetuning_scheduler.core.exceptions import FTSConfigurationError
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
