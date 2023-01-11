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

from typing import Set, Type

import torch
from packaging.version import Version

from fts_examples import _HF_AVAILABLE
from fts_examples.cli_experiment_utils import instantiate_class
from fts_examples.fts_superglue import RteBoolqModule

if _HF_AVAILABLE:
    from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Embeddings, DebertaV2Encoder, DebertaV2Layer

deberta_transformer_layer_cls: Set = {DebertaV2Layer, DebertaV2Embeddings, DebertaV2Encoder}


def deberta_awp(
    module: torch.nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Set[Type[torch.nn.Module]] = deberta_transformer_layer_cls,
) -> bool:
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or remainder
        return isinstance(module, tuple(transformer_layer_cls))


class RteBoolqModuleFSDP(RteBoolqModule):
    # we override `configure_optimizers` because use of the `no_decay` lightning module attribute is not currently
    # supported with FTS FSDP strategy adapter
    def configure_optimizers(self):
        if Version(torch.__version__) == Version("1.12.0") or torch.__version__.startswith("1.12.0"):
            # we need to use a patched version of AdamW to fix https://github.com/pytorch/pytorch/issues/80809
            # and allow examples to succeed with torch 1.12.0 (this torch bug is fixed in 1.12.1)
            self.hparams.optimizer_init["class_path"] = "fts_examples.patched_adamw.AdamW"
        parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = instantiate_class(args=parameters, init=self.hparams.optimizer_init)
        scheduler = {
            "scheduler": instantiate_class(args=optimizer, init=self.hparams.lr_scheduler_init),
            **self.hparams.pl_lrs_cfg,
        }
        return [optimizer], [scheduler]
