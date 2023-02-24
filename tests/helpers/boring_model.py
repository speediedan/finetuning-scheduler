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
# Initially based on https://bit.ly/3oQ8Vqf
import re
from functools import partial
from typing import List, Optional, Tuple
from warnings import WarningMessage

import torch
from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset


def multiwarn_check(
    rec_warns: List, expected_warns: List, expected_mode: bool = False
) -> List[Optional[WarningMessage]]:
    msg_search = lambda w1, w2: re.compile(w1).search(w2.message.args[0])  # noqa: E731
    if expected_mode:  # we're directed to check that multiple expected warns are obtained
        return [w_msg for w_msg in expected_warns if not any([msg_search(w_msg, w) for w in rec_warns])]
    else:  # by default we're checking that no unexpected warns are obtained
        return [w_msg for w_msg in rec_warns if not any([msg_search(w, w_msg) for w in expected_warns])]


unexpected_warns = partial(multiwarn_check, expected_mode=False)


unmatched_warns = partial(multiwarn_check, expected_mode=True)


class LinearWarmupLR(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        super().__init__(optimizer, lr_lambda, last_epoch)


class CustomLRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, epoch):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...


class RandomDictDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {"a": a, "b": b}

    def __len__(self):
        return self.len


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class RandomIterableDataset(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):
    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self):
        return self.count


class BoringModel(LightningModule):
    def __init__(self):
        """Testing PL Module.

        Use as follows:
        - subclass
        - modify the behavior for what you want

        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing

        or:

        model = BaseTestModel()
        model.training_step_end = None  # disable hook
        """
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def loss(self, preds: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if labels is None:
            labels = torch.ones_like(preds)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(preds, labels)

    def step(self, batch: Tensor) -> Tensor:
        output = self(batch)
        return self.loss(output)

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        return {"loss": self.step(batch)}

    # def training_step(self, batch, batch_idx):
    #     output = self(batch)
    #     loss = self.loss(batch, output)
    #     return {"loss": loss}

    def training_step_end(self, training_step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        return training_step_output

    def validation_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return {"x": self.step(batch)}

    # def validation_step(self, batch, batch_idx):
    #     output = self(batch)
    #     loss = self.loss(batch, output)
    #     return {"x": loss}

    # def on_validation_epoch_end(self, outputs) -> None:
    #     torch.stack([x["x"] for x in outputs]).mean()

    # def test_step(self, batch, batch_idx):
    #     output = self(batch)
    #     loss = self.loss(batch, output)
    #     return {"y": loss}

    def test_step(self, batch: Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return {"y": self.step(batch)}

    # def test_epoch_end(self, outputs) -> None:
    #     torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[_TORCH_LRSCHEDULER]]:
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class BoringDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.random_train = Subset(self.random_full, indices=range(64))

        if stage in ("fit", "validate") or stage is None:
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))

        if stage == "test" or stage is None:
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))

        if stage == "predict" or stage is None:
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))

    def train_dataloader(self):
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)

    def predict_dataloader(self):
        return DataLoader(self.random_predict)


class ManualOptimBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    # def training_step(self, batch, batch_idx):
    #     opt = self.optimizers()
    #     output = self(batch)
    #     loss = self.loss(batch, output)
    #     opt.zero_grad()
    #     self.manual_backward(loss)
    #     opt.step()
    #     return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        opt = self.optimizers()
        assert isinstance(opt, (Optimizer, LightningOptimizer))
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss
