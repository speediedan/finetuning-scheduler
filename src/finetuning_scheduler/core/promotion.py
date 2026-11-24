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
r"""
Fine-Tuning Scheduler Promotion Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A promoted tensor's value when it was promoted, as each process keeps it: its own shard, on the CPU, when the live
tensor is a ``DTensor`` (FSDP2 or tensor parallelism), the whole value otherwise. A checkpoint saved before the
promotion lacks the tensor, so restoring one puts this value back, without any process holding a tensor it has only a
shard of.
"""
from collections.abc import Iterable, Mapping

import torch
from torch import nn, Tensor


def _is_dtensor(t: Tensor) -> bool:
    return type(t).__name__ == "DTensor"


def local_value(t: Tensor) -> Tensor:
    """A CPU copy of the part of ``t`` this process holds."""
    t = t.detach()
    return (t.to_local() if _is_dtensor(t) else t).to("cpu", copy=True)  # type: ignore[attr-defined]


def live_tensors(model: nn.Module) -> dict[str, Tensor]:
    """The model's parameters and buffers by name, aliases of a tied tensor included."""
    return {**dict(model.named_buffers(remove_duplicate=False)),
            **dict(model.named_parameters(remove_duplicate=False))}


@torch.no_grad()
def put_back(t: Tensor, value: Tensor) -> None:
    """Copy ``value`` (a :func:`local_value`) into the part of ``t`` this process holds."""
    t = t.detach()
    (t.to_local() if _is_dtensor(t) else t).copy_(value)  # type: ignore[attr-defined]


def in_layout_of(t: Tensor, value: Tensor) -> Tensor:
    """``value`` (a :func:`local_value` of ``t``) as a tensor laid out as ``t`` is: a ``DTensor`` of the same mesh
    and placements when ``t`` is one, ``value`` itself otherwise."""
    if not _is_dtensor(t):
        return value
    from torch.distributed.tensor import DTensor

    assert isinstance(t, DTensor)
    # the mesh's device rather than the shard's: FSDP2's CPU offload keeps the shards of a GPU mesh on the CPU
    return DTensor.from_local(value.to(t.device_mesh.device_type, copy=True), t.device_mesh, t.placements,
                              shape=t.shape, stride=t.stride())


def full_value(t: Tensor, value: Tensor) -> Tensor:
    """The whole of ``value`` (a :func:`local_value` of ``t``) on the CPU (collective when ``t`` is a
    ``DTensor``)."""
    laid_out = in_layout_of(t, value)
    return laid_out.full_tensor().to("cpu") if _is_dtensor(laid_out) else value  # type: ignore[attr-defined]


def tied_names(model: nn.Module, names: Iterable[str]) -> dict[str, list[str]]:
    """Each of ``names`` that shares its tensor with other names, mapped to them.

    A ``DTensor`` is compared by identity, since sharding gives distinct tensors storage of their own (and a process's
    shard may be empty); any other tensor by its storage.
    """
    def key(t: Tensor) -> object:
        return id(t) if _is_dtensor(t) or t.device.type == "meta" else t.untyped_storage().data_ptr() or id(t)

    live: Mapping[str, Tensor] = live_tensors(model)
    holders: dict[object, list[str]] = {}
    for n, t in live.items():
        holders.setdefault(key(t), []).append(n)
    tied = {n: sorted(h for h in holders[key(live[n])] if h != n) for n in names}
    return {n: others for n, others in tied.items() if others}
