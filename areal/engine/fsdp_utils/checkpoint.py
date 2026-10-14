# SPDX-License-Identifier: Apache-2.0

"""FSDP checkpointing utilities for DCP (Distributed Checkpoint) integration."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

LR_SCHEDULER_STATE_FILENAME = "lr_scheduler.pt"
DCP_PREFETCH_CHUNK_BYTES = 16 * 1024 * 1024


def prefetch_dcp_shard(checkpoint_dir: str, rank: int) -> tuple[Path, int] | None:
    """Read one rank-local DCP shard sequentially to warm the filesystem cache."""

    root = Path(checkpoint_dir)
    shard = root / f"__{rank}_0.distcp"
    if not shard.is_file():
        shards = sorted(root.glob("__*_0.distcp"))
        if not shards:
            return None
        shard = shards[rank % len(shards)]

    bytes_read = 0
    buffer = bytearray(DCP_PREFETCH_CHUNK_BYTES)
    with shard.open("rb", buffering=0) as stream:
        while size := stream.readinto(buffer):
            bytes_read += size
    return shard, bytes_read


def save_lr_scheduler_state(scheduler: Any, checkpoint_dir: str) -> None:
    torch.save(
        scheduler.state_dict(),
        Path(checkpoint_dir) / LR_SCHEDULER_STATE_FILENAME,
    )


def load_lr_scheduler_state(scheduler: Any, checkpoint_dir: str) -> bool:
    path = Path(checkpoint_dir) / LR_SCHEDULER_STATE_FILENAME
    if not path.is_file():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    scheduler.load_state_dict(state)
    return True


class DCPState(Stateful):
    """Wrapper for checkpointing the State using DCP.

    This class implements the Stateful protocol, so DCP will automatically call
    state_dict/load_state_dict as needed in the dcp.save/load APIs.

    It handles calling distributed state dict methods on the model and optimizer.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
    ):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        """
        Get state dict for model and optimizer using DCP utilities.
        This automatically manages FSDP FQN's and
        sets default state dict type to FSDP.SHARDED_STATE_DICT
        """
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
            state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
        else:
            state_dict = {"model": get_model_state_dict(self.model)}
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state dicts onto model and optimizer.
        """
        if self.optimizer is not None:
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
            )
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
            )
