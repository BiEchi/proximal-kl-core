# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

from .version import __version__  # noqa

from .infra import (
    RolloutController,
    StalenessManager,
    TrainController,
    WorkflowExecutor,
    current_platform,
    workflow_context,
)


def __getattr__(name: str):
    if name == "DPOTrainer":
        from .trainer.dpo_trainer import DPOTrainer

        globals()[name] = DPOTrainer
        return DPOTrainer
    if name == "PPOTrainer":
        from .trainer.rl_trainer import PPOTrainer

        globals()[name] = PPOTrainer
        return PPOTrainer
    if name == "RWTrainer":
        from .trainer.rw_trainer import RWTrainer

        globals()[name] = RWTrainer
        return RWTrainer
    if name == "SFTTrainer":
        from .trainer.sft_trainer import SFTTrainer

        globals()[name] = SFTTrainer
        return SFTTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DPOTrainer",
    "PPOTrainer",
    "RolloutController",
    "RWTrainer",
    "SFTTrainer",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "current_platform",
    "workflow_context",
]
