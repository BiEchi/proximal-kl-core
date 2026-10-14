# SPDX-License-Identifier: Apache-2.0
"""Small walltime guard for checkpoint-safe Slurm exits."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class WalltimeDecision:
    should_exit: bool
    should_save_recovery: bool
    remaining_seconds: float | None
    next_step_budget_seconds: float
    checkpoint_budget_seconds: float


class WalltimeGuard:
    """Decide whether another train step fits before the allocation ends.

    The launcher exports ``AREAL_JOB_END_TIME`` as a Unix timestamp.  When it
    is absent (for example, in a local run), the guard is disabled and leaves
    training behavior unchanged.
    """

    def __init__(
        self,
        end_time: float | None,
        *,
        initial_step_seconds: float = 300.0,
        initial_checkpoint_seconds: float = 420.0,
        safety_factor: float = 1.25,
        shutdown_margin_seconds: float = 120.0,
    ) -> None:
        self.end_time = end_time
        self.initial_step_seconds = initial_step_seconds
        self.initial_checkpoint_seconds = initial_checkpoint_seconds
        self.safety_factor = safety_factor
        self.shutdown_margin_seconds = shutdown_margin_seconds
        self._step_seconds: deque[float] = deque(maxlen=16)
        self._checkpoint_seconds: deque[float] = deque(maxlen=4)

    @classmethod
    def from_env(cls) -> WalltimeGuard:
        raw_end_time = os.getenv("AREAL_JOB_END_TIME") or os.getenv(
            "SLURM_JOB_END_TIME"
        )
        try:
            end_time = float(raw_end_time) if raw_end_time else None
        except ValueError:
            end_time = None

        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return max(float(raw), 0.0)
            except ValueError:
                return default

        return cls(
            end_time,
            initial_step_seconds=_env_float(
                "AREAL_WALLTIME_STEP_ESTIMATE_SECONDS", 300.0
            ),
            initial_checkpoint_seconds=_env_float(
                "AREAL_WALLTIME_CHECKPOINT_ESTIMATE_SECONDS", 420.0
            ),
            safety_factor=_env_float("AREAL_WALLTIME_SAFETY_FACTOR", 1.25),
            shutdown_margin_seconds=_env_float(
                "AREAL_WALLTIME_SHUTDOWN_MARGIN_SECONDS", 120.0
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.end_time is not None

    def observe_step(self, seconds: float) -> None:
        if seconds > 0:
            self._step_seconds.append(seconds)

    def observe_checkpoint(self, seconds: float) -> None:
        if seconds > 0:
            self._checkpoint_seconds.append(seconds)

    def decide(
        self,
        *,
        current_step_elapsed_seconds: float,
        now: float | None = None,
    ) -> WalltimeDecision:
        step_estimate = (
            max(self._step_seconds) if self._step_seconds else self.initial_step_seconds
        )
        checkpoint_estimate = (
            max(self._checkpoint_seconds)
            if self._checkpoint_seconds
            else self.initial_checkpoint_seconds
        )
        next_step_budget = (
            max(step_estimate, current_step_elapsed_seconds) * self.safety_factor
        )
        checkpoint_budget = (
            checkpoint_estimate * self.safety_factor + self.shutdown_margin_seconds
        )
        if self.end_time is None:
            return WalltimeDecision(
                should_exit=False,
                should_save_recovery=False,
                remaining_seconds=None,
                next_step_budget_seconds=next_step_budget,
                checkpoint_budget_seconds=checkpoint_budget,
            )

        remaining = self.end_time - (time.time() if now is None else now)
        should_exit = remaining <= next_step_budget + checkpoint_budget
        should_save = should_exit and remaining >= checkpoint_budget
        return WalltimeDecision(
            should_exit=should_exit,
            should_save_recovery=should_save,
            remaining_seconds=remaining,
            next_step_budget_seconds=next_step_budget,
            checkpoint_budget_seconds=checkpoint_budget,
        )
