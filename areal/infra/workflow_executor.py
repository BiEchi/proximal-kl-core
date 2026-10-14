# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

import asyncio
import json
import os
import random
import socket
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast
from collections.abc import Generator
from collections import deque
import torch
import requests
import torch.distributed as dist

import aiofiles
import aiofiles.os
from torchdata.stateful_dataloader import StatefulDataLoader

# Trajectory dump format: JSONL, one record per line, one file per worker per
# role. Append-only; readable at any time without close (no footer needed); a
# SIGKILL loses at most the in-flight buffered line, never prior records. The
# 2026-06-02 rewrite replaced a chunked-parquet design whose footer-on-close
# semantics made SIGKILL/scancel drop entire chunks (~200 unreadable files per
# trial accumulated across OOM events), and whose cross-cycle accumulation in
# the same chunk caused pair-5 reward=0 data to bleed into pair-4 plot points.
# Provenance fields ``slurm_job_id`` + ``worker_uid`` are written on every
# record so the reader can filter out bad runs without re-running.
_TRAJECTORY_FIELDS = (
    "task_id", "sample_idx", "seqlen", "prompt_len",
    "head_version", "tail_version", "reward",
    "prompt", "completion",
    "slurm_job_id", "worker_uid",
)

from areal.api.cli_args import InferenceEngineConfig
from areal.api import RolloutWorkflow
from .async_task_runner import (
    AsyncTaskRunner,
    TaskQueueFullError,
    TimedResult,
)
from .staleness_manager import StalenessManager
from areal.infra import workflow_context
from .workflow_context import WorkflowContext
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    concat_string_interactions,
)
from areal.utils import logging, perf_tracer, stats_tracker
from areal.infra.utils.concurrent import get_executor
from areal.infra.utils.exp_metadata import get_metadata_dir
from areal.utils.data import concat_padded_tensors, cycle_dataloader
from areal.utils.math import ceil_div
from areal.utils.perf_tracer import trace_perf, trace_session_event
from logging import Logger

if TYPE_CHECKING:
    from .remote_inf_engine import RemoteInfEngine


def check_trajectory_format(
    data: dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward],
    batch_size: int | None = None,
    expected_keys: set | None = None,
    logger: Any = None,
) -> bool:
    """Check the format of trajectory data returned by workflow.arun_episode.

    This function validates trajectory data to ensure it conforms to one of three
    expected formats:

    1. **None**: Indicates a rejected trajectory that will not be used for
       training
    2. **Dict[str, InteractionWithTokenLogpReward]**: Completion/Response results from
       the workflow
    3. **Dict[str, torch.Tensor]**: Tensor format with specific shape and
       key requirements

    For tensor format validation, the function ensures:

    - Required keys ``input_ids`` and ``attention_mask`` are present
    - All tensors have consistent batch size and sequence length dimensions
    - Tensor shapes follow the pattern ``[batch_size, max_seqlen]``
    - Keys are consistent across different episodes when ``expected_keys`` is
      provided

    Special handling is provided for:

    - **multi_modal_input**: Expected to be a non-empty list of dictionaries
      containing ``pixel_values``
    - **Non-tensor data**: Logged for informational purposes

    Parameters
    ----------
    data : Dict[str, Any] | None | Dict[str, InteractionWithTokenLogpReward]
        The trajectory data to validate. Can be:

        - ``None`` for rejected trajectories
        - Dictionary mapping strings to ``InteractionWithTokenLogpReward`` objects
        - Dictionary mapping strings to PyTorch tensors or other data types

    batch_size : int | None, optional
        Expected batch size for tensor validation. If ``None``, batch size is inferred
        from the first dimension of ``input_ids``. Default is ``None``.

    expected_keys : set | None, optional
        Set of expected keys for consistency checking across multiple episodes.
        If provided, validates that the current trajectory contains all expected keys.
        Default is ``None``.

    logger : Any, optional
        Logger instance for warning and info messages. If ``None``, creates a default
        logger named "Workflow API". Default is ``None``.

    Returns
    -------
    bool
        ``True`` if the trajectory format is valid, ``False`` otherwise.

    Raises
    ------
    ValueError
        If the trajectory format is invalid. Error messages provide detailed information
        about the specific validation failure, including:

        - Missing required keys
        - Incorrect tensor dimensions
        - Inconsistent batch sizes or sequence lengths
        - Invalid multi-modal input format
        - Key inconsistencies across episodes

    Examples
    --------
    Basic usage with tensor data:

    >>> import torch
    >>> data = {
    ...     'input_ids': torch.randint(0, 1000, (2, 10)),
    ...     'attention_mask': torch.ones(2, 10)
    ... }
    >>> check_trajectory_format(data, batch_size=2)
    True

    Validation with expected keys:

    >>> expected = {'input_ids', 'attention_mask', 'labels'}
    >>> data_with_labels = {
    ...     'input_ids': torch.randint(0, 1000, (2, 10)),
    ...     'attention_mask': torch.ones(2, 10),
    ...     'labels': torch.randint(0, 1000, (2, 10))
    ... }
    >>> check_trajectory_format(data_with_labels, expected_keys=expected)
    True

    Rejected trajectory:

    >>> check_trajectory_format(None)
    True

    See Also
    --------
    RolloutWorkflow.arun_episode : Method that returns trajectory data
    WorkflowExecutor : Class that uses this function when
        ``check_trajectory_format`` is enabled
    """
    if logger is None:
        logger = logging.getLogger("WorkflowExecutor")
    if data is None:
        return True

    if not isinstance(data, dict):
        raise ValueError(f"Expected data to be None or dict, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Data dict cannot be empty")

    # Check if all values are InteractionWithTokenLogpReward
    if all(isinstance(v, InteractionWithTokenLogpReward) for v in data.values()):
        return True

    # Check required keys
    # At least require `input_ids` and `attention_mask`
    required_keys = {"input_ids", "attention_mask"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in tensor data: {missing_keys}")

    # Check tensor shapes
    input_ids = data["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors with shape [batch_size, max_seqlen], "
            f"got {input_ids.dim()}D"
        )

    inferred_batch_size, max_seqlen = input_ids.shape

    if batch_size is not None and inferred_batch_size != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {inferred_batch_size}")

    # Check all tensors have consistent shape
    for key, value in data.items():
        if torch.is_tensor(value):
            if value.shape[0] != inferred_batch_size:
                logger.warning(
                    f"The first dim of tensor `{key}` is {value.shape[0]}, "
                    f"rather than the batch size of input_ids ({inferred_batch_size})."
                )
            if value.ndim >= 2 and value.shape[1] != max_seqlen:
                logger.warning(
                    f"The second dim of tensor `{key}` is {value.shape[1]}, "
                    f"rather than the max seqlen of input_ids ({max_seqlen})."
                )
        elif key == "multi_modal_input":
            if (
                not isinstance(value, list)
                or len(value) == 0
                or any(not isinstance(v, dict) for v in value)
            ):
                raise ValueError(
                    "multi_modal_input should be a non-empty list of dicts"
                )
            if not all("pixel_values" in v for v in value):
                raise ValueError(
                    "multi_modal_input should at least contain the "
                    "`pixel_values` field."
                )
        else:
            logger.info(f"Encounter non-tensor data with key `{key}`: {value}")

    # Check key consistency if expected_keys is provided
    if expected_keys is not None:
        missing_keys = expected_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Inconsistent keys compared to expected: "
                f"expected {expected_keys}, but {missing_keys} are missing."
            )

    return True


@dataclass
class _RolloutTaskInput:
    """Internal wrapper for rollout-specific task input."""

    task_id: int
    data: dict[str, Any]
    workflow: RolloutWorkflow
    should_accept_fn: Callable[[dict[str, Any]], bool] | None = None
    is_eval: bool = False


@dataclass
class _RolloutResult:
    task_id: int
    trajectory: dict[str, Any]


# Batch size for fetching from the async task runner
_MAX_FETCH_BATCH_SIZE = 100
# Timeout for shutting down threads
_SHUTDOWN_TIMEOUT_SECONDS = 2.0
# Timeout for "wait" and "wait_for_task" if timeout parameter is None
_DEFAULT_WAIT_TIMEOUT_SECONDS = float(7 * 24 * 3600)
# Fail-fast timeout for prepare_batch when no task can be submitted and no
# result arrives. The old behavior looped silently until the Slurm allocation
# was killed, which hid the real blocked state.
_DEFAULT_ACTIVE_WAIT_STALL_TIMEOUT_SECONDS = 300.0
_ACTIVE_WAIT_STATUS_LOG_INTERVAL_SECONDS = 60.0


class WithTaskID(Protocol):
    task_id: int


TInput = TypeVar("TInput", bound=WithTaskID)
TResult = TypeVar("TResult")


@dataclass
class _BatchPrefetch(Generic[TInput]):
    """One rollout batch submitted while the trainer updates the actor."""

    task_ids: list[int]
    input_generator: Generator[TInput, None, None]
    batch_size: int
    dynamic_bs: bool
    min_batch_size: int
    expected_version: int | None


class BatchTaskDispatcher(Generic[TInput, TResult]):
    """Generic dispatcher for asynchronous task execution with staleness control.

    Manages background threads for task submission and result collection.
    Uses producer-consumer pattern with AsyncTaskRunner for async execution.

    Architecture:
    - Producer thread: Submits tasks from _pending_inputs to AsyncTaskRunner
      based on staleness capacity
    - Consumer thread: Collects results from AsyncTaskRunner to _pending_results
    - Main thread: submit_task_input() enqueues, wait_results() polls results
    """

    def __init__(
        self,
        max_queue_size: int,
        task_factory: Callable[[TInput], Callable[[], Awaitable[TResult | None]]],
        staleness_manager: StalenessManager,
        enable_tracing: bool = False,
    ):
        self.runner = AsyncTaskRunner(
            max_queue_size=max_queue_size,
            enable_tracing=enable_tracing,
        )
        self.task_factory = task_factory
        self.staleness_manager = staleness_manager
        self.enable_tracing = enable_tracing
        self.logger: Logger

        # Unbounded deques for producer/consumer pattern
        self._pending_inputs: deque[TInput] = deque()
        self._pending_results: dict[int, TimedResult[TResult]] = {}
        self._active_task_ids: set[int] = set()

        # A fixed-lag buffer is bootstrapped synchronously, then replenished by
        # exactly one batch while the trainer performs its update.  The
        # prefetch is drained by pause() before rollout weights can advance.
        self._ready_batches: deque[list[TResult]] = deque()
        self._ready_batch_buffer: int | None = None
        self._ready_buffer_fill_target: int | None = None
        self._ready_buffer_initialized = False
        self._batch_prefetch: _BatchPrefetch[TInput] | None = None
        self._last_prefetch_version: int | None = None

        # Condition variables for coordination
        self._input_lock = threading.Lock()
        self._input_cv = threading.Condition(self._input_lock)
        self._result_lock = threading.Lock()
        self._result_cv = threading.Condition(self._result_lock)

        # Background thread infrastructure
        self._shutdown_event = threading.Event()
        self._commit_thread: threading.Thread | None = None
        self._fetch_thread: threading.Thread | None = None

        # Exception propagation for fail-fast behavior
        self._thread_exception: Exception | None = None
        self._thread_exception_lock = threading.Lock()

        # Callback support: task_id -> callback_addr
        self._task_callbacks: dict[int, str] = {}
        self._discard_result_task_ids: set[int] = set()

    def _set_thread_exception(self, exc: Exception):
        """Store exception from background thread for fail-fast behavior."""
        with self._thread_exception_lock:
            if self._thread_exception is None:
                self._thread_exception = exc

    def _check_thread_exception(self):
        """Check if any background thread has failed and raise if so."""
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(
                    f"Background thread failed: {self._thread_exception}"
                ) from self._thread_exception

    def _has_runner_capacity(self) -> bool:
        return (
            not self.runner.paused.is_set()
            and self.staleness_manager.get_capacity() > 0
            and self.runner.get_input_queue_size() < self.runner.max_queue_size
        )

    def register_callback(self, task_id: int, callback_addr: str):
        """Register a callback address for a task."""
        with self._result_cv:
            self._task_callbacks[task_id] = callback_addr

    def cancel_callback(self, task_id: int):
        """Remove a registered callback for a task (e.g., on timeout)."""
        with self._result_cv:
            self._task_callbacks.pop(task_id, None)

    def cancel_task(self, task_id: int) -> bool:
        """Cancel a pending or running task and discard its eventual result."""
        self._check_thread_exception()
        self.cancel_callback(task_id)

        with self._result_cv:
            found_result = self._pending_results.pop(task_id, None)
            if found_result is not None:
                self._active_task_ids.discard(task_id)
                self._result_cv.notify_all()
                # _fetch_loop accounts a non-None result as accepted before
                # publishing it in _pending_results.  A controller timeout can
                # race with that publication and cancel the task before it is
                # fetched.  Removing the result without releasing that accepted
                # slot permanently exhausts staleness capacity even though no
                # runnable work remains.
                if found_result.data is not None:
                    self.staleness_manager.on_accepted_rollouts_discarded(1)
                return True
            # Mark the task before calling into the runner so a concurrently
            # arriving cancellation result cannot update staleness twice.
            self._discard_result_task_ids.add(task_id)

        cancelled_before_submit = False
        with self._input_cv:
            for idx, task_input in enumerate(self._pending_inputs):
                if task_input.task_id == task_id:
                    del self._pending_inputs[idx]
                    self.staleness_manager.on_rollout_cancelled_before_submit()
                    cancelled_before_submit = True
                    self._input_cv.notify()
                    break

        if cancelled_before_submit:
            with self._result_cv:
                self._discard_result_task_ids.discard(task_id)
                self._active_task_ids.discard(task_id)
                self._result_cv.notify_all()
            return True

        cancelled = self.runner.cancel_task(task_id)
        with self._result_cv:
            discard_pending = task_id in self._discard_result_task_ids
            if not cancelled and discard_pending:
                self._discard_result_task_ids.discard(task_id)
            should_release_running = cancelled or not discard_pending
            if should_release_running:
                self._active_task_ids.discard(task_id)
                self._result_cv.notify_all()

        if should_release_running:
            # The task was already submitted to the runner, so it consumed one
            # running slot in StalenessManager. Release it immediately on
            # timeout/cancel; delayed runner output is ignored by _fetch_loop.
            self.staleness_manager.on_rollout_rejected()
            with self._result_cv:
                self._result_cv.notify_all()
            with self._input_cv:
                self._input_cv.notify()
        return cancelled

    def _should_discard_result(self, task_id: int) -> bool:
        if task_id not in self._discard_result_task_ids:
            return False
        self._discard_result_task_ids.discard(task_id)
        return True

    def _send_callback(self, addr: str, task_id: int, result: TResult):
        """Send task result to callback address (fire-and-forget)."""

        def post():
            try:
                resp = requests.post(
                    addr,
                    json={"task_id": task_id},
                    timeout=30,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                self.logger.error(f"Callback to {addr} failed: {e}")

        get_executor().submit(post)

    def _commit_loop(self) -> None:
        """Producer thread - continuously submits tasks based on capacity."""
        while not self._shutdown_event.is_set():
            try:
                # Check for errors from other threads (fail-fast)
                self._check_thread_exception()

                task_input = self._get_next_task_for_submission()
                if task_input is None:
                    continue

                task_fn = self.task_factory(task_input)
                try:
                    self.runner.submit(task_fn, task_id=task_input.task_id)
                    self.staleness_manager.on_rollout_submitted()
                    if self.enable_tracing:
                        self.logger.info(f"Submit rollout. {self._rollout_stats()}")
                except TaskQueueFullError:
                    with self._input_cv:
                        self._pending_inputs.appendleft(task_input)
                        self._input_cv.wait_for(
                            lambda: (
                                self._shutdown_event.is_set()
                                or self._has_runner_capacity()
                            )
                        )
                    # Allow other threads to make progress before retrying
                    continue

            except Exception as e:
                self.logger.error("Producer thread failed", exc_info=True)
                self._set_thread_exception(e)
                with self._result_cv:
                    self._result_cv.notify_all()
                break

    def _fetch_loop(self) -> None:
        """Consumer thread - continuously collects results from runner."""
        while not self._shutdown_event.is_set():
            try:
                # Check for errors from other threads (fail-fast)
                self._check_thread_exception()

                # Poll runner for available results (non-blocking)
                output_queue_size = self.runner.get_output_queue_size()
                count = max(1, min(output_queue_size, _MAX_FETCH_BATCH_SIZE))

                try:
                    # Use short timeout for responsiveness (latency-optimized)
                    results = self.runner.wait(
                        count=count, timeout=0.05, with_timing=True
                    )
                except TimeoutError:
                    continue

                with self._result_cv:
                    for result in results:
                        if self._should_discard_result(result.task_id):
                            continue
                        if result.data is None:
                            self.staleness_manager.on_rollout_rejected()
                        else:
                            self.staleness_manager.on_rollout_accepted()
                        self._pending_results[result.task_id] = result
                        # Trigger callback if registered
                        cb_addr = self._task_callbacks.pop(result.task_id, None)
                        if cb_addr:
                            self._send_callback(cb_addr, result.task_id, result.data)
                    self._result_cv.notify_all()

                # Newly available capacity after result processing should wake producers
                with self._input_cv:
                    self._input_cv.notify()

            except Exception as e:
                self.logger.error("Consumer thread failed", exc_info=True)
                self._set_thread_exception(e)
                with self._result_cv:
                    self._result_cv.notify_all()
                with self._input_cv:
                    self._input_cv.notify()
                break

    def _get_next_task_for_submission(self) -> TInput | None:
        with self._input_cv:
            while not self._shutdown_event.is_set():
                self._check_thread_exception()
                # There is capacity and pending inputs
                if (
                    not self.runner.paused.is_set()
                    and self.staleness_manager.get_capacity() > 0
                    and self._pending_inputs
                ):
                    return self._pending_inputs.popleft()
                self._input_cv.wait()

        return None

    def initialize(self, logger: Logger):
        self.logger = logger
        self.runner.initialize(logger=logger)

        self._shutdown_event.clear()

        self._commit_thread = threading.Thread(target=self._commit_loop, daemon=True)
        self._commit_thread.start()

        self._fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._fetch_thread.start()

    def destroy(self):
        self._shutdown_event.set()
        with self._input_cv:
            self._input_cv.notify()
        with self._result_cv:
            self._result_cv.notify_all()

        if self._commit_thread and self._commit_thread.is_alive():
            self._commit_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._commit_thread.is_alive() and self.logger:
                self.logger.warning(
                    "Producer thread did not exit cleanly within timeout"
                )

        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._fetch_thread.is_alive() and self.logger:
                self.logger.warning(
                    "Consumer thread did not exit cleanly within timeout"
                )

        # Clear pending callbacks to prevent memory leak
        self._task_callbacks.clear()

        # Shutdown the async task runner
        self.runner.destroy()

    def pause(self):
        """Pause request submission for async tasks.

        After calling pause(), no new tasks will be started from the
        input queue, but existing running tasks will continue to completion.
        """
        # The trainer calls pause before updating rollout weights.  Finish the
        # batch started by active_submit_and_wait while all tasks still use the
        # version recorded at prefetch time; otherwise a partial batch could
        # straddle two policy versions and break the fixed-lag contract.
        # Keep this timing separate from rollout.pause so residual generation
        # latency remains visible instead of looking like eliminated wait.
        with stats_tracker.record_timing("rollout_prefetch_wait"):
            self._finish_batch_prefetch()
        self.runner.pause()
        with self._input_cv:
            self._input_cv.notify()

    def resume(self):
        """Resume request submission for async tasks.

        Allows new tasks to be pulled from the input queue and started.
        """
        self.runner.resume()
        with self._input_cv:
            self._input_cv.notify()

    def is_paused(self) -> bool:
        """Check if the dispatcher is currently paused.

        Returns
        -------
        bool
            True if paused, False otherwise.
        """
        return self.runner.paused.is_set()

    def _rollout_stats(self) -> str:
        stats = self.staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _active_wait_debug_state(self) -> dict[str, Any]:
        stats = self.staleness_manager.get_stats()
        with self._input_cv:
            pending_inputs = len(self._pending_inputs)
        with self._result_cv:
            pending_results = len(self._pending_results)
            active_task_ids = len(self._active_task_ids)
        return {
            "staleness_capacity": self.staleness_manager.get_capacity(),
            "pending_limit": self.staleness_manager.get_pending_limit(),
            "pending_inputs": pending_inputs,
            "pending_results": pending_results,
            "active_task_ids": active_task_ids,
            "runner_input_queue": self.runner.get_input_queue_size(),
            "runner_output_queue": self.runner.get_output_queue_size(),
            "runner_paused": self.runner.paused.is_set(),
            "accepted": stats.accepted,
            "enqueued": stats.enqueued,
            "running": stats.running,
            "rejected": stats.rejected,
        }

    @staticmethod
    def _format_active_wait_debug_state(state: dict[str, Any]) -> str:
        return ", ".join(f"{key}={value}" for key, value in state.items())

    def submit_task_input(self, task_input: TInput) -> None:
        """Submit a task input for processing.

        Parameters
        ----------
        task_input : TInput
            Task input to be processed.
        """
        self._check_thread_exception()
        with self._input_cv:
            self._pending_inputs.append(task_input)
            self.staleness_manager.on_rollout_enqueued()
            if self.enable_tracing:
                self.logger.info(f"Enqueue rollout. {self._rollout_stats()}")
            self._input_cv.notify()
        with self._result_cv:
            self._active_task_ids.add(task_input.task_id)

    def wait_results(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
        drain_available: bool = False,
    ) -> list[TResult | None]:
        """Wait for the completion of `count` tasks.

        Parameters
        ----------
        count : int
            Number of results to wait for.
        timeout : float | None
            Maximum time to wait in seconds.
        raise_timeout : bool
            Whether to raise TimeoutError on timeout.

        Returns
        -------
        list[TResult | None]
            List of task results, None for rejected tasks.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        start_time = time.perf_counter()
        if timeout is None:
            timeout = _DEFAULT_WAIT_TIMEOUT_SECONDS

        with self._result_cv:
            while len(self._pending_results) < count:
                self._check_thread_exception()

                elapsed = time.perf_counter() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    if raise_timeout:
                        raise TimeoutError(
                            f"Timed out waiting for {count} results, "
                            f"only received {len(self._pending_results)}"
                        )
                    if not drain_available or not self._pending_results:
                        return []
                    break

                self._result_cv.wait(timeout=remaining)

            drained: list[TimedResult[TResult]] = list(self._pending_results.values())
            self._pending_results.clear()

        drained.sort(key=lambda x: x.create_time)
        selected, pending = drained[:count], drained[count:]
        with self._result_cv:
            if pending:
                for result in pending:
                    self._pending_results[result.task_id] = result
                self._result_cv.notify_all()
            for r in selected:
                self._active_task_ids.discard(r.task_id)

        random.shuffle(selected)

        return [r.data for r in selected]

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> TResult | None:
        """Wait for a specific task result by task_id."""
        start_time = time.perf_counter()
        if timeout is None:
            timeout = _DEFAULT_WAIT_TIMEOUT_SECONDS

        with self._result_cv:
            if task_id not in self._active_task_ids:
                raise ValueError(f"Task {task_id} is never submitted.")

            while task_id not in self._pending_results:
                self._check_thread_exception()

                elapsed = time.perf_counter() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    if raise_timeout:
                        raise TimeoutError(f"Timed out waiting for task {task_id}.")
                    return None

                self._result_cv.wait(timeout=remaining)

            found_result = self._pending_results.pop(task_id)
            self._active_task_ids.remove(task_id)
            self._result_cv.notify_all()
            return found_result.data

    def wait_for_tasks(
        self,
        task_ids: list[int],
        timeout: float | None = None,
        raise_timeout: bool = True,
        missing_as_rejected: bool = False,
    ) -> list[TResult | None]:
        """Wait for a specific set of tasks without draining unrelated results."""
        if not task_ids:
            return []
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be unique")

        start_time = time.perf_counter()
        if timeout is None:
            timeout = _DEFAULT_WAIT_TIMEOUT_SECONDS
        wanted = set(task_ids)

        with self._result_cv:
            known = self._active_task_ids | set(self._pending_results)
            unknown = wanted - known
            if unknown and not missing_as_rejected:
                raise ValueError(f"Tasks were never submitted: {sorted(unknown)}")

            def unresolved_task_ids() -> set[int]:
                pending = set(self._pending_results)
                unresolved = wanted - pending
                if missing_as_rejected:
                    # A timed-out or cancelled rollout is removed from the
                    # active set without publishing a result. Fixed-lag
                    # prefetch treats that task exactly like a rejected
                    # rollout and refills it at the same policy version.
                    unresolved &= self._active_task_ids
                return unresolved

            while unresolved_task_ids():
                self._check_thread_exception()
                elapsed = time.perf_counter() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    if raise_timeout:
                        missing = sorted(unresolved_task_ids())
                        raise TimeoutError(f"Timed out waiting for tasks {missing}")
                    return []
                self._result_cv.wait(timeout=remaining)

            selected = [
                self._pending_results.pop(task_id)
                for task_id in task_ids
                if task_id in self._pending_results
            ]
            rejected_count = len(task_ids) - len(selected)
            self._active_task_ids.difference_update(wanted)
            self._result_cv.notify_all()

        selected.sort(key=lambda x: x.create_time)
        random.shuffle(selected)
        return [result.data for result in selected] + [None] * rejected_count

    def _current_rollout_version(self) -> int | None:
        provider = getattr(self.staleness_manager, "version_provider", None)
        get_version = getattr(provider, "get_version", None)
        return int(get_version()) if callable(get_version) else None

    def _start_batch_prefetch(
        self,
        input_generator: Generator[TInput, None, None],
        batch_size: int,
        dynamic_bs: bool,
        min_batch_size: int,
    ) -> None:
        if self._batch_prefetch is not None:
            raise RuntimeError("A rollout batch prefetch is already active")

        expected_version = self._current_rollout_version()
        if (
            expected_version is not None
            and expected_version == self._last_prefetch_version
        ):
            raise RuntimeError(
                "Fixed-lag buffering cannot prefetch two batches at the same "
                f"rollout version {expected_version}"
            )

        task_inputs = []
        for _ in range(batch_size):
            try:
                task_inputs.append(next(input_generator))
            except StopIteration:
                raise RuntimeError(
                    "Input generator exhausted before batch prefetch. "
                    "Use cycle_dataloader() or provide an infinite generator."
                ) from None

        for task_input in task_inputs:
            self.submit_task_input(task_input)
        self._batch_prefetch = _BatchPrefetch(
            task_ids=[task.task_id for task in task_inputs],
            input_generator=input_generator,
            batch_size=batch_size,
            dynamic_bs=dynamic_bs,
            min_batch_size=min_batch_size,
            expected_version=expected_version,
        )
        self._last_prefetch_version = expected_version

    def _finish_batch_prefetch(self) -> None:
        prefetch = self._batch_prefetch
        if prefetch is None:
            return

        prefetch_timeout = float(
            os.environ.get(
                "AREAL_FIXED_LAG_PREFETCH_TIMEOUT",
                _DEFAULT_ACTIVE_WAIT_STALL_TIMEOUT_SECONDS,
            )
        )
        try:
            results = self.wait_for_tasks(
                prefetch.task_ids,
                timeout=prefetch_timeout,
                missing_as_rejected=True,
            )
        except TimeoutError:
            # A lost runner result used to leave fixed-lag recovery blocked in
            # wait_for_tasks() for the seven-day generic wait timeout.  That
            # presents as zero pending/running rollout work until the cluster's
            # idle-GPU reaper kills the job.  Expire only this prefetch's
            # unresolved tasks and refill them at the same policy version.
            with self._result_cv:
                unresolved = [
                    task_id
                    for task_id in prefetch.task_ids
                    if task_id in self._active_task_ids
                    and task_id not in self._pending_results
                ]
            self.logger.warning(
                "Fixed-lag prefetch stalled for %.1fs; expiring %d unresolved "
                "tasks and refilling the batch.",
                prefetch_timeout,
                len(unresolved),
            )
            for task_id in unresolved:
                self.cancel_task(task_id)

            # A runner can report an unknown task after losing its completion
            # record.  cancel_task() deliberately waits for a racing result in
            # that case; after the full prefetch timeout, force those stale
            # bookkeeping entries to rejected and discard any late result.
            with self._result_cv:
                still_unresolved = [
                    task_id
                    for task_id in unresolved
                    if task_id in self._active_task_ids
                    and task_id not in self._pending_results
                ]
                self._discard_result_task_ids.update(still_unresolved)
                self._active_task_ids.difference_update(still_unresolved)
                self._result_cv.notify_all()
            for _ in still_unresolved:
                self.staleness_manager.on_rollout_rejected()

            results = self.wait_for_tasks(
                prefetch.task_ids,
                timeout=1.0,
                missing_as_rejected=True,
            )
        batch = [result for result in results if result is not None]
        refill_target = (
            prefetch.min_batch_size
            if prefetch.dynamic_bs
            else prefetch.batch_size
        )
        if len(batch) < refill_target:
            batch.extend(
                self._collect_batch(
                    prefetch.input_generator,
                    batch_size=refill_target - len(batch),
                    dynamic_bs=False,
                    strict_capacity=True,
                )
            )

        current_version = self._current_rollout_version()
        if (
            prefetch.expected_version is not None
            and current_version != prefetch.expected_version
        ):
            raise RuntimeError(
                "Rollout version advanced before fixed-lag prefetch completed: "
                f"expected {prefetch.expected_version}, got {current_version}"
            )

        self._ready_batches.append(batch)
        self._batch_prefetch = None

    def active_submit_and_wait(
        self,
        input_generator: Generator[TInput, None, None],
        batch_size: int,
        dynamic_bs: bool = False,
        ready_batch_buffer: int = 1,
        min_batch_size: int = 1,
    ) -> list[TResult]:
        if ready_batch_buffer == 1:
            # Rejection sampling may need several refill rounds for one
            # effective batch.  The ordinary cap-only path can queue a full
            # pending limit on every refill while earlier attempts are still
            # running.  At lag zero that can exhaust staleness capacity with
            # all remaining work stranded in the pending-input queue.  Use
            # the same exact capacity accounting as fixed-lag buffering when
            # an explicit replacement reserve is enabled.
            strict_capacity = (
                getattr(
                    self.staleness_manager,
                    "rejection_sampling_reserve",
                    0,
                )
                > 0
            )
            return self._collect_batch(
                input_generator,
                batch_size,
                dynamic_bs,
                strict_capacity=strict_capacity,
            )
        if not 1 <= ready_batch_buffer <= self.staleness_manager.max_staleness + 1:
            raise ValueError("ready_batch_buffer must be between 1 and max_head_offpolicyness + 1")
        if not 1 <= min_batch_size <= batch_size:
            raise ValueError("min_batch_size must be between 1 and batch_size")

        # Preserve the lightweight object.__new__ test seam used by downstream
        # callers that replace _collect_batch and do not initialize dispatcher
        # threads.  Real dispatchers always own _pending_inputs.
        if not hasattr(self, "_pending_inputs"):
            if not hasattr(self, "_ready_batches"):
                self._ready_batches = deque()
            while len(self._ready_batches) < ready_batch_buffer:
                self._ready_batches.append(
                    self._collect_batch(input_generator, batch_size, dynamic_bs)
                )
            return self._ready_batches.popleft()

        if (
            self._ready_batch_buffer is not None
            and self._ready_batch_buffer != ready_batch_buffer
        ):
            raise RuntimeError(
                "ready_batch_buffer cannot change after fixed-lag buffering starts"
            )
        self._ready_batch_buffer = ready_batch_buffer

        # A fresh run fills L+1 complete batches at version zero, consumes one,
        # and deliberately does not prefetch an extra startup batch.  That
        # gives the established 0..L lag ramp without ever creating a batch
        # that would later be consumed at lag L+1.
        #
        # Repeating that synchronous bootstrap after every checkpoint recovery
        # can leave all trainer GPUs idle for hours when L is large.  Recoveries
        # therefore start from depth one and add one ready batch per update
        # until the configured depth is restored.  The consumed lag remains
        # bounded by L throughout the ramp, while useful training resumes after
        # the first batch instead of waiting for all L+1 batches.
        if not self._ready_buffer_initialized:
            current_version = self._current_rollout_version()
            is_recovery = current_version is not None and current_version > 0
            fill_target = 1 if is_recovery else ready_batch_buffer
            self._ready_buffer_fill_target = fill_target
            while len(self._ready_batches) < fill_target:
                batch = self._collect_batch(
                    input_generator,
                    batch_size,
                    dynamic_bs,
                    strict_capacity=True,
                )
                if dynamic_bs and len(batch) < min_batch_size:
                    batch.extend(
                        self._collect_batch(
                            input_generator,
                            min_batch_size - len(batch),
                            dynamic_bs=False,
                            strict_capacity=True,
                        )
                    )
                self._ready_batches.append(batch)
            self._ready_buffer_initialized = True
            batch = self._ready_batches.popleft()
            if fill_target < ready_batch_buffer:
                self.logger.info(
                    "Recovering fixed-lag rollout buffer gradually from depth "
                    "1 to %d at actor version %d.",
                    ready_batch_buffer,
                    current_version,
                )
                self._start_batch_prefetch(
                    input_generator, batch_size, dynamic_bs, min_batch_size
                )
            return batch

        # pause() normally completed the previous prefetch before weights
        # advanced.  This fallback also makes direct dispatcher use fail fast
        # on an unexpected version transition.
        self._finish_batch_prefetch()
        fill_target = self._ready_buffer_fill_target
        if fill_target is None:
            fill_target = ready_batch_buffer - 1
        steady_fill_target = ready_batch_buffer - 1
        if fill_target < steady_fill_target:
            fill_target += 1
            self._ready_buffer_fill_target = fill_target
            while len(self._ready_batches) < fill_target:
                batch = self._collect_batch(
                    input_generator,
                    batch_size,
                    dynamic_bs,
                    strict_capacity=True,
                )
                if dynamic_bs and len(batch) < min_batch_size:
                    batch.extend(
                        self._collect_batch(
                            input_generator,
                            min_batch_size - len(batch),
                            dynamic_bs=False,
                            strict_capacity=True,
                        )
                    )
                self._ready_batches.append(batch)
        if not self._ready_batches:
            raise RuntimeError("Fixed-lag rollout buffer is unexpectedly empty")

        batch = self._ready_batches.popleft()
        self._start_batch_prefetch(
            input_generator, batch_size, dynamic_bs, min_batch_size
        )
        return batch

    def _collect_batch(
        self,
        input_generator: Generator[TInput, None, None],
        batch_size: int,
        dynamic_bs: bool = False,
        strict_capacity: bool = False,
    ) -> list[TResult]:
        """Continuously submit tasks and wait until a full batch of results is ready.

        This method maintains overlap between submission and result collection
        to maximize throughput.

        Parameters
        ----------
        input_generator : Generator[TInput, None, None]
            An **infinite** generator that yields task inputs. The generator
            must not be exhausted before the batch is complete. Use
            :func:`~areal.utils.data.cycle_dataloader` to wrap finite data sources.
        batch_size : int
            Number of results to collect before returning.
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. The method will stop collecting
            when (accepted + rejected) >= batch_size, returning only accepted results.
            This results in variable-sized batches of valid data. Default is False.

        Returns
        -------
        list[TResult]
            A list of task results. When ``dynamic_bs=False``, returns exactly
            ``batch_size`` results. When ``dynamic_bs=True``, returns up to
            ``batch_size`` accepted results (variable-sized).

        Raises
        ------
        RuntimeError
            If the input generator is exhausted before the batch is complete.
        """
        accepted_cnt = 0
        total_attempts = 0
        strict_inflight = 0
        results = []
        last_progress_time = time.perf_counter()
        last_status_log_time = last_progress_time
        stall_timeout = float(
            os.environ.get(
                "AREAL_PREPARE_BATCH_STALL_TIMEOUT",
                _DEFAULT_ACTIVE_WAIT_STALL_TIMEOUT_SECONDS,
            )
        )

        while True:
            # Submit tasks to maintain overlap
            if strict_capacity:
                # Fixed-lag prefetches must not sit behind speculative inputs
                # from bootstrap. Read enqueued first, then capacity: if the
                # producer moves an item to running between the snapshots, the
                # result is conservatively low rather than allowing an extra
                # task through the hand-off race.
                enqueued = self.staleness_manager.get_stats().enqueued
                cap_staleness = self.staleness_manager.get_capacity() - enqueued
            else:
                # Preserve the established cap-only async path, which keeps a
                # deeper input queue to maximize overlap.
                with self._input_cv:
                    pending_inputs = len(self._pending_inputs)
                cap_staleness = (
                    self.staleness_manager.get_pending_limit() - pending_inputs
                )
            if self.runner.max_queue_size < batch_size:
                raise ValueError(
                    f"Inference engine config's queue size is too small: {self.runner.max_queue_size} < batch size {batch_size}."
                )
            cap_queue = self.runner.max_queue_size - (
                self.runner.get_input_queue_size() + batch_size
            )
            capacity = min(cap_staleness, cap_queue)
            submitted = 0
            if strict_capacity:
                if dynamic_bs:
                    submit_budget = (
                        batch_size - total_attempts - strict_inflight
                    )
                else:
                    submit_budget = (
                        batch_size - accepted_cnt - strict_inflight
                    )
            else:
                submit_budget = batch_size
            if capacity > 0 and submit_budget > 0:
                if self.enable_tracing:
                    perf_tracer.instant(
                        "batch_task_dispatcher.continously_submit",
                        category="scheduler",
                        args={"data": capacity},
                    )
                for _ in range(min(submit_budget, capacity)):
                    try:
                        self.submit_task_input(next(input_generator))
                        submitted += 1
                    except StopIteration:
                        raise RuntimeError(
                            "Input generator exhausted before batch completion. "
                            "Use cycle_dataloader() or provide an infinite generator."
                        ) from None
            if submitted:
                if strict_capacity:
                    strict_inflight += submitted
                last_progress_time = time.perf_counter()
            try:
                arrived = self.wait_results(
                    count=batch_size - accepted_cnt,
                    timeout=1,
                    raise_timeout=False,
                    drain_available=True,
                )
            except TimeoutError:
                arrived = []
            if arrived:
                last_progress_time = time.perf_counter()
            elif submitted == 0:
                now = time.perf_counter()
                state = self._active_wait_debug_state()
                state_text = self._format_active_wait_debug_state(state)
                if (
                    now - last_status_log_time
                    >= _ACTIVE_WAIT_STATUS_LOG_INTERVAL_SECONDS
                ):
                    logger_obj = getattr(self, "logger", None)
                    if logger_obj is not None:
                        logger_obj.warning(
                            "prepare_batch made no progress for %.1fs while "
                            "waiting for %d/%d accepted results: %s",
                            now - last_progress_time,
                            accepted_cnt,
                            batch_size,
                            state_text,
                        )
                    last_status_log_time = now
                blocked_without_work = (
                    state["running"] <= 0
                    and state["runner_input_queue"] == 0
                    and state["runner_output_queue"] == 0
                    and state["pending_results"] == 0
                )
                if (
                    blocked_without_work
                    and stall_timeout > 0
                    and now - last_progress_time >= stall_timeout
                ):
                    raise RuntimeError(
                        "prepare_batch made no progress for "
                        f"{now - last_progress_time:.1f}s while waiting for "
                        f"{accepted_cnt}/{batch_size} accepted results: {state_text}"
                    )

            for res in arrived:
                if strict_capacity:
                    strict_inflight = max(0, strict_inflight - 1)
                is_accepted = res is not None

                if not is_accepted:
                    if dynamic_bs:
                        total_attempts += 1
                        if total_attempts >= batch_size:
                            break
                    continue

                # Accepted sample
                accepted_cnt += 1
                total_attempts += 1
                results.append(res)

                if dynamic_bs:
                    if total_attempts >= batch_size:
                        break
                elif accepted_cnt >= batch_size:
                    break
            else:
                continue
            break

        return results


class TaskIdGenerator:
    def __init__(self):
        self._task_cnt = 0
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            task_id = self._task_cnt
            self._task_cnt += 1
        return task_id


class WorkflowExecutor:
    """Executor for asynchronous workflow-based rollout generation.

    Orchestrates workflow execution with AReaL-specific features including
    staleness management, trajectory validation, and result filtering.
    Delegates task dispatching to BatchTaskDispatcher.
    """

    def __init__(
        self,
        config: InferenceEngineConfig,
        inference_engine: RemoteInfEngine,
        staleness_manager: StalenessManager | None = None,
    ):
        self.max_concurrent_rollouts = (
            config.max_concurrent_rollouts or config.consumer_batch_size
        )
        self.consumer_batch_size = config.consumer_batch_size
        self.max_staleness = config.max_head_offpolicyness

        self.config = config
        self.inference_engine = inference_engine

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager = staleness_manager

        # For trajectory format checking
        self._expected_trajectory_keys: set | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RolloutTaskInput, _RolloutResult] | None
        ) = None

        self._task_id_generator = TaskIdGenerator()

        # Lazy-loaded tokenizer for trajectory dumping
        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()

        # Per-worker, per-role JSONL append. One file per (dump_dir, worker)
        # for the lifetime of the worker. Asyncio Lock per dump_dir serializes
        # appends from concurrent dump coroutines in this process; the kernel
        # handles cross-process file isolation (different filenames per pid).
        # No chunk rotation, no close-on-shutdown — JSONL is readable at any
        # point, including mid-write.
        self._dump_locks: dict[str, asyncio.Lock] = {}
        self._worker_uid = (
            f"{socket.gethostname().split('.')[0]}_{os.getpid()}"
        )
        self._slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))

        # Host-RAM leak diagnostic: a daemon thread logs RSS + key dict sizes
        # + top object-type counts every AREAL_MEMLOG_INTERVAL seconds.
        # Toggle off with AREAL_MEMLOG=0.
        self._memlog_stop = threading.Event()
        self._memlog_thread: threading.Thread | None = None

    def _resolve_dp_world_size(self):
        if not dist.is_initialized():
            return 1

        try:
            from megatron.core import parallel_state as mpu

            if mpu.is_initialized():
                return mpu.get_data_parallel_world_size()
            return dist.get_world_size()
        except ImportError:
            return dist.get_world_size()

    def _get_tokenizer(self):
        """Lazy-load tokenizer for trajectory text decoding."""
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizer_path = self.config.tokenizer_path
        if not tokenizer_path:
            return None

        with self._tokenizer_lock:
            if self._tokenizer is not None:
                return self._tokenizer

            from areal.utils.hf_utils import load_hf_tokenizer

            self._tokenizer = load_hf_tokenizer(tokenizer_path)
            return self._tokenizer

    def _get_dump_dir(self, is_eval: bool) -> str | None:
        """Get the dump directory based on config and is_eval flag.

        Eval-rollout JSONL is sharded under ``eval-rollout/job_{SLURM_JOB_ID}/``
        so that chain-link resumes (each a fresh SLURM job → fresh process
        → fresh task_id counter starting at 0) don't pool their task_ids
        into the previous link's records. Pre-fix, aggregate_eval_jsonl
        keyed by (tail_version, task_id) collided across chain links and
        falsely doubled/tripled pass_at_n / mean_at_n task counts on the
        same eval cycle. With per-job sharding each link has an isolated
        subtree; the active aggregator only reads its own job's subdir.
        Train-rollout keeps the flat dir since train doesn't aggregate
        across cycles by task_id.
        """
        config = self.config
        if not config.fileroot or not config.experiment_name or not config.trial_name:
            return None

        log_path = get_metadata_dir(
            self.config.fileroot,
            self.config.experiment_name,
            self.config.trial_name,
        )
        if is_eval:
            job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
            return os.path.join(log_path, "eval-rollout", f"job_{job_id}")
        return os.path.join(log_path, "rollout")

    async def _dump_trajectory(
        self,
        traj: dict[str, Any] | None,
        task_id: int,
        is_eval: bool,
    ) -> tuple[bool, str]:
        if traj is None:
            return False, "trajectory is None"

        # Diagnostic kill-switch: AREAL_DISABLE_TRAJECTORY_DUMP=1 makes this a
        # no-op. Used 2026-06-03 to test whether the per-rollout JSONL append
        # (with its asyncio.Lock per dump_dir) was the proximate cause of the
        # repeatable step-30 hang across pair-14/16/17. Keep the switch in
        # place: useful for future debugging, costs nothing when unset.
        if os.environ.get("AREAL_DISABLE_TRAJECTORY_DUMP") == "1":
            return False, "trajectory dump disabled via env"

        # Eval rollouts bypass the dump_limit. The eval RolloutController has
        # its own monotonic task_id counter that starts at 0 and increments
        # forever; with the limit (default 2048) and a valid_dataset of 1819
        # prompts, only the first ≈1 eval cycle dumps fully and the second
        # cycle dumps ≈228 prompts before task_id crosses the cap — every
        # later eval cycle dumps NOTHING (records still get computed; they're
        # just dropped). aggregate_eval_jsonl finds an empty dump dir for
        # cycles 3+ and returns {}, so no `per-source eval logged to wandb`
        # line lands. Eval volume is small (one cycle ≈ 1819 prompts × 4
        # samples × ~1 KB JSON = ~7 MB; ~50 cycles per 4 h walltime ≈ 350 MB).
        if not is_eval:
            dump_limit_raw = os.environ.get(
                "AREAL_TRAJECTORY_DUMP_TASK_LIMIT",
                os.environ.get("AREAL_TRAJECTORY_DUMP_LIMIT"),
            )
            if dump_limit_raw is not None:
                try:
                    dump_limit = int(dump_limit_raw)
                except ValueError:
                    return False, f"invalid trajectory dump task limit: {dump_limit_raw}"
                if dump_limit >= 0 and task_id >= dump_limit:
                    return True, ""

        dump_dir = self._get_dump_dir(is_eval)
        if dump_dir is None:
            return False, "dump dir is empty"

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return False, "tokenizer not configured"

        # Extract tensors
        input_ids = traj.get("input_ids")
        rewards = traj.get("rewards")
        loss_mask = traj.get("loss_mask")
        attention_mask = traj.get("attention_mask")

        if (
            input_ids is None
            or rewards is None
            or loss_mask is None
            or attention_mask is None
        ):
            return (
                False,
                "missing required tensor fields: input_ids, rewards, attention_mask, or loss_mask",
            )

        if "versions" not in traj:
            self.logger.warning(
                "Trajectory missing 'versions' field, defaulting to current inference engine version."
            )
            versions = [self.inference_engine.get_version()]
        else:
            versions = traj["versions"].flatten().tolist()

        tail_version = max(versions)
        head_version = min(versions)
        await aiofiles.os.makedirs(dump_dir, exist_ok=True)

        batch_size = input_ids.shape[0]

        # Build records for this dump call (one row per non-empty sample).
        records: list[dict[str, Any]] = []
        for i in range(batch_size):
            seqlen = int(attention_mask[i].sum().item())
            if seqlen == 0:
                continue
            ids = input_ids[i, :seqlen].tolist()
            mask = loss_mask[i, :seqlen].tolist()
            # Skip samples with empty completions (all prompt, no completion tokens)
            if mask[-1] != 1:
                continue

            prompt_end = seqlen - sum(mask)
            prompt_ids = ids[:prompt_end]
            completion_ids = ids[prompt_end:]

            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion_text = tokenizer.decode(
                completion_ids, skip_special_tokens=False
            )

            records.append({
                "task_id": int(task_id),
                "sample_idx": int(i),
                "seqlen": int(seqlen),
                "prompt_len": int(prompt_end),
                "head_version": int(head_version),
                "tail_version": int(tail_version),
                "reward": float(rewards[i].item()),
                "prompt": prompt_text,
                "completion": completion_text,
                "slurm_job_id": self._slurm_job_id,
                "worker_uid": self._worker_uid,
            })

        if not records:
            return True, ""

        # JSONL append: one line per record. The dump_dir-keyed asyncio Lock
        # serializes concurrent dump coroutines within this process so lines
        # never interleave; different processes write to different filenames
        # (worker_uid embeds host + pid). The blob is built outside the lock
        # so we hold it only across the OS write (~hundreds of µs for ~16 KB
        # of JSON).
        blob = (
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
            + "\n"
        )
        file_path = os.path.join(dump_dir, f"worker_{self._worker_uid}.jsonl")
        lock = self._dump_locks.get(dump_dir)
        if lock is None:
            # asyncio.Lock() must be created on the running loop; lazy-init.
            # dict.setdefault is thread-safe under the GIL for CPython.
            lock = self._dump_locks.setdefault(dump_dir, asyncio.Lock())
        async with lock:
            # Sync open/write is fine — kernel append + close is fast for
            # ~16 KB and runs ~20×/s/worker max, so event-loop stall is
            # well under 1 % wall time. No fsync: SIGKILL leaves prior
            # records durable in the page cache (kernel-side flush handles
            # them) and only the in-flight buffered line, if any, is lost.
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(blob)

        return True, ""

    def initialize(self, logger=None, train_data_parallel_size: int | None = None):
        """Initialize the workflow executor and start background threads.

        Creates and initializes BatchTaskDispatcher with StalenessManager.
        The dispatcher starts producer and consumer threads for async execution.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for debugging and tracing. If None, creates a
            default logger.
        train_data_parallel_size : int | None, optional
            Data parallel world size for capacity scaling. If None, will be inferred
            from distributed state.
        """
        if logger is None:
            dist_ready = dist.is_initialized()
            name = (
                f"WorkflowExecutor Rank {dist.get_rank()}"
                if dist_ready
                else "WorkflowExecutor"
            )
            logger = logging.getLogger(name)
        self.logger = logger

        # Initialize staleness manager if not provided
        if self._staleness_manager is None:
            if train_data_parallel_size is not None:
                dp_world_size = train_data_parallel_size
            else:
                dp_world_size = self._resolve_dp_world_size()

            # The controller enforces the exact global staleness limit.  A worker
            # must therefore not impose an equal-share staleness quota: requests
            # are routed round-robin, but reward filtering makes accepted counts
            # uneven.  Near a full fixed-lag window, requests can otherwise be
            # stranded on locally-full workers while other workers have spare
            # capacity, leaving no runnable rollout that can finish the batch.
            # Keep only concurrency local and give every worker the global batch
            # allowance; the controller remains the authoritative global bound.
            max_concurrent_rollouts = ceil_div(
                self.max_concurrent_rollouts, dp_world_size
            )
            consumer_batch_size = self.consumer_batch_size

            self._staleness_manager = StalenessManager(
                version_provider=self.inference_engine,
                max_concurrent_rollouts=max_concurrent_rollouts,
                consumer_batch_size=consumer_batch_size,
                max_staleness=self.config.max_head_offpolicyness,
                rejection_sampling_reserve=ceil_div(
                    self.config.rejection_sampling_reserve, dp_world_size
                ),
            )

        # Create and initialize the dispatcher
        qsize = self.config.queue_size or self.max_concurrent_rollouts * 16
        self._dispatcher = BatchTaskDispatcher[_RolloutTaskInput, _RolloutResult](
            max_queue_size=qsize,
            task_factory=self._create_workflow_task,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )

        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

        # Start the memory leak diagnostic logger (daemon thread).
        if os.environ.get("AREAL_MEMLOG", "1") != "0":
            self._memlog_thread = threading.Thread(
                target=self._memlog_loop,
                name=f"MemLog-{self._worker_uid}",
                daemon=True,
            )
            self._memlog_thread.start()

    def _memlog_loop(self):
        """Periodic host-RAM / object-count snapshot for leak hunting.

        Logged fields:
          rss_gb          : process resident set size
          peak_gb         : high-water rss
          fds             : open file descriptors
          dump_dirs       : # of dump_dir asyncio Locks held (= # roles dumping)
          ray_refs        : ray.ObjectRef instance count
          tensor_objs     : torch.Tensor instance count
          top_types       : top 10 Python type counts by instance
        """
        interval = float(os.environ.get("AREAL_MEMLOG_INTERVAL", "60"))
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            proc = None
        peak_rss = 0.0
        import gc
        while not self._memlog_stop.is_set():
            try:
                if proc is not None:
                    rss_gb = proc.memory_info().rss / 1e9
                    peak_rss = max(peak_rss, rss_gb)
                    try:
                        fd_count = proc.num_fds()
                    except Exception:
                        fd_count = -1
                else:
                    rss_gb = -1.0
                    fd_count = -1
                # Cheap object-type tally; gc.get_objects can be 1M+ entries
                # on a fat actor — keep it bounded by sampling.
                type_counts: dict[str, int] = {}
                ray_refs = 0
                tensor_objs = 0
                for o in gc.get_objects():
                    t = type(o).__name__
                    type_counts[t] = type_counts.get(t, 0) + 1
                    if t == "ObjectRef":
                        ray_refs += 1
                    elif t == "Tensor":
                        tensor_objs += 1
                top10 = sorted(type_counts.items(), key=lambda kv: -kv[1])[:10]
                top10_str = ",".join(f"{n}={c}" for n, c in top10)
                dispatcher_state: dict[str, Any] = {}
                if self._dispatcher is not None:
                    dispatcher_state = self._dispatcher._active_wait_debug_state()
                    dispatcher_state["runner_running_tasks"] = (
                        self._dispatcher.runner.get_running_task_count()
                    )
                from areal.api.reward_api import AsyncRewardWrapper

                reward_state = AsyncRewardWrapper.debug_state()
                self.logger.info(
                    f"[MemLog] uid={self._worker_uid} "
                    f"rss={rss_gb:.2f}GB peak={peak_rss:.2f}GB "
                    f"fds={fd_count} dump_dirs={len(self._dump_locks)} "
                    f"ray_refs={ray_refs} tensor_objs={tensor_objs} "
                    f"version={self.inference_engine.get_version()} "
                    f"generation={self.inference_engine.get_generation_debug_state()} "
                    f"dispatcher={dispatcher_state} reward={reward_state} "
                    f"top10=[{top10_str}]"
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[MemLog] failed: {e}")
            self._memlog_stop.wait(interval)

    def destroy(self):
        """Shutdown the workflow executor and clean up resources.

        Destroys the dispatcher (stops background threads + AsyncTaskRunner)
        and flushes the performance tracer. Trajectory dumps are JSONL
        append-only, so there's nothing to close or finalize on the dump
        side — files are valid at every line boundary.
        """
        # Stop the memory log thread (best-effort; it's daemon so it dies with proc).
        self._memlog_stop.set()
        if self._memlog_thread is not None:
            self._memlog_thread.join(timeout=2)

        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        # Flush performance tracer
        tracer = perf_tracer.get_session_tracer()
        if tracer is not None:
            tracer.flush(force=True)

    def get_capacity(self):
        """Get current available capacity for new rollouts.

        Returns
        -------
        int
            Number of new rollout slots available based on staleness constraints.
        """
        return self.staleness_manager.get_capacity()

    def _rollout_stats(self) -> str:
        stats = self.staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_workflow_task(
        self, pending_task: _RolloutTaskInput
    ) -> Callable[[], Awaitable[_RolloutResult | None]]:
        """Wrapper to create an async function that will be executed by AsyncTaskRunner.

        This is a synchronous function that returns an async function, which allows
        us to capture the pending_task context.

        Parameters
        ----------
        pending_task : _RolloutTaskInput
            The rollout task input containing workflow, data, and filter callback.

        Returns
        -------
        Callable
            An async function that executes the workflow and applies
            filtering/validation.
        """

        async def _execute_workflow() -> _RolloutResult | None:
            """Execute workflow.arun_episode and apply AReaL-specific logic."""
            task_id = pending_task.task_id

            # Set task_id in ContextVar before entering arun_episode
            perf_tracer.set_task_id(task_id)

            # Set workflow execution context
            workflow_context.set(
                WorkflowContext(is_eval=pending_task.is_eval, task_id=task_id)
            )

            traj: dict[str, Any] | None = None
            should_accept_fn = pending_task.should_accept_fn
            should_accept: bool | None = None
            reason: str | None = None

            try:
                # 2026-06-18: belt-and-suspenders trajectory timeout enforced HERE
                # (workflow_executor) in addition to GroupedRolloutWorkflow's inner
                # timeout. If a single arun_episode call (one whole group) takes
                # > AREAL_TRAJECTORY_TIMEOUT_SECONDS (default 360s — 2x train sample
                # 180s to account for group_size×sample concurrent budget), we
                # cancel the task and treat it as rejected. Surfaces in log as
                # "TrajectoryTimeout" — distinct from inner GroupedRollout warnings.
                _traj_timeout = float(
                    os.environ.get("AREAL_TRAJECTORY_TIMEOUT_SECONDS", "360")
                )
                if _traj_timeout > 0 and not pending_task.is_eval:
                    try:
                        traj = await asyncio.wait_for(
                            pending_task.workflow.arun_episode(
                                self.inference_engine, pending_task.data
                            ),
                            timeout=_traj_timeout,
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            "TrajectoryTimeout: workflow.arun_episode timed out after "
                            "%.1fs for task_id=%s (workflow class=%s); rejecting",
                            _traj_timeout,
                            task_id,
                            type(pending_task.workflow).__name__,
                        )
                        traj = None
                else:
                    traj = await pending_task.workflow.arun_episode(
                        self.inference_engine, pending_task.data
                    )

                # Trajectory format checking
                if self.config.check_trajectory_format and traj is not None:
                    check_trajectory_format(
                        traj,
                        expected_keys=self._expected_trajectory_keys,
                        logger=self.logger,
                    )
                    # Track expected keys for consistency checking
                    if isinstance(traj, dict) and "input_ids" in traj:
                        if self._expected_trajectory_keys is None:
                            self._expected_trajectory_keys = set(traj.keys())
                            self.logger.info(
                                "Trajectory format check: tracking keys %s",
                                self._expected_trajectory_keys,
                            )

                # Convert InteractionWithTokenLogpReward to tensor dict if needed.
                # External-API interactions have no tensor data; fall back to
                # concat_string_interactions which produces a plain dict of
                # request/response strings instead of padded tensors.
                if isinstance(traj, dict) and all(
                    isinstance(v, InteractionWithTokenLogpReward) for v in traj.values()
                ):
                    if all(v.has_tensor_data for v in traj.values()):
                        traj = concat_padded_tensors(
                            [v.to_tensor_dict() for v in traj.values()]
                        )
                    else:
                        traj = concat_string_interactions(traj)

                assert traj is None or isinstance(traj, dict), traj

                if traj is None:
                    should_accept_traj = False
                    reason = "returned_none"
                else:
                    if should_accept_fn is None:
                        should_accept = True
                    else:
                        should_accept = bool(should_accept_fn(traj))
                    should_accept_traj = bool(should_accept)
                    if not should_accept_traj and should_accept_fn is not None:
                        reason = "rejected"

                # Dump trajectory to file
                if self.config.dump_to_file:
                    dump_success, dump_reason = await self._dump_trajectory(
                        traj, task_id, pending_task.is_eval
                    )
                    if not dump_success:
                        self.logger.warning(
                            f"Failed to dump trajectory for task {task_id}: {dump_reason}"
                        )

                if should_accept_traj:
                    stats_tracker.get("rollout").scalar(accepted=1)
                    trace_session_event(
                        "mark_finalized",
                        task_id=task_id,
                        status="accepted",
                    )
                    if self.config.enable_rollout_tracing:
                        self.logger.info(
                            f"Finish and accept rollout task {task_id}.",
                        )
                    assert traj is not None
                    return _RolloutResult(task_id=task_id, trajectory=traj)

                stats_tracker.get("rollout").scalar(rejected=1)
                trace_session_event(
                    "mark_finalized",
                    task_id=task_id,
                    status="rejected",
                    reason=reason,
                )
                if self.config.enable_rollout_tracing:
                    self.logger.info(
                        f"Finish but reject rollout task {task_id}.",
                    )
                return None

            except Exception as exc:  # pragma: no cover - workflow execution errors
                stats_tracker.get("rollout").scalar(rejected=1)
                trace_session_event(
                    "mark_finalized",
                    task_id=task_id,
                    status="failed",
                    reason="workflow_exception",
                )
                if self.logger is not None:
                    self.logger.error(
                        "Workflow execution failed: %s", exc, exc_info=True
                    )
                return None

        return _execute_workflow

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow,
        should_accept_fn: Callable[[dict[str, Any]], bool] = None,
        task_id: int | None = None,
        is_eval: bool = False,
    ) -> int:
        """Submit a rollout request to the workflow executor.

        Enqueues the request to _pending_inputs. The background producer thread
        will submit it to AsyncTaskRunner when staleness capacity allows. Non-blocking.

        See :meth:`~areal.api.engine_api.InferenceEngine.submit` for parameters.
        """
        if task_id is None:
            task_id = self._task_id_generator.next()
        perf_tracer.register_task(task_id)
        task_input = _RolloutTaskInput(
            data=data,
            workflow=workflow,
            should_accept_fn=should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)
        return task_id

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for the completion of `count` workflows.

        Returns a list of trajectory dictionaries (or None for rejected rollouts).
        Results are sorted by creation time and shuffled for diversity.

        See :meth:`~areal.api.engine_api.InferenceEngine.wait` for parameters.
        """
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            self.logger.info("Rollout results are ready!")
        return [r.trajectory if r is not None else None for r in results]

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """
        Wait for a specific workflow task to complete.
        Parameters
        ----------
        task_id : int
            The ID of the workflow task to wait for.
        timeout : float or None, optional
            Maximum time to wait for the task to complete, in seconds. If None, wait indefinitely.
        raise_timeout : bool, optional
            If True, raise TimeoutError if the task does not complete within the timeout.
        Returns
        -------
        dict[str, Any] or None
            The trajectory dictionary for the completed task, or None if the rollout was rejected.
        Raises
        ------
        ValueError
            If the task_id is invalid.
        TimeoutError
            If the task does not complete within the specified timeout and raise_timeout is True.
        See Also
        --------
        :meth:`~areal.api.engine_api.InferenceEngine.wait_for_task`
        """
        result = self.dispatcher.wait_for_task(task_id, timeout, raise_timeout)

        if result is not None and self.config.enable_rollout_tracing:
            self.logger.info(f"Task {task_id} completed successfully")
        return result.trajectory if result is not None else None

    def cancel_task(self, task_id: int) -> bool:
        """Cancel a submitted workflow task if it is still pending or running."""
        return self.dispatcher.cancel_task(task_id)

    @trace_perf("workflow_executor.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow,
    ) -> list[dict[str, Any]]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        See :meth:`~areal.api.engine_api.InferenceEngine.rollout_batch` for
        detailed documentation.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        perf_tracer.instant(
            "workflow_executor.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
            )
        results = self.wait(count=len(data))
        # Return list of trajectory dicts (filter out None)
        return [r for r in results if r is not None]

    @trace_perf("workflow_executor.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow,
        should_accept_fn: Callable[[dict[str, Any]], bool] = None,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        .. warning::

            This method caches an internal data generator on the first call.
            The ``dataloader``, ``workflow``, and ``should_accept_fn`` parameters
            are captured at the first invocation and reused in all subsequent calls.
            Passing different arguments in later calls will **not** take effect.

            If you need to switch configurations mid-training, consider:

            - Using a separate :class:`WorkflowExecutor` (or engine) instance
            - Using the :meth:`submit` / :meth:`wait` pattern for finer control

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """

        def task_input_generator():
            for data in cycle_dataloader(dataloader):
                for item in data:
                    # Workflow is already resolved by RemoteInfEngine
                    task_id = self._task_id_generator.next()
                    perf_tracer.register_task(task_id)
                    yield _RolloutTaskInput(
                        data=item,
                        workflow=workflow,
                        should_accept_fn=should_accept_fn,
                        task_id=task_id,
                    )

        if not hasattr(self, "data_generator"):
            self.data_generator = task_input_generator()

        # Delegate to dispatcher
        assert dataloader.batch_size is not None
        results = self.dispatcher.active_submit_and_wait(
            self.data_generator, batch_size=dataloader.batch_size, dynamic_bs=dynamic_bs,
            ready_batch_buffer=getattr(self.config, "ready_batch_buffer", 1),
        )

        # Return list of trajectory dicts (filter out None)
        return [r.trajectory for r in results if r is not None]

    def pause(self):
        """Pause request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.pause` for detailed
        documentation.
        """
        self.dispatcher.pause()

    def resume(self):
        """Resume request submission for async rollout.

        See :meth:`~areal.api.engine_api.InferenceEngine.resume` for detailed
        documentation.
        """
        self.dispatcher.resume()

    def is_paused(self):
        return self.dispatcher.is_paused()

    @property
    def staleness_manager(self) -> StalenessManager:
        manager = self._staleness_manager
        if manager is None:
            raise RuntimeError(
                "WorkflowExecutor.initialize() must be called before scheduling rollouts."
            )
        return manager

    @property
    def dispatcher(self) -> BatchTaskDispatcher[_RolloutTaskInput, _RolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "WorkflowExecutor.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
