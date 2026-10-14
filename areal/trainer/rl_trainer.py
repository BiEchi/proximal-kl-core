# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    RolloutWorkflow,
    SaveLoadMeta,
    Scheduler,
    StepInfo,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SGLangConfig,
    TrainDatasetConfig,
    ValidDatasetConfig,
    vLLMConfig,
)
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.experimental.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.infra import (
    LocalScheduler,
    RayScheduler,
    RolloutController,
    SlurmScheduler,
    current_platform,
)
from areal.infra.data_service import DataController
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.rdataset import RDataset
from areal.infra.utils.concurrent import call_maybe_async
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.environ import is_single_controller
from areal.utils.eval_per_source import (
    aggregate_eval_jsonl,
    build_prompt_source_map,
    compute_train_pass_at_n,
)
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.utils.walltime import WalltimeGuard

if TYPE_CHECKING:
    from datasets import Dataset

    from areal.engine import (
        FSDPPPOActor,
        FSDPPPOCritic,
        MegatronPPOActor,
        MegatronPPOCritic,
    )
    from areal.experimental.engine.archon_engine import ArchonPPOActor, ArchonPPOCritic
    from areal.trainer.ppo.actor import PPOActorController
    from areal.trainer.ppo.critic import PPOCriticController

logger = logging.getLogger("RLTrainer")


def _should_create_ref_model(config: PPOConfig) -> bool:
    if config.ref is None or config.actor.kl_ctl <= 0:
        return False
    return not config.actor.use_frpo_loss_term or (
        config.actor.frpo_anchor == "ref"
    )


def _reference_step_for_global_step(global_step: int, interval: int) -> int:
    """Return the actor version held fixed as reference for ``global_step``."""
    return global_step - global_step % interval


def _reference_snapshots_to_keep(
    current_reference_step: int,
    last_recovery_step: int,
    interval: int,
) -> set[int]:
    """Return the minimal reference snapshots needed for live and resumed work."""
    keep = {current_reference_step}
    if last_recovery_step < 0:
        return keep

    resume_step = last_recovery_step + 1
    recovery_reference_step = _reference_step_for_global_step(
        resume_step, interval
    )
    # At a refresh boundary the recovered actor already is the new reference,
    # so _ensure_reference_for_step can recreate it directly.
    if recovery_reference_step != resume_step:
        keep.add(recovery_reference_step)
    return keep


class _EmptyDataLoader:
    """Minimal dataloader for online mode that yields empty dicts.

    Compatible with ``cycle_dataloader()`` and ``len()`` expectations.
    ``steps_per_epoch`` controls how many steps constitute one epoch,
    derived from ``total_train_steps // total_train_epochs`` to ensure
    epoch-frequency-gated components (Saver, RecoverHandler) behave correctly.
    """

    def __init__(self, batch_size: int = 1, steps_per_epoch: int = 1):
        self.batch_size = batch_size
        self._steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self._steps_per_epoch

    def __iter__(self):
        while True:
            yield [{} for _ in range(self.batch_size)]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:  # noqa: ARG002
        pass


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        train_dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
    ):
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            # Set up file logging for controller process
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()
        self.data_controller: DataController | None = None
        self._train_rdataset: RDataset | None = None
        self._valid_rdataset: RDataset | None = None

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        # Parse per-engine allocations from config.
        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")
        self.rollout_alloc = ModelAllocation.from_str(
            config.rollout.backend, name="rollout"
        )
        self._should_offload_rollout = self._is_actor_rollout_colocated(config)
        self._should_offload_actor = (
            self._should_offload_rollout or config.actor.offload
        )
        self._should_offload_critic = (
            config.critic is not None and config.critic.offload
        )
        self._should_offload_ref = config.ref is not None and config.ref.offload
        self._should_offload_teacher = (
            config.teacher is not None and config.teacher.offload
        )

        # Validate config before proceeding with weight initialization
        self._validate_cfg()

        self._amend_xccl_weight_update_envvar()

        agent_cfg = config.rollout.agent
        self._online_mode = agent_cfg is not None and agent_cfg.mode == "online"

        if self._online_mode and config.valid_dataset is not None:
            raise ValueError(
                "valid_dataset must not be set when using online RL mode "
                "(openai.mode='online'). Online mode does not support "
                "validation datasets."
            )

        # -- Dataset loading --------------------------------------------------
        if not self._online_mode and train_dataset is None:
            raise ValueError(
                "train_dataset must be provided unless using online RL mode "
                "(openai.mode='online')."
            )

        # Create models: actor, critic, ref — each with its own allocation.
        self.actor = self._create_train_engine(config.actor, self.actor_alloc)
        self.critic = None
        if config.critic is not None:
            critic_alloc = ModelAllocation.from_str(
                config.critic.backend, name="critic"
            )
            self.critic = self._create_critic(config.critic, critic_alloc)
        self.ref = None
        if _should_create_ref_model(config):
            ref_alloc = ModelAllocation.from_str(config.ref.backend, name="ref")
            self.ref = self._create_train_engine(config.ref, ref_alloc)

        self.teacher = None
        if config.teacher is not None:
            teacher_alloc = ModelAllocation.from_str(
                config.teacher.backend, name="teacher"
            )
            self.teacher = self._create_train_engine(config.teacher, teacher_alloc)

        steps_per_epoch: int | None = None
        self.train_dataloader: StatefulDataLoader | _EmptyDataLoader
        if self._online_mode:
            if config.total_train_steps is None:
                raise ValueError(
                    "total_train_steps must be set for online mode. "
                    "Both total_train_epochs and total_train_steps are needed "
                    "to compute steps_per_epoch."
                )
            steps_per_epoch = config.total_train_steps // config.total_train_epochs
            if steps_per_epoch < 1:
                raise ValueError(
                    f"total_train_steps ({config.total_train_steps}) must be >= "
                    f"total_train_epochs ({config.total_train_epochs}) so that "
                    f"steps_per_epoch >= 1."
                )
            self.train_dataloader = _EmptyDataLoader(
                batch_size=config.train_dataset.batch_size,
                steps_per_epoch=steps_per_epoch,
            )
        else:
            assert train_dataset is not None
            if is_single_controller() and isinstance(train_dataset, RDataset):
                ds_cfg = DataServiceConfig.from_dataset_config(
                    config.train_dataset, seed=config.seed
                )
                assert self.scheduler is not None
                controller = DataController(ds_cfg, self.scheduler)
                controller.initialize(
                    role="data", num_dataset_workers=ds_cfg.num_workers
                )
                self.data_controller = controller
                train_dataset.connect(
                    controller,
                    dataset_id=f"{config.experiment_name}_{config.trial_name}_train",
                    tokenizer_or_processor_path=config.tokenizer_path,
                    shuffle=config.train_dataset.shuffle,
                    drop_last=config.train_dataset.drop_last,
                )
                self._train_rdataset = train_dataset

            self.train_dataloader = self._create_dataloader(
                train_dataset,
                dataset_config=self.config.train_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        self.valid_dataloader: StatefulDataLoader | None = None
        if self.config.valid_dataset is not None and valid_dataset is not None:
            assert self.config.valid_dataset is not None
            if is_single_controller() and isinstance(valid_dataset, RDataset):
                assert self.data_controller is not None
                valid_dataset.connect(
                    self.data_controller,
                    dataset_id=f"{config.experiment_name}_{config.trial_name}_valid",
                    tokenizer_or_processor_path=config.tokenizer_path,
                    shuffle=self.config.valid_dataset.shuffle,
                    drop_last=self.config.valid_dataset.drop_last,
                )
                self._valid_rdataset = valid_dataset

            self.valid_dataloader = self._create_dataloader(
                valid_dataset,
                dataset_config=self.config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        # -- FinetuneSpec -----------------------------------------------------
        if self._online_mode:
            assert steps_per_epoch is not None
            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=steps_per_epoch * config.train_dataset.batch_size,
                train_batch_size=config.train_dataset.batch_size,
            )
        else:
            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=len(self.train_dataloader)
                * config.train_dataset.batch_size,
                train_batch_size=config.train_dataset.batch_size,
                max_train_steps=config.total_train_steps,
            )

        # Initialize engines first — the scheduler must know about roles
        # before the data controller can colocate with them.
        engine_init_kwargs = {"addr": None, "ft_spec": ft_spec}
        self.rollout, initial_lora_path = self._initialize_engines_and_rollout(
            engine_init_kwargs
        )
        self._reference_version: int | None = None

        self.eval_rollout = None
        if not self._online_mode:
            self.eval_rollout = self._init_rollout(
                config.rollout, is_eval=True, lora_path=initial_lora_path
            )

        # Proxy worker initialization (lazy, for AgentWorkflow support)
        self._proxy_started = False

        # Prepare weight update meta and connect to inference engine
        if self.config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            if config.actor.use_lora:
                disk_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                    }
                )
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif self.config.actor.weight_update_mode == "xccl":
            # NCCL/XCCL weight update
            xccl_kwargs: dict[str, Any] = {
                "gen_allocation": self.rollout_alloc,
            }

            if config.actor.use_lora:
                xccl_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                    }
                )

            if self.actor_alloc.backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    **xccl_kwargs
                )
            else:
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        else:
            raise ValueError(
                f"Invalid weight update mode: {self.config.actor.weight_update_mode}"
            )

        self.actor.connect_engine(self.rollout, self.weight_update_meta)

        # Set up evaluation (skip in online mode)
        self.evaluator = Evaluator(config.evaluator, ft_spec)
        # Lazily-built map for per-source eval metric logging (rank 0 only).
        self._eval_prompt_to_source: dict[str, str] | None = None

        # Set up save as HF model
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)
        self.walltime_guard = WalltimeGuard.from_env()

        # Set up statistics logging (wandb, tensoboard, etc.)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Set up checkpointing for recover
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )

        # After recovery, sync the staleness manager so its capacity formula
        # stays bounded despite the version jumping from 0 to recovery_version.
        if self.recover_info is not None:
            recovery_version = self.recover_info.last_step_info.global_step + 1
            self._sync_rollout_staleness_after_version_jump(recovery_version)
        elif self.weight_update_meta.type == "xccl":
            self._sync_initial_rollout_weights()

        self._config_perf_tracer()
        self._apply_initial_offload_policy()

    @staticmethod
    def _is_colocation(strategy: SchedulingStrategy | None) -> bool:
        if strategy is None:
            return False
        return strategy.type in (
            SchedulingStrategyType.colocation,
            SchedulingStrategyType.colocation.value,
            "colocation",
        )

    def _is_actor_rollout_colocated(self, config: PPOConfig) -> bool:
        actor_s = config.actor.scheduling_strategy
        rollout_s = config.rollout.scheduling_strategy
        return (self._is_colocation(actor_s) and actor_s.target == "rollout") or (
            self._is_colocation(rollout_s) and rollout_s.target == "actor"
        )

    def _onload_model(self, engine, role: str) -> None:
        with (
            stats_tracker.record_timing(f"{role}_onload"),
            perf_tracer.trace_scope(
                f"train.{role}_onload",
                category=Category.IO,
            ),
        ):
            engine.onload()

    def _offload_model(self, engine, role: str) -> None:
        with (
            stats_tracker.record_timing(f"{role}_offload"),
            perf_tracer.trace_scope(
                f"train.{role}_offload",
                category=Category.IO,
            ),
        ):
            engine.offload()

    def _offload_rollout(self, is_eval: bool = False):
        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        with (
            stats_tracker.record_timing("rollout_pause"),
            perf_tracer.trace_scope(
                "train.rollout_pause",
                category=Category.INSTR,
            ),
        ):
            rollout.pause()

        with (
            stats_tracker.record_timing("rollout_pause_generation"),
            perf_tracer.trace_scope(
                "train.rollout_pause_generation",
                category=Category.INSTR,
            ),
        ):
            call_maybe_async(rollout.pause_generation)

        with (
            stats_tracker.record_timing("rollout_offload"),
            perf_tracer.trace_scope(
                "train.rollout_offload",
                category=Category.IO,
            ),
        ):
            rollout.offload()

    def _onload_rollout(self, is_eval: bool = False) -> None:
        cleanup_error: Exception | None = None

        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        try:
            with (
                stats_tracker.record_timing("rollout_onload"),
                perf_tracer.trace_scope(
                    "train.rollout_onload",
                    category=Category.IO,
                ),
            ):
                rollout.onload()
        except Exception as exc:  # noqa: BLE001
            cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_continue_generation"),
                perf_tracer.trace_scope(
                    "train.rollout_continue_generation",
                    category=Category.INSTR,
                ),
            ):
                call_maybe_async(rollout.continue_generation)
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_resume"),
                perf_tracer.trace_scope(
                    "train.rollout_resume",
                    category=Category.INSTR,
                ),
            ):
                rollout.resume()
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        if cleanup_error is not None:
            raise cleanup_error

    def _apply_initial_offload_policy(self) -> None:
        if self._should_offload_rollout:
            self._offload_rollout()
        if self._should_offload_ref:
            self._offload_model(self.ref, role="ref")
        if self._should_offload_critic:
            self._offload_model(self.critic, role="critic")
        if self._should_offload_teacher:
            self._offload_model(self.teacher, role="teacher")
        if self._should_offload_actor:
            self._offload_model(self.actor, role="actor")

    def _sync_rollout_staleness_after_version_jump(self, version: int) -> None:
        if is_single_controller():
            sm = self.rollout.staleness_manager
        else:
            sm = self.rollout.workflow_executor.staleness_manager
        if sm is not None:
            sm.on_version_recovered(version)

    def _sync_initial_rollout_weights(self) -> None:
        """Mirror verl: push actor weights to rollout before step-0 eval."""
        logger.info("Synchronizing initial actor weights to rollout at version 0")
        initial_meta = self.weight_update_meta.with_version(0)
        self.actor.update_weights(initial_meta)
        self.actor.set_version(0)
        if self.critic is not None:
            self.critic.set_version(0)
        self.rollout.set_version(0)
        if self.eval_rollout is not None:
            self.eval_rollout.set_version(0)

    def _reference_snapshot_path(self, reference_step: int) -> Path:
        root = Saver.get_save_root(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.cluster.fileroot,
        )
        return Path(root) / "reference_snapshots" / f"step{reference_step}"

    def _prune_reference_snapshots(self, current_reference_step: int) -> None:
        """Keep only the live reference and the one needed by recovery."""
        interval = self.config.ref_refresh_steps
        if interval is None:
            return

        last_recovery_step = self.recover_handler.last_step_info.global_step
        if self.recover_info is not None:
            last_recovery_step = max(
                last_recovery_step,
                self.recover_info.last_step_info.global_step,
            )
        keep_steps = _reference_snapshots_to_keep(
            current_reference_step,
            last_recovery_step,
            interval,
        )
        snapshots_root = self._reference_snapshot_path(0).parent
        if not snapshots_root.is_dir():
            return

        for snapshot_path in snapshots_root.iterdir():
            if not snapshot_path.is_dir() or not snapshot_path.name.startswith("step"):
                continue
            try:
                snapshot_step = int(snapshot_path.name.removeprefix("step"))
            except ValueError:
                continue
            if snapshot_step in keep_steps:
                continue
            shutil.rmtree(snapshot_path)
            logger.info("Removed obsolete reference snapshot %s", snapshot_path)

    def _ensure_reference_for_step(self, global_step: int) -> None:
        """Load or create the fixed reference required by this training step."""
        interval = self.config.ref_refresh_steps
        if interval is None:
            return
        if self.ref is None:
            raise RuntimeError("ref_refresh_steps requires an initialized ref engine")

        reference_step = _reference_step_for_global_step(global_step, interval)
        if self._reference_version == reference_step:
            return
        if reference_step == 0:
            self.ref.set_version(0)
            self._reference_version = 0
            logger.info(
                "Reference policy set to actor version 0 for training steps 0--%s",
                interval - 1,
            )
            return

        snapshot_path = self._reference_snapshot_path(reference_step)
        complete_marker = snapshot_path / ".complete"
        meta = SaveLoadMeta(
            path=str(snapshot_path),
            weight_format="dcp",
            with_optim=False,
        )
        if not complete_marker.exists():
            if global_step != reference_step:
                raise RuntimeError(
                    "Missing reference snapshot for resumed training: "
                    f"step={global_step}, expected_reference={reference_step}, "
                    f"path={snapshot_path}"
                )
            logger.info(
                "Saving actor version %s as the next fixed reference at %s",
                reference_step,
                snapshot_path,
            )
            self.actor.save(meta)
            complete_marker.write_text(f"{reference_step}\n")

        self.ref.load(meta)
        self.ref.set_version(reference_step)
        self._reference_version = reference_step
        self._prune_reference_snapshots(reference_step)
        logger.info(
            "Reference policy set to actor version %s for training step %s",
            reference_step,
            global_step,
        )

    def train(
        self,
        workflow: WorkflowLike | None = None,
        eval_workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        total_epochs: int | None = None,
    ):
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )
        fake_start_step = os.environ.get("AREAL_DEBUG_FAKE_START_STEP")
        if fake_start_step:
            start_step = int(fake_start_step)
            if start_step < 0:
                raise ValueError(
                    "AREAL_DEBUG_FAKE_START_STEP must be non-negative, "
                    f"got {start_step}."
                )
            logger.warning(
                "Using debug fake start step %s without loading a matching "
                "training checkpoint. This is only valid for scheduler/rollout "
                "mechanism debugging.",
                start_step,
            )
            self.actor.set_version(start_step)
            if self.critic is not None:
                self.critic.set_version(start_step)
            self.rollout.set_version(start_step)
            if self.eval_rollout is not None:
                self.eval_rollout.set_version(start_step)
            self._sync_rollout_staleness_after_version_jump(start_step)

        self._ensure_reference_for_step(start_step)

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive: {total_epochs}")
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch
        last_completed_step: int | None = None
        stopped_for_walltime = False

        # Initialize proxy workers if not using RolloutWorkflow
        if workflow is None:
            agent_cfg = self.config.rollout.agent
            if agent_cfg is not None and agent_cfg.mode == "online":
                self._ensure_proxy_started()
            else:
                raise ValueError(
                    "workflow must be specified for train() unless "
                    "agent.mode='online' is configured. "
                    "Pass a RolloutWorkflow, AgentWorkflow, or callable."
                )
        elif self._requires_proxy_workflow(workflow):
            self._ensure_proxy_started()

        if start_step == 0 and self.evaluator.config.eval_before_train:
            if self._should_offload_rollout:
                self._onload_rollout(is_eval=True)
            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval_before_train",
                    category=Category.COMPUTE,
                    args={"global_step": start_step},
                ),
            ):
                self._evaluate(
                    eval_workflow=eval_workflow,
                    eval_workflow_kwargs=eval_workflow_kwargs,
                    epoch=0,
                    epoch_step=0,
                    global_step=start_step,
                    step_version=start_step,
                )
            if self._should_offload_rollout:
                self._offload_rollout(is_eval=True)

        for global_step in range(start_step, max_steps):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break
            step_started_at = time.monotonic()
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            self._ensure_reference_for_step(global_step)

            if self._should_offload_rollout:
                self._onload_rollout()
            with (
                stats_tracker.record_timing("rollout"),
                perf_tracer.trace_scope(
                    "train.rollout",
                    category=Category.COMPUTE,
                    args={
                        "global_step": global_step,
                        "epoch_step": step,
                    },
                ),
            ):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=dynamic_filter_fn,
                    group_size=config.gconfig.n_samples,
                    dynamic_bs=self.config.dynamic_bs,
                )

            # train/pass_at_n: same data path as the natively-logged
            # ppo_actor/task_reward/avg — read in-memory off rollout_batch,
            # stash for _export_and_commit_stats() so wandb sees both at the
            # same global_step. Replaces the prior JSONL-scan that the dump
            # cap silently starved.
            self._pending_train_metrics = compute_train_pass_at_n(rollout_batch)

            if self._should_offload_rollout:
                self._offload_rollout()

            if self.critic is not None:
                if self._should_offload_critic:
                    self._onload_model(self.critic, role="critic")
                with (
                    stats_tracker.record_timing("critic_values"),
                    perf_tracer.trace_scope(
                        "train.compute_values",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    values = self.critic.compute_values(rollout_batch)
                    for traj, v in zip(rollout_batch, values):
                        traj["values"] = v
                    self.critic.get_device_stats().log("critic values")
                # Critic stays onloaded — offloaded after ppo_update below

            if self.ref is not None:
                if self._should_offload_ref:
                    self._onload_model(self.ref, role="ref")
                with (
                    stats_tracker.record_timing("ref_logp"),
                    perf_tracer.trace_scope(
                        "train.ref_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    ref_logps = self.ref.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, ref_logps):
                        traj["ref_logp"] = logp
                    self.ref.get_device_stats().log("ref logp")
                if self._should_offload_ref:
                    self._offload_model(self.ref, role="ref")

            if self.teacher is not None:
                if self._should_offload_teacher:
                    self._onload_model(self.teacher, role="teacher")
                with (
                    stats_tracker.record_timing("teacher_logp"),
                    perf_tracer.trace_scope(
                        "train.teacher_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    teacher_logps = self.teacher.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, teacher_logps):
                        traj["teacher_logp"] = logp
                        traj["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                        traj["distill_loss_weight"] = (
                            self.config.teacher.distill_loss_weight
                        )
                    self.teacher.get_device_stats().log("teacher logp")
                if self._should_offload_teacher:
                    self._offload_model(self.teacher, role="teacher")

            if self._should_offload_actor:
                self._onload_model(self.actor, role="actor")
            if config.actor.should_compute_prox_logp():
                with (
                    stats_tracker.record_timing("recompute_logp"),
                    perf_tracer.trace_scope(
                        "train.recompute_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    prox_logps = self.actor.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, prox_logps):
                        traj["prox_logp"] = logp
                    self.actor.get_device_stats().log("recompute logp")

            with (
                stats_tracker.record_timing("compute_advantage"),
                perf_tracer.trace_scope(
                    "train.compute_advantage",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                adv_batch = self.actor.compute_advantages(rollout_batch)
                self.actor.get_device_stats().log("compute advantages")

            # Wait for async checkpoint staging to complete before modifying parameters
            self.saver.maybe_wait_for_staging()

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
            ):
                self.actor.start_memory_profile(config.memory_profiler.max_entries)

            with (
                stats_tracker.record_timing("train_step"),
                perf_tracer.trace_scope(
                    "train.ppo_update",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self.actor.ppo_update(adv_batch)
                self.actor.step_lr_scheduler()
                self.actor.get_device_stats().log("ppo update")

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
            ):
                log_dir = StatsLogger.get_log_path(config.stats_logger)
                snapshot_dir = os.path.join(
                    log_dir, "memory_snapshots", f"step_{global_step}"
                )
                os.makedirs(snapshot_dir, exist_ok=True)
                self.actor.stop_memory_profile(snapshot_dir)
                logger.info(f"Memory snapshots saved to {snapshot_dir}")

            if self.critic is not None:
                with (
                    stats_tracker.record_timing("critic_train_step"),
                    perf_tracer.trace_scope(
                        "train.critic_ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.critic.ppo_update(adv_batch)
                    self.critic.step_lr_scheduler()
                    self.critic.get_device_stats().log("ppo critic update")
                if self._should_offload_critic:
                    self._offload_model(self.critic, role="critic")

            # pause inference for updating weights, save, and evaluation
            self.rollout.pause()

            with (
                stats_tracker.record_timing("rollout_pause_generation"),
                perf_tracer.trace_scope(
                    "train.rollout_pause_generation",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                call_maybe_async(self.rollout.pause_generation)

            # Actor already onloaded; engine-internal _offload_aware_context
            # calls in update_weights/save are no-ops.

            with (
                stats_tracker.record_timing("update_weights"),
                perf_tracer.trace_scope(
                    "train.update_weights",
                    category=Category.COMM,
                    args={"global_step": global_step},
                ),
            ):
                # Use versioned path for weight updates
                new_version = global_step + 1
                versioned_meta = self.weight_update_meta.with_version(new_version)
                self.actor.update_weights(versioned_meta)

                self.actor.set_version(new_version)
                if self.critic is not None:
                    self.critic.set_version(new_version)
                self.rollout.set_version(new_version)
                if self.eval_rollout is not None:
                    self.eval_rollout.set_version(new_version)

            with (
                stats_tracker.record_timing("rollout_continue_generation"),
                perf_tracer.trace_scope(
                    "train.rollout_continue_generation",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                call_maybe_async(self.rollout.continue_generation)

            walltime_decision = self.walltime_guard.decide(
                current_step_elapsed_seconds=time.monotonic() - step_started_at
            )
            if walltime_decision.should_exit:
                logger.warning(
                    "Walltime guard stopping after global_step=%s: remaining=%.1fs, "
                    "next_step_budget=%.1fs, checkpoint_budget=%.1fs, final_save=%s",
                    global_step,
                    walltime_decision.remaining_seconds,
                    walltime_decision.next_step_budget_seconds,
                    walltime_decision.checkpoint_budget_seconds,
                    walltime_decision.should_save_recovery,
                )
            else:
                with (
                    stats_tracker.record_timing("save"),
                    perf_tracer.trace_scope(
                        "train.save",
                        category=Category.IO,
                        args={"global_step": global_step},
                    ),
                ):
                    # A step limit can end training before the next epoch save.
                    self._save_hf(
                        epoch=epoch,
                        epoch_step=step,
                        global_step=global_step,
                        force=(
                            global_step + 1 == max_steps
                            or global_step + 1 == config.total_train_steps
                        ),
                    )

            if (
                not walltime_decision.should_exit
                or walltime_decision.should_save_recovery
            ):
                checkpoint_started_at = time.monotonic()
                with (
                    stats_tracker.record_timing("checkpoint_for_recover"),
                    perf_tracer.trace_scope(
                        "train.checkpoint",
                        category=Category.IO,
                        args={"global_step": global_step},
                    ),
                ):
                    checkpoint_saved = self._save_recover_checkpoint(
                        epoch=epoch,
                        epoch_step=step,
                        global_step=global_step,
                        force=walltime_decision.should_save_recovery,
                    )
                if checkpoint_saved:
                    self.walltime_guard.observe_checkpoint(
                        time.monotonic() - checkpoint_started_at
                    )

            # Offload actor before eval
            if self._should_offload_actor:
                self._offload_model(self.actor, role="actor")

            if self._should_offload_rollout and not walltime_decision.should_exit:
                self._onload_rollout(is_eval=True)
            if not walltime_decision.should_exit:
                with (
                    stats_tracker.record_timing("eval"),
                    perf_tracer.trace_scope(
                        "train.eval",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self._evaluate(
                        eval_workflow=eval_workflow,
                        eval_workflow_kwargs=eval_workflow_kwargs,
                        epoch=epoch,
                        epoch_step=step,
                        global_step=global_step,
                    )
            if self._should_offload_rollout and not walltime_decision.should_exit:
                self._offload_rollout(is_eval=True)

            with (
                stats_tracker.record_timing("clear_batches"),
                perf_tracer.trace_scope(
                    "train.clear_batches",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                # Each role runs in its own Python process with a
                # process-local ``_fetch_buffer``; one HTTP DELETE to the
                # storage owner clears ``_storage`` but not per-consumer
                # caches. Fan out ``clear_batches`` to every role that
                # localized the batch — see areal-project/AReaL#1209.
                # SPMD mode never populates ``_fetch_buffer`` (no RTensor
                # round-trip), so the fan-out is single-controller only.
                if is_single_controller():
                    self.actor.clear_batches(rollout_batch, adv_batch)
                    if self.critic is not None:
                        self.critic.clear_batches(rollout_batch, adv_batch)
                    if self.ref is not None:
                        self.ref.clear_batches(rollout_batch)
                    if self.data_controller is not None:
                        self.data_controller.clear_batches()

            with perf_tracer.trace_scope(
                "train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            # Resume rollout
            self.rollout.resume()

            self._save_perf_tracer(step=global_step)
            last_completed_step = global_step
            self.walltime_guard.observe_step(time.monotonic() - step_started_at)
            if walltime_decision.should_exit:
                stopped_for_walltime = True
                break

        if self.evaluator.config.eval_after_train and not stopped_for_walltime:
            final_global_step = (
                last_completed_step
                if last_completed_step is not None
                else max(start_step - 1, 0)
            )
            final_epoch = final_global_step // steps_per_epoch
            final_epoch_step = final_global_step % steps_per_epoch
            final_version = (
                last_completed_step + 1
                if last_completed_step is not None
                else start_step
            )
            logger.info(
                "Running final evaluation at global_step=%s, model_version=%s",
                final_global_step,
                final_version,
            )
            self.rollout.pause()
            try:
                if self._should_offload_rollout:
                    self._onload_rollout(is_eval=True)
                try:
                    with (
                        stats_tracker.record_timing("eval"),
                        perf_tracer.trace_scope(
                            "train.eval_after_train",
                            category=Category.COMPUTE,
                            args={"global_step": final_global_step},
                        ),
                    ):
                        self._evaluate(
                            eval_workflow=eval_workflow,
                            eval_workflow_kwargs=eval_workflow_kwargs,
                            epoch=final_epoch,
                            epoch_step=final_epoch_step,
                            global_step=final_global_step,
                            step_version=final_version,
                            force=True,
                        )
                finally:
                    if self._should_offload_rollout:
                        self._offload_rollout(is_eval=True)
            finally:
                self.rollout.resume()
        elif stopped_for_walltime and self.evaluator.config.eval_after_train:
            logger.info("Skipping final evaluation to stay within the job walltime.")

    def close(self):
        self.saver.finalize()
        if hasattr(self, "_train_rdataset") and self._train_rdataset is not None:
            self._train_rdataset.close()
        if hasattr(self, "_valid_rdataset") and self._valid_rdataset is not None:
            self._valid_rdataset.close()
        if hasattr(self, "data_controller") and self.data_controller is not None:
            self.data_controller.destroy()
        self.stats_logger.close()
        if self.eval_rollout is not None:
            self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.ref is not None:
            self.ref.destroy()
        if self.critic is not None:
            self.critic.destroy()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def _config_perf_tracer(self):
        rank = int(os.getenv("RANK", "0"))
        if self.config.perf_tracer is None:
            return
        perf_tracer.configure(self.config.perf_tracer, rank=rank, role="master")

        if not is_single_controller():
            return

        self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        if self.critic is not None:
            self.critic.config_perf_tracer(self.config.perf_tracer, role="critic")
        if self.ref is not None:
            self.ref.config_perf_tracer(self.config.perf_tracer, role="ref")
        self.rollout.config_perf_tracer(self.config.perf_tracer, role="rollout")
        if self.eval_rollout is not None:
            self.eval_rollout.config_perf_tracer(
                self.config.perf_tracer, role="eval-rollout"
            )

    def _save_perf_tracer(self, step: int):
        cfg = self.config.perf_tracer
        session_cfg = getattr(cfg, "session_tracer", None) if cfg is not None else None
        if cfg is None or (
            not cfg.enabled and not getattr(session_cfg, "enabled", False)
        ):
            return

        self.actor.save_perf_tracer(step=step)
        if self.ref is not None:
            self.ref.save_perf_tracer(step=step)
        if self.critic is not None:
            self.critic.save_perf_tracer(step=step)
        if self.eval_rollout is not None:
            self.eval_rollout.save_perf_tracer(step=step)
        self.rollout.save_perf_tracer(step=step)
        perf_tracer.save(step=step)

    def _init_scheduler(self) -> Scheduler:
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        elif cfg.type == "ray":
            return RayScheduler(exp_config=self.config)
        elif cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _create_dataloader(
        self,
        dataset: Dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
        )

    def _amend_xccl_weight_update_envvar(self):
        if not is_single_controller():
            # These environs are set by the launcher in the SPMD mode.
            return
        if self.rollout_alloc.backend != "sglang":
            return

        # Disable some environ for NCCL weight update.
        for spec in self.config.actor.scheduling_spec:
            spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

    def _create_train_engine(
        self, actor_config: PPOActorConfig, alloc: ModelAllocation
    ) -> FSDPPPOActor | MegatronPPOActor | ArchonPPOActor | PPOActorController:
        """Create a training engine (actor or ref) based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOActor

            actor_cls = FSDPPPOActor
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOActor

            actor_cls = MegatronPPOActor
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOActor

            actor_cls = ArchonPPOActor
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor

    def _create_critic(
        self, critic_config: PPOCriticConfig, alloc: ModelAllocation
    ) -> FSDPPPOCritic | MegatronPPOCritic | ArchonPPOCritic | PPOCriticController:
        """Create a critic engine based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOCritic

            critic_cls = FSDPPPOCritic
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOCritic

            critic_cls = MegatronPPOCritic
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOCritic

            critic_cls = ArchonPPOCritic
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            critic = critic_cls.as_controller(critic_config, self.scheduler)
        else:
            critic = critic_cls(config=critic_config)
        critic.create_process_group(parallel_strategy=alloc.parallel)
        return critic

    def _initialize_engines_and_rollout(
        self, engine_init_kwargs: dict[str, Any]
    ) -> tuple[InferenceEngine | RolloutController, str | None]:
        """Initialize training and rollout engines, optionally in parallel.

        Ray assigns separate GPU sets to the actor and rollout controllers. On
        large jobs, initializing those controllers serially can leave half of
        the allocation idle long enough for a cluster idle reaper to cancel an
        otherwise healthy job. Parallel initialization is therefore available
        as an explicit opt-in for the simple actor-plus-rollout topology.

        More complex topologies and LoRA retain the established serial path so
        their initialization ordering is unchanged.
        """
        parallel_init = os.getenv("AREAL_PARALLEL_TRAIN_ROLLOUT_INIT", "0") == "1"
        parallel_init = parallel_init and is_single_controller()
        parallel_init = parallel_init and not self.config.actor.use_lora
        parallel_init = parallel_init and self.critic is None
        parallel_init = parallel_init and self.ref is None
        parallel_init = parallel_init and self.teacher is None

        if parallel_init:
            logger.info("Initializing actor and rollout controllers in parallel")
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="engine_init"
            ) as executor:
                actor_future = executor.submit(
                    self.actor.initialize, **engine_init_kwargs, role="actor"
                )
                rollout_future = executor.submit(
                    self._init_rollout,
                    self.config.rollout,
                    is_eval=False,
                    lora_path=None,
                )
                actor_future.result()
                rollout = rollout_future.result()
            return rollout, None

        self.actor.initialize(**engine_init_kwargs, role="actor")
        if self.critic is not None:
            self.critic.initialize(**engine_init_kwargs, role="critic")
        if self.ref is not None:
            self.ref.initialize(**engine_init_kwargs, role="ref")
        if self.teacher is not None:
            self.teacher.initialize(**engine_init_kwargs, role="teacher")

        initial_lora_path = self._save_initial_lora_weights()
        rollout = self._init_rollout(
            self.config.rollout,
            is_eval=False,
            lora_path=initial_lora_path,
        )
        return rollout, initial_lora_path

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
        lora_path: str | None = None,
    ) -> InferenceEngine | RolloutController:
        if lora_path is not None and not is_single_controller():
            raise ValueError(
                "LoRA is only supported in single-controller mode. "
                "Use `python3 train.py scheduler.type=local` instead of "
                "`python3 -m areal.infra.launcher.local`."
            )
        # Create a working copy of config
        config = deepcopy(rollout_config)
        if is_eval:
            # NOTE: eval does not have any offpolicyness control
            config.max_head_offpolicyness = int(1e12)
            # eval-rollout uses the same inference servers as rollout
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        # Determine engine class and server args based on backend
        rollout_backend = self.rollout_alloc.backend
        if rollout_backend == "sglang":
            if self.config.rollout.return_routed_experts:
                self.config.sglang.enable_return_routed_experts = True
            if lora_path is not None and self.config.actor.use_lora:
                self.config.sglang.lora_paths = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
                base_gpu_id=0,
            )
        elif rollout_backend == "vllm":
            if self.config.rollout.return_routed_experts:
                raise ValueError(
                    "return_routed_experts is not supported with vLLM backend. Please disable return_routed_experts or switch to SGLang backend."
                )
            if lora_path is not None and self.config.actor.use_lora:
                self.config.vllm.lora_modules = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
            )
            # vLLM does not require LoRA paths during initialization.
            # LoRA is attached to generation requests.
        else:
            raise ValueError(
                f"Invalid backend: {rollout_backend}, expected sglang or vllm"
            )

        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine

        # Single-controller mode - no engine instantiation needed
        if config._version == "v2":
            controller = RolloutControllerV2(
                config=config, scheduler=cast(Scheduler, self.scheduler)
            )
        else:
            controller = engine_cls.as_controller(config, self.scheduler)
        init_kwargs = dict(
            role="rollout",
            server_args=server_args,
        )
        if config._version != "v2":
            # Match the non-controller path.  The legacy controller already
            # enforces the global staleness bound; its remote workers only
            # partition that capacity.  Defaulting each worker to rollout DP
            # over-partitions a 128-group batch across 48 rollout workers and
            # can strand replacement tasks behind worker-local capacity even
            # while the global controller is waiting for them.
            init_kwargs["train_data_parallel_size"] = (
                self.actor_alloc.parallel.dp_size
            )
        if is_eval:
            assert len(self.rollout.server_infos) > 0
            init_kwargs["server_infos"] = self.rollout.server_infos
            init_kwargs["role"] = "eval-rollout"
        controller.initialize(**init_kwargs)
        return controller

    def _save_initial_lora_weights(self) -> str | None:
        """Save initial LoRA weights for inference server pre-loading.

        Returns path to saved LoRA weights, or None if LoRA is disabled.
        """
        if not self.config.actor.use_lora:
            return None

        path = os.path.join(
            Saver.get_model_save_root(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.cluster.fileroot,
                name="actor",
            ),
            "initial_lora",
        )

        meta = SaveLoadMeta(
            path=path,
            weight_format="hf",
            with_optim=False,
            tokenizer=self.tokenizer,
            processor=self.processor,
            base_model_path=self.config.actor.path,
        )
        # Save LoRA weights using engine's HuggingFace save
        self.actor.save(meta=meta)

        return path

    def _save_hf(
        self, epoch: int, epoch_step: int, global_step: int, *, force: bool = False
    ):
        # Save as HF models for evaluation
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
            force=force,
        )
        if self.critic is not None:
            self.saver.save(
                self.critic,
                epoch,
                epoch_step,
                global_step,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name="critic",
                force=force,
            )
        # Async mode: synchronization handled by AsyncCheckpointManager
        if not self.saver.is_async:
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _save_recover_checkpoint(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
        *,
        force: bool = False,
    ) -> bool:
        # Save recoverable checkpoints
        to_save: dict = dict(default=self.actor)
        if self.critic is not None:
            to_save["critic"] = self.critic
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=len(self.train_dataloader),
        )
        saved = self.recover_handler.dump(
            to_save,
            step_info,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            tokenizer=self.tokenizer,
            processor=self.processor,
            force=force,
        )

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
        if saved and self._reference_version is not None:
            self._prune_reference_snapshots(self._reference_version)
        return saved

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs,
    ):
        if self.actor.is_data_parallel_head():
            cnt = 0
            flat_eval_samples = os.environ.get("AREAL_EVAL_FLAT_SAMPLES") == "1"
            eval_n = self.config.eval_gconfig.n_samples
            for data in self.valid_dataloader:
                for item in data:
                    if flat_eval_samples:
                        for _ in range(eval_n):
                            self.eval_rollout.submit(
                                dict(item),
                                eval_workflow,
                                eval_workflow_kwargs,
                                group_size=1,
                                is_eval=True,
                            )
                            cnt += 1
                    else:
                        self.eval_rollout.submit(
                            item,
                            eval_workflow,
                            eval_workflow_kwargs,
                            group_size=eval_n,
                            is_eval=True,
                        )
                        cnt += 1
            self.eval_rollout.wait(cnt, timeout=None)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _evaluate(
        self,
        eval_workflow: WorkflowLike | None,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
        step_version: int | None = None,
        force: bool = False,
    ):
        if (
            self.eval_rollout is None
            or self.valid_dataloader is None
            or eval_workflow is None
        ):
            return

        ran_eval = [False]
        eval_fn = functools.partial(
            self._evaluate_fn,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )

        def _marked_eval_fn():
            eval_fn()
            ran_eval[0] = True

        self.evaluator.evaluate(
            _marked_eval_fn,
            epoch,
            epoch_step,
            global_step,
            force=force,
        )
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

        # Per-source eval aggregation (rank 0 → wandb). Runs after barrier so
        # all eval-rollout JSONL writes are flushed. Idempotent: filters on
        # head_version == global_step, so multiple calls at same step would
        # log the same numbers (we still gate on ran_eval to avoid wasted
        # directory scans on non-eval steps).
        if ran_eval[0] and dist.get_rank() == 0:
            try:
                if self._eval_prompt_to_source is None:
                    parquet_path = (
                        self.config.valid_dataset.path
                        if self.config.valid_dataset is not None
                        else None
                    )
                    self._eval_prompt_to_source = build_prompt_source_map(
                        parquet_path or ""
                    )
                if self._eval_prompt_to_source:
                    log_root = StatsLogger.get_log_path(
                        experiment_name=self.config.experiment_name,
                        trial_name=self.config.trial_name,
                        fileroot=self.config.cluster.fileroot,
                    )
                    # Eval workers stamp records with the active rollout
                    # version. Pre-train eval runs at version 0; after an
                    # update, line 789 above bumps eval_rollout to
                    # global_step+1 before calling _evaluate().
                    # train/pass_at_n is no longer in this path — it lands
                    # via _pending_train_metrics in _export_and_commit_stats,
                    # computed inline off rollout_batch.
                    # Per-job sharding (see workflow_executor._get_dump_dir):
                    # this link's eval JSONL lives in eval-rollout/job_<SLURM_JOB_ID>/.
                    # We aggregate only that subtree so prior links' task_id
                    # ranges (which reset to 0 on resume) don't pool with ours.
                    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
                    # Dual-eval override (set by entry point): per-source n_samples
                    # cap as a {raw_source_string: int} dict (e.g. {'math500': 1,
                    # 'deepscaler/aime24': 16}). Falls back to single eval_gconfig.
                    n_samples_arg = getattr(
                        self, "_n_samples_per_source_override", None
                    )
                    if n_samples_arg is None:
                        n_samples_arg = (
                            self.config.eval_gconfig.n_samples
                            if self.config.eval_gconfig is not None
                            else self.config.gconfig.n_samples
                        )
                    metrics = aggregate_eval_jsonl(
                        os.path.join(log_root, "eval-rollout", f"job_{job_id}"),
                        self._eval_prompt_to_source,
                        step_version=(
                            step_version if step_version is not None else global_step + 1
                        ),
                        n_samples_per_task=n_samples_arg,
                    )
                    if metrics:
                        self.stats_logger.commit(
                            epoch, epoch_step, global_step, metrics
                        )
                        logger.info(
                            f"per-source eval logged to wandb: "
                            f"{ {k: round(v, 4) for k, v in metrics.items()} }"
                        )
            except Exception as e:
                logger.warning(
                    f"per-source eval aggregation failed (non-fatal): {e!r}"
                )

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        # Upload statistics to the logger (e.g., wandb)
        stats = self.actor.export_stats()
        # FRPO records the first two token-level moments of C^(H).  Publish the
        # population variance explicitly while retaining both raw moments for
        # reproducible offline aggregation.
        for second_moment_key, second_moment in list(stats.items()):
            if not second_moment_key.endswith("/frpo_tail_sq/avg"):
                continue
            mean_key = second_moment_key.removesuffix("_sq/avg") + "/avg"
            if mean_key not in stats:
                continue
            variance_key = second_moment_key.removesuffix("_sq/avg") + "_var"
            stats[variance_key] = max(
                0.0, float(second_moment) - float(stats[mean_key]) ** 2
            )
        stats.update(self.rollout.export_stats())
        if self.eval_rollout is not None:
            stats.update(self.eval_rollout.export_stats())
        pending = getattr(self, "_pending_train_metrics", None)
        if pending:
            stats.update(pending)
            self._pending_train_metrics = None
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _validate_cfg(self):
        """validate config for incompatible settings before weight initialization, to avoid wasted resources on spawning workers and loading models."""
        rollout_backend = self.rollout_alloc.backend
        actor_backend = self.actor_alloc.backend
        requires_train_engine_offload = any(
            (
                self._should_offload_rollout,
                self._should_offload_actor,
                self._should_offload_critic,
                self._should_offload_ref,
                self._should_offload_teacher,
            )
        )

        if requires_train_engine_offload and not self.config.enable_offload:
            raise ValueError(
                "enable_offload must be True when colocation scheduling or train-engine "
                "offload is enabled. Please set enable_offload=True."
            )

        if self.config.ref_refresh_steps is not None:
            if self.config.ref is None:
                raise ValueError("ref_refresh_steps requires a ref engine")
            if not self.config.actor.use_frpo_loss_term:
                raise ValueError(
                    "ref_refresh_steps currently requires use_frpo_loss_term=True"
                )
            if self.config.actor.frpo_anchor != "ref":
                raise ValueError(
                    "ref_refresh_steps requires actor.frpo_anchor='ref'"
                )
            if self.config.ref.backend != self.config.actor.backend:
                raise ValueError(
                    "ref_refresh_steps requires matching actor/ref backends so "
                    "the DCP snapshot is load-compatible"
                )

        exclude_xccl_source = os.getenv(
            "AREAL_PARTIAL_COLOCATION_EXCLUDE_XCCL_SOURCE", ""
        ).lower() in {"1", "true", "yes", "on"}
        safe_partial_xccl = (
            self.config.actor.weight_update_mode == "xccl"
            and os.getenv("AREAL_PARTIAL_COLOCATION", "").lower()
            in {"1", "true", "yes", "on"}
            and exclude_xccl_source
            and self.rollout_alloc.parallel.world_size
            > self.actor_alloc.parallel.world_size
        )
        if (
            self._is_actor_rollout_colocated(self.config)
            and self.config.actor.weight_update_mode != "disk"
            and not safe_partial_xccl
        ):
            raise ValueError(
                "weight_update_mode must be 'disk' when colocation scheduling is enabled. "
                "Please set actor.weight_update_mode=disk."
            )

        if rollout_backend == "vllm" and self.config.rollout.return_routed_experts:
            raise ValueError(
                "return_routed_experts is only supported with SGLang backend. "
                "Please disable return_routed_experts or switch to SGLang backend."
            )
        if (
            actor_backend == "megatron"
            and self.config.actor.use_lora
            and rollout_backend == "sglang"
        ):
            raise ValueError(
                "Megatron actor with LoRA is not supported with SGLang rollout in "
                "RL trainer. Please use vLLM rollout backend, or disable LoRA, or "
                "switch actor backend from Megatron."
            )

    def _requires_proxy_workflow(self, workflow: WorkflowLike | None) -> bool:
        """Check if workflow requires proxy workers (i.e., not a RolloutWorkflow).

        Returns True if:
        - Workflow is NOT a RolloutWorkflow instance
        - Workflow is NOT a RolloutWorkflow class
        - Workflow is a string that does NOT import to a RolloutWorkflow

        This enables any callable object with a compatible signature to work
        without requiring inheritance from AgentWorkflow.
        """
        # None workflow is handled separately in train()
        if workflow is None:
            return False

        # Direct RolloutWorkflow instances
        if isinstance(workflow, RolloutWorkflow):
            return False

        # RolloutWorkflow classes
        if isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return False

        # String import paths
        if isinstance(workflow, str):
            from areal.utils.dynamic_import import import_from_string

            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError):
                # If import fails, assume it needs proxy (fail-safe)
                return True

            # Check if imported object is RolloutWorkflow
            if isinstance(imported_obj, RolloutWorkflow):
                return False
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                return False

        # Everything else requires proxy workers
        return True

    def _ensure_proxy_started(self) -> None:
        """Lazily initialize proxy workers when agent workflows are used.

        This method is called before training when a non-RolloutWorkflow is detected
        or when online mode is configured. It creates proxy workers colocated with
        rollout workers to handle OpenAI-compatible API requests.

        In online mode, also starts the proxy gateway for external access.
        """
        if self._proxy_started:
            return

        # Only initialize proxy in single-controller mode with RolloutController
        if not is_single_controller():
            raise NotImplementedError("Proxy workers not supported in SPMD mode")

        if self.config.scheduler.type == "ray":
            raise NotImplementedError("Proxy workers not supported with RayScheduler")

        assert isinstance(self.rollout, RolloutController)

        logger.info("Initializing proxy workers for AgentWorkflow support")
        self.rollout.start_proxy()
        if self.eval_rollout is not None:
            self.eval_rollout.start_proxy()

        # Start proxy gateway for online mode.
        agent_cfg = self.config.rollout.agent
        if agent_cfg is not None and agent_cfg.mode == "online":
            self.rollout.start_proxy_gateway()
            logger.info(
                "Proxy gateway available at %s",
                self.rollout.proxy_gateway_addr,
            )

        self._proxy_started = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Training failed with exception: {exc_value}", exc_info=True)
        self.close()
        return False
