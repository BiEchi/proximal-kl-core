# SPDX-License-Identifier: Apache-2.0

import getpass
import os
import time
from dataclasses import asdict

import torch.distributed as dist
import wandb

try:
    import swanlab
except ImportError:  # optional when stats_logger.swanlab.mode is disabled
    swanlab = None

try:
    import trackio
except ImportError:  # optional when stats_logger.trackio.mode is disabled
    trackio = None

try:
    from tensorboardX import SummaryWriter
except ImportError:  # optional when stats_logger.tensorboard.path is unset
    SummaryWriter = None

from areal.api import FinetuneSpec
from areal.api.cli_args import BaseExperimentConfig, StatsLoggerConfig
from areal.utils import logging
from areal.utils.printing import tabulate_stats
from areal.version import version_info

logger = logging.getLogger("StatsLogger", "system")


def _resolve_wandb_mode(mode: str) -> str:
    override = os.environ.get("WANDB_MODE")
    if override is None:
        return mode
    valid_modes = {"online", "offline", "disabled", "shared"}
    if override not in valid_modes:
        raise ValueError(
            f"Invalid WANDB_MODE={override!r}; expected one of {sorted(valid_modes)}"
        )
    if override != mode:
        logger.warning("WANDB_MODE overrides configured wandb mode: %s -> %s", mode, override)
    return override


class StatsLogger:
    def __init__(self, config: BaseExperimentConfig, ft_spec: FinetuneSpec):
        if isinstance(config, StatsLoggerConfig):
            raise ValueError(
                "Passing config.stats_logger as the config is deprecated. "
                "Please pass the full config instead."
            )
        self.exp_config = config
        self.config = config.stats_logger
        self.ft_spec = ft_spec
        self._last_commit_step = -1
        self.init()

    def init(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        self._swanlab_enabled = False
        self._trackio_enabled = False

        if self.config.wandb.wandb_base_url:
            os.environ["WANDB_BASE_URL"] = self.config.wandb.wandb_base_url
        if self.config.wandb.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.config.wandb.wandb_api_key

        self.start_time = time.perf_counter()
        wandb_mode = _resolve_wandb_mode(self.config.wandb.mode)

        suffix = self.config.wandb.id_suffix
        if suffix == "timestamp":
            suffix = time.strftime("%Y_%m_%d_%H_%M_%S")

        exp_config_dict = asdict(self.exp_config)
        exp_config_dict["version_info"] = {
            "commit_id": version_info.commit,
            "branch": version_info.branch,
            "is_dirty": version_info.is_dirty,
            "version": version_info.full_version_with_dirty_description,
        }

        # Allow sbatch (or any caller) to override the run id via env var.
        # We use this to handle the wandb 409 ("run was previously created and
        # deleted; try a new run id") permanent ban — wandb server forever
        # forbids reusing a deleted run id, so on a state wipe sbatch generates
        # a fresh timestamped id and pins it in $CKPT_DIR/wandb_run_id.txt
        # (so the same id survives walltime-kill relaunches).
        run_id = os.environ.get("WANDB_RUN_ID") or (
            f"{self.config.experiment_name}_{self.config.trial_name}_{suffix}"
        )
        init_kwargs = dict(
            mode=wandb_mode,
            entity=self.config.wandb.entity,
            project=self.config.wandb.project or self.config.experiment_name,
            name=self.config.wandb.name or self.config.trial_name,
            job_type=self.config.wandb.job_type,
            group=self.config.wandb.group
            or f"{self.config.experiment_name}_{self.config.trial_name}",
            notes=self.config.wandb.notes,
            tags=self.config.wandb.tags,
            config=exp_config_dict,  # save all experiment config to wandb
            dir=self.get_log_path(self.config),
            force=True,
            id=run_id,
        )

        # Resume strategy.
        # - Fresh run on the server → `resume="allow"` (creates new run).
        # - Existing run on the server → `resume_from=f"{id}?_step=N"` to truncate
        #   history at step N (server-side, in-place; same run id). Avoids the
        #   delete-then-recreate-same-id race that times out wandb.init.
        #     • Wipe case (no checkpoint)  → N = 0  (full history truncate)
        #     • Resume case (with checkpoint) → N = ckpt's global_step (drop orphan)
        # `WANDB_RESUME_FROM_STEP` is set by sbatch (which inspects the local
        # checkpoint dir). If unset, we fall back to server-side max step (no-op
        # truncate, equivalent to legacy `resume="allow"`).
        existing_run = None
        if wandb_mode in ("online", "shared"):
            try:
                api = wandb.Api()
                project_path = (
                    f"{init_kwargs['entity']}/{init_kwargs['project']}"
                    if init_kwargs.get("entity")
                    else init_kwargs["project"]
                )
                existing_run = api.run(f"{project_path}/{run_id}")
            except Exception as e:
                # Run-not-found / network blip: fall through to fresh init.
                logger.info(
                    f"wandb run lookup for {run_id!r} treated as 'not found': {e!r}"
                )

        # Unconditionally use resume="allow" for both fresh and existing-run
        # paths. The earlier resume_from=...?_step=N pattern (server-side
        # history truncation to ckpt's step) collided with wandb 0.22+'s
        # internal Settings validation — "fork_from, resume, or resume_from
        # are mutually exclusive" — even after we tried clearing WANDB_RESUME
        # env var pre-init. The wandb sdk's `maybe_login` builds an internal
        # Settings instance that ends up with BOTH resume and resume_from
        # populated through a path that's hard to disable cleanly. Living
        # with orphaned wandb history points past ckpt step is cheaper than
        # this startup failure: AReaL recover.mode=auto handles the actual
        # training progress correctly from the ckpt, and downstream plots
        # use ckpt-aligned data anyway.
        init_kwargs["resume"] = "allow"
        if existing_run is None:
            logger.info(f"wandb.init: fresh run id={run_id}")
        else:
            logger.info(
                f"wandb.init: resume=allow run id={run_id} "
                f"(existing wandb run at step={int(existing_run.summary.get('_step', 0) or 0)}; "
                "no server-side truncate — orphaned history past ckpt step accepted)"
            )
            existing_step = int(existing_run.summary.get("_step", 0) or 0)
            self._last_commit_step = max(self._last_commit_step, existing_step)

        env_resume_step = os.environ.get("WANDB_RESUME_FROM_STEP")
        if env_resume_step:
            try:
                self._last_commit_step = max(
                    self._last_commit_step, int(env_resume_step)
                )
            except ValueError:
                logger.warning(
                    "Ignoring invalid WANDB_RESUME_FROM_STEP=%r", env_resume_step
                )
        if self._last_commit_step >= 0:
            logger.info(
                "wandb local log-step floor set to %s", self._last_commit_step
            )

        # Clear the env var as a defensive measure (still independent of the
        # resume mechanism above).
        os.environ.pop("WANDB_RESUME", None)

        wandb.init(**init_kwargs)

        swanlab_config = self.config.swanlab
        if swanlab_config.mode != "disabled":
            if swanlab is None:
                raise ImportError(
                    "swanlab is required when stats_logger.swanlab.mode is not disabled"
                )
            if swanlab_config.api_key:
                swanlab.login(swanlab_config.api_key)
            else:
                swanlab.login()

            swanlab.init(
                project=swanlab_config.project or self.config.experiment_name,
                experiment_name=swanlab_config.name
                or self.config.trial_name + "_train",
                # NOTE: change from swanlab_config.config to log all experiment config, to be tested
                config=exp_config_dict,
                logdir=self.get_log_path(self.config),
                mode=swanlab_config.mode,
            )
            self._swanlab_enabled = True

        # trackio init
        trackio_config = self.config.trackio
        if trackio_config.mode != "disabled":
            if trackio is None:
                raise ImportError(
                    "trackio is required when stats_logger.trackio.mode is not disabled"
                )
            trackio.init(
                project=trackio_config.project or self.config.experiment_name,
                name=trackio_config.name or self.config.trial_name,
                config=exp_config_dict,
                space_id=trackio_config.space_id,
            )
            self._trackio_enabled = True

        # tensorboard logging
        self.summary_writer = None
        if self.config.tensorboard.path is not None:
            if SummaryWriter is None:
                raise ImportError(
                    "tensorboardX is required when stats_logger.tensorboard.path is set"
                )
            self.summary_writer = SummaryWriter(log_dir=self.config.tensorboard.path)

    def state_dict(self):
        return {
            "last_commit_step": self._last_commit_step,
        }

    def load_state_dict(self, state_dict):
        self._last_commit_step = max(
            self._last_commit_step, state_dict["last_commit_step"]
        )

    def close(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Training completes! Total time elapsed {time.monotonic() - self.start_time:.2f}."
        )
        wandb.finish()
        if getattr(self, "_swanlab_enabled", False):
            swanlab.finish()
        if getattr(self, "_trackio_enabled", False):
            trackio.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def commit(self, epoch: int, step: int, global_step: int, data: dict | list[dict]):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Epoch {epoch + 1}/{self.ft_spec.total_train_epochs} "
            f"Step {step + 1}/{self.ft_spec.steps_per_epoch} "
            f"Train step {global_step + 1}/{self.ft_spec.total_train_steps} done."
        )
        if isinstance(data, dict):
            data = [data]
        log_step = max(global_step, self._last_commit_step + 1)
        for i, item in enumerate(data):
            # Filter out counter keys for scalar variables
            item = {k: v for k, v in item.items() if not k.endswith("__count")}
            item["trainer/global_step"] = global_step

            logger.info(f"Stats ({i + 1}/{len(data)}):")
            self.print_stats(item)
            wandb.log(item, step=log_step + i)
            if getattr(self, "_swanlab_enabled", False):
                swanlab.log(item, step=log_step + i)
            if getattr(self, "_trackio_enabled", False):
                trackio.log(item, step=log_step + i)
            if self.summary_writer is not None:
                for key, val in item.items():
                    self.summary_writer.add_scalar(f"{key}", val, log_step + i)
        self._last_commit_step = log_step + len(data) - 1

    def print_stats(self, stats: dict[str, float]):
        logger.info("\n" + tabulate_stats(stats))

    @staticmethod
    def get_log_path(
        config: StatsLoggerConfig | None = None,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
    ) -> str:
        if config is not None:
            experiment_name = config.experiment_name
            trial_name = config.trial_name
            fileroot = config.fileroot
        if not fileroot or not experiment_name or not trial_name:
            raise ValueError(
                "fileroot, experiment_name, and trial_name must be provided."
            )
        path = f"{fileroot}/logs/{getpass.getuser()}/{experiment_name}/{trial_name}"
        os.makedirs(path, exist_ok=True)
        return path
