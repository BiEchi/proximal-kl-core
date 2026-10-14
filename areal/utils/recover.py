# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import shutil
import uuid
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import AutoProcessor

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    SaveLoadMeta,
    StepInfo,
    TrainEngine,
    WeightUpdateMeta,
)
from areal.api.cli_args import RecoverConfig
from areal.infra import TrainController
from areal.utils import logging, timeutil
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver

if TYPE_CHECKING:
    from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Recover")


class InValidRecoverInfo(Exception):
    pass


@dataclasses.dataclass
class RecoverInfo:
    # Last step info is the counter of the saved checkpoint.
    # Recover will start from the next iteration, obtained by `last_step_info.next()`.
    last_step_info: StepInfo

    saver_info: dict
    evaluator_info: dict
    stats_logger_info: dict
    dataloader_info: dict | list[dict]
    checkpoint_info: dict

    def dump(self, dump_dir: str):
        # Dumps the recover info to multiple files in `dump_dir`:
        # 1. step_info.json: contains the recover info
        # 2. *_info.json or *_info.pkl: contains other informantion required for recover.

        if dist.is_initialized():
            # Since dataloader state is different across distributed ranks,
            # we need to all gather the dataloader state from all ranks.
            # In this situation, saved dataloader_info is a list of states from all ranks.
            dataloader_info = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dataloader_info, self.dataloader_info)

            # To avoid contention, do not dump on multiple ranks
            if dist.get_rank() != 0:
                return
        else:
            dataloader_info = self.dataloader_info

        os.makedirs(dump_dir, exist_ok=True)
        step_info_path = os.path.join(dump_dir, "step_info.json")
        with open(step_info_path, "w") as f:
            json.dump(dataclasses.asdict(self.last_step_info), f, indent=4)

        saver_info_path = os.path.join(dump_dir, "saver_info.json")
        with open(saver_info_path, "w") as f:
            json.dump(self.saver_info, f, indent=4)

        evaluator_info_path = os.path.join(dump_dir, "evaluator_info.json")
        with open(evaluator_info_path, "w") as f:
            json.dump(self.evaluator_info, f, indent=4)

        stats_logger_info_path = os.path.join(dump_dir, "stats_logger_info.json")
        with open(stats_logger_info_path, "w") as f:
            json.dump(self.stats_logger_info, f, indent=4)

        checkpoint_info_path = os.path.join(dump_dir, "checkpoint_info.json")
        with open(checkpoint_info_path, "w") as f:
            json.dump(self.checkpoint_info, f, indent=4)

        dataloader_info_path = os.path.join(dump_dir, "dataloader_info.pkl")
        with open(dataloader_info_path, "wb") as f:
            pickle.dump(dataloader_info, f)

    @classmethod
    def load(cls, load_dir: str):
        # Loads the recover info from multiple files in `load_dir`:
        if not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"Recover info directory {load_dir} does not exist."
            )

        try:
            step_info_path = os.path.join(load_dir, "step_info.json")
            with open(step_info_path) as f:
                step_info_dict = json.load(f)
                last_step_info = StepInfo(**step_info_dict)

            evaluator_info_path = os.path.join(load_dir, "evaluator_info.json")
            with open(evaluator_info_path) as f:
                evaluator_info = json.load(f)

            saver_info_path = os.path.join(load_dir, "saver_info.json")
            with open(saver_info_path) as f:
                saver_info = json.load(f)

            stats_logger_info_path = os.path.join(load_dir, "stats_logger_info.json")
            with open(stats_logger_info_path) as f:
                stats_logger_info = json.load(f)

            checkpoint_info_path = os.path.join(load_dir, "checkpoint_info.json")
            with open(checkpoint_info_path) as f:
                checkpoint_info = json.load(f)

            dataloader_info_path = os.path.join(load_dir, "dataloader_info.pkl")
            with open(dataloader_info_path, "rb") as f:
                dataloader_info = pickle.load(f)
                if isinstance(dataloader_info, list):
                    # If dataloader_info a list, it means it is saved from a distributed run.
                    if dist.is_initialized():
                        # Loading dataloader states in a distributed context.
                        assert dist.get_world_size() == len(dataloader_info), (
                            f"Dataloader info list length {len(dataloader_info)} does not match "
                            f"the world size {dist.get_world_size()}."
                        )
                        dataloader_info = dataloader_info[dist.get_rank()]

            return cls(
                last_step_info=last_step_info,
                saver_info=saver_info,
                evaluator_info=evaluator_info,
                stats_logger_info=stats_logger_info,
                dataloader_info=dataloader_info,
                checkpoint_info=checkpoint_info,
            )
        except Exception as e:
            logger.error(f"Failed to load recover info from {load_dir}: {e}")
            raise InValidRecoverInfo(f"Invalid recover info in {load_dir}") from e


class RecoverHandler:
    _GENERATIONS_DIR = ".recover_generations"
    _GENERATION_MARKER = "recover_generation.json"

    def __init__(self, config: RecoverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.last_step_info = StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
            steps_per_epoch=ft_spec.steps_per_epoch,
        )
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    @classmethod
    def recover_info_path(
        cls,
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        save_root = Saver.get_save_root(experiment_name, trial_name, fileroot)
        generation = cls._read_active_generation(save_root)
        if generation is not None:
            return os.path.join(
                save_root, cls._GENERATIONS_DIR, generation, "recover_info"
            )
        return os.path.join(save_root, "recover_info")

    @classmethod
    def recover_checkpoint_path(
        cls,
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        *,
        name: str = "default",
    ) -> str:
        save_root = Saver.get_save_root(experiment_name, trial_name, fileroot)
        generation = cls._read_active_generation(save_root)
        if generation is not None:
            return os.path.join(
                save_root,
                cls._GENERATIONS_DIR,
                generation,
                "models",
                name,
            )
        return Saver.get_recover_checkpoint_path(
            experiment_name, trial_name, fileroot, name=name
        )

    @classmethod
    def _generation_marker_path(cls, save_root: str) -> str:
        return os.path.join(save_root, cls._GENERATION_MARKER)

    @classmethod
    def _read_active_generation(cls, save_root: str) -> str | None:
        marker_path = cls._generation_marker_path(save_root)
        if not os.path.isfile(marker_path):
            return None
        try:
            with open(marker_path) as f:
                generation = json.load(f)["generation"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InValidRecoverInfo(
                f"Invalid recovery generation marker at {marker_path}"
            ) from exc
        if not isinstance(generation, str) or not generation:
            raise InValidRecoverInfo(
                f"Invalid recovery generation in marker at {marker_path}"
            )
        return generation

    @staticmethod
    def _barrier() -> None:
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _broadcast_generation(generation: str | None) -> str:
        if not dist.is_initialized():
            assert generation is not None
            return generation
        value = [generation if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(value, src=0)
        assert isinstance(value[0], str)
        return value[0]

    @classmethod
    def _commit_generation(cls, save_root: str, generation: str) -> None:
        marker_path = cls._generation_marker_path(save_root)
        marker_tmp = f"{marker_path}.tmp-{os.getpid()}"
        with open(marker_tmp, "w") as f:
            json.dump({"generation": generation}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(marker_tmp, marker_path)

        generations_root = os.path.join(save_root, cls._GENERATIONS_DIR)
        for entry in os.listdir(generations_root):
            if entry != generation:
                shutil.rmtree(os.path.join(generations_root, entry), ignore_errors=True)

        # The first transactional save migrates from the legacy in-place paths.
        # They are removed only after the generation marker points at a complete
        # checkpoint and matching RecoverInfo.
        shutil.rmtree(os.path.join(save_root, "recover_info"), ignore_errors=True)
        models_root = os.path.join(
            save_root, cls._GENERATIONS_DIR, generation, "models"
        )
        for name in os.listdir(models_root):
            shutil.rmtree(
                os.path.join(save_root, name, "recover_checkpoint"),
                ignore_errors=True,
            )

    @classmethod
    def _cleanup_uncommitted_generations(cls, save_root: str) -> None:
        generations_root = os.path.join(save_root, cls._GENERATIONS_DIR)
        if not os.path.isdir(generations_root):
            return
        active_generation = cls._read_active_generation(save_root)
        for entry in os.listdir(generations_root):
            if entry != active_generation:
                shutil.rmtree(os.path.join(generations_root, entry), ignore_errors=True)

    @staticmethod
    def _is_gateway_train_controller(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> bool:
        from areal.experimental.training_service.controller.controller import (
            GatewayTrainController,
        )

        if isinstance(engine, GatewayTrainController):
            return True
        if isinstance(engine, dict):
            return any(
                isinstance(controller, GatewayTrainController)
                for controller in engine.values()
            )
        return False

    def _ensure_recover_supported(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> None:
        if self._is_gateway_train_controller(engine):
            raise NotImplementedError(
                "Recovery is not supported with GatewayTrainController "
                '(`_version="v2"`) yet. Disable `recover.mode` or use '
                '`_version="v1"`.'
            )

    @staticmethod
    def _normalize_recover_engines(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> dict[str, TrainEngine | TrainController]:
        if isinstance(engine, dict):
            return engine
        return {"default": engine}

    def dump(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
        step_info: StepInfo,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
        force: bool = False,
    ) -> bool:
        if self.config.mode in ("disabled", "off"):
            return False
        self._ensure_recover_supported(engine)
        # currently only support recover on one engine
        scheduled = self.freq_ctl.check(
            epochs=int(step_info.epoch_step == self.ft_spec.steps_per_epoch - 1),
            steps=1,
        )
        if not (scheduled or force):
            return False
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )

        save_root = Saver.get_save_root(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        generation = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            generation = f"step-{step_info.global_step}-{uuid.uuid4().hex}"
        generation = self._broadcast_generation(generation)
        generation_root = os.path.join(
            save_root, self._GENERATIONS_DIR, generation
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            self._cleanup_uncommitted_generations(save_root)
            shutil.rmtree(generation_root, ignore_errors=True)
        self._barrier()

        for name, engine_ in normalized_engine.items():
            checkpoint_path = os.path.join(generation_root, "models", name)
            os.makedirs(checkpoint_path, exist_ok=True)
            self._save_checkpoint(
                engine_,
                name=name,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
                path=checkpoint_path,
            )

        self.last_step_info = step_info
        recover_info = RecoverInfo(
            last_step_info=self.last_step_info,
            saver_info=saver.state_dict(),
            evaluator_info=evaluator.state_dict(),
            stats_logger_info=stats_logger.state_dict(),
            dataloader_info=dataloader.state_dict(),
            checkpoint_info=self.freq_ctl.state_dict(),
        )

        recover_info_path = os.path.join(generation_root, "recover_info")
        recover_info.dump(recover_info_path)
        self._barrier()
        if not dist.is_initialized() or dist.get_rank() == 0:
            self._commit_generation(save_root, generation)
        self._barrier()
        return True

    def load(
        self,
        engine: TrainEngine | dict[str, TrainEngine] | TrainController,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        inference_engine: InferenceEngine | None = None,
        weight_update_meta: WeightUpdateMeta | None = None,
        inference_engine_update_from: str = "default",
    ) -> RecoverInfo | None:
        if self.config.mode in ("disabled", "off"):
            return
        self._ensure_recover_supported(engine)
        if inference_engine is not None and weight_update_meta is None:
            raise ValueError("Weight update meta is required for recovery.")

        # TODO(agent): GatewayTrainController is currently duck-typed and does
        # not satisfy this TrainController type check. Extend recovery to accept
        # controller-v2 instances (or make v2 inherit TrainController) before
        # relying on resumed runs with `_version="v2"`.
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )

        recover_info_path = self.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        logger.info(f"Loading recover info from {recover_info_path}")
        try:
            recover_info: RecoverInfo = RecoverInfo.load(recover_info_path)
            logger.info(f"Recovering from {recover_info.last_step_info.next()}.")
            saver.load_state_dict(recover_info.saver_info)
            self.freq_ctl.load_state_dict(recover_info.checkpoint_info)
            # Restore elapsed counters, but keep the continuation's explicit
            # recovery cadence.  A bootstrap checkpoint may intentionally use
            # freq_steps=1 even though the resumed production run disables
            # periodic recovery saves.
            self.freq_ctl.epoch_ctl.frequency_steps = self.config.freq_epochs
            self.freq_ctl.step_ctl.frequency_steps = self.config.freq_steps
            self.freq_ctl.time_ctl.frequency_seconds = self.config.freq_secs
            evaluator.load_state_dict(recover_info.evaluator_info)
            # Restore evaluator counters without reviving the scheduling
            # policy saved by an older continuation.  The active config must
            # decide whether and how often this continuation evaluates.
            evaluator.freq_ctl.epoch_ctl.frequency_steps = (
                evaluator.config.freq_epochs
            )
            evaluator.freq_ctl.step_ctl.frequency_steps = evaluator.config.freq_steps
            evaluator.freq_ctl.time_ctl.frequency_seconds = evaluator.config.freq_secs
            stats_logger.load_state_dict(recover_info.stats_logger_info)
            dataloader.load_state_dict(recover_info.dataloader_info)

            for name, engine_ in normalized_engine.items():
                self._load_checkpoint(engine_, name=name)
            global_step = recover_info.last_step_info.global_step

            if inference_engine is not None:
                assert weight_update_meta is not None
                update_engine = normalized_engine[inference_engine_update_from]
                recovery_version = global_step + 1
                versioned_meta = weight_update_meta.with_version(recovery_version)
                update_engine.connect_engine(inference_engine, versioned_meta)
                inference_engine.pause()
                update_engine.update_weights(versioned_meta)
                inference_engine.resume()
                update_engine.set_version(recovery_version)
                inference_engine.set_version(recovery_version)
            return recover_info
        except (FileNotFoundError, InValidRecoverInfo):
            logger.warning(
                f"Resume info not found at {recover_info_path}. "
                f"This should not be a resumed experiment!"
            )

    def _save_checkpoint(
        self,
        engine: TrainEngine,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
        path: str | None = None,
    ):
        if path is None:
            path = self.recover_checkpoint_path(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.fileroot,
                name=name,
            )
        weight_format = "dcp"
        with_optim = not self.config.no_save_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        logger.info(f"Saved recover checkpoint to {path} (with_optim={with_optim})")

    def _load_checkpoint(
        self,
        engine: TrainEngine | TrainController,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        path = self.recover_checkpoint_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            name=name,
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        weight_format = "dcp"
        with_optim = not self.config.no_load_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=None,
            processor=None,
            base_model_path=None,
        )
        engine.load(meta)


def check_if_auto_recover(config: RecoverConfig) -> bool:
    # This method is called by check_if_recover to check if the experiment should
    # recover from a previous run when recovery is enabled ("on" or "auto" mode).
    experiment_name = config.experiment_name
    trial_name = config.trial_name
    fileroot = config.fileroot
    try:
        recover_info_path = RecoverHandler.recover_info_path(
            experiment_name, trial_name, fileroot
        )
    except InValidRecoverInfo as exc:
        logger.warning(str(exc))
        return False
    logger.info(f"Searching for recover info file in {recover_info_path}.")
    if os.path.exists(str(recover_info_path)):
        try:
            info = RecoverInfo.load(recover_info_path)
        except Exception as e:
            logger.warning(f"Failed to load recover info from {recover_info_path}: {e}")
            return False
        if info.last_step_info.epoch < 0:
            msg = (
                f"Recover checkpoint is not valid. "
                f"Expected last_step_info.epoch >= 0, "
                f"but found {info.last_step_info.epoch}"
            )
            logger.warning(msg)
            return False

        save_root = Saver.get_save_root(experiment_name, trial_name, fileroot)
        try:
            generation = RecoverHandler._read_active_generation(save_root)
        except InValidRecoverInfo as exc:
            logger.warning(str(exc))
            return False
        if generation is not None:
            models_root = os.path.join(
                save_root,
                RecoverHandler._GENERATIONS_DIR,
                generation,
                "models",
            )
            if not os.path.isdir(models_root):
                logger.warning(
                    f"Recover model directory does not exist at {models_root}."
                )
                return False
            model_names = [
                name
                for name in os.listdir(models_root)
                if os.path.isdir(os.path.join(models_root, name))
            ]
            if not model_names:
                logger.warning(f"No recover models found in {models_root}.")
                return False
            return True

        for name in os.listdir(save_root):
            if (
                name.startswith(".")
                or name == "recover_info"
                or not os.path.isdir(os.path.join(save_root, name))
            ):
                continue
            path = RecoverHandler.recover_checkpoint_path(
                experiment_name, trial_name, fileroot, name=name
            )
            if not os.path.exists(path):
                logger.warning(f"Recover checkpoint for model {name} does not exist.")
                return False
        return True
    logger.warning(f"Recover info not found at: {recover_info_path}")
    return False


def check_if_recover(config: RecoverConfig, _run_id: int) -> bool:
    """Check if the experiment should be a recover run.

    When recovery is enabled ('on' or 'auto'), this checks if valid recover
    info and checkpoints are available for automatic recovery.

    Args:
        config: Recovery configuration.
        _run_id: Unused. Kept for API compatibility.

    Returns:
        True if the experiment should recover from a previous run.
    """
    if config.mode in ("disabled", "off"):
        return False
    # Both "on" and "auto" use auto-recovery behavior
    return check_if_auto_recover(config)
