# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Any

import torch

from areal.api import TrainEngine
from areal.api.cli_args import MicroBatchSpec, PPOActorConfig, RejectionSamplingConfig
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.data import (
    KLEstimator,
    Normalization,
    batched_call,
    concat_batch,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    ppo_actor_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("PPOActor")


def _response_position_masks(
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Split valid response tokens into normalized-position quartiles."""
    valid = loss_mask.bool()
    if valid.dim() == 2:
        ranks = valid.long().cumsum(dim=1) - 1
        totals = valid.long().sum(dim=1, keepdim=True)
        fractions = ranks.clamp(min=0).float() / (totals - 1).clamp(min=1).float()
    elif valid.dim() == 1:
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(
                [0, valid.numel()], device=valid.device, dtype=torch.long
            )
        else:
            cu_seqlens = cu_seqlens.to(device=valid.device, dtype=torch.long)
        prefix = torch.cat(
            [
                torch.zeros(1, device=valid.device, dtype=torch.long),
                valid.long().cumsum(dim=0),
            ]
        )
        segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        segment_ids = torch.repeat_interleave(
            torch.arange(segment_lengths.numel(), device=valid.device),
            segment_lengths,
        )
        starts = prefix[cu_seqlens[:-1]][segment_ids]
        totals = (prefix[cu_seqlens[1:]] - prefix[cu_seqlens[:-1]])[segment_ids]
        ranks = prefix[:-1] - starts
        fractions = ranks.float() / (totals - 1).clamp(min=1).float()
    else:
        raise ValueError(f"loss_mask must be 1D or 2D, got shape {valid.shape}")

    return {
        "q1": valid & (fractions < 0.25),
        "q2": valid & (fractions >= 0.25) & (fractions < 0.5),
        "q3": valid & (fractions >= 0.5) & (fractions < 0.75),
        "q4": valid & (fractions >= 0.75),
    }


def _masked_sequence_means(
    values: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Return one equally weighted masked mean per non-empty sequence."""
    values = values.detach().float()
    mask = mask.bool()
    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have the same shape, got {values.shape} and {mask.shape}"
        )
    if values.dim() == 2:
        counts = mask.sum(dim=1)
        nonempty = counts > 0
        means = (values * mask).sum(dim=1) / counts.clamp(min=1)
        return means[nonempty]
    if values.dim() != 1:
        raise ValueError(f"values must be 1D or 2D, got shape {values.shape}")
    boundaries = (
        torch.tensor([0, values.numel()], device=values.device, dtype=torch.long)
        if cu_seqlens is None
        else cu_seqlens.to(device=values.device, dtype=torch.long)
    )
    means = []
    for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        segment_mask = mask[start:end]
        if segment_mask.any():
            means.append(values[start:end][segment_mask].mean())
    return torch.stack(means) if means else values.new_empty(0)


@torch.no_grad()
def _apply_group_baseline_shift(
    normalized_rewards: torch.Tensor,
    group_sizes: list[int] | tuple[int, ...],
    shift: float,
    margin_quantile: float = 0.0,
    margin_side: str = "positive",
    margin_scale: float = 1.0,
    center_quantile: float = 0.0,
    center_scale: float = 1.0,
    random_signs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same signed baseline shift to every informative prompt group."""
    if sum(group_sizes) != normalized_rewards.shape[0]:
        raise ValueError("group sizes must cover every normalized reward")
    if margin_quantile and center_quantile:
        raise ValueError("only one baseline perturbation mode can be enabled")
    if random_signs is not None and (margin_quantile or center_quantile):
        raise ValueError("random signs require the fixed baseline shift mode")
    if random_signs is not None and random_signs.numel() != len(group_sizes):
        raise ValueError("random signs must contain one value per prompt group")
    if margin_quantile and not (0.0 < margin_quantile < 1.0):
        raise ValueError("margin_quantile must be in (0, 1) when enabled")
    if margin_quantile and margin_side not in {"positive", "negative", "auto"}:
        raise ValueError("margin_side must be 'positive', 'negative', or 'auto'")
    if margin_quantile and margin_scale <= 0.0:
        raise ValueError("margin_scale must be positive")
    if center_quantile and not (0.0 < center_quantile < 1.0):
        raise ValueError("center_quantile must be in (0, 1) when enabled")
    if center_quantile and center_scale <= 0.0:
        raise ValueError("center_scale must be positive")

    baseline_shift = torch.zeros_like(normalized_rewards)
    start = 0
    for group_index, group_size in enumerate(group_sizes):
        end = start + group_size
        group_rewards = normalized_rewards[start:end]
        active = torch.any(group_rewards != 0).to(normalized_rewards.dtype)
        if margin_quantile and bool(active):
            positive_rewards = group_rewards[group_rewards > 0]
            negative_rewards = group_rewards[group_rewards < 0]
            if margin_side == "positive" or negative_rewards.numel() == 0:
                candidates = positive_rewards
                quantile = margin_quantile
            elif margin_side == "negative" or positive_rewards.numel() == 0:
                candidates = negative_rewards
                quantile = 1.0 - margin_quantile
            else:
                positive_margin = torch.quantile(
                    positive_rewards.float(), margin_quantile
                )
                negative_margin = torch.quantile(
                    negative_rewards.float(), 1.0 - margin_quantile
                )
                if positive_margin <= negative_margin.abs():
                    candidates = positive_rewards
                    quantile = margin_quantile
                else:
                    candidates = negative_rewards
                    quantile = 1.0 - margin_quantile
            if candidates.numel() > 0:
                group_shift = torch.quantile(
                    candidates.float(), quantile
                ).to(dtype=normalized_rewards.dtype, device=normalized_rewards.device)
                group_shift = group_shift * margin_scale
                baseline_shift[start:end] = group_shift
        elif center_quantile and bool(active):
            group_shift = torch.quantile(
                group_rewards.float(), center_quantile
            ).to(dtype=normalized_rewards.dtype, device=normalized_rewards.device)
            baseline_shift[start:end] = group_shift * center_scale
        else:
            group_shift = shift
            if random_signs is not None:
                group_shift = abs(shift) * random_signs[group_index]
            baseline_shift[start:end] = group_shift * active
        start = end

    perturbed = normalized_rewards - baseline_shift
    sign_flipped = normalized_rewards * perturbed < 0
    return perturbed, baseline_shift, sign_flipped.float()


@torch.no_grad()
def _prompt_group_random_signs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    group_sizes: list[int] | tuple[int, ...],
    seed: int,
) -> torch.Tensor:
    """Choose a reproducible +/-1 sign from each prompt's token ids."""
    if input_ids.dim() != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must be matching 2D tensors")
    if input_ids.shape != loss_mask.shape:
        raise ValueError("input_ids and loss_mask must have matching shapes")

    starts = []
    offset = 0
    for group_size in group_sizes:
        starts.append(offset)
        offset += group_size
    if offset != input_ids.shape[0]:
        raise ValueError("group sizes must cover every input sequence")

    first = torch.tensor(starts, device=input_ids.device, dtype=torch.long)
    tokens = input_ids.index_select(0, first).long()
    prompt_mask = attention_mask.index_select(0, first).bool().logical_and(
        ~loss_mask.index_select(0, first).bool()
    )
    prompt_positions = prompt_mask.long().cumsum(dim=1)
    mixed = (
        (tokens + 1) * 1_000_003
        + prompt_positions * 97_409
        + int(seed) * 65_537
    )
    mixed = torch.bitwise_xor(mixed, mixed >> 13)
    hashes = (mixed * prompt_mask.long()).sum(dim=1)
    hashes = torch.bitwise_xor(hashes, hashes >> 17)
    return hashes.remainder(2).mul(2).sub(1)


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip
        self.reward_group_baseline_shift = config.reward_group_baseline_shift
        self.reward_group_baseline_random_sign = (
            config.reward_group_baseline_random_sign
        )
        self.reward_group_baseline_random_seed = (
            config.reward_group_baseline_random_seed
        )
        self.reward_group_baseline_margin_quantile = (
            config.reward_group_baseline_margin_quantile
        )
        self.reward_group_baseline_margin_side = (
            config.reward_group_baseline_margin_side
        )
        self.reward_group_baseline_margin_scale = (
            config.reward_group_baseline_margin_scale
        )
        self.reward_group_baseline_center_quantile = (
            config.reward_group_baseline_center_quantile
        )
        self.reward_group_baseline_center_scale = (
            config.reward_group_baseline_center_scale
        )

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        direct_frpo_weight = config.importance_sampling_level in (
            "sequence_token", "none"
        )
        if direct_frpo_weight and (
            not config.use_decoupled_loss or config.use_sapo_loss
        ):
            raise ValueError(
                "Direct Proximal KL controls require use_decoupled_loss=True to keep "
                "logged behavior probabilities separate from the KL reference, and "
                "use_sapo_loss=False. The importance ratio itself is not decoupled."
            )

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            if direct_frpo_weight:
                logger.info(
                    "Mode: Proximal KL with direct importance weighting (off-policy)"
                )
            else:
                logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.rejection_sampling is not None:
                rs = config.rejection_sampling
                logger.info(
                    f"  Rejection sampling: level={rs.level}, metric={rs.metric}, "
                    f"action={rs.action}, upper={rs.upper}"
                    + (f", lower={rs.lower}" if rs.lower is not None else "")
                    + (f", agg={rs.agg}" if rs.level == "sequence" else "")
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(
            "  reward_group_baseline_shift: "
            f"{config.reward_group_baseline_shift}"
        )
        logger.info(
            "  reward_group_baseline_random_sign: "
            f"{config.reward_group_baseline_random_sign}"
        )
        if config.use_cispo:
            logger.info(
                "  objective: CISPO "
                f"(ratio=pi_theta/pi_behave, eps_low={config.cispo_eps_low}, "
                f"eps_high={config.cispo_eps_high})"
            )
        if direct_frpo_weight:
            logger.info(
                "  Proximal KL importance coefficient: "
                + (
                    "GM(pi_theta/pi_behave)"
                    if config.importance_sampling_level == "sequence_token"
                    else "1 (no IS)"
                )
            )
            logger.info("  Token-specific future KL credit; no ordinary PPO/dual clipping")
        logger.info(
            f"  eps_clip: {config.eps_clip}"
            + (" (unused)" if direct_frpo_weight else "")
        )
        logger.info("=" * 70)

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        return batched_call(self._compute_logp, data)

    def _compute_logp(self, data: dict[str, Any]) -> torch.Tensor | None:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batched, meta = concat_batch(data)
        batched["_traj_group_sizes"] = meta.traj_group_sizes
        return split_batch(self._compute_advantages(batched), meta)

    def _compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(
                reward_score, group_sizes=data.get("_traj_group_sizes")
            )
        if (
            self.reward_group_baseline_shift
            or self.reward_group_baseline_margin_quantile
            or self.reward_group_baseline_center_quantile
        ):
            group_sizes = data.get("_traj_group_sizes")
            if group_sizes is None:
                raise ValueError(
                    "reward group baseline perturbation requires trajectory groups"
                )
            random_signs = None
            if self.reward_group_baseline_random_sign:
                random_signs = _prompt_group_random_signs(
                    data["input_ids"],
                    data["attention_mask"],
                    data["loss_mask"],
                    group_sizes,
                    self.reward_group_baseline_random_seed,
                )
            reward_score, baseline_shift, sign_flipped = (
                _apply_group_baseline_shift(
                    reward_score,
                    group_sizes,
                    self.reward_group_baseline_shift,
                    self.reward_group_baseline_margin_quantile,
                    self.reward_group_baseline_margin_side,
                    self.reward_group_baseline_margin_scale,
                    self.reward_group_baseline_center_quantile,
                    self.reward_group_baseline_center_scale,
                    random_signs,
                )
            )
            data["reward_group_baseline_shift"] = baseline_shift
            data["reward_advantage_sign_flipped"] = sign_flipped
        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Pick the KL anchor: pi_ref (frozen reference) or pi_prox (current policy
        # at start of PPO update, recomputed via compute_logp). The latter makes
        # KL-in-reward + GAE(gamma=1,lambda=1) match the FRPO advantage formula.
        if self.config.kl_target == "prox":
            prox_logp_raw = data.get("prox_logp")
            if prox_logp_raw is None:
                kl_anchor = old_logp
            else:
                kl_anchor = torch.roll(prox_logp_raw, shifts=-1, dims=-1) * loss_mask
        else:
            kl_anchor = ref_logp

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        if self.config.use_frpo_loss_term:
            # FRPO loss-side path: KL contribution lives inside ppo_actor_loss_fn
            # as an advantage modification, recomputed per PPO inner step. Bypass
            # the reward injection here to avoid double-counting.
            rewards = torch.zeros_like(old_logp)
        else:
            rewards = -self.kl_ctl * self.kl_estimator(old_logp, kl_anchor)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp
        data.pop("_traj_group_sizes", None)

        return data

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._ppo_update, data, unpack=False)

    def _ppo_update(self, data: dict[str, Any]) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=infer_token_denominator(data, loss_mask),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        if "reward_group_baseline_shift" in data:
            baseline_shift = data["reward_group_baseline_shift"].float()
            seq_stats["reward_group_baseline_shift"] = baseline_shift
            seq_stats["reward_group_baseline_shift_positive"] = (
                baseline_shift > 0
            ).float()
            seq_stats["reward_group_baseline_shift_negative"] = (
                baseline_shift < 0
            ).float()
            seq_stats["reward_advantage_sign_flipped"] = data[
                "reward_advantage_sign_flipped"
            ].float()
        elif "reward_advantage_sign_flipped" in data:
            seq_stats["reward_advantage_sign_flipped"] = data[
                "reward_advantage_sign_flipped"
            ].float()
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.rejection_sampling is not None:
            rs = self.config.rejection_sampling
            scalars["rs_upper"] = rs.upper
            if rs.lower is not None:
                scalars["rs_lower"] = rs.lower
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept if needed for approximation/metrics in loss function
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()

            minibatch_stats_enabled = any(
                os.environ.get(name) == "1"
                for name in (
                    "AREAL_THETA_PROX_MINIBATCH_STATS",
                    "AREAL_FRPO_MINIBATCH_STATS",
                )
            )
            for minibatch_index, mb in enumerate(mb_inputs.mbs):
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        eps_clip=self.config.eps_clip,
                        eps_clip_higher=self.config.eps_clip_higher,
                        c_clip=self.config.c_clip,
                        rejection_sampling=self.config.rejection_sampling,
                        m2_threshold=self.m2_threshold,
                        importance_sampling_level=self.config.importance_sampling_level,
                        current_version=current_version,
                        prox_logp_method=self.config.prox_logp_method,
                        recompute_logprob=self.config.recompute_logprob,
                        use_sapo_loss=self.config.use_sapo_loss,
                        sapo_tau_pos=self.config.sapo_tau_pos,
                        sapo_tau_neg=self.config.sapo_tau_neg,
                        use_decoupled_loss=self.config.use_decoupled_loss,
                        use_frpo_loss_term=self.config.use_frpo_loss_term,
                        frpo_clip_eps=self.config.frpo_clip_eps,
                        frpo_beta=self.kl_ctl,
                        frpo_normalize_tail=self.config.frpo_normalize_tail,
                        frpo_tail_horizon=self.config.frpo_tail_horizon,
                        frpo_reset_tail_on_loss_mask_gap=(
                            self.config.frpo_reset_tail_on_loss_mask_gap
                        ),
                        frpo_local_boundary_horizon=self.config.frpo_local_boundary_horizon,
                        frpo_semantic_gate=(
                            self.config.frpo_semantic_gate
                            or self.config.frpo_attention_gate
                        ),
                        frpo_tail_gamma=self.config.frpo_tail_gamma,
                        frpo_tail_linear_decay_start=(
                            self.config.frpo_tail_linear_decay_start
                        ),
                        frpo_use_tail_correction=self.config.frpo_use_tail_correction,
                        frpo_use_local_correction=self.config.frpo_use_local_correction,
                        frpo_anchor=self.config.frpo_anchor,
                        frpo_minibatch_index=(
                            minibatch_index if minibatch_stats_enabled else None
                        ),
                        use_fipo=self.config.use_fipo,
                        fipo_beta=self.config.fipo_beta,
                        fipo_gamma=self.config.fipo_gamma,
                        fipo_dual_clip_c=self.config.fipo_dual_clip_c,
                        fipo_eps_low=self.config.fipo_eps_low,
                        fipo_eps_high=self.config.fipo_eps_high,
                        use_cispo=self.config.use_cispo,
                        cispo_eps_low=self.config.cispo_eps_low,
                        cispo_eps_high=self.config.cispo_eps_high,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)


class PPOActorController(TrainController):
    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_logp", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def compute_advantages(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_advantages", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def ppo_update(self, *args, **kwargs) -> None:
        self._custom_function_call(
            "ppo_update", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class PPOActorControllerV2(GatewayTrainController):
    def compute_logp(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/actor/compute_logp", payload)

    def compute_advantages(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/actor/compute_advantages", payload)

    def ppo_update(self, *args, **kwargs) -> None:
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post("/ppo/actor/update", payload)


def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_decoupled_loss: bool = False,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
    frpo_semantic_gate_mask: torch.Tensor | None = None,
    use_frpo_loss_term: bool = False,
    frpo_clip_eps: float | None = 0.2,
    frpo_beta: float = 0.0,
    frpo_normalize_tail: bool = False,
    frpo_tail_horizon: int = 0,
    frpo_reset_tail_on_loss_mask_gap: bool = False,
    frpo_local_boundary_horizon: bool = False,
    frpo_tail_lengths: torch.Tensor | None = None,
    frpo_semantic_gate: bool = False,
    frpo_tail_gamma: float = 1.0,
    frpo_tail_linear_decay_start: int = 0,
    frpo_use_tail_correction: bool = True,
    frpo_use_local_correction: bool = False,
    frpo_anchor: str = "prox",
    frpo_minibatch_index: int | None = None,
    use_fipo: bool = False,
    fipo_beta: float = 1.0,
    fipo_gamma: float = 0.99,
    fipo_dual_clip_c: float = 10.0,
    fipo_eps_low: float = 0.2,
    fipo_eps_high: float = 0.2,
    use_cispo: bool = False,
    cispo_eps_low: float | None = None,
    cispo_eps_high: float = 0.2,
    recompute_logprob: bool = False,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    entropy = entropy.detach()

    # Resolve proximal log-probabilities based on method
    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # Apply M2PO masking if threshold is set
    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    if use_cispo and use_sapo_loss:
        raise ValueError("CISPO and SAPO are mutually exclusive.")

    # Use SAPO or PPO-family loss
    if use_sapo_loss:
        if use_decoupled_loss:
            raise ValueError(
                "SAPO is not compatible with `use_decoupled_loss=True`. "
                "Please set `actor.use_decoupled_loss=false` in your configuration."
            )
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=c_clip,
            proximal_logprobs=prox_logp,
            rejection_sampling=rejection_sampling,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
            use_frpo_loss_term=use_frpo_loss_term,
            frpo_clip_eps=frpo_clip_eps,
            frpo_beta=frpo_beta,
            frpo_normalize_tail=frpo_normalize_tail,
            frpo_tail_horizon=frpo_tail_horizon,
            frpo_reset_tail_on_loss_mask_gap=(
                frpo_reset_tail_on_loss_mask_gap
            ),
            frpo_tail_lengths=(
                frpo_tail_lengths if frpo_local_boundary_horizon else None
            ),
            frpo_semantic_gate_mask=(
                frpo_semantic_gate_mask if frpo_semantic_gate else None
            ),
            frpo_tail_gamma=frpo_tail_gamma,
            frpo_tail_linear_decay_start=frpo_tail_linear_decay_start,
            frpo_use_tail_correction=frpo_use_tail_correction,
            frpo_use_local_correction=frpo_use_local_correction,
            frpo_anchor=frpo_anchor,
            ref_logprobs=input_data.get("ref_logp"),
            use_fipo=use_fipo,
            fipo_beta=fipo_beta,
            fipo_gamma=fipo_gamma,
            fipo_dual_clip_c=fipo_dual_clip_c,
            fipo_eps_low=fipo_eps_low,
            fipo_eps_high=fipo_eps_high,
            use_cispo=use_cispo,
            cispo_eps_low=cispo_eps_low,
            cispo_eps_high=cispo_eps_high,
        )

    # Joint Distillation KL Loss
    teacher_logp = input_data.get("teacher_logp")
    rkl_stat = None
    if teacher_logp is not None:
        # Coefficients for RL and Knowledge Distillation
        rl_loss_weight = input_data.get("rl_loss_weight", 1.0)
        distill_loss_weight = input_data.get("distill_loss_weight", 0.005)

        teacher_logp = (
            teacher_logp.detach()
        )  # detach to prevent gradient backprop to teacher

        if rl_loss_weight == 0:
            # Pure KD using reverse KL (importance-sampling)
            rkl_reward = teacher_logp - logprobs.detach()
            importance_weight = torch.exp(logprobs - old_logp)

            rkl_weighted_term = importance_weight * rkl_reward * loss_mask

            kd_coef = -1 * distill_loss_weight
            loss = kd_coef * rkl_weighted_term.sum() / loss_mask.sum().clamp(min=1)

            rkl_stat = -1 * rkl_weighted_term
        else:
            # KDRL: Knowledge Distillation + Reinforcement Learning (joint loss)
            rkl_penalty_per_token = (logprobs - teacher_logp) * loss_mask
            rkl_penalty = rkl_penalty_per_token.sum() / loss_mask.sum().clamp(min=1)

            loss = rl_loss_weight * loss + distill_loss_weight * rkl_penalty

            rkl_stat = rkl_penalty_per_token

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, loss_mask),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    if rkl_stat is not None:
        stats_tracker.stat(
            rkl_loss=rkl_stat,
            denominator="n_valid_tokens",
        )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        theta_prox_abs_logratio=stat["theta_prox_abs_logratio"],
        theta_prox_logratio_sq=stat["theta_prox_logratio_sq"],
        theta_prox_k3_sample=stat["theta_prox_k3_sample"],
        theta_prox_k3_prox_weighted=stat["theta_prox_k3_prox_weighted"],
        theta_prox_abs_logratio_gt_005=stat["theta_prox_abs_logratio_gt_005"],
        theta_prox_abs_logratio_gt_02=stat["theta_prox_abs_logratio_gt_02"],
        theta_prox_abs_logratio_gt_1=stat["theta_prox_abs_logratio_gt_1"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
            actor_loss=stat["loss"],
            clip_ratio=stat["clip_mask"].float(),
            dual_clip_ratio=stat["dual_clip_mask"].float(),
            pg_raw_advantage=stat["pg_raw_advantage"].float(),
            pg_raw_advantage_sq=stat["pg_raw_advantage_sq"].float(),
            pg_effective_advantage=stat["pg_effective_advantage"].float(),
            pg_effective_advantage_sq=stat["pg_effective_advantage_sq"].float(),
            pg_advantage_delta=stat["pg_advantage_delta"].float(),
            pg_raw_positive_fraction=stat["pg_raw_positive_fraction"].float(),
            pg_effective_positive_fraction=stat[
                "pg_effective_positive_fraction"
            ].float(),
            pg_pos_to_nonpos_flip=stat["pg_pos_to_nonpos_flip"].float(),
            pg_neg_to_pos_flip=stat["pg_neg_to_pos_flip"].float(),
            pg_raw_positive_mass=stat["pg_raw_positive_mass"].float(),
            pg_effective_positive_mass=stat["pg_effective_positive_mass"].float(),
            pg_reinforce_weight=stat["pg_reinforce_weight"].float(),
            pg_reinforce_weight_sq=stat["pg_reinforce_weight_sq"].float(),
            pg_raw_reinforce_weight=stat["pg_raw_reinforce_weight"].float(),
            pg_raw_reinforce_weight_sq=stat[
                "pg_raw_reinforce_weight_sq"
            ].float(),
            pg_reinforce_delta=stat["pg_reinforce_delta"].float(),
            pg_reinforce_delta_sq=stat["pg_reinforce_delta_sq"].float(),
            pg_outer_clip_suppressed_abs_mass=stat[
                "pg_outer_clip_suppressed_abs_mass"
            ].float(),
            pg_outer_clip_suppressed_signed_mass=stat[
                "pg_outer_clip_suppressed_signed_mass"
            ].float(),
            pg_outer_clip_suppressed_effective_abs_mass=stat[
                "pg_outer_clip_suppressed_effective_abs_mass"
            ].float(),
            pg_outer_clip_suppressed_effective_signed_mass=stat[
                "pg_outer_clip_suppressed_effective_signed_mass"
            ].float(),
            pg_outer_clip_suppressed_delta_abs_mass=stat[
                "pg_outer_clip_suppressed_delta_abs_mass"
            ].float(),
            pg_outer_clip_suppressed_delta_signed_mass=stat[
                "pg_outer_clip_suppressed_delta_signed_mass"
            ].float(),
            pg_reinforce_positive_mass=stat[
                "pg_reinforce_positive_mass"
            ].float(),
            pg_reinforce_negative_mass=stat[
                "pg_reinforce_negative_mass"
            ].float(),
            denominator="n_valid_tokens",
        )
    if use_cispo:
        stats_tracker.stat(
            cispo_logratio=stat["cispo_logratio"].float(),
            cispo_ratio=stat["cispo_ratio"].float(),
            cispo_weight=stat["cispo_weight"].float(),
            cispo_clip_ratio=stat["cispo_clip_mask"].float(),
            denominator="n_valid_tokens",
        )
    if os.environ.get("AREAL_PPO_POSITION_STATS") == "1":
        position_masks = _response_position_masks(
            loss_mask, input_data.get("cu_seqlens")
        )
        for label, position_mask in position_masks.items():
            denominator = f"ppo_position_{label}_tokens"
            stats_tracker.denominator(**{denominator: position_mask})
            stats_tracker.stat(
                **{
                    f"clip_ratio_{label}": (
                        stat["clip_mask"].float() * position_mask
                    ),
                    f"theta_prox_k3_sample_{label}": stat[
                        "theta_prox_k3_sample"
                    ].float()
                    * position_mask,
                    f"theta_prox_k3_prox_weighted_{label}": stat[
                        "theta_prox_k3_prox_weighted"
                    ].float()
                    * position_mask,
                    f"pg_outer_clip_suppressed_effective_abs_mass_{label}": stat[
                        "pg_outer_clip_suppressed_effective_abs_mass"
                    ].float()
                    * position_mask,
                    f"pg_outer_clip_suppressed_delta_abs_mass_{label}": stat[
                        "pg_outer_clip_suppressed_delta_abs_mass"
                    ].float()
                    * position_mask,
                },
                denominator=denominator,
            )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            behave_prox_abs_logratio=stat["behave_prox_abs_logratio"],
            behave_prox_logratio_sq=stat["behave_prox_logratio_sq"],
            behave_prox_k3_sample=stat["behave_prox_k3_sample"],
            denominator="unclipped_behave_tokens",
        )
    if frpo_minibatch_index is not None:
        suffix = f"mb{frpo_minibatch_index}"
        denominator = f"theta_prox_{suffix}_tokens"
        stats_tracker.denominator(**{denominator: loss_mask.bool()})
        stats_tracker.stat(
            **{
                f"theta_prox_abs_logratio_{suffix}": stat[
                    "theta_prox_abs_logratio"
                ],
                f"theta_prox_logratio_sq_{suffix}": stat[
                    "theta_prox_logratio_sq"
                ],
                f"theta_prox_k3_sample_{suffix}": stat["theta_prox_k3_sample"],
                f"theta_prox_abs_logratio_gt_005_{suffix}": stat[
                    "theta_prox_abs_logratio_gt_005"
                ],
                f"theta_prox_abs_logratio_gt_02_{suffix}": stat[
                    "theta_prox_abs_logratio_gt_02"
                ],
                f"theta_prox_abs_logratio_gt_1_{suffix}": stat[
                    "theta_prox_abs_logratio_gt_1"
                ],
            },
            denominator=denominator,
        )
    if "frpo_clipped_logratio" in stat:
            stats_tracker.stat(
                frpo_clipped_logratio=stat["frpo_clipped_logratio"].float(),
                frpo_clip_ratio=stat["frpo_clip_mask"].float(),
                frpo_tail=stat["frpo_tail"].float(),
                frpo_tail_sq=stat["frpo_tail_sq"].float(),
                frpo_correction=stat["frpo_correction"].float(),
                frpo_correction_sq=stat["frpo_correction_sq"].float(),
                frpo_local_correction=stat["frpo_local_correction"].float(),
                frpo_total_correction=stat["frpo_total_correction"].float(),
                frpo_total_correction_sq=stat["frpo_total_correction_sq"].float(),
                frpo_correction_over_abs_adv=stat[
                    "frpo_correction_over_abs_adv"
                ].float(),
                frpo_tail_positive_fraction=stat[
                    "frpo_tail_positive_fraction"
                ].float(),
                frpo_semantic_gate=stat["frpo_semantic_gate_mask"].float(),
                denominator="n_valid_tokens",
            )
            if os.environ.get("AREAL_FRPO_POSITION_STATS") == "1":
                position_masks = _response_position_masks(
                    loss_mask, input_data.get("cu_seqlens")
                )
                for label, position_mask in position_masks.items():
                    denominator = f"frpo_position_{label}_tokens"
                    stats_tracker.denominator(**{denominator: position_mask})
                    stats_tracker.stat(
                        **{
                            f"frpo_correction_{label}": stat[
                                "frpo_correction"
                            ].float()
                            * position_mask,
                            f"frpo_correction_sq_{label}": stat[
                                "frpo_correction_sq"
                            ].float()
                            * position_mask,
                            f"pg_pos_to_nonpos_flip_{label}": stat[
                                "pg_pos_to_nonpos_flip"
                            ].float()
                            * position_mask,
                            f"pg_reinforce_weight_{label}": stat[
                                "pg_reinforce_weight"
                            ].float()
                            * position_mask,
                        },
                        denominator=denominator,
                    )
            if frpo_minibatch_index is not None:
                suffix = f"mb{frpo_minibatch_index}"
                denominator = f"frpo_{suffix}_tokens"
                stats_tracker.denominator(**{denominator: loss_mask.bool()})
                stats_tracker.stat(
                    **{
                        f"frpo_clipped_logratio_{suffix}": stat[
                            "frpo_clipped_logratio"
                        ].float(),
                        f"frpo_tail_{suffix}": stat["frpo_tail"].float(),
                        f"frpo_tail_sq_{suffix}": stat["frpo_tail_sq"].float(),
                        f"pg_effective_advantage_{suffix}": stat[
                            "pg_effective_advantage"
                        ].float(),
                        f"pg_effective_advantage_sq_{suffix}": stat[
                            "pg_effective_advantage_sq"
                        ].float(),
                    },
                    denominator=denominator,
                )
                if os.environ.get("AREAL_FRPO_ALIGNMENT_STATS") == "1":
                    def track_alignment(
                        alignment_suffix: str, alignment_mask: torch.Tensor
                    ) -> None:
                        sequence_tail = _masked_sequence_means(
                            stat["frpo_tail"],
                            alignment_mask,
                            input_data.get("cu_seqlens"),
                        )
                        sequence_raw_advantage = _masked_sequence_means(
                            stat["pg_raw_advantage"],
                            alignment_mask,
                            input_data.get("cu_seqlens"),
                        )
                        if sequence_tail.numel() != sequence_raw_advantage.numel():
                            raise RuntimeError(
                                "tail and raw-advantage sequence counts do not match"
                            )
                        if sequence_tail.numel() == 0:
                            return
                        sequence_denominator = (
                            f"frpo_alignment_{alignment_suffix}_sequences"
                        )
                        stats_tracker.denominator(
                            **{
                                sequence_denominator: torch.ones_like(
                                    sequence_tail, dtype=torch.bool
                                )
                            }
                        )
                        stats_tracker.stat(
                            **{
                                f"frpo_sequence_tail_{alignment_suffix}": sequence_tail,
                                f"frpo_sequence_tail_sq_{alignment_suffix}": (
                                    sequence_tail.square()
                                ),
                                f"frpo_sequence_raw_advantage_{alignment_suffix}": (
                                    sequence_raw_advantage
                                ),
                                f"frpo_sequence_raw_advantage_sq_{alignment_suffix}": (
                                    sequence_raw_advantage.square()
                                ),
                                f"frpo_sequence_tail_raw_advantage_{alignment_suffix}": (
                                    sequence_tail * sequence_raw_advantage
                                ),
                            },
                            denominator=sequence_denominator,
                        )

                    track_alignment(suffix, loss_mask)
                    for position_label, position_mask in _response_position_masks(
                        loss_mask, input_data.get("cu_seqlens")
                    ).items():
                        track_alignment(
                            f"{suffix}_{position_label}", position_mask
                        )
    if "fipo_f_t" in stat:
        stats_tracker.stat(
            fipo_delta_logp=stat["fipo_delta_logp"].float(),
            fipo_tail=stat["fipo_tail"].float(),
            fipo_f_t=stat["fipo_f_t"].float(),
            fipo_outlier_keep_frac=stat["fipo_outlier_mask"].float(),
            denominator="n_valid_tokens",
        )
    if "filtered_fraction" in stat:
        stats_tracker.scalar(rs_filtered_fraction=stat["filtered_fraction"])

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    # Log SAPO-specific statistics
    if use_sapo_loss:
        stats_tracker.stat(
            sapo_soft_gate=stat["sapo_soft_gate"],
            sapo_scaled_gate_pos=stat["sapo_scaled_gate_pos"],
            sapo_scaled_gate_neg=stat["sapo_scaled_gate_neg"],
            denominator="n_valid_tokens",
        )
    else:
        # Log clipping statistics (PPO only)
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )

    # Log proximal approximation metrics
    compute_logp_mask = stat.get("behave_mask", loss_mask)
    _log_proximal_approximation_stats(
        prox_logp_method=prox_logp_method,
        prox_logp_gt=prox_logp_gt,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
        compute_logp_mask=compute_logp_mask,
    )

    # Log version staleness metrics
    if "versions" in input_data and current_version is not None:
        version_metrics_mask = stat.get("behave_mask", loss_mask)
        # This is the PPO proximal snapshot, independent of the KL reference
        # selected by frpo_anchor. Recomputed logp is frozen at the beginning
        # of this outer update; without recomputation it is behavior logp.
        # Interpolated logp does not identify an actual policy snapshot.
        if prox_logp_gt is None:
            proximal_version = None
        elif use_decoupled_loss or recompute_logprob:
            proximal_version = current_version
        else:
            proximal_version = input_data["versions"]
        _log_version_staleness_stats(
            versions=input_data["versions"],
            current_version=current_version,
            version_metrics_mask=version_metrics_mask,
            proximal_version=proximal_version,
        )

    return loss


# =============================================================================
# Core Functions
# =============================================================================


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, metrics).
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (should be detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor.

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


def _apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance.

    Args:
        old_logp: Behavior policy log-probabilities.
        prox_logp: Proximal policy log-probabilities.
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering.

    Returns:
        Updated loss mask with M2PO filtering applied.
    """
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask


# =============================================================================
# Logging Helper Functions
# =============================================================================

_EPSILON = 1e-8  # Small constant for numerical stability in relative error calculations


def _compute_importance_weight(
    logp_numerator: torch.Tensor,
    logp_denominator: torch.Tensor,
) -> torch.Tensor:
    """Compute importance weight as exp(logp_num - logp_denom)."""
    return torch.exp(logp_numerator - logp_denominator).float()


def _compute_approximation_errors(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute error metrics between ground truth and approximation.

    Returns:
        Dictionary with abs_error, rel_error, and squared_error tensors.
    """
    diff = ground_truth - approximation
    abs_error = torch.abs(diff).float()
    rel_error = torch.abs(diff / (torch.abs(ground_truth) + _EPSILON)).float()
    squared_error = (diff * diff).float()
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": squared_error,
    }


def _tensor_scalar_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Compute scalar statistics (avg, max, min) for a tensor.

    Args:
        tensor: Input tensor to compute statistics on.

    Returns:
        Dictionary with avg, max, min as Python floats.
    """
    t = tensor.float()
    return {
        "avg": t.mean().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def _log_approximation_metrics_for_method(
    method_name: str,
    approx_logp: torch.Tensor,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    prox_logp_gt: torch.Tensor | None = None,
) -> None:
    """
    Log metrics for a single approximation method.

    Args:
        method_name: Name of the approximation method (e.g., "loglinear").
        approx_logp: Approximated proximal log-probabilities.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities.
        prox_logp_gt: Ground truth proximal logp, or None if unavailable.
    """
    # Compute importance weights from approximation
    behave_imp_weight = _compute_importance_weight(approx_logp, old_logp)
    importance_weight = _compute_importance_weight(logprobs, approx_logp)

    metrics = {
        f"{method_name}/approx_logp": approx_logp.float(),
        f"{method_name}/behave_imp_weight": behave_imp_weight,
        f"{method_name}/importance_weight": importance_weight,
    }

    # Add error metrics if ground truth is available
    if prox_logp_gt is not None:
        # Log-probability errors
        logp_errors = _compute_approximation_errors(prox_logp_gt, approx_logp)
        metrics.update(
            {
                f"{method_name}/abs_error": logp_errors["abs_error"],
                f"{method_name}/rel_error": logp_errors["rel_error"],
                f"{method_name}/squared_error": logp_errors["squared_error"],
            }
        )

        # Ground truth importance weights for comparison
        behave_imp_weight_gt = _compute_importance_weight(prox_logp_gt, old_logp)
        importance_weight_gt = _compute_importance_weight(logprobs, prox_logp_gt)

        # Importance weight errors
        behave_errors = _compute_approximation_errors(
            behave_imp_weight_gt, behave_imp_weight
        )
        imp_errors = _compute_approximation_errors(
            importance_weight_gt, importance_weight
        )

        metrics.update(
            {
                f"{method_name}/behave_imp_weight_abs_error": behave_errors[
                    "abs_error"
                ],
                f"{method_name}/behave_imp_weight_rel_error": behave_errors[
                    "rel_error"
                ],
                f"{method_name}/importance_weight_abs_error": imp_errors["abs_error"],
                f"{method_name}/importance_weight_rel_error": imp_errors["rel_error"],
            }
        )

    stats_tracker.stat(**metrics, denominator="n_valid_tokens")


def _log_proximal_approximation_stats(
    prox_logp_method: str,
    prox_logp_gt: torch.Tensor | None,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
    compute_logp_mask: torch.Tensor,
) -> None:
    """
    Log proximal policy approximation metrics based on the method.

    Args:
        prox_logp_method: The proximal logp method being used.
        prox_logp_gt: Ground truth proximal logp, or None if skipped.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.
        compute_logp_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("compute_logp"):
        stats_tracker.denominator(n_valid_tokens=compute_logp_mask.bool())

        # Log ground truth when available
        if prox_logp_gt is not None:
            stats_tracker.stat(
                prox_logp_gt=prox_logp_gt.float(),
                denominator="n_valid_tokens",
            )

        # Skip if versions not available
        if versions is None or current_version is None:
            return

        if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
            # Loglinear mode: log approximation without error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=None,  # No ground truth in loglinear mode
                )

        elif prox_logp_method == PROX_LOGP_METHOD_METRICS and prox_logp_gt is not None:
            # Metrics mode: compute all methods with error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=None,  # Compute all methods
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=prox_logp_gt,
                )

        if logprobs is not None:
            # Log KL divergence estimators to check for policy drift between the
            # training-time policy (logprobs) and the inference-time policy (old_logp).
            log_ratio = (logprobs.float() - old_logp.float()).detach()

            # Implementation of different estimators for KL divergence.
            # See: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl
            kl_div_estimator_direct = -log_ratio
            kl_div_estimator_taylor = log_ratio**2 / 2.0
            kl_div_estimator_dual = log_ratio.exp() - 1 - log_ratio

            # Register these to TensorBoard
            stats_tracker.stat(
                kl_div_direct=kl_div_estimator_direct,
                kl_div_taylor=kl_div_estimator_taylor,
                kl_div_dual=kl_div_estimator_dual,
                denominator="n_valid_tokens",
            )


def _log_version_staleness_stats(
    versions: torch.Tensor,
    current_version: int,
    version_metrics_mask: torch.Tensor,
    proximal_version: int | torch.Tensor | None,
) -> None:
    """
    Log sample staleness metrics based on policy versions.

    Args:
        versions: Per-token policy versions from rollout.
        current_version: Current training version.
        version_metrics_mask: Mask for valid tokens.
        proximal_version: Recomputed snapshot version, per-token behavior
            versions, or None when logp is approximated without a snapshot.
    """
    with stats_tracker.scope("version_stats"):
        stats_tracker.denominator(n_valid_tokens=version_metrics_mask.bool())

        v_theta = current_version
        v_behave = versions.float()

        # Filter to generated tokens only (version >= 0)
        valid_generated_mask = version_metrics_mask & (versions >= 0)

        if not valid_generated_mask.any():
            return

        # Compute staleness for valid tokens
        staleness_theta = (v_theta - v_behave)[valid_generated_mask]

        # Compute and log statistics
        theta_stats = _tensor_scalar_stats(staleness_theta)

        metrics = dict(
            sample_staleness_theta_avg=theta_stats["avg"],
            sample_staleness_theta_max=theta_stats["max"],
            sample_staleness_theta_min=theta_stats["min"],
            v_theta=v_theta,
            n_valid_generated_tokens=valid_generated_mask.sum().item(),
        )
        if proximal_version is not None:
            if isinstance(proximal_version, torch.Tensor):
                v_proximal = proximal_version.float()[valid_generated_mask]
                version_stats = _tensor_scalar_stats(v_proximal)
                # Behavior versions may differ between tokens. Keep the
                # existing scalar as their mean and expose the range.
                metrics.update(
                    v_proximal=version_stats["avg"],
                    v_proximal_min=version_stats["min"],
                    v_proximal_max=version_stats["max"],
                )
            else:
                v_proximal = proximal_version
                metrics["v_proximal"] = v_proximal
            proximal_stats = _tensor_scalar_stats(
                v_proximal - v_behave[valid_generated_mask]
            )
            metrics.update(
                sample_staleness_proximal_avg=proximal_stats["avg"],
                sample_staleness_proximal_max=proximal_stats["max"],
                sample_staleness_proximal_min=proximal_stats["min"],
            )
        stats_tracker.scalar(**metrics)
