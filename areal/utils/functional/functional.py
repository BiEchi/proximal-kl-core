# SPDX-License-Identifier: Apache-2.0

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from areal.api.cli_args import RejectionSamplingConfig
from areal.utils.data import KLEstimator


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level geometric mean ratios and average advantages per sequence (GSPO).

    Args:
        log_ratio: Log of probability ratios (logprobs - proximal_logprobs)
        advantages: Per-token advantages
        loss_mask: Boolean mask indicating valid tokens
        cu_seqlens: Cumulative sequence lengths. Required for 1D tensors (packed format).
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            For a single sequence, use cu_seqlens=torch.tensor([0, seq_len]).

    Returns:
        ratio: Sequence-level importance sampling ratios (broadcast to all tokens)
        advantages: Sequence-averaged advantages (broadcast to all tokens)
            Note: We use mean instead of sum to keep gradient magnitude independent
            of sequence length. When multiplied by ratio and summed over tokens,
            this gives the correct total gradient contribution per sequence.
    """
    # Handle both 1D (packed) and 2D (padded) tensor shapes
    if log_ratio.ndim == 1:
        # For 1D tensors (packed format), cu_seqlens is required
        if cu_seqlens is None:
            raise ValueError(
                "cu_seqlens is required for 1D tensors (packed format). "
                "In AReaL, 1D tensors are produced by pack_tensor_dict() and always have cu_seqlens. "
                "For a single sequence, use cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32)."
            )

        # Packed sequences: use cu_seqlens boundaries
        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # Create sequence index for each token: [0,0,0,1,1,2,2,2,2,...]
        sequence_idx = torch.arange(
            batch_size, device=log_ratio.device
        ).repeat_interleave(seq_lengths)

        # Use scatter_add for vectorized summation per sequence (faster than Python loop)
        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(
            batch_size, device=log_ratio.device, dtype=log_ratio.dtype
        ).scatter_add_(0, sequence_idx, masked_log_ratio)

        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(
            batch_size, device=advantages.device, dtype=advantages.dtype
        ).scatter_add_(0, sequence_idx, masked_advantages)

        valid_count_per_seq = (
            torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
            .scatter_add_(0, sequence_idx, loss_mask.int())
            .clamp(min=1)
        )

        # Compute sequence-level means
        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(
            log_ratio.dtype
        )
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(
            advantages.dtype
        )

        # Broadcast sequence-level values back to token-level
        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)

        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        # For 2D tensors (padded sequences)
        # Input shape: [batch_size, seq_len]
        # Compute mean log ratio over sequence length for each sample
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / (
            loss_mask.sum(dim=1).clamp(min=1)
        )
        # Broadcast back to original shape: each sequence gets its own geometric mean ratio
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        # Apply mask
        ratio = torch.where(loss_mask, ratio, 0.0)

        # Average token advantages per sequence
        # This ensures gradient magnitude is independent of sequence length
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages = (
            masked_advantages.sum(dim=-1, keepdim=True) / seq_lengths
        ).expand_as(log_ratio)
        advantages = torch.where(loss_mask, advantages, 0.0)

    return ratio, advantages


def _compute_sequence_token_weight(
    log_ratio: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Broadcast a response geometric-mean weight without averaging token credit.

    The caller supplies detached current-to-behavior log ratios. This helper
    aggregates action tokens only; prompt and padding positions never contribute.
    """
    masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
    if log_ratio.ndim == 1:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for packed sequence-token IS.")
        n_seqs = cu_seqlens.shape[0] - 1
        sequence_idx = torch.arange(n_seqs, device=log_ratio.device).repeat_interleave(
            cu_seqlens[1:] - cu_seqlens[:-1], output_size=log_ratio.numel()
        )
        sums = log_ratio.new_zeros(n_seqs).scatter_add_(
            0, sequence_idx, masked_log_ratio
        )
        counts = log_ratio.new_zeros(n_seqs).scatter_add_(
            0, sequence_idx, loss_mask.to(log_ratio.dtype)
        )
        weight = torch.exp(sums / counts.clamp(min=1))[sequence_idx]
    elif log_ratio.ndim == 2:
        counts = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        means = masked_log_ratio.sum(dim=-1, keepdim=True) / counts
        weight = torch.exp(means).expand_as(log_ratio)
    else:
        raise ValueError("Sequence-token IS expects packed 1D or padded 2D tensors.")
    return torch.where(loss_mask, weight, 0.0)


@dataclass
class RejectionSamplingResult:
    """Result of rejection sampling, used by ppo_actor_loss_fn.

    Attributes:
        loss_mask: Updated loss mask (mask mode) or original loss mask (clamp mode).
        behave_imp_weight: Importance weight (clamped in clamp mode, raw in mask mode).
        filtered_fraction: Fraction of valid tokens that were filtered/clamped (for logging).
    """

    loss_mask: torch.Tensor
    behave_imp_weight: torch.Tensor
    filtered_fraction: float


def _check_bounds(
    metric: torch.Tensor, config: RejectionSamplingConfig
) -> torch.Tensor:
    """Check if metric values are within configured bounds.

    Args:
        metric: Per-token or per-sequence metric values.
        config: Rejection sampling configuration with upper and optional lower bounds.

    Returns:
        Boolean tensor, True where metric is within bounds.
    """
    if config.lower is not None:
        return (metric >= config.lower) & (metric <= config.upper)
    else:
        return metric <= config.upper


def apply_rejection_sampling(
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    config: RejectionSamplingConfig,
) -> RejectionSamplingResult:
    """Apply rejection sampling based on divergence between proximal and behavior policy.

    Supports two action modes:
    - 'mask': zero out loss_mask for tokens/sequences exceeding threshold (rejection)
    - 'clamp': clamp importance weight to bounds for tokens/sequences exceeding
      threshold (truncation, tokens still participate in gradient)

    Args:
        proximal_logprobs: Proximal policy log-probabilities,
            shape [batch, seq_len] (2D padded) or [total_tokens] (1D packed).
        old_logprobs: Behavior policy log-probabilities from inference engine,
            same shape as proximal_logprobs.
        loss_mask: Original loss mask (1 for valid tokens), same shape as proximal_logprobs.
        cu_seqlens: Cumulative sequence lengths for 1D packed format. Shape: [batch_size + 1].
            Required when inputs are 1D. None for 2D padded inputs.
        config: Configuration for rejection sampling.

    Returns:
        RejectionSamplingResult with updated loss_mask, behave_imp_weight, and filtered_fraction.
    """
    # Step 0: Validate input shapes.
    if proximal_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"proximal_logprobs shape {proximal_logprobs.shape} != "
            f"old_logprobs shape {old_logprobs.shape}"
        )
    if proximal_logprobs.shape != loss_mask.shape:
        raise ValueError(
            f"proximal_logprobs shape {proximal_logprobs.shape} != "
            f"loss_mask shape {loss_mask.shape}"
        )
    if proximal_logprobs.ndim not in (1, 2):
        raise ValueError(
            f"Expected 1D (packed) or 2D (padded) tensors, "
            f"got ndim={proximal_logprobs.ndim}"
        )

    # Step 1: Compute log ratio = log(π_proximal / π_behave)
    # Upcast operands to fp32 before subtraction to avoid precision loss in bf16/fp16.
    log_ratio = proximal_logprobs.detach().float() - old_logprobs.detach().float()
    # Sanitize non-finite values (e.g. -inf - (-inf) = NaN) to prevent NaN propagation.
    log_ratio = torch.where(torch.isfinite(log_ratio), log_ratio, 0.0)

    # Step 2: Compute metric value (reuse existing KLEstimator sign conventions)
    if config.metric == "ratio":
        # Direct ratio π_proximal / π_behave
        metric = torch.exp(log_ratio)
    elif config.metric in ("kl_k1", "kl_k2", "kl_k3"):
        # Use existing KLEstimator (note: _compute_approx_kl takes log_probs, log_probs_base)
        estimator_name = config.metric.replace("kl_", "")  # "k1", "k2", "k3"
        metric = KLEstimator._compute_approx_kl(
            log_probs=proximal_logprobs.detach(),
            log_probs_base=old_logprobs.detach(),
            kl_estimator=estimator_name,
            apply_clamp=False,  # Don't clamp; threshold check handles bounds
        )
    else:
        raise ValueError(f"Unknown metric: {config.metric}")

    # Step 3: Compute behave_imp_weight (needed for both modes)
    behave_imp_weight = torch.exp(log_ratio)
    # Save original weight before any clamping, to compute clamped fraction later.
    original_weight = behave_imp_weight

    # Step 4: Aggregate and filter
    #
    # For ratio metric, aggregate in log space (geometric mean) to match GSPO
    # semantics and avoid the "length trap" where arithmetic mean inflates
    # sequence-level ratios. For KL metrics, aggregate in metric space
    # (arithmetic) since KL divergence is additive.
    _use_log_agg = config.metric == "ratio"

    if config.level == "sequence":
        # Pre-compute sequence indexing (shared by filtering and weight broadcast).
        if loss_mask.ndim == 1:
            # 1D packed format: use cu_seqlens
            if cu_seqlens is None:
                raise ValueError(
                    "cu_seqlens is required for 1D packed tensors "
                    "in sequence-level filtering."
                )
            batch_size = cu_seqlens.shape[0] - 1
            seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            sequence_idx = torch.arange(
                batch_size, device=metric.device
            ).repeat_interleave(seq_lengths)

            # For ratio metric: aggregate log_ratio (geometric); else: aggregate metric (arithmetic).
            agg_values = log_ratio if _use_log_agg else metric
            masked_agg = torch.where(loss_mask.bool(), agg_values, 0.0)
            valid_count_per_seq = (
                torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
                .scatter_add_(0, sequence_idx, loss_mask.int())
                .clamp(min=1)
            )

            # Ratio metric + sequence level: use geometric mean as uniform weight
            # for all tokens (matches old sequence_mask/sequence_truncate semantics).
            if _use_log_agg:
                # masked_agg is already log_ratio masked by loss_mask (computed above).
                seq_log_sum = torch.zeros(
                    batch_size, device=log_ratio.device, dtype=log_ratio.dtype
                ).scatter_add_(0, sequence_idx, masked_agg)
                seq_log_mean = seq_log_sum / valid_count_per_seq.to(log_ratio.dtype)
                behave_imp_weight = torch.exp(seq_log_mean)[sequence_idx]
                original_weight = behave_imp_weight

            if config.agg == "sum":
                seq_agg = torch.zeros(
                    batch_size, device=metric.device, dtype=agg_values.dtype
                ).scatter_add_(0, sequence_idx, masked_agg)
            elif config.agg == "mean":
                seq_agg_sum = torch.zeros(
                    batch_size, device=metric.device, dtype=agg_values.dtype
                ).scatter_add_(0, sequence_idx, masked_agg)
                seq_agg = seq_agg_sum / valid_count_per_seq.to(agg_values.dtype)
            elif config.agg == "max":
                agg_for_max = agg_values.masked_fill(~loss_mask.bool(), float("-inf"))
                seq_agg = torch.full(
                    (batch_size,),
                    float("-inf"),
                    device=metric.device,
                    dtype=agg_values.dtype,
                ).scatter_reduce_(0, sequence_idx, agg_for_max, reduce="amax")
                # All-masked sequences stay -inf; treat them as in-bounds (no valid
                # tokens to filter, and their loss_mask is already all-zero).
                # Recompute from raw counts to detect true zero.
                raw_valid = torch.zeros(
                    batch_size, device=loss_mask.device, dtype=torch.int32
                ).scatter_add_(0, sequence_idx, loss_mask.int())
                all_masked = raw_valid == 0
                seq_agg = torch.where(all_masked, torch.zeros_like(seq_agg), seq_agg)
            else:
                raise ValueError(f"Unknown agg method: {config.agg}")

            # Convert back to metric space for threshold comparison.
            seq_metric = torch.exp(seq_agg) if _use_log_agg else seq_agg

            # Check each sequence against bounds
            in_bounds_per_seq = _check_bounds(seq_metric, config)

            if config.action == "mask":
                # Broadcast back to token level, filter entire sequence
                in_bounds = in_bounds_per_seq[sequence_idx]
            else:
                # clamp mode: clamp tokens in out-of-bounds sequences
                out_of_bounds = (~in_bounds_per_seq)[sequence_idx]
                behave_imp_weight = torch.where(
                    out_of_bounds,
                    behave_imp_weight.clamp(
                        min=config.lower if config.lower is not None else 0.0,
                        max=config.upper,
                    ),
                    behave_imp_weight,
                )
        else:
            # 2D padded format
            agg_values = log_ratio if _use_log_agg else metric
            masked_agg = torch.where(loss_mask.bool(), agg_values, 0.0)
            valid_count = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)

            # Ratio metric + sequence level: geometric mean as uniform weight.
            if _use_log_agg:
                seq_log_mean = masked_agg.sum(dim=-1, keepdim=True) / valid_count
                behave_imp_weight = torch.exp(seq_log_mean).expand_as(log_ratio)
                original_weight = behave_imp_weight

            if config.agg == "sum":
                seq_agg = masked_agg.sum(dim=-1, keepdim=True)
            elif config.agg == "mean":
                seq_agg = masked_agg.sum(dim=-1, keepdim=True) / valid_count
            elif config.agg == "max":
                agg_for_max = agg_values.masked_fill(~loss_mask.bool(), float("-inf"))
                seq_agg = agg_for_max.max(dim=-1, keepdim=True)[0]
                # All-masked sequences stay -inf; treat them as in-bounds (no valid
                # tokens to filter, and their loss_mask is already all-zero).
                all_masked = loss_mask.sum(dim=-1, keepdim=True) == 0
                seq_agg = torch.where(all_masked, torch.zeros_like(seq_agg), seq_agg)
            else:
                raise ValueError(f"Unknown agg method: {config.agg}")

            # Convert back to metric space for threshold comparison.
            seq_metric = torch.exp(seq_agg) if _use_log_agg else seq_agg

            if config.action == "mask":
                in_bounds = _check_bounds(seq_metric, config).expand_as(loss_mask)
            else:
                # clamp mode: clamp tokens in out-of-bounds sequences
                out_of_bounds = (~_check_bounds(seq_metric, config)).expand_as(
                    loss_mask
                )
                behave_imp_weight = torch.where(
                    out_of_bounds,
                    behave_imp_weight.clamp(
                        min=config.lower if config.lower is not None else 0.0,
                        max=config.upper,
                    ),
                    behave_imp_weight,
                )
    else:
        # Token level
        if config.action == "mask":
            in_bounds = _check_bounds(metric, config)
        else:
            # clamp mode: directly clamp importance weight
            behave_imp_weight = behave_imp_weight.clamp(
                min=config.lower if config.lower is not None else 0.0,
                max=config.upper,
            )

    # Step 5: Update loss_mask or keep it based on action mode
    if config.action == "mask":
        candidates = loss_mask.bool()
        updated_mask = (candidates & in_bounds).to(loss_mask.dtype)
        filtered_count = (candidates & ~in_bounds).sum().item()
        total_count = candidates.sum().item()
        filtered_fraction = filtered_count / max(total_count, 1)
    else:
        # clamp mode: loss_mask unchanged
        updated_mask = loss_mask
        # Report fraction of clamped tokens (for logging)
        clamped_count = (
            (loss_mask.bool() & (original_weight != behave_imp_weight)).sum().item()
        )
        total_count = loss_mask.bool().sum().item()
        filtered_fraction = clamped_count / max(total_count, 1)

    # Apply loss_mask to behave_imp_weight
    behave_imp_weight = torch.where(updated_mask.bool(), behave_imp_weight, 0.0)

    return RejectionSamplingResult(
        loss_mask=updated_mask,
        behave_imp_weight=behave_imp_weight,
        filtered_fraction=filtered_fraction,
    )


def _frpo_reverse_cumsum_per_seq(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    horizon: int | None = None,
) -> torch.Tensor:
    """Per-sequence tail-sum, optionally truncated to ``horizon`` tokens.

    Handles 2D [bs, seq] padded tensors (no cu_seqlens needed) and 1D packed
    tensors (require cu_seqlens of shape [n_seq + 1]).
    """
    if horizon is not None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive or None, got {horizon}")
        if x.dim() == 2:
            seq_len = x.shape[1]
            prefix = torch.cat(
                [
                    torch.zeros(
                        (x.shape[0], 1), device=x.device, dtype=x.dtype
                    ),
                    torch.cumsum(x, dim=1),
                ],
                dim=1,
            )
            starts = torch.arange(seq_len, device=x.device)
            ends = (starts + horizon).clamp(max=seq_len)
            return prefix[:, ends] - prefix[:, starts]

        n = x.shape[0]
        prefix = torch.cat(
            [torch.zeros(1, device=x.device, dtype=x.dtype), torch.cumsum(x, dim=0)]
        )
        starts = torch.arange(n, device=x.device)
        if cu_seqlens is None:
            ends = (starts + horizon).clamp(max=n)
        else:
            seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            seg_end = torch.repeat_interleave(cu_seqlens[1:], seg_lens)
            ends = torch.minimum(starts + horizon, seg_end)
        return prefix[ends] - prefix[starts]

    if x.dim() == 2:
        return torch.flip(torch.cumsum(torch.flip(x, dims=[1]), dim=1), dims=[1])
    n = x.shape[0]
    if cu_seqlens is None:
        return torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0), dims=[0])
    # Global reverse cumsum, padded with 0 at the end as the "after-end" reference.
    global_rcs = torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0), dims=[0])
    global_rcs = torch.cat(
        [global_rcs, torch.zeros(1, device=x.device, dtype=x.dtype)]
    )
    # For each token t, the index of "one past the end of its own sequence" is
    # cu_seqlens[seg_id + 1] where seg_id is the segment containing t. The
    # per-token tail-sum within its segment is then
    #   global_rcs[t] - global_rcs[seg_end[t]].
    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seg_end = torch.repeat_interleave(cu_seqlens[1:], seg_lens)
    return global_rcs[:n] - global_rcs[seg_end]


def _frpo_decayed_reverse_cumsum_per_seq(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    gamma: float,
    horizon: int | None = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Stable per-sequence sum of ``gamma**distance * x``.

    Absolute powers underflow on long responses. Processing from the end in
    bounded chunks keeps every local power well scaled while carrying the
    exact discounted suffix value across chunk boundaries.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if horizon is not None and horizon <= 0:
        raise ValueError(f"horizon must be positive or None, got {horizon}")
    if gamma >= 1.0 - 1e-9:
        return _frpo_reverse_cumsum_per_seq(x, cu_seqlens, horizon=horizon)

    def padded(values: torch.Tensor) -> torch.Tensor:
        squeeze = values.dim() == 1
        if squeeze:
            values = values.unsqueeze(0)
        batch_size, seq_len = values.shape
        result = torch.empty_like(values)
        carry = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
        gamma_tensor = values.new_tensor(gamma)
        for end in range(seq_len, 0, -chunk_size):
            start = max(0, end - chunk_size)
            block = values[:, start:end]
            offsets = torch.arange(
                block.shape[1], device=values.device, dtype=values.dtype
            )
            scales = torch.pow(gamma_tensor, offsets)
            local = torch.flip(
                torch.cumsum(torch.flip(block * scales, dims=[1]), dim=1),
                dims=[1],
            ) / scales
            carry_powers = torch.pow(
                gamma_tensor,
                block.shape[1] - offsets,
            )
            local = local + carry.unsqueeze(1) * carry_powers
            result[:, start:end] = local
            carry = local[:, 0]
        if horizon is not None and horizon < seq_len:
            result[:, :-horizon] -= (gamma**horizon) * result[:, horizon:]
        return result.squeeze(0) if squeeze else result

    if x.dim() == 2 or cu_seqlens is None:
        return padded(x)
    if x.dim() != 1:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    segments = list(torch.split(x, lengths))
    padded_x = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)
    padded_result = padded(padded_x)
    return torch.cat(
        [padded_result[index, :length] for index, length in enumerate(lengths)]
    )


def _frpo_decayed_reverse_cumsum_per_valid_segment(
    x: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    gamma: float,
    horizon: int | None = None,
) -> torch.Tensor:
    """Tail sums within contiguous valid-token runs.

    In multi-turn trajectories, each run of ``loss_mask == 1`` is one agent
    response. Invalid observation tokens terminate the response and reset the
    FRPO tail, so credit never crosses into a later agent turn.
    """
    if x.shape != loss_mask.shape:
        raise ValueError(
            "x and loss_mask must have identical shapes, got "
            f"{x.shape} and {loss_mask.shape}"
        )
    if x.dim() not in (1, 2):
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")

    original_shape = x.shape
    if x.dim() == 2:
        row_length = x.shape[1]
        flat_x = x.reshape(-1)
        flat_mask = loss_mask.reshape(-1).bool()
        sequence_start = torch.zeros_like(flat_mask)
        sequence_start[::row_length] = True
    else:
        flat_x = x
        flat_mask = loss_mask.bool()
        sequence_start = torch.zeros_like(flat_mask)
        sequence_start[0] = True
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.long)
            if cu_seqlens[0] != 0 or cu_seqlens[-1] != x.numel():
                raise ValueError("cu_seqlens must cover x exactly")
            sequence_start[cu_seqlens[:-1]] = True

    previous_valid = torch.cat(
        [flat_mask.new_zeros(1), flat_mask[:-1]], dim=0
    )
    run_start = flat_mask & (~previous_valid | sequence_start)
    run_ids = torch.cumsum(run_start.long(), dim=0) - 1
    valid_values = flat_x[flat_mask]
    if valid_values.numel() == 0:
        return torch.zeros_like(x)
    valid_run_ids = run_ids[flat_mask]
    run_lengths = torch.bincount(valid_run_ids)
    run_cu_seqlens = torch.cat(
        [run_lengths.new_zeros(1), torch.cumsum(run_lengths, dim=0)]
    )
    valid_tail = _frpo_decayed_reverse_cumsum_per_seq(
        valid_values,
        run_cu_seqlens,
        gamma=gamma,
        horizon=horizon,
    )
    flat_result = torch.zeros_like(flat_x)
    flat_result[flat_mask] = valid_tail
    return flat_result.reshape(original_shape)


def _frpo_local_boundary_tail_lengths(
    boundary_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    minimum_future_horizon: int,
) -> torch.Tensor:
    """Convert local boundaries into current-inclusive per-token tail lengths."""
    if boundary_mask.dim() != 1:
        raise ValueError(
            f"boundary_mask must be 1D, got shape {boundary_mask.shape}"
        )
    if minimum_future_horizon < 0:
        raise ValueError(
            "minimum_future_horizon must be non-negative, got "
            f"{minimum_future_horizon}"
        )
    n = boundary_mask.numel()
    if n == 0:
        return torch.empty_like(boundary_mask, dtype=torch.long)

    device = boundary_mask.device
    indices = torch.arange(n, device=device)
    if cu_seqlens is None:
        segment_ends = torch.full_like(indices, n)
    else:
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if int(segment_lengths.sum()) != n:
            raise ValueError(
                "cu_seqlens must cover boundary_mask exactly, got "
                f"{int(segment_lengths.sum())} tokens for {n} entries"
            )
        segment_ends = torch.repeat_interleave(cu_seqlens[1:], segment_lengths)

    effective_boundary = boundary_mask.bool().clone()
    effective_boundary[segment_ends - 1] = True
    candidates = torch.where(effective_boundary, indices, n)
    next_boundary = torch.flip(
        torch.cummin(torch.flip(candidates, dims=[0]), dim=0).values,
        dims=[0],
    )
    remaining_future = segment_ends - 1 - indices
    boundary_distance = next_boundary - indices
    future_horizon = torch.maximum(
        boundary_distance,
        torch.minimum(
            torch.full_like(indices, minimum_future_horizon),
            remaining_future,
        ),
    )
    return future_horizon + 1


def _frpo_variable_decayed_tail_sum(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    gamma: float,
    tail_lengths: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence discounted tail sum with a length for every packed token."""
    if x.dim() != 1 or tail_lengths.shape != x.shape:
        raise ValueError(
            "x and tail_lengths must be identically shaped 1D tensors, got "
            f"{x.shape} and {tail_lengths.shape}"
        )
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    tail_lengths = tail_lengths.to(device=x.device, dtype=torch.long)
    if torch.any(tail_lengths <= 0):
        raise ValueError("tail_lengths must be positive")

    n = x.numel()
    indices = torch.arange(n, device=x.device)
    if cu_seqlens is None:
        segment_ends = torch.full_like(indices, n)
    else:
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.long)
        segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        segment_ends = torch.repeat_interleave(cu_seqlens[1:], segment_lengths)
    ends = torch.minimum(indices + tail_lengths, segment_ends)

    full_tail = _frpo_decayed_reverse_cumsum_per_seq(x, cu_seqlens, gamma)
    padded_tail = torch.cat([full_tail, x.new_zeros(1)])
    continuation = padded_tail[ends]
    continuation = torch.where(ends < segment_ends, continuation, 0.0)
    discount = torch.pow(x.new_tensor(gamma), (ends - indices).to(x.dtype))
    return full_tail - discount * continuation


def _frpo_linear_decay_tail_sum(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    full_weight_horizon: int,
) -> torch.Tensor:
    """Tail sum with unit weight through H, then linear decay to zero.

    For a token with ``R`` remaining positions, distances ``0..H-1`` receive
    weight one.  If ``R > H``, distances ``H..R-1`` receive weights that
    decrease linearly from below one to exactly zero at the final position.
    The implementation uses two reverse cumulative sums and is linear in the
    token count rather than materializing a quadratic token-pair matrix.
    """
    if full_weight_horizon <= 0:
        raise ValueError(
            "full_weight_horizon must be positive, got "
            f"{full_weight_horizon}"
        )
    if x.dim() not in (1, 2):
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")

    original_shape = x.shape
    if x.dim() == 2:
        batch_size, seq_len = x.shape
        flat_x = x.reshape(-1)
        segment_lengths = torch.full(
            (batch_size,), seq_len, device=x.device, dtype=torch.long
        )
        flat_cu_seqlens = torch.cat(
            [segment_lengths.new_zeros(1), torch.cumsum(segment_lengths, dim=0)]
        )
    else:
        flat_x = x
        if cu_seqlens is None:
            flat_cu_seqlens = torch.tensor(
                [0, x.numel()], device=x.device, dtype=torch.long
            )
        else:
            flat_cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.long)
            if (
                flat_cu_seqlens.numel() < 2
                or flat_cu_seqlens[0] != 0
                or flat_cu_seqlens[-1] != x.numel()
            ):
                raise ValueError("cu_seqlens must cover x exactly")

    n = flat_x.numel()
    if n == 0:
        return torch.zeros_like(x)
    segment_lengths = flat_cu_seqlens[1:] - flat_cu_seqlens[:-1]
    if torch.any(segment_lengths < 0):
        raise ValueError("cu_seqlens must be nondecreasing")
    segment_ends = torch.repeat_interleave(
        flat_cu_seqlens[1:], segment_lengths
    )
    indices = torch.arange(n, device=x.device, dtype=torch.long)
    beyond_starts = torch.minimum(indices + full_weight_horizon, segment_ends)

    full_tail = _frpo_reverse_cumsum_per_seq(flat_x, flat_cu_seqlens)
    index_weighted_tail = _frpo_reverse_cumsum_per_seq(
        flat_x * indices.to(flat_x.dtype), flat_cu_seqlens
    )
    padded_full_tail = torch.cat([full_tail, flat_x.new_zeros(1)])
    padded_index_tail = torch.cat([index_weighted_tail, flat_x.new_zeros(1)])
    beyond_sum = padded_full_tail[beyond_starts]
    beyond_index_sum = padded_index_tail[beyond_starts]
    # A packed sequence boundary can point at the first token of the next
    # sequence.  Such a continuation is empty, not the next sequence's tail.
    has_beyond_start = beyond_starts < segment_ends
    beyond_sum = torch.where(has_beyond_start, beyond_sum, 0.0)
    beyond_index_sum = torch.where(has_beyond_start, beyond_index_sum, 0.0)

    unit_weight_prefix = full_tail - beyond_sum
    final_indices = segment_ends - 1
    decay_denominator = (segment_ends - indices - full_weight_horizon).clamp(min=1)
    decayed_suffix = (
        final_indices.to(flat_x.dtype) * beyond_sum - beyond_index_sum
    ) / decay_denominator.to(flat_x.dtype)
    has_suffix = segment_ends - indices > full_weight_horizon
    result = unit_weight_prefix + torch.where(
        has_suffix, decayed_suffix, torch.zeros_like(decayed_suffix)
    )
    return result.reshape(original_shape)


def _fipo_decayed_reverse_cumsum_per_seq(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    """Per-sequence decayed tail-sum: out[t] = sum_{k>=t in same seq} gamma**(k-t) * x[k].

    FIPO §4.2.1 "soft decay window" with horizon tau = -1 / log2(gamma).
    For gamma=1.0 this reduces to _frpo_reverse_cumsum_per_seq.

    Vectorised via the scale-then-cumsum identity: let r_k = k - seg_start(k),
    g_k = gamma**r_k. Then with y_k = g_k * x_k and Y[t] = sum_{k>=t in same seg} y_k
    (an ordinary reverse cumsum), out[t] = Y[t] / g_t. No Python loops.

    Numerical caveat: for gamma=0.99 and T=4096, gamma**T ~ 1e-18 which still
    fits float32. For very small gamma + long sequences, scale_min may underflow;
    we clamp to 1e-30 to keep the division finite.
    """
    if gamma >= 1.0 - 1e-9:
        return _frpo_reverse_cumsum_per_seq(x, cu_seqlens)

    if x.dim() == 2:
        bs, T = x.shape
        r = torch.arange(T, device=x.device, dtype=x.dtype)
        g = torch.pow(x.new_tensor(gamma), r)  # [T]
        y = x * g.unsqueeze(0)  # [bs, T]
        Y = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1), dims=[1])
        return Y / g.unsqueeze(0).clamp(min=1e-30)

    n = x.shape[0]
    if cu_seqlens is None:
        r = torch.arange(n, device=x.device, dtype=x.dtype)
        g = torch.pow(x.new_tensor(gamma), r)
        y = x * g
        Y = torch.flip(torch.cumsum(torch.flip(y, dims=[0]), dim=0), dims=[0])
        return Y / g.clamp(min=1e-30)

    # Packed: within-segment position r_k = k - seg_start(k).
    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seg_id = torch.repeat_interleave(
        torch.arange(seg_lens.numel(), device=x.device), seg_lens
    )
    seg_start_per_token = cu_seqlens[:-1][seg_id]
    abs_idx = torch.arange(n, device=x.device)
    r = (abs_idx - seg_start_per_token).to(x.dtype)
    g = torch.pow(x.new_tensor(gamma), r)
    y = x * g
    # Use the existing per-segment reverse cumsum on y, then divide by g.
    Y = _frpo_reverse_cumsum_per_seq(y, cu_seqlens)
    return Y / g.clamp(min=1e-30)


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
    use_frpo_loss_term: bool = False,
    frpo_clip_eps: float | None = 0.2,
    frpo_beta: float = 0.0,
    frpo_normalize_tail: bool = False,
    frpo_tail_horizon: int = 0,
    frpo_reset_tail_on_loss_mask_gap: bool = False,
    frpo_tail_lengths: torch.Tensor | None = None,
    frpo_semantic_gate_mask: torch.Tensor | None = None,
    frpo_tail_gamma: float = 1.0,
    frpo_tail_linear_decay_start: int = 0,
    frpo_use_tail_correction: bool = True,
    frpo_use_local_correction: bool = False,
    # FRPO anchor: "prox" uses log-probabilities recomputed once before the outer
    # update and held fixed across its minibatches. "ref" uses frozen SFT/base
    # log-probabilities (delta = log pi_theta / pi_ref).
    frpo_anchor: str = "prox",
    ref_logprobs: torch.Tensor | None = None,
    # FIPO (Future-KL Induced Policy Optimization, arXiv:2603.19835): a contemporaneous
    # framework that uses the SAME per-token primitive Δlog p as FRPO but couples it
    # MULTIPLICATIVELY to the advantage and applies a soft-decay window + dual-clip
    # mask. The two are mutually exclusive: setting use_fipo=True disables the FRPO
    # subtractive coupling above. Defaults below are the paper's near-defaults adjusted
    # for 1.7B + 8K-tok max_new (smaller scale than the paper's 32B + 8K).
    use_fipo: bool = False,
    fipo_beta: float = 1.0,
    fipo_gamma: float = 0.99,         # soft-decay; effective horizon = -1/log2(0.99) ~ 69 tokens
    fipo_dual_clip_c: float = 10.0,   # zero out summand whose ratio > c (FIPO Eq. 5)
    fipo_eps_low: float = 0.2,        # f_t = clip(exp(...), 1-eps_low, 1+eps_high)
    fipo_eps_high: float = 0.2,
    # CISPO: clip the behavior-policy importance ratio only in the detached
    # score-function coefficient. Unlike PPO, tokens outside the interval keep
    # a non-zero policy gradient.
    use_cispo: bool = False,
    cispo_eps_low: float | None = None,
    cispo_eps_high: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """PPO actor loss function with optional rejection sampling.

    The ``rejection_sampling`` parameter replaces the removed
    ``behave_imp_weight_cap`` / ``behave_imp_weight_mode``.

    - ``action='mask'``: modifies loss_mask before loss computation (rejection)
    - ``action='clamp'``: clamps importance weight to bounds (truncation)

    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.

    The legacy ``token`` and ``sequence`` modes aggregate π_θ/π_proximal
    and optionally multiply a separate behavior weight. The Proximal KL controls
    ``sequence_token`` and ``none`` instead use a detached direct π_θ/π_behave
    response weight or unit weight, respectively, in an unclipped score loss.
    They preserve per-token corrected advantages and never decouple the ratio.

    Args:
        logprobs: Current policy log-probabilities (π_θ).
        proximal_logprobs: Proximal policy log-probabilities (π_proximal).
        old_logprobs: Behavior policy log-probabilities from inference (π_behave).
        advantages: Per-token advantage estimates.
        eps_clip: PPO clipping factor for policy ratio.
        loss_mask: Mask for valid tokens (1 = valid).
        eps_clip_higher: Upper clipping factor (decoupled clipping). None = use eps_clip.
        c_clip: Dual clipping factor, must be > 1.0. None disables dual clipping.
        rejection_sampling: Rejection sampling configuration. None disables filtering.
        importance_sampling_level: Level at which to compute importance sampling ratios.
            - 'token': Per-token ratios (standard PPO)
            - 'sequence': Legacy sequence ratios and sequence-averaged advantages
            - 'sequence_token': Direct θ/behavior geometric-mean weight, token credit
            - 'none': Unit weight, token credit (Proximal KL without IS)
        cu_seqlens: Cumulative sequence lengths for packed sequences (1D tensors).
            Required for packed 'sequence' and 'sequence_token' inputs.
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            Not needed for 2D padded inputs (sequences identified by batch dimension).
    """
    # Save original count BEFORE rejection sampling may modify loss_mask.
    # This keeps the denominator consistent with loss_weight_fn in actor.py,
    # which always uses the original loss_mask from input_data. Without this,
    # mask mode would inflate per-token gradients by N_original / N_kept.
    loss_mask_count = loss_mask.count_nonzero() or 1

    direct_frpo_weight = importance_sampling_level in ("sequence_token", "none")
    if direct_frpo_weight:
        if not use_frpo_loss_term or use_fipo or use_cispo:
            raise ValueError(
                "Direct Proximal KL weighting requires FRPO without FIPO/CISPO."
            )
        if rejection_sampling is not None:
            raise ValueError(
                "Direct Proximal KL weighting requires rejection_sampling=None; "
                "no separate proximal-to-behavior weight is applied."
            )
        if c_clip is not None:
            raise ValueError("Direct Proximal KL weighting does not use dual clipping.")

    if use_cispo:
        if use_frpo_loss_term or use_fipo:
            raise ValueError("CISPO is mutually exclusive with FRPO and FIPO.")
        if c_clip is not None:
            raise ValueError("CISPO does not use PPO dual clipping (c_clip).")
        if rejection_sampling is not None:
            raise ValueError(
                "CISPO already uses pi_theta/pi_behave; rejection sampling would "
                "apply a second behavior-policy correction."
            )
        if importance_sampling_level != "token":
            raise ValueError("CISPO requires token-level importance ratios.")
        if cispo_eps_high < 0 or (
            cispo_eps_low is not None and cispo_eps_low < 0
        ):
            raise ValueError("CISPO clipping epsilons must be non-negative.")

    # === Apply rejection sampling (replaces old compute_behave_imp_weight) ===
    if rejection_sampling is not None:
        rs_result = apply_rejection_sampling(
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            config=rejection_sampling,
        )
        # mask mode updates loss_mask; clamp mode keeps it unchanged
        loss_mask = rs_result.loss_mask
        behave_imp_weight = rs_result.behave_imp_weight
        filtered_fraction = rs_result.filtered_fraction
    else:
        filtered_fraction = 0.0

    raw_advantages = advantages.detach()

    # FRPO advantage modification (loss-side, per PPO inner step):
    #   A^FRPO_t = A_t - frpo_beta * sum_{t' >= t in same seq}
    #                clip(log(pi_theta.detach() / pi_prox), -frpo_clip_eps, +frpo_clip_eps) * loss_mask_{t'}
    # When frpo_normalize_tail=True the tail-sum is additionally divided by the FULL
    # valid-token count of the token's trajectory (T_i), not by the remaining count
    # (T_i - t). Trajectory-level normalisation removes systematic length-dependence
    # between trajectories while preserving the per-position structure of the
    # tail-sum within a trajectory.
    # Detached log-ratio so the FRPO term contributes only as a scalar shift on
    # the advantage; gradient still flows through the ratio in the pg loss.
    frpo_enabled = (
        use_frpo_loss_term
        and frpo_beta != 0
        and (frpo_clip_eps is None or frpo_clip_eps > 0)
    )
    if frpo_enabled:
        with torch.no_grad():
            # Pick FRPO anchor: pi_prox (AReaL default, recomputed per outer step) or
            # pi_ref (frozen, paper-faithful). If "ref" is requested but ref_logprobs
            # wasn't supplied, fall back to prox with a no-op so training doesn't
            # silently train against a zero tensor.
            if frpo_anchor == "ref":
                if ref_logprobs is None:
                    raise ValueError(
                        "frpo_anchor='ref' requires ref_logprobs to be passed in. "
                        "Make sure the yaml has a `ref:` engine configured so the "
                        "trainer computes ref_logp each rollout."
                    )
                anchor_logp = ref_logprobs.detach().float()
            else:
                anchor_logp = proximal_logprobs.detach().float()
            raw_frpo_logratio = logprobs.detach().float() - anchor_logp
            if frpo_clip_eps is None:
                frpo_clip_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
            else:
                frpo_clip_mask = raw_frpo_logratio.abs() > frpo_clip_eps
            frpo_clip_mask.logical_and_(loss_mask)
            clipped_logratio = raw_frpo_logratio
            if frpo_clip_eps is not None:
                clipped_logratio = clipped_logratio.clamp(
                    -frpo_clip_eps, frpo_clip_eps
                )
            clipped_logratio = clipped_logratio * loss_mask.float()
            if frpo_tail_linear_decay_start and (
                frpo_tail_horizon
                or frpo_tail_lengths is not None
                or frpo_reset_tail_on_loss_mask_gap
                or abs(frpo_tail_gamma - 1.0) > 1.0e-9
            ):
                raise ValueError(
                    "frpo_tail_linear_decay_start requires a full tail, gamma=1, "
                    "fixed sequence boundaries, and no token-specific horizon"
                )
            if frpo_tail_lengths is None:
                if frpo_tail_linear_decay_start:
                    tail = _frpo_linear_decay_tail_sum(
                        clipped_logratio,
                        cu_seqlens,
                        full_weight_horizon=frpo_tail_linear_decay_start,
                    )
                elif frpo_reset_tail_on_loss_mask_gap:
                    tail = _frpo_decayed_reverse_cumsum_per_valid_segment(
                        clipped_logratio,
                        loss_mask,
                        cu_seqlens,
                        gamma=frpo_tail_gamma,
                        horizon=frpo_tail_horizon or None,
                    )
                else:
                    tail = _frpo_decayed_reverse_cumsum_per_seq(
                        clipped_logratio,
                        cu_seqlens,
                        gamma=frpo_tail_gamma,
                        horizon=frpo_tail_horizon or None,
                    )
            else:
                tail = _frpo_variable_decayed_tail_sum(
                    clipped_logratio,
                    cu_seqlens,
                    gamma=frpo_tail_gamma,
                    tail_lengths=frpo_tail_lengths,
                )
            if frpo_semantic_gate_mask is not None:
                if frpo_semantic_gate_mask.shape != tail.shape:
                    raise ValueError(
                        "frpo_semantic_gate_mask and tail must have identical shapes, "
                        f"got {frpo_semantic_gate_mask.shape} and {tail.shape}"
                    )
                tail = tail * frpo_semantic_gate_mask.to(tail.dtype)
            if frpo_normalize_tail:
                # Per-token denominator = total valid tokens in THIS token's trajectory
                # (constant across tokens within a segment). Broadcast a per-segment
                # mask-sum back to each token.
                if cu_seqlens is None:
                    if loss_mask.dim() == 2:
                        traj_total = loss_mask.float().sum(dim=-1, keepdim=True)
                    else:
                        traj_total = loss_mask.float().sum()
                    traj_total = traj_total.clamp(min=1.0)
                    tail = tail / traj_total
                else:
                    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
                    n_segs = seg_lens.numel()
                    seg_id = torch.repeat_interleave(
                        torch.arange(n_segs, device=loss_mask.device), seg_lens
                    )
                    seg_totals = torch.zeros(
                        n_segs, device=loss_mask.device, dtype=torch.float32
                    )
                    seg_totals.scatter_add_(0, seg_id, loss_mask.float())
                    traj_total_per_token = seg_totals[seg_id].clamp(min=1.0)
                    tail = tail / traj_total_per_token
            if frpo_use_tail_correction:
                frpo_correction = (frpo_beta * tail).to(advantages.dtype)
            else:
                frpo_correction = torch.zeros_like(advantages)
            # Optional local term β·f'(ρ_i)·ρ_i from the w_i derivation. For
            # f(ρ) = -log(ρ), it is the action-independent constant -β. Its
            # conditional score-function expectation is zero, so retaining it does
            # not change the ideal expected gradient; it only changes finite-sample
            # variance. Keep the switch explicit so the two estimators can be
            # compared without changing the future-tail correction.
            if frpo_use_local_correction:
                frpo_local_correction = (
                    frpo_beta * loss_mask.to(advantages.dtype)
                )
            else:
                frpo_local_correction = torch.zeros_like(advantages)
            advantages = advantages - frpo_correction - frpo_local_correction

    # FIPO advantage modification (mutually exclusive with FRPO above):
    #   Delta_logp_t = log pi_theta - log pi_old  (using pi_behave as the
    #     rollout-time anchor, NOT the per-inner-step pi_prox; FIPO's anchor
    #     is fixed for the whole outer step which is critical to its design.)
    #   FutureKL_t   = sum_{k>=t in same seg} M_k * gamma^(k-t) * Delta_logp_k
    #     where M_k = indicator(pi_theta(o_k)/pi_old(o_k) <= dual_clip_c)
    #     zeros out outlier tokens (FIPO Eq. 5).
    #   f_t          = clip(exp(beta * FutureKL_t), 1-eps_low, 1+eps_high)
    #     multiplied INTO the advantage (FIPO Eq. 7).
    if use_fipo and fipo_beta != 0:
        with torch.no_grad():
            # FIPO uses the rollout-time behavior policy as its old-policy anchor.
            # In asynchronous training it can differ from the newer pi_prox, even
            # though both tensors stay fixed throughout this outer update.
            delta_logp = (logprobs.detach().float() - old_logprobs.detach().float())
            # Dual-clip mask: importance ratio > c => set summand to 0.
            ratio_for_mask = torch.exp(delta_logp)
            outlier_mask = (ratio_for_mask <= fipo_dual_clip_c).float()
            masked = delta_logp * outlier_mask * loss_mask.float()
            fipo_tail = _fipo_decayed_reverse_cumsum_per_seq(
                masked, cu_seqlens, fipo_gamma
            )
            f_t_raw = torch.exp(fipo_beta * fipo_tail)
            f_t = f_t_raw.clamp(1.0 - fipo_eps_low, 1.0 + fipo_eps_high)
            f_t = f_t.to(advantages.dtype)
            advantages = advantages * f_t

    if direct_frpo_weight:
        with torch.no_grad():
            if importance_sampling_level == "sequence_token":
                # Direct current-to-behavior ratio: pi_prox is ONLY the KL anchor.
                ratio = _compute_sequence_token_weight(
                    logprobs.detach().float() - old_logprobs.detach().float(),
                    loss_mask,
                    cu_seqlens,
                )
            else:
                # Do not retain a theta/prox ratio when removing IS.
                ratio = loss_mask.to(dtype=logprobs.dtype)
    elif importance_sampling_level == "sequence":
        # GSPO: Compute sequence-level geometric mean of probability ratios
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        # Standard PPO: per-token ratio
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token', 'sequence', 'sequence_token', or 'none'."
        )

    with torch.no_grad():
        effective_advantages = advantages.detach()
        raw_advantage_f = raw_advantages.float()
        effective_advantage_f = effective_advantages.float()
        valid_mask_f = loss_mask.float()
        theta_prox_logratio = (
            logprobs.detach().float() - proximal_logprobs.detach().float()
        )
        theta_prox_logratio_c20 = theta_prox_logratio.clamp(-20.0, 20.0)
        theta_prox_k3_sample = (
            torch.expm1(theta_prox_logratio_c20) - theta_prox_logratio_c20
        )
        raw_positive = (raw_advantage_f > 0).logical_and(loss_mask)
        effective_positive = (effective_advantage_f > 0).logical_and(loss_mask)
        reinforce_weight = effective_advantage_f * ratio.detach().float()
        raw_reinforce_weight = raw_advantage_f * ratio.detach().float()
        reinforce_delta = reinforce_weight - raw_reinforce_weight
        behavior_correction = (
            behave_imp_weight.detach().float()
            if rejection_sampling is not None
            else torch.ones_like(reinforce_weight)
        )
        theta_prox_k3_prox_weighted = theta_prox_k3_sample * behavior_correction
        corrected_reinforce_weight = reinforce_weight * behavior_correction
        corrected_reinforce_delta = reinforce_delta * behavior_correction

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_clip,
        1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
    )

    def clipped_surrogate(
        branch_advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pg_loss1 = -branch_advantages * ratio
        pg_loss2 = -branch_advantages * clipped_ratio
        branch_clip_mask = pg_loss1.detach() < pg_loss2.detach()
        branch_loss = torch.max(pg_loss1, pg_loss2)
        if c_clip is not None:
            assert c_clip > 1.0, c_clip
            pg_loss3 = torch.sign(branch_advantages) * c_clip * branch_advantages
            branch_dual_clip_mask = pg_loss3.detach() < branch_loss.detach()
            branch_loss = torch.min(branch_loss, pg_loss3)
        else:
            branch_dual_clip_mask = torch.zeros_like(branch_clip_mask)
        return branch_loss, branch_clip_mask, branch_dual_clip_mask

    if direct_frpo_weight:
        # The shared sequence weight must NOT backpropagate across tokens:
        # dL/dlogp_i = -weight_i * (A_i - C_i) / N_tokens. In particular,
        # opposite token credits must not cancel through a sequence mean.
        pg_loss = -ratio.detach() * advantages.detach() * logprobs
        clip_mask = torch.zeros_like(loss_mask)
        dual_clip_mask = torch.zeros_like(loss_mask)
    elif use_cispo:
        # CISPO (MiniMax-M1, Eq. 10): the clipped importance ratio is a
        # stop-gradient coefficient multiplying A_t log pi_theta(a_t|s_t).
        # The denominator is the rollout-time behavior policy, which matters in
        # asynchronous training where pi_behave can differ from pi_prox.
        cispo_logratio = logprobs - old_logprobs
        cispo_ratio = torch.exp(cispo_logratio)
        cispo_weight = cispo_ratio.detach().clamp(max=1.0 + cispo_eps_high)
        if cispo_eps_low is not None:
            cispo_weight = cispo_weight.clamp(min=1.0 - cispo_eps_low)
        pg_loss = -cispo_weight * advantages.detach() * logprobs
        clip_mask = cispo_weight.ne(cispo_ratio.detach())
        dual_clip_mask = torch.zeros_like(clip_mask)
    else:
        pg_loss, clip_mask, dual_clip_mask = clipped_surrogate(advantages)

    # Apply behavioural importance weight from rejection sampling
    if rejection_sampling is not None:
        behave_approx_kl = proximal_logprobs.detach() - old_logprobs.detach()
        behave_mask = (behave_imp_weight > 0).logical_and(loss_mask.bool())
        behave_approx_kl = torch.where(behave_mask, behave_approx_kl, 0.0)
        pg_loss = pg_loss * behave_imp_weight

    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    outer_clip_mask_f = clip_mask.float()
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        theta_prox_abs_logratio=theta_prox_logratio.abs(),
        theta_prox_logratio_sq=theta_prox_logratio.square(),
        theta_prox_k3_sample=theta_prox_k3_sample,
        theta_prox_k3_prox_weighted=theta_prox_k3_prox_weighted,
        theta_prox_abs_logratio_gt_005=(theta_prox_logratio.abs() > 0.05).float(),
        theta_prox_abs_logratio_gt_02=(theta_prox_logratio.abs() > 0.2).float(),
        theta_prox_abs_logratio_gt_1=(theta_prox_logratio.abs() > 1.0).float(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
        pg_raw_advantage=raw_advantage_f,
        pg_raw_advantage_sq=raw_advantage_f.square(),
        pg_effective_advantage=effective_advantage_f,
        pg_effective_advantage_sq=effective_advantage_f.square(),
        pg_advantage_delta=effective_advantage_f - raw_advantage_f,
        pg_raw_positive_fraction=raw_positive.float(),
        pg_effective_positive_fraction=effective_positive.float(),
        pg_pos_to_nonpos_flip=raw_positive.logical_and(
            effective_advantage_f <= 0
        ).float(),
        pg_neg_to_pos_flip=(
            (raw_advantage_f <= 0).logical_and(effective_positive)
        ).float(),
        pg_raw_positive_mass=torch.relu(raw_advantage_f) * valid_mask_f,
        pg_effective_positive_mass=torch.relu(effective_advantage_f) * valid_mask_f,
        pg_reinforce_weight=reinforce_weight,
        pg_reinforce_weight_sq=reinforce_weight.square(),
        pg_raw_reinforce_weight=raw_reinforce_weight,
        pg_raw_reinforce_weight_sq=raw_reinforce_weight.square(),
        pg_reinforce_delta=reinforce_delta,
        pg_reinforce_delta_sq=reinforce_delta.square(),
        pg_outer_clip_suppressed_abs_mass=(
            raw_reinforce_weight.abs() * outer_clip_mask_f
        ),
        pg_outer_clip_suppressed_signed_mass=(
            raw_reinforce_weight * outer_clip_mask_f
        ),
        pg_outer_clip_suppressed_effective_abs_mass=(
            corrected_reinforce_weight.abs() * outer_clip_mask_f
        ),
        pg_outer_clip_suppressed_effective_signed_mass=(
            corrected_reinforce_weight * outer_clip_mask_f
        ),
        pg_outer_clip_suppressed_delta_abs_mass=(
            corrected_reinforce_delta.abs() * outer_clip_mask_f
        ),
        pg_outer_clip_suppressed_delta_signed_mass=(
            corrected_reinforce_delta * outer_clip_mask_f
        ),
        pg_reinforce_positive_mass=torch.relu(reinforce_weight) * valid_mask_f,
        pg_reinforce_negative_mass=torch.relu(-reinforce_weight) * valid_mask_f,
    )
    if frpo_enabled:
        frpo_total_correction = frpo_correction + frpo_local_correction
        stat.update(
            frpo_clipped_logratio=clipped_logratio.detach(),
            frpo_clip_mask=frpo_clip_mask.detach(),
            frpo_tail=tail.detach(),
            frpo_tail_sq=tail.detach().float().square(),
            frpo_correction=frpo_correction.detach(),
            frpo_correction_sq=frpo_correction.detach().float().square(),
            frpo_local_correction=frpo_local_correction.detach(),
            frpo_total_correction=frpo_total_correction.detach(),
            frpo_total_correction_sq=(
                frpo_total_correction.detach().float().square()
            ),
            frpo_correction_over_abs_adv=(
                frpo_total_correction.detach().float().abs()
                / raw_advantage_f.abs().clamp(min=1e-3)
            ),
            frpo_tail_positive_fraction=(tail.detach().float() > 0).float(),
            frpo_semantic_gate_mask=(
                frpo_semantic_gate_mask.detach().float()
                if frpo_semantic_gate_mask is not None
                else torch.ones_like(tail.detach(), dtype=torch.float32)
            ),
        )
        if frpo_tail_lengths is not None:
            stat["frpo_tail_length"] = frpo_tail_lengths.detach().float()
    if use_fipo and fipo_beta != 0:
        stat.update(
            fipo_delta_logp=delta_logp.detach(),
            fipo_outlier_mask=outlier_mask.detach(),
            fipo_tail=fipo_tail.detach(),
            fipo_f_t=f_t.detach(),
        )
    if use_cispo:
        stat.update(
            cispo_logratio=cispo_logratio.detach(),
            cispo_ratio=cispo_ratio.detach(),
            cispo_weight=cispo_weight.detach(),
            cispo_clip_mask=clip_mask.detach(),
        )
    if rejection_sampling is not None:
        behave_logratio = behave_approx_kl.detach().float()
        behave_logratio_c20 = behave_logratio.clamp(-20.0, 20.0)
        stat.update(
            behave_approx_kl=behave_approx_kl.detach(),
            behave_prox_abs_logratio=behave_logratio.abs(),
            behave_prox_logratio_sq=behave_logratio.square(),
            behave_prox_k3_sample=(
                torch.expm1(behave_logratio_c20) - behave_logratio_c20
            ),
            behave_imp_weight=behave_imp_weight.detach(),
            behave_mask=behave_mask,
            filtered_fraction=filtered_fraction,
        )
    return pg_loss, stat


def sapo_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    loss_mask: torch.Tensor,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """SAPO (Soft Adaptive Policy Optimization) loss with asymmetric sigmoid gates.

    SAPO replaces PPO clipping with soft sigmoid gates, providing smooth gradients.
    Note: SAPO requires use_decoupled_loss=False.

    Args:
        logprobs: Current policy log probabilities
        old_logprobs: Old policy log probabilities
        advantages: Advantage values
        tau_pos: Temperature for positive advantages (higher = sharper gate)
        tau_neg: Temperature for negative advantages (higher = sharper gate)
        loss_mask: Mask for valid tokens
        importance_sampling_level: "token" or "sequence" level importance sampling
        cu_seqlens: Cumulative sequence lengths for sequence-level IS

    Returns:
        Tuple of (loss, statistics dict compatible with PPO)
    """
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("SAPO temperatures (tau_pos, tau_neg) must be positive.")
    loss_mask_count = loss_mask.count_nonzero() or 1
    advantages = advantages.detach()
    log_ratio = logprobs - old_logprobs

    if importance_sampling_level == "sequence":
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        ratio = torch.exp(log_ratio)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    # SAPO: Asymmetric sigmoid gates with 4/τ gradient normalization
    gate_pos = torch.sigmoid(tau_pos * (ratio - 1.0))
    gate_neg = torch.sigmoid(tau_neg * (ratio - 1.0))
    scale_pos = 4.0 / tau_pos
    scale_neg = 4.0 / tau_neg
    scaled_gate_pos = gate_pos * scale_pos
    scaled_gate_neg = gate_neg * scale_neg

    # Select gate based on advantage sign
    is_positive = advantages > 0
    soft_gate = torch.where(is_positive, scaled_gate_pos, scaled_gate_neg)

    # Compute loss
    pg_loss = -soft_gate * advantages
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count

    # Return stat dict compatible with PPO (fake clip_mask for logging compatibility)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),  # SAPO doesn't clip
        dual_clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),
        # SAPO-specific stats (scaled gates for consistency)
        sapo_soft_gate=soft_gate.detach(),
        sapo_scaled_gate_pos=scaled_gate_pos.detach(),
        sapo_scaled_gate_neg=scaled_gate_neg.detach(),
    )

    return pg_loss, stat


def dpo_pair_logratios(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate per-sequence masked logprobs over a packed batch and pair them up.

    Sequences are packed and interleaved as ``[chosen_0, rejected_0, chosen_1,
    rejected_1, ...]``. ``loss_mask`` is shifted by one to align with
    next-token logprobs, and each sequence's final position is zeroed (no next
    token exists for it). Aggregation uses ``fp64`` scatter-add to avoid
    precision loss on long (~2k+ tok) pairs where ``fp32`` accumulation can
    flip the log-ratio sign.

    Returns:
        ``(policy_logps, ref_logps, completion_lens)`` each of shape ``(K, 2)``
        where ``K = valid_pairs.sum()``, column 0 is chosen, column 1 is
        rejected. ``completion_lens`` counts the shifted-mask response tokens
        per sequence (needed by IPO for per-token normalization).
    """
    device = logprobs.device

    shifted_mask = torch.zeros_like(loss_mask)
    if loss_mask.shape[0] > 1:
        shifted_mask[:-1] = loss_mask[1:]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    shifted_mask.index_fill_(0, (cu_seqlens[1:] - 1)[seqlens > 0], False)

    n_seqs = seqlens.shape[0]
    seq_ids = torch.repeat_interleave(torch.arange(n_seqs, device=device), seqlens)
    masked = shifted_mask.to(dtype=logprobs.dtype)
    policy_logps = torch.zeros(n_seqs, dtype=torch.float64, device=device)
    ref_logps = torch.zeros(n_seqs, dtype=torch.float64, device=device)
    policy_logps.index_add_(0, seq_ids, (logprobs * masked).double())
    ref_logps.index_add_(0, seq_ids, (ref_logprobs * masked).double())

    completion_lens = torch.zeros(n_seqs, dtype=torch.float64, device=device)
    completion_lens.index_add_(0, seq_ids, shifted_mask.double())

    return (
        policy_logps.view(-1, 2)[valid_pairs],
        ref_logps.view(-1, 2)[valid_pairs],
        completion_lens.view(-1, 2)[valid_pairs],
    )


def dpo_preference_loss(
    logits: torch.Tensor, *, beta: float, loss_type: str = "sigmoid"
) -> torch.Tensor:
    """Per-pair preference loss from DPO log-ratio logits.

    For ``"sigmoid"``, ``logits`` is the un-normalized pair delta.
    For ``"ipo"``, ``logits`` must be **per-token averaged** (length-normalized)
    before being passed here, matching trl's confirmed-with-authors convention.
    """
    if loss_type == "sigmoid":
        return -torch.nn.functional.logsigmoid(beta * logits.float())
    if loss_type == "ipo":
        return (logits.float() - 1.0 / (2.0 * beta)) ** 2
    raise ValueError(f"Unsupported DPO loss_type: {loss_type!r}")


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float):
    diff = torch.abs(x - y)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))


def _mse_loss(x: torch.Tensor, y: torch.Tensor):
    return 0.5 * (x - y) ** 2


def ppo_critic_loss_fn(
    value: torch.FloatTensor,
    old_value: torch.FloatTensor,
    target_value: torch.FloatTensor,
    value_eps_clip: float,
    loss_mask: torch.Tensor | None = None,
    loss_fn_type: str = "mse",
) -> tuple[torch.Tensor, dict]:
    """Compute PPO critic loss function given padded batch inputs.

    There is no shape requirements for the inputs, but they must have the same shape.
    Either [bs, max_seqlen] for batch padded inputs or [tot_seqlen] for padded inputs.

    Args:
        value (torch.FloatTensor): Values. The position of the final token is not included.
            (The whole generated sequence is not a state.)
        old_value (torch.FloatTensor): Old values.
        target_value (torch.FloatTensor): Returns computed by GAE.
        value_eps_clip (float): Clip ratio.
        loss_mask (Optional[torch.Tensor], optional): Mask for loss computation.
            1 if valid else 0. Defaults to None.
        loss_fn_type (str, optional): Type of loss function. Defaults to 'mse'.

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """
    assert value.dtype == torch.float32
    assert old_value.dtype == torch.float32
    assert target_value.dtype == torch.float32

    if loss_fn_type == "huber":
        loss_fn = functools.partial(_huber_loss, delta=10.0)
    elif loss_fn_type == "mse":
        loss_fn = _mse_loss
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    if target_value.is_inference():
        target_value = target_value.clone()  # clone a inference tensor

    value_loss_original = loss_fn(value, target_value)

    value_clipped = old_value + (value - old_value).clamp(
        -value_eps_clip, value_eps_clip
    )

    value_loss_clipped = loss_fn(value_clipped, target_value)

    value_loss = torch.max(value_loss_original, value_loss_clipped)

    with torch.no_grad():
        clip_mask = value_loss_clipped.detach() > value_loss_original.detach()
        if loss_mask is not None:
            clip_mask.logical_and_(loss_mask)

        stat = dict(clip_mask=clip_mask, loss=value_loss.detach())

    if loss_mask is not None:
        value_loss = (
            torch.where(loss_mask, value_loss, 0).sum() / loss_mask.count_nonzero()
        )
    else:
        value_loss = value_loss.mean()

    return value_loss, stat


# code modified from VERL: https://github.com/volcengine/src/verl/blob/main/src/verl/workers/reward_manager/dapo.py
def reward_overlong_penalty(
    data: dict[str, Any],
    overlong_tokens: int,
    overlong_penalty_factor: float,
    max_response_length: int,
) -> dict[str, Any]:
    reward_score = data["rewards"]
    input_ids = data["input_ids"]
    response_lengths = (data["loss_mask"].sum(dim=-1)).long()
    batch_size = input_ids.shape[0]
    for sample_idx in range(batch_size):
        reward_score_cur = reward_score[sample_idx]
        response_length_cur = response_lengths[sample_idx]
        expected_len = max_response_length - overlong_tokens
        exceed_len = response_length_cur - expected_len
        overlong_reward = min(
            -exceed_len / overlong_tokens * overlong_penalty_factor, 0
        )
        reward_score_cur += overlong_reward
        reward_score[sample_idx] = reward_score_cur

    data["rewards"] = reward_score
    return data
