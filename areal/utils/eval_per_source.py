# SPDX-License-Identifier: Apache-2.0
"""Per-source eval aggregation for blog/wandb logging.

Reads eval-rollout JSONL files written by workflow_executor, groups records
by the `data_source` column from the valid_dataset parquet (matched via
substring of the user-message content), and computes mean@N / pass@N per
source. Designed to be called by rank-0 of the RL trainer right after an
eval cycle so the metrics land in wandb at the correct global_step.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from areal.utils import logging

logger = logging.getLogger("EvalPerSource")


def build_prompt_source_map(parquet_path) -> dict[str, str]:
    """Return {prompt_user_content: data_source}.

    Accepts either a single parquet path (str) or a list of paths. Multi-path
    is used by dual-eval flows (math500 + aime24 in separate parquets, each
    with its own eval_gconfig).
    """
    if isinstance(parquet_path, (list, tuple)):
        paths = list(parquet_path)
    else:
        paths = [parquet_path] if parquet_path else []
    out: dict[str, str] = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        if not str(p).lower().endswith((".parquet", ".pq")):
            continue
        import pyarrow.parquet as pq

        table = pq.read_table(p, columns=["data_source", "prompt"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            prompt = row["prompt"]
            try:
                first = prompt[0]
                if isinstance(first, dict):
                    content = first.get("content", "")
                else:
                    content = str(prompt)
            except (TypeError, IndexError, KeyError):
                content = str(prompt)
            if content:
                out[content] = str(row["data_source"])
    return out


def _classify(prompt_text: str, prompt_to_source: dict[str, str]) -> str | None:
    for content, src in prompt_to_source.items():
        if content in prompt_text:
            return src
    return None


def aggregate_eval_jsonl(
    dump_dir: str,
    prompt_to_source: dict[str, str],
    step_version: int,
    n_samples_per_task=None,
) -> dict[str, float]:
    """``n_samples_per_task`` accepts int OR dict[str, int]. When dict, each
    source's expected n is looked up by the (raw, pre-sanitize) source string;
    sources not in the dict fall back to no cap. This supports dual-eval where
    e.g. math500=1 (greedy) and deepscaler/aime24=16 (sampled)."""
    """Scan ${dump_dir}/worker_*.jsonl for records of THIS eval cycle, group by
    data_source (math500, openai/gsm8k, ...), and compute:
      - eval/<src>/mean_at_n  : per-task mean of `reward`, then average across tasks
      - eval/<src>/pass_at_n  : per task_id, any(reward>0); then mean across tasks
      - eval/<src>/n_samples  : count of records contributing (post-cap)

    Filtering: eval records carry head_version=-1 (no off-policy concept in
    eval) and tail_version = actor's current update step at eval submission
    time, so we match on `tail_version == step_version`.

    Per-task cap (2026-06-15): If ``n_samples_per_task`` is provided, we keep
    only the FIRST ``n_samples_per_task`` records per (src, task_id) pair.
    Reason: ``GroupedRolloutWorkflow`` re-submits the whole rollout group on
    sample timeout, so a task can accumulate 5–11 records under one tail_version
    instead of the intended N. Without a cap, ``mean_at_n`` is biased toward
    high-retry (= long-response = hard) prompts, and ``pass_at_n`` is biased
    upward (extra rollouts = more chances to pass). The cap restores the
    "exactly N rollouts per task" invariant the metric names imply.

    Per-task averaging (also 2026-06-15): we now compute ``mean_at_n`` as the
    mean of per-task means rather than the mean across all records. With the
    cap, both definitions coincide for tasks that hit N records; for tasks
    that hit fewer (due to permanent failures), per-task averaging avoids
    weighting full-coverage tasks more than partial-coverage tasks.
    """
    paths = sorted(glob.glob(os.path.join(dump_dir, "worker_*.jsonl")))
    if not paths:
        return {}

    by_src_task_rewards: dict[str, dict[int | str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skipped_unmapped = 0
    group_by_prompt = os.environ.get("AREAL_EVAL_GROUP_BY_PROMPT") == "1"

    def _cap_for(src: str) -> int | None:
        if isinstance(n_samples_per_task, dict):
            return n_samples_per_task.get(src)
        return n_samples_per_task

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(rec.get("tail_version", -1)) != int(step_version):
                        continue
                    prompt_text = rec.get("prompt", "")
                    src = _classify(prompt_text, prompt_to_source)
                    if src is None:
                        skipped_unmapped += 1
                        continue
                    tid = int(rec.get("task_id", -1))
                    if tid < 0:
                        continue
                    bucket_key: int | str = prompt_text if group_by_prompt else tid
                    bucket = by_src_task_rewards[src][bucket_key]
                    cap = _cap_for(src)
                    if cap is None or len(bucket) < cap:
                        bucket.append(float(rec.get("reward", 0.0)))
        except FileNotFoundError:
            continue

    def _sanitize(s: str) -> str:
        # Strip a single "owner/" prefix (e.g. "openai/gsm8k" -> "gsm8k")
        # and collapse any remaining slashes — wandb treats "/" as group
        # nesting, which we don't want here.
        s = s.split("/", 1)[-1]
        return s.replace("/", "-")

    out: dict[str, float] = {}
    # Use the cap (which is the EXPECTED n_samples per task from
    # eval_gconfig.n_samples) as the denominator for per-task mean_at_n.
    # Rationale: a task with 3 successful rollouts + 1 timeout was supposed
    # to be evaluated 4 times; the failed sample is "no generation within
    # budget" → reward 0 contribution. So per-task mean = sum(actual rewards)
    # divided by the EXPECTED group size (n_samples_per_task), NOT by the
    # number of actually-recorded rewards. Counting timeouts as zero is the
    # scientifically honest accounting and keeps the metric comparable across
    # tasks with different completion rates.
    # pass_at_n is unaffected: "any of n succeeded" is determined by the
    # actual successes (timeouts can never raise pass_at_n).
    for src, task_rewards in by_src_task_rewards.items():
        if not task_rewards:
            continue
        s = _sanitize(src)
        cap_src = _cap_for(src)
        expected_n = cap_src if (cap_src and cap_src > 0) else None
        per_task_means = []
        per_task_pass = []
        for rs in task_rewards.values():
            if not rs:
                continue
            denom = expected_n if expected_n else len(rs)
            per_task_means.append(sum(rs) / denom)
            per_task_pass.append(1.0 if any(r > 0.0 for r in rs) else 0.0)
        total_records = sum(len(rs) for rs in task_rewards.values())
        if not per_task_means:
            continue
        out[f"eval/{s}/mean_at_n"] = sum(per_task_means) / len(per_task_means)
        out[f"eval/{s}/pass_at_n"] = sum(per_task_pass) / len(per_task_pass)
        out[f"eval/{s}/n_samples"] = float(total_records)
        out[f"eval/{s}/n_tasks"] = float(len(per_task_means))
    if skipped_unmapped:
        logger.warning(
            f"aggregate_eval_jsonl: {skipped_unmapped} records had prompt "
            f"not matching any data_source key — check chat-template alignment"
        )
    return out


def compute_train_pass_at_n(
    rollout_batch: list[dict],
) -> dict[str, float]:
    """Compute `train/pass_at_n` inline from a freshly-arrived train rollout
    batch. Each item is one prompt's K samples (rewards tensor of shape [K]);
    pass@n is the fraction of prompts where ≥1 sample got reward > 0.

    Replaces the earlier JSONL-scanning version, which broke whenever the
    rollout dataset's task_ids exceeded ``AREAL_TRAJECTORY_DUMP_TASK_LIMIT``
    (silent dump skip) — that gate left only the early-epoch task_ids on
    disk, all stamped with tail_versions that never matched the trainer's
    current global_step, so the metric never landed in wandb.

    Returns an empty dict if no usable rewards were found.
    """
    n_tasks = 0
    n_passed = 0
    n_samples = 0
    skipped_reasons: dict[str, int] = {}

    def _bump(reason: str) -> None:
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    for traj in rollout_batch:
        if not isinstance(traj, dict):
            _bump(f"non-dict:{type(traj).__name__}")
            continue
        rewards = traj.get("rewards")
        if rewards is None:
            _bump(f"no-rewards-key:keys={sorted(traj.keys())[:8]}")
            continue
        # rollout_batch lives controller-side, where trajectory tensors are
        # ``RTensor`` handles (data on meta device, fetched on demand). Pull
        # rewards local before the arithmetic. Each rewards tensor is only
        # K floats (K = gconfig.n_samples ≈ 4), so the per-step fetch cost
        # is negligible and `_fetch_buffer` already caches across this step.
        if hasattr(rewards, "to_local"):
            rewards = rewards.to_local()
        try:
            numel = int(rewards.numel())
        except AttributeError:
            _bump(f"no-numel:{type(rewards).__name__}")
            continue
        if numel == 0:
            _bump("empty-rewards")
            continue
        n_tasks += 1
        n_samples += numel
        if bool((rewards > 0).any()):
            n_passed += 1

    if n_tasks == 0:
        # Diagnostic: surface why we found zero trajectories. One-shot warning
        # per process so it doesn't spam the log every step.
        if not getattr(compute_train_pass_at_n, "_warned_empty", False):
            logger.warning(
                "compute_train_pass_at_n: 0 usable trajectories in "
                f"rollout_batch (len={len(rollout_batch)}); "
                f"skipped reasons={skipped_reasons}"
            )
            compute_train_pass_at_n._warned_empty = True  # type: ignore[attr-defined]
        return {}
    return {
        "train/pass_at_n": n_passed / n_tasks,
        "train/n_samples_at_step": float(n_samples),
    }
