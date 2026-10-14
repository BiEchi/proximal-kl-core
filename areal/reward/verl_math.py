# SPDX-License-Identifier: Apache-2.0
"""Reward function whose score is byte-identical to verl's
``verl.utils.reward_score.math_verify.compute_score``. Use for eval to make AReaL
numbers directly comparable to a verl baseline.

Differences from ``areal.reward.gsm8k.gsm8k_reward_fn``:

- gold is wrapped in ``\\boxed{}`` before parsing; gold parsed with
  ``LatexExtractionConfig`` only.
- pred parsed with ``(ExprExtractionConfig(), LatexExtractionConfig())`` — i.e.
  ``try_extract_without_anchor=False`` (default), unlike gsm8k_reward_fn which
  uses ``try_extract_without_anchor=True`` on both gold and pred.
- An 8 s timeout bounds parsing of long LaTeX expressions in MATH-style responses.
- Wraps each call in a per-invocation ``ThreadPoolExecutor`` with
  ``shutdown(wait=False)`` on timeout, so the caller does not wait for a stuck
  parser thread. Input checks and worker recycling bound resource use.
"""

import concurrent.futures
import json
import os
import resource
import select
import subprocess
import sys

from math_verify.grader import verify as math_verify_verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

# Deeply nested expressions can keep parser threads running after timeout.
# Input pre-checks, bounded worker lifetimes, and RSS watchdogs complement
# parser-cache cleanup; cache cleanup alone cannot interrupt a stuck parse.
try:
    from latex2sympy2_extended.antlr_parser import PSLexer, PSParser
except Exception:  # pragma: no cover
    PSLexer = None
    PSParser = None

from areal.reward import (
    _schedule_worker_suicide,
    _set_rlimit_as,
    _start_rss_killer_subprocess,
    _start_rss_watchdog,
)
from areal.utils import logging

logger = logging.getLogger("VerlMathReward")

_GOLD_TARGETS = (LatexExtractionConfig(),)
_PRED_TARGETS = (ExprExtractionConfig(), LatexExtractionConfig())
_TIMEOUT_S = 8.0  # Longer-running parses time out and receive reward zero.

# Recycle workers above this RSS threshold. The subprocess watchdog below
# remains responsive when parser code holds the Python GIL.
_RSS_KILL_MB = 2500

# Optional libc malloc_trim handle — resolved lazily so unit tests on macOS
# don't fail to import. Called from the finally block to release unused arenas.
_LIBC_MALLOC_TRIM = None


def _malloc_trim() -> None:
    global _LIBC_MALLOC_TRIM
    if _LIBC_MALLOC_TRIM is None:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            _LIBC_MALLOC_TRIM = libc.malloc_trim
        except Exception:
            _LIBC_MALLOC_TRIM = False  # mark as tried-and-failed
    if _LIBC_MALLOC_TRIM and _LIBC_MALLOC_TRIM is not False:
        try:
            _LIBC_MALLOC_TRIM(0)
        except Exception:
            pass


_MAX_LATEX_DEPTH = 40  # Maximum nesting of parentheses/brackets.
_MAX_BACKSLASH_COUNT = 300
_MAX_RESPONSE_LEN_PRECHECK = 8192  # tightened from 16384.


# Per-worker counters so we can attribute reward=0 to its cause.
# Logged every 200 calls so the rate is visible without log-spamming.
_REJECT_COUNTERS = {
    "precheck_len": 0,
    "precheck_backslash": 0,
    "precheck_depth": 0,
    "math_exception": 0,
    "timeout": 0,
    "ok_nonzero": 0,
    "ok_zero": 0,
}

_HARD_WORKER_TIMEOUT_S = float(
    os.environ.get("AREAL_VERL_MATH_TIMEOUT_SECONDS", "30")
)
_HARD_WORKER_STARTUP_TIMEOUT_S = float(
    os.environ.get("AREAL_VERL_MATH_STARTUP_TIMEOUT_SECONDS", "180")
)
_HARD_WORKER_PROC = None
_HARD_WORKER_REQ_ID = 0


def _log_counters_if_due() -> None:
    total = sum(_REJECT_COUNTERS.values())
    if total > 0 and total % 200 == 0:
        try:
            logger.warning(f"[reward-attribution after {total} calls] {_REJECT_COUNTERS}")
        except Exception:
            pass


def _looks_pathological(response: str) -> bool:
    """Cheap pre-check: is this completion likely to push math_verify into
    a multi-second / multi-GB parse?

    Deeply nested expressions can require excessive parsing time and memory.
    Returning True skips math_verify and gives reward zero, including any
    correct answer that exceeds these resource-protection limits.

    Increments the appropriate _REJECT_COUNTERS entry on rejection so we
    can later attribute training-time reward collapse to its real cause.
    """
    if len(response) > _MAX_RESPONSE_LEN_PRECHECK:
        _REJECT_COUNTERS["precheck_len"] += 1
        return True
    bs = response.count("\\")
    if bs > _MAX_BACKSLASH_COUNT:
        _REJECT_COUNTERS["precheck_backslash"] += 1
        return True
    # Maximum simultaneous open-paren depth across the string
    depth = max_depth = 0
    for c in response:
        if c == "(" or c == "[" or c == "{":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif c == ")" or c == "]" or c == "}":
            depth -= 1
    if max_depth > _MAX_LATEX_DEPTH:
        _REJECT_COUNTERS["precheck_depth"] += 1
        return True
    return False


def _clear_antlr_dfa_cache() -> None:
    """Drop the class-level DFA states and shared context cache that
    latex2sympy2_extended's ANTLR4 parser accumulates per unique input.

    Belt-and-suspenders with _drop_math_verify_caches() above: even after
    we evict the sympy trees, the ANTLR prediction DFAs keep growing with
    input variety.
    Clearing them costs nothing — the next parse rebuilds what it needs.
    """
    if PSParser is None or PSLexer is None:
        return
    try:
        for dfa in PSParser.decisionsToDFA:
            dfa.states.clear()
        for dfa in PSLexer.decisionsToDFA:
            dfa.states.clear()
        PSParser.sharedContextCache.cache.clear()
    except Exception:
        pass


def _maybe_schedule_rss_suicide() -> None:
    """If our own RSS is past _RSS_KILL_MB, schedule worker self-exit so the
    next reward call lands on a fresh process. Cheap: getrusage is a syscall."""
    try:
        # ru_maxrss is in kilobytes on Linux (POSIX leaves units implementation-
        # defined, but glibc uses KB here).
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return
    mb = kb / 1024.0
    if mb > _RSS_KILL_MB:
        logger.warning(
            f"Reward worker RSS={mb:.0f} MB exceeds {_RSS_KILL_MB} MB; "
            f"scheduling self-exit to recycle leaked C-level state"
        )
        _schedule_worker_suicide()


def _verify_impl(response: str, ground_truth_boxed: str) -> float:
    gold = parse(
        ground_truth_boxed, extraction_config=_GOLD_TARGETS, parsing_timeout=None
    )
    pred = parse(response, extraction_config=_PRED_TARGETS, parsing_timeout=None)
    if not gold or not pred:
        return 0.0
    return max(
        1.0
        if any(
            math_verify_verify(g, p, float_rounding=6, timeout_seconds=None)
            for g in gold
        )
        else 0.0
        for p in pred
    )


def _verl_compute_score_impl(model_output: str, ground_truth: str) -> float:
    """Exact verl math_verify scoring body, without AReaL prechecks/timeouts."""

    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())
    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(
            1.0 if any(math_verify_verify(g, p) for g in extracted_gold) else 0.0
            for p in extracted_pred
        )
    return 0.0


def _run_hard_timeout_worker_stdio() -> None:
    """Line-oriented subprocess worker for exact verl math_verify scoring."""

    sys.stdout.write(json.dumps({"ready": True}, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    for line in sys.stdin.buffer:
        req_id = None
        try:
            item = json.loads(line)
            req_id = item["req_id"]
            model_output = item["model_output"]
            ground_truth = item["ground_truth"]
            score = _verl_compute_score_impl(model_output, ground_truth)
        except Exception:
            score = 0.0
        sys.stdout.write(
            json.dumps(
                {
                    "req_id": req_id,
                    "score": float(score),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        sys.stdout.flush()


def _close_hard_timeout_worker(kill: bool = False) -> None:
    global _HARD_WORKER_PROC
    proc = _HARD_WORKER_PROC
    _HARD_WORKER_PROC = None

    if proc is not None:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                if kill:
                    proc.kill()
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass


def _ensure_hard_timeout_worker():
    global _HARD_WORKER_PROC
    if _HARD_WORKER_PROC is not None and _HARD_WORKER_PROC.poll() is None:
        return _HARD_WORKER_PROC

    _close_hard_timeout_worker(kill=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    _HARD_WORKER_PROC = subprocess.Popen(
        [sys.executable, "-m", "areal.reward.verl_math", "--hard-timeout-worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
        env=env,
        start_new_session=True,
    )
    try:
        assert _HARD_WORKER_PROC.stdout is not None
        ready, _, _ = select.select(
            [_HARD_WORKER_PROC.stdout], [], [], _HARD_WORKER_STARTUP_TIMEOUT_S
        )
        if not ready:
            raise TimeoutError
        line = _HARD_WORKER_PROC.stdout.readline()
        item = json.loads(line)
        if not item.get("ready"):
            raise RuntimeError(f"unexpected verifier startup message: {item!r}")
    except Exception:
        logger.warning(
            "verl math_verify subprocess failed to become ready within "
            f"{_HARD_WORKER_STARTUP_TIMEOUT_S}s; restarting",
            exc_info=True,
        )
        _close_hard_timeout_worker(kill=True)
        raise
    return _HARD_WORKER_PROC


def _verl_math_score_hard_timeout(
    model_output: str,
    ground_truth: str,
    timeout: float = _HARD_WORKER_TIMEOUT_S,
) -> float:
    global _HARD_WORKER_REQ_ID
    try:
        proc = _ensure_hard_timeout_worker()
    except Exception:
        return 0.0
    _HARD_WORKER_REQ_ID += 1
    req_id = _HARD_WORKER_REQ_ID
    payload = json.dumps(
        {
            "req_id": req_id,
            "model_output": model_output,
            "ground_truth": ground_truth,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    try:
        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.flush()
    except Exception:
        _close_hard_timeout_worker(kill=True)
        return 0.0

    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError
        line = proc.stdout.readline()
        if not line:
            _close_hard_timeout_worker(kill=True)
            return 0.0
        item = json.loads(line)
        if item.get("req_id") != req_id:
            logger.warning(
                "verl math_verify subprocess returned mismatched request id; "
                "restarting verifier subprocess"
            )
            _close_hard_timeout_worker(kill=True)
            return 0.0
        return float(item.get("score", 0.0))
    except TimeoutError:
        logger.warning(
            f"verl math_verify hard timeout after {timeout}s; "
            "killing and restarting verifier subprocess"
        )
        _close_hard_timeout_worker(kill=True)
        return 0.0
    except Exception:
        logger.warning("verl math_verify subprocess failed; restarting", exc_info=True)
        _close_hard_timeout_worker(kill=True)
        return 0.0


import atexit

atexit.register(_close_hard_timeout_worker)


# See MathVerifyWorker._LONG_RESPONSE_CHARS for why we short-circuit long
# responses. verl_math is used for MATH500 eval, where the same degenerate
# long-CoT runaway shows up.
_LONG_RESPONSE_CHARS = 16384


def verl_math_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    # Recycle workers that exceed the configured RSS threshold.
    _start_rss_watchdog(threshold_mb=_RSS_KILL_MB)
    # GIL-independent backup: fork()s a killer subprocess on first call. The
    # in-process watchdog above starves under sympy's GIL-hogging C parses;
    # this one polls /proc and SIGKILLs the parent — kernel-level, no GIL.
    # Threshold is intentionally lower (2 GB vs 2.5 GB) so the killer fires
    # before the Python watchdog would have, catching the runaway earlier.
    _start_rss_killer_subprocess(threshold_mb=2000, interval_s=0.25)
    response = str(completions)
    if _looks_pathological(response):
        # See module docstring: pathological depth/length nests blow up sympy
        # parse to multi-GB single allocations and abandoned threads. Short-
        # circuit to 0 — losing a small fraction of valid rewards is cheaper
        # than worker-process OOM. (_REJECT_COUNTERS bumped inside.)
        _log_counters_if_due()
        return 0.0
    ground_truth_boxed = "\\boxed{" + str(answer) + "}"
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_verify_impl, response, ground_truth_boxed)
        result = future.result(timeout=_TIMEOUT_S)
        # Bucket the legitimate math_verify outcomes — distinguishes "model
        # answered correctly" from "model answered but wrong" from the
        # reject paths above. Together with precheck_* and math_exception
        # counters, this gives a complete attribution of every reward=0.
        if result > 0:
            _REJECT_COUNTERS["ok_nonzero"] += 1
        else:
            _REJECT_COUNTERS["ok_zero"] += 1
        _log_counters_if_due()
        return result
    except concurrent.futures.TimeoutError:
        logger.warning(
            f"Timeout ({_TIMEOUT_S}s) in verl_math_reward_fn for "
            f"ground_truth={ground_truth_boxed!r}; scheduling worker self-exit "
            f"so the GIL-hogging sympy thread cannot poison the next task"
        )
        _schedule_worker_suicide()
        _REJECT_COUNTERS["timeout"] += 1
        _log_counters_if_due()
        return 0.0
    except Exception:
        logger.warning("Exception in verl_math_reward_fn", exc_info=True)
        _REJECT_COUNTERS["math_exception"] += 1
        _log_counters_if_due()
        return 0.0
    finally:
        executor.shutdown(wait=False)
        _clear_antlr_dfa_cache()
        # malloc_trim before checking RSS so the threshold compares "real"
        # working set, not unreclaimed glibc arenas.
        _malloc_trim()
        _maybe_schedule_rss_suicide()


def _strip_single_outer_boxed(answer: str) -> str:
    answer = answer.strip()
    prefix = "\\boxed{"
    if answer.startswith(prefix) and answer.endswith("}"):
        return answer[len(prefix) : -1]
    return answer


def verl_math_boxed_answer_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """verl math scorer for AReaL datasets whose loader already boxes answers."""

    return _verl_math_score_hard_timeout(
        str(completions),
        _strip_single_outer_boxed(str(answer)),
    )


if __name__ == "__main__" and "--hard-timeout-worker" in sys.argv:
    _run_hard_timeout_worker_stdio()
