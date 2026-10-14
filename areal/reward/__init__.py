# SPDX-License-Identifier: Apache-2.0

import os
import resource
import signal
import threading
import time

from math_verify.grader import verify as math_verify_verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

# See verl_math.py for the full memory-leak investigation. Both reward
# paths share the same defense layers.
try:
    from latex2sympy2_extended.antlr_parser import PSLexer, PSParser
except Exception:  # pragma: no cover
    PSLexer = None
    PSParser = None


_MAX_LATEX_DEPTH = 40  # tightened — see verl_math.py
_MAX_BACKSLASH_COUNT = 300
_MAX_RESPONSE_LEN_PRECHECK = 8192


def _looks_pathological(response: str) -> bool:
    if len(response) > _MAX_RESPONSE_LEN_PRECHECK:
        return True
    bs = response.count("\\")
    if bs > _MAX_BACKSLASH_COUNT:
        return True
    depth = max_depth = 0
    for c in response:
        if c == "(" or c == "[" or c == "{":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif c == ")" or c == "]" or c == "}":
            depth -= 1
    return max_depth > _MAX_LATEX_DEPTH


_LIBC_MALLOC_TRIM = None


def _malloc_trim() -> None:
    global _LIBC_MALLOC_TRIM
    if _LIBC_MALLOC_TRIM is None:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            _LIBC_MALLOC_TRIM = libc.malloc_trim
        except Exception:
            _LIBC_MALLOC_TRIM = False
    if _LIBC_MALLOC_TRIM and _LIBC_MALLOC_TRIM is not False:
        try:
            _LIBC_MALLOC_TRIM(0)
        except Exception:
            pass

from areal.utils import logging

logger = logging.getLogger("RewardUtils")

# See verl_math.py for rationale. Same threshold here so both reward paths
# auto-recycle workers that crossed the runaway-leak boundary.
_RSS_KILL_MB = 4000  # tightened from 6 GB — see verl_math.py.


def _clear_antlr_dfa_cache() -> None:
    """See verl_math.py._clear_antlr_dfa_cache. Duplicated to keep this
    module a leaf (no cross-import from verl_math) — verl_math depends on
    this module's _schedule_worker_suicide, so the dependency only flows one
    way."""
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
    next reward call lands on a fresh process. Cheap: one getrusage syscall."""
    try:
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


_RSS_WATCHDOG_STARTED = False
_RSS_WATCHDOG_LOCK = threading.Lock()


def _read_vmrss_kb() -> int:
    """Read VmRSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


_RLIMIT_AS_SET = False


def _set_rlimit_as(limit_mb: int = 6000) -> None:
    """Set the OS-level RLIMIT_AS so the kernel enforces the per-process
    address-space cap. Critical when Python-level watchdogs starve under
    sympy's long C-level computations (no GIL release → daemon thread
    can't run → no suicide fires). RLIMIT_AS is checked by the kernel
    itself on every brk()/mmap(), so it triggers even if Python is stuck.

    Set on every reward call — idempotent (calling setrlimit twice with
    same args is a no-op). Workers are spawn-context so we must (re)set
    in each new process.
    """
    global _RLIMIT_AS_SET
    if _RLIMIT_AS_SET:
        return
    try:
        import resource as _resource

        limit = limit_mb * 1024 * 1024
        soft, hard = _resource.getrlimit(_resource.RLIMIT_AS)
        # Only tighten — never loosen. Use min with existing hard cap.
        new_soft = limit if hard == _resource.RLIM_INFINITY else min(limit, hard)
        new_hard = hard if hard != _resource.RLIM_INFINITY else limit
        _resource.setrlimit(_resource.RLIMIT_AS, (new_soft, new_hard))
        _RLIMIT_AS_SET = True
    except Exception as e:
        try:
            logger.warning(f"_set_rlimit_as failed: {e!r}")
        except Exception:
            pass


_RSS_KILLER_STARTED = False
_RSS_KILLER_LOCK = threading.Lock()


def _start_rss_killer_subprocess(
    threshold_mb: int = 2000, interval_s: float = 0.25
) -> None:
    """GIL-independent backup watchdog. fork()s a child whose only job is to
    poll the *parent's* /proc/<pid>/status and SIGKILL the parent if its
    VmRSS exceeds threshold_mb.

    Why this exists in addition to ``_start_rss_watchdog`` (in-process daemon
    thread): the in-process watchdog calls ``os._exit(137)`` from PYTHON. When
    sympy / latex2sympy2_extended ANTLR holds the GIL for minutes (no release
    points in tight C inner loops), the daemon thread STARVES and can never
    fire — by the time it gets the GIL, RSS is already 450 GB. Confirmed root
    cause of 2026-06-09 FRPO retry-N OOM (4 spawn workers × 450 GB → node 98%
    → Ray kill).

    The forked killer subprocess is independent of the parent's GIL: it polls
    via ``open('/proc/<parent>/status')`` and calls ``os.kill(parent, SIGKILL)``
    — both kernel-level, no Python execution in the parent required. SIGKILL
    is uncatchable, so the runaway parent dies immediately. ProcessPoolExecutor
    sees ``BrokenProcessPool`` and respawns; the existing retry logic in
    ``AsyncRewardWrapper.__call__`` handles up to ``max_retries`` recoveries.

    Trade-off vs RLIMIT_AS (which we removed previously): RLIMIT_AS=12GB hit
    on every task that allocated >12 GB, including legitimate heavy parses,
    causing 211 BrokenProcessPool / 32 steps. The fork-killer at 2 GB
    threshold-on-RSS fires far less often because RSS only grows under
    runaway leak — legitimate parses peak at ~1.5 GB working set then
    deallocate. The threshold is also *resident set size* (only what's
    paged in), not virtual address space.

    Idempotent — calling twice is a no-op."""
    global _RSS_KILLER_STARTED
    with _RSS_KILLER_LOCK:
        if _RSS_KILLER_STARTED:
            return
        _RSS_KILLER_STARTED = True

    parent_pid = os.getpid()
    threshold_kb = threshold_mb * 1024

    try:
        child_pid = os.fork()
    except OSError:
        return  # cannot fork — fall back to existing in-process watchdog

    if child_pid != 0:
        # parent: child runs detached in the background; nothing to do here
        return

    # killer child — clean up Python state we don't need, then loop
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.setsid()
    except Exception:
        pass

    while True:
        try:
            with open(f"/proc/{parent_pid}/status") as f:
                vmrss_kb = 0
                for line in f:
                    if line.startswith("VmRSS:"):
                        vmrss_kb = int(line.split()[1])
                        break
            if vmrss_kb > threshold_kb:
                try:
                    os.kill(parent_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os._exit(0)
        except FileNotFoundError:
            # parent died (Ray killed it, normal exit, etc) — we're done
            os._exit(0)
        except Exception:
            pass
        time.sleep(interval_s)


def _start_rss_watchdog(threshold_mb: int = 2500, interval_s: float = 0.5) -> None:
    """Spawn a daemon thread that polls /proc/self/status every interval_s
    and triggers os._exit(137) if VmRSS crosses threshold_mb.

    Critical insight (2026-06-07): the existing _maybe_schedule_rss_suicide
    only runs in verl_math_reward_fn's finally block — i.e., AFTER each
    call returns. If a single call's inner ThreadPoolExecutor task runs
    away allocating 100+ GB of sympy state, the outer wrapper times out
    at 30 s, returns 0, and then checks RSS in finally. By that point
    the abandoned thread is still allocating; next call comes in; the
    pattern repeats. Worker can balloon to 450+ GB before the next
    finally-block RSS check catches it.

    A watchdog poll bypasses this: it fires REGARDLESS of which thread
    is busy, including from inside a pathological in-flight parse. The
    poll thread runs as daemon so it doesn't block process exit.

    Idempotent — calling twice in the same process is a no-op.
    """
    global _RSS_WATCHDOG_STARTED
    with _RSS_WATCHDOG_LOCK:
        if _RSS_WATCHDOG_STARTED:
            return
        _RSS_WATCHDOG_STARTED = True

    threshold_kb = threshold_mb * 1024

    def _loop():
        while True:
            try:
                rss = _read_vmrss_kb()
                if rss > threshold_kb:
                    # Bypass the 50 ms suicide delay used elsewhere — we
                    # want IMMEDIATE exit before another reward call
                    # piles more state onto an already-stuck worker.
                    try:
                        logger.warning(
                            f"[RSS-watchdog] VmRSS={rss/1024:.0f} MB > "
                            f"threshold {threshold_mb} MB; calling os._exit(137)"
                        )
                    except Exception:
                        pass
                    os._exit(137)
            except Exception:
                pass
            time.sleep(interval_s)

    threading.Thread(target=_loop, daemon=True, name="rss-watchdog").start()


def _schedule_worker_suicide(delay: float = 0.05) -> None:
    """Exit the current reward-worker process shortly after the caller returns.

    Called from inside the ProcessPoolExecutor child after a math_verify timeout.
    The stuck sympy thread continues to hog the GIL even after `ThreadPoolExecutor.
    shutdown(wait=False)` abandons it, so the worker becomes progressively slower
    and eventually deadlocks (root cause of the 2026-06-03 step-30 stall). Letting
    the worker exit cleanly *after* the current future has returned 0 lets
    ProcessPoolExecutor's `_adjust_process_count()` respawn a fresh child without
    the BrokenProcessPool fallout that an in-place SIGKILL would cause.
    """

    def _exit():
        time.sleep(delay)
        os._exit(137)

    threading.Thread(target=_exit, daemon=True).start()

VALID_REWARD_FN = ["clevr_count_70k", "geometry3k"]


def get_custom_reward_fn(path: str, **kwargs):
    if "clevr_count_70k" in path:
        from .clevr_count_70k import clevr_count_70k_reward_fn

        return clevr_count_70k_reward_fn
    elif "geometry3k" in path:
        from .geometry3k import geometry3k_reward_fn

        return geometry3k_reward_fn
    else:
        raise ValueError(
            f"Reward function {path} is not supported. "
            f"Supported reward functions are: {VALID_REWARD_FN}. "
        )


class MathVerifyWorker:
    """Thin wrapper over math_verify with configurable extraction/precision.

    Timed calls run in a dedicated scorer subprocess. Killing that subprocess
    on a pathological parser input leaves the shared reward-pool worker alive.

    Args:
        try_extract_without_anchor: When False, only answers with explicit anchors
            (e.g., "answer = 1", "final answer = 1") are matched. When True,
            any numeric string in the text may be extracted.
        precision: Number of significant digits that must match.
        timeout: Thread-safe timeout in seconds for the entire verify call
            (parsing + comparison). ``None`` disables the timeout.
    """

    def __init__(
        self,
        try_extract_without_anchor=True,
        precision: int = 6,
        timeout: float | None = 5.0,
    ):
        self.try_extract_without_anchor = try_extract_without_anchor
        self.gold_extraction_target = (
            ExprExtractionConfig(try_extract_without_anchor=try_extract_without_anchor),
            LatexExtractionConfig(),
        )
        self.pred_extraction_target = (
            ExprExtractionConfig(try_extract_without_anchor=try_extract_without_anchor),
            LatexExtractionConfig(),
        )
        self.precision = precision
        self.timeout = timeout

    def _verify_impl(self, response: str, ground_truth: str) -> float:
        gold_parsed = parse(
            ground_truth,
            extraction_config=self.gold_extraction_target,
            parsing_timeout=None,
        )
        pred_parsed = parse(
            response,
            extraction_config=self.pred_extraction_target,
            parsing_timeout=None,
        )
        if not gold_parsed or not pred_parsed:
            return 0.0
        result = math_verify_verify(
            gold_parsed,
            pred_parsed,
            float_rounding=self.precision,
            timeout_seconds=None,
        )
        return 1.0 if result else 0.0

    # Long-CoT short-circuit: responses above this length almost always come
    # from a degenerate "Step 1: ... Step 2: ..." loop that confuses sympy
    # into a runaway computation. Real math500/gsm8k correct answers fit well
    # under 16K chars. Returning 0 directly bypasses the sympy slow path that
    # caused the 2026-06-03 step-30 cascade: each long-CoT verify burned a
    # spawn-ctx worker process (~3 s respawn), and a handful of them per
    # rollout-batch was enough to starve the reward pool and deadlock the
    # trainer. The cost of a few false-zeros on legitimately-long correct
    # answers is far smaller than the cost of stalling training.
    _LONG_RESPONSE_CHARS = 16384

    def verify(self, response: str, ground_truth: str) -> float:
        if _looks_pathological(response):
            return 0.0
        if self.timeout is None:
            try:
                return self._verify_impl(response, ground_truth)
            except Exception:
                logger.warning(
                    f"Exception in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                    exc_info=True,
                )
                return 0.0
        from .math_verify_subprocess import score_with_hard_timeout

        return score_with_hard_timeout(
            response,
            ground_truth,
            self.try_extract_without_anchor,
            self.precision,
            self.timeout,
        )


_MATH_VERIFY_WORKER: MathVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        # AREAL_MATH_VERIFY_TIMEOUT_SECONDS lets rollout-debug runs bump the
        # MathVerifyWorker per-call timeout above the 5 s default without
        # touching the production code path. Default preserves prior behavior.
        timeout_env = os.environ.get("AREAL_MATH_VERIFY_TIMEOUT_SECONDS")
        kwargs = {}
        if timeout_env:
            kwargs["timeout"] = float(timeout_env)
        _MATH_VERIFY_WORKER = MathVerifyWorker(**kwargs)
    return _MATH_VERIFY_WORKER


__all__ = [
    "VALID_REWARD_FN",
    "get_custom_reward_fn",
    "MathVerifyWorker",
    "get_math_verify_worker",
    "gsm8k_reward_fn",
    "geometry3k_reward_fn",
    "clevr_count_70k_reward_fn",
    "verl_math_reward_fn",
    "verl_math_boxed_answer_reward_fn",
]


_LAZY_IMPORTS = {
    "gsm8k_reward_fn": "areal.reward.gsm8k",
    "geometry3k_reward_fn": "areal.reward.geometry3k",
    "clevr_count_70k_reward_fn": "areal.reward.clevr_count_70k",
    "verl_math_reward_fn": "areal.reward.verl_math",
    "verl_math_boxed_answer_reward_fn": "areal.reward.verl_math",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
