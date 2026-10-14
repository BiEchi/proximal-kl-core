# SPDX-License-Identifier: Apache-2.0
"""Hard-timeout isolation for the training math verifier.

Each reward-pool worker owns one small scorer subprocess. A pathological
parser input can therefore be killed without terminating the shared
``ProcessPoolExecutor`` worker or invalidating unrelated reward futures.
"""

import atexit
import json
import os
import select
import socket
import subprocess
import sys
import time
import zlib

from math_verify.grader import verify as math_verify_verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

from areal.utils import logging

logger = logging.getLogger("MathVerifySubprocess")

_WORKER_PROC: subprocess.Popen | None = None
_REQUEST_ID = 0
_WORKER_REQUEST_COUNT = 0
_DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0


def _startup_timeout_seconds() -> float:
    """Return the scorer-process startup timeout.

    Importing ``math_verify`` can take longer when many rollout workers start
    scorer subprocesses at once. Keep the historical default for callers that
    do not opt in, while allowing launchers to raise only the startup guard;
    the per-request scoring timeout remains unchanged.
    """
    raw = os.environ.get("AREAL_MATH_VERIFY_STARTUP_TIMEOUT_SECONDS")
    if raw is None:
        return _DEFAULT_STARTUP_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return _DEFAULT_STARTUP_TIMEOUT_SECONDS
    if timeout <= 0:
        return _DEFAULT_STARTUP_TIMEOUT_SECONDS
    return timeout


def _jittered_recycle_limit(
    max_requests: int, jitter_fraction: float, identity_hash: int
) -> int:
    """Stagger periodic scorer restarts across reward workers.

    Without jitter, workers created together all recycle after the same request
    count. Importing math_verify in all replacement subprocesses at once can
    exceed the startup timeout even though each import succeeds in isolation.
    """
    if max_requests <= 0:
        return max_requests
    jitter_fraction = min(max(jitter_fraction, 0.0), 1.0)
    jitter_window = int(max_requests * jitter_fraction)
    if jitter_window == 0:
        return max_requests
    return max(1, max_requests - identity_hash % (jitter_window + 1))


def _recycle_limit(max_requests: int) -> int:
    try:
        jitter_fraction = float(
            os.environ.get("AREAL_MATH_VERIFY_RECYCLE_JITTER_FRACTION", "0")
        )
    except ValueError:
        jitter_fraction = 0.0
    identity = f"{socket.gethostname()}:{os.getpid()}".encode()
    return _jittered_recycle_limit(
        max_requests, jitter_fraction, zlib.crc32(identity)
    )


def _score_impl(
    response: str,
    ground_truth: str,
    try_extract_without_anchor: bool,
    precision: int,
) -> float:
    extraction_targets = (
        ExprExtractionConfig(
            try_extract_without_anchor=try_extract_without_anchor
        ),
        LatexExtractionConfig(),
    )
    gold_parsed = parse(
        ground_truth,
        extraction_config=extraction_targets,
        parsing_timeout=None,
    )
    pred_parsed = parse(
        response,
        extraction_config=extraction_targets,
        parsing_timeout=None,
    )
    if not gold_parsed or not pred_parsed:
        return 0.0
    result = math_verify_verify(
        gold_parsed,
        pred_parsed,
        float_rounding=precision,
        timeout_seconds=None,
    )
    return 1.0 if result else 0.0


def _run_worker_stdio() -> None:
    sys.stdout.write(json.dumps({"ready": True}, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    for line in sys.stdin.buffer:
        request_id = None
        try:
            item = json.loads(line)
            request_id = item["request_id"]
            delay = float(item.get("test_delay_seconds", 0.0))
            if delay > 0:
                time.sleep(delay)
            score = _score_impl(
                str(item["response"]),
                str(item["ground_truth"]),
                bool(item["try_extract_without_anchor"]),
                int(item["precision"]),
            )
        except Exception:
            score = 0.0
        sys.stdout.write(
            json.dumps(
                {"request_id": request_id, "score": float(score)},
                separators=(",", ":"),
            )
            + "\n"
        )
        sys.stdout.flush()


def _close_worker(kill: bool = False) -> None:
    global _WORKER_PROC, _WORKER_REQUEST_COUNT
    proc = _WORKER_PROC
    _WORKER_PROC = None
    _WORKER_REQUEST_COUNT = 0
    if proc is None:
        return

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


def _ensure_worker() -> subprocess.Popen:
    global _WORKER_PROC
    if _WORKER_PROC is not None and _WORKER_PROC.poll() is None:
        return _WORKER_PROC

    _close_worker(kill=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    _WORKER_PROC = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "areal.reward.math_verify_subprocess",
            "--worker",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
        env=env,
        start_new_session=True,
    )
    try:
        assert _WORKER_PROC.stdout is not None
        ready, _, _ = select.select(
            [_WORKER_PROC.stdout], [], [], _startup_timeout_seconds()
        )
        if not ready:
            raise TimeoutError("math verifier subprocess startup timed out")
        item = json.loads(_WORKER_PROC.stdout.readline())
        if not item.get("ready"):
            raise RuntimeError(f"unexpected verifier startup message: {item!r}")
    except Exception:
        logger.warning("Math verifier subprocess failed to start", exc_info=True)
        _close_worker(kill=True)
        raise
    return _WORKER_PROC


def _score_with_hard_timeout(
    response: str,
    ground_truth: str,
    try_extract_without_anchor: bool,
    precision: int,
    timeout: float,
    *,
    test_delay_seconds: float = 0.0,
) -> float:
    global _REQUEST_ID, _WORKER_REQUEST_COUNT
    proc = _ensure_worker()

    _REQUEST_ID += 1
    request_id = _REQUEST_ID
    payload = json.dumps(
        {
            "request_id": request_id,
            "response": response,
            "ground_truth": ground_truth,
            "try_extract_without_anchor": try_extract_without_anchor,
            "precision": precision,
            "test_delay_seconds": test_delay_seconds,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    try:
        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.flush()
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("math verifier subprocess exited")
        item = json.loads(line)
        if item.get("request_id") != request_id:
            raise RuntimeError(
                f"mismatched verifier request id: {item.get('request_id')!r}"
            )
        score = float(item.get("score", 0.0))
        _WORKER_REQUEST_COUNT += 1
        max_requests = int(
            os.environ.get("AREAL_MATH_VERIFY_MAX_REQUESTS_PER_SCORER", "256")
        )
        recycle_limit = _recycle_limit(max_requests)
        if recycle_limit > 0 and _WORKER_REQUEST_COUNT >= recycle_limit:
            logger.info(
                "Recycling math verifier scorer after "
                f"{_WORKER_REQUEST_COUNT} requests (limit={recycle_limit})"
            )
            _close_worker()
        return score
    except TimeoutError:
        logger.warning(
            f"Math verifier hard timeout after {timeout}s; restarting scorer"
        )
        _close_worker(kill=True)
        return 0.0
    except Exception:
        logger.warning("Math verifier subprocess failed; restarting", exc_info=True)
        _close_worker(kill=True)
        raise


def score_with_hard_timeout(
    response: str,
    ground_truth: str,
    try_extract_without_anchor: bool,
    precision: int,
    timeout: float,
) -> float:
    return _score_with_hard_timeout(
        response,
        ground_truth,
        try_extract_without_anchor,
        precision,
        timeout,
    )


atexit.register(_close_worker)


if __name__ == "__main__" and "--worker" in sys.argv:
    _run_worker_stdio()
