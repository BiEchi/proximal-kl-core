# Proximal KL

Minimal training code for the paper's controlled rollout staleness setting,
using the AReaL framework.

[Paper](https://arxiv.org/abs/2610.18738) ·
[Blog](https://biechi.github.io/blog/offpolicy-proximal-kl/)

## Run

Use Linux x86_64, Python 3.12, and compatible CUDA GPUs. The configs use a local
scheduler; adjust resource allocation to your environment.

```sh
python -m pip install uv
uv sync --frozen --python 3.12 --extra sglang
mkdir -p logs/areal logs/name_resolve
uv run --extra sglang python train.py --config configs/frpo.yaml
```

For GRPO:

```sh
uv run --extra sglang python train.py --config configs/grpo.yaml
```

Use `uv` to apply the framework's dependency overrides. The SGLang extra
includes FlashAttention 2 for packed training; SDPA is not a compatible
replacement for this path. Use a short local `TMPDIR` to stay within Unix
socket pathname limits. Keep checkpoint output on persistent storage.

## Configuration

Both configs specify `lr=5e-6` and use public Qwen3-1.7B and GSM8K data.
To reproduce a particular paper result, match its model, data, generation
budget, batch size, training budget, and evaluation protocol.

Controlled lag requires both settings:

```yaml
rollout:
  max_head_offpolicyness: 4
  ready_batch_buffer: 5
```

For lag `L`, set the buffer to `L + 1`. Lag is measured in outer learner
updates, each containing four PPO minibatches. Fresh and resumed runs warm
up the FIFO before reaching the requested lag. Inspect
`version_stats/sample_staleness_theta_avg` for the measured value.

FRPO uses full Future-KL, `beta=0.30`, and per-token cost bound
`epsilon_F=0.05`; the matched GRPO config uses PPO clipping. External
experiment logging is disabled.

Core loss: `areal/utils/functional/functional.py`.
Actor: `areal/trainer/ppo/actor.py`.
Controlled rollout buffering: `areal/infra/workflow_executor.py`.

See `LICENSE` and `NOTICE` for third-party attribution.
