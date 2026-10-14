# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory local attention gate for FRPO future corrections."""

from __future__ import annotations

from typing import Any

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


@torch.no_grad()
def local_attention_strongest_mean_ratio(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scaling: float,
    position_ids: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Strongest future attention to each source versus a local uniform null."""
    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1:
        raise ValueError("FRPO attention gate expects packed Q/K shaped [1, heads, tokens, dim]")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    heads, length = query.shape[1:3]
    logits = torch.full(
        (heads, length, horizon),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    valid = torch.zeros((length, horizon), dtype=torch.bool, device=query.device)
    positions = position_ids.reshape(-1)
    for distance in range(1, min(horizon, length - 1) + 1):
        same_sequence = positions[distance:] == positions[:-distance] + distance
        score = (
            query[0, :, distance:, :].float()
            * key[0, :, :-distance, :].float()
        ).sum(dim=-1) * float(scaling)
        logits[:, distance:, distance - 1] = torch.where(
            same_sequence.unsqueeze(0), score, -torch.inf
        )
        valid[distance:, distance - 1] = same_sequence
    has_history = valid.any(dim=-1)
    probabilities = torch.zeros_like(logits)
    probabilities[:, has_history] = torch.softmax(logits[:, has_history], dim=-1)
    mean_probability = probabilities.mean(dim=0)
    counts = valid.sum(dim=-1).clamp(min=1).float()
    strongest = torch.zeros(length, dtype=torch.float32, device=query.device)
    for distance in range(1, min(horizon, length - 1) + 1):
        source_valid = valid[distance:, distance - 1]
        ratio = mean_probability[distance:, distance - 1] * counts[distance:]
        strongest[:-distance] = torch.maximum(
            strongest[:-distance], torch.where(source_valid, ratio, 0.0)
        )
    return strongest


class FRPOAttentionGateCollector:
    """Collect last-layer Q/K during FlashAttention without materializing T x T."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layers: int,
        horizon: int,
        threshold: float,
    ) -> None:
        decoder_layers = model.model.layers
        if layers <= 0 or layers > len(decoder_layers):
            raise ValueError(f"invalid attention gate layer count {layers}")
        self.horizon = int(horizon)
        self.threshold = float(threshold)
        self._active = False
        self._position_ids: torch.Tensor | None = None
        self._cu_seqlens: torch.Tensor | None = None
        self._scores: list[torch.Tensor] = []
        self._state: dict[int, dict[str, torch.Tensor]] = {}
        for index, layer in enumerate(decoder_layers[-layers:]):
            attention = layer.self_attn
            attention.register_forward_pre_hook(
                self._attention_pre_hook(index), with_kwargs=True
            )
            attention.q_norm.register_forward_hook(self._query_hook(index))
            attention.k_norm.register_forward_hook(self._key_hook(index, attention))

    def begin(
        self,
        model_inputs: dict[str, Any],
        cu_seqlens: torch.Tensor | None,
    ) -> None:
        if self._active:
            raise RuntimeError("FRPO attention collector was started twice")
        input_ids = model_inputs["input_ids"]
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
            if cu_seqlens is not None:
                position_ids = position_ids.clone()
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                    position_ids[0, start:end] = torch.arange(end - start, device=input_ids.device)
        self._position_ids = position_ids
        self._cu_seqlens = cu_seqlens
        self._scores = []
        self._state = {}
        self._active = True

    def abort(self) -> None:
        self._active = False
        self._position_ids = None
        self._cu_seqlens = None
        self._scores = []
        self._state = {}

    def finish(self) -> torch.Tensor:
        if not self._active or self._position_ids is None:
            raise RuntimeError("FRPO attention collector is not active")
        if not self._scores:
            self.abort()
            raise RuntimeError("FRPO attention collector captured no layers")
        source_gate = torch.stack(self._scores).mean(dim=0) >= self.threshold
        gate = torch.roll(source_gate, shifts=-1, dims=0)
        gate[-1] = False
        if self._cu_seqlens is not None:
            for end in self._cu_seqlens[1:].tolist():
                gate[int(end) - 1] = False
        self.abort()
        return gate

    def _attention_pre_hook(self, index: int):
        def hook(module, args, kwargs):
            if not self._active:
                return
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            if position_embeddings is None:
                raise RuntimeError("Qwen attention did not provide rotary embeddings")
            cos, sin = position_embeddings
            self._state[index] = {"cos": cos.detach(), "sin": sin.detach()}

        return hook

    def _query_hook(self, index: int):
        def hook(module, args, output):
            if self._active:
                self._state[index]["query"] = output.detach()

        return hook

    def _key_hook(self, index: int, attention: torch.nn.Module):
        def hook(module, args, output):
            if not self._active:
                return
            state = self._state[index]
            query = state["query"].transpose(1, 2)
            key = output.detach().transpose(1, 2)
            query, key = apply_rotary(query, key, state["cos"], state["sin"])
            self._scores.append(
                local_attention_strongest_mean_ratio(
                    query,
                    key,
                    scaling=float(attention.scaling),
                    position_ids=self._position_ids,
                    horizon=self.horizon,
                )
            )
            self._state.pop(index, None)

        return hook
