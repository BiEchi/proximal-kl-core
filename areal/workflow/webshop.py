# SPDX-License-Identifier: Apache-2.0

"""Direct multi-turn rollout workflow for the WebShop text environment."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import stats_tracker
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer


_ACTION_RE = re.compile(r"(?i)(search|click)\[([^\]\r\n]+)\]")
_EXPLICIT_ACTION_RE = re.compile(
    r"(?is)(?:^|[^a-z])(?:action|answer)\s*[:：]\s*(?:\[)?\s*"
    r"(search|click)\[([^\]\r\n]+)\]"
)


def extract_webshop_action(text: str) -> str:
    """Return the intended executable WebShop command in a model response."""
    explicit = list(_EXPLICIT_ACTION_RE.finditer(text))
    if explicit:
        match = explicit[-1]
        return f"{match.group(1).lower()}[{match.group(2).strip()}]"
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        return text.strip()
    # Models sometimes echo the available-action list before giving the answer.
    # The final command is usually the selected one in those responses.
    match = matches[-1]
    return f"{match.group(1).lower()}[{match.group(2).strip()}]"


def format_webshop_observation(payload: dict[str, Any], max_chars: int) -> str:
    observation = str(payload["observation"])
    if len(observation) > max_chars:
        observation = observation[:max_chars] + "\n[observation truncated]"

    available = payload.get("available_actions", {})
    lines = [f"Observation:\n{observation}", "Available actions:"]
    if available.get("has_search_bar"):
        lines.append("- search[keywords]")
    for value in available.get("clickables", []):
        lines.append(f"- click[{value}]")
    lines.append("Reply with exactly one available action and no other text.")
    return "\n".join(lines)


_SERVER: Any | None = None
_ENV_TYPE: Any | None = None
_SERVER_KEY: tuple[str, int] | None = None
_SERVER_LOCK = threading.Lock()


def _goal_indexed_env_type(base_type: Any) -> Any:
    class GoalIndexedWebAgentTextEnv(base_type):
        def __init__(self, *args: Any, goal_index: int, **kwargs: Any):
            self._goal_index = int(goal_index)
            super().__init__(*args, **kwargs)

        def reset(
            self, session: int | None = None, instruction_text: str | None = None
        ) -> Any:
            selected_session = self._goal_index if session is None else session
            return super().reset(
                session=selected_session, instruction_text=instruction_text
            )

    return GoalIndexedWebAgentTextEnv


def _load_webshop(webshop_root: str, num_products: int) -> tuple[Any, Any]:
    global _SERVER, _ENV_TYPE, _SERVER_KEY
    key = (str(Path(webshop_root).resolve()), num_products)
    with _SERVER_LOCK:
        if _SERVER is not None:
            if _SERVER_KEY != key:
                raise RuntimeError("one process cannot mix different WebShop datasets")
            return _SERVER, _ENV_TYPE

        sys.path.insert(0, key[0])
        dependency_path = os.environ.get("WEBSHOP_DEPS")
        if dependency_path and dependency_path not in sys.path:
            # Keep the training image's core packages ahead of WebShop-only deps.
            sys.path.append(dependency_path)
        if "JAVA_HOME" not in os.environ:
            java = shutil.which("java")
            if java:
                os.environ["JAVA_HOME"] = str(Path(java).resolve().parent.parent)
        from web_agent_site.envs.web_agent_text_env import (  # noqa: PLC0415
            SimServer,
            WebAgentTextEnv,
        )
        from web_agent_site.utils import DEFAULT_FILE_PATH  # noqa: PLC0415

        _SERVER = SimServer(
            "http://127.0.0.1:3000",
            file_path=DEFAULT_FILE_PATH,
            num_products=num_products,
            human_goals=0,
        )
        _ENV_TYPE = _goal_indexed_env_type(WebAgentTextEnv)
        _SERVER_KEY = key
        return _SERVER, _ENV_TYPE


class WebShopEpisode:
    """One isolated session backed by process-shared WebShop search data."""

    def __init__(self, webshop_root: str, num_products: int, goal_index: int):
        server, env_type = _load_webshop(webshop_root, num_products)
        self.env = env_type(
            observation_mode="text",
            server=server,
            goal_index=goal_index,
            session_prefix=f"{uuid.uuid4().hex}-",
        )

    def state(self) -> dict[str, Any]:
        return {
            "observation": self.env.observation,
            "available_actions": self.env.get_available_actions(),
        }

    def step(self, action: str) -> dict[str, Any]:
        available = self.env.get_available_actions()
        valid_action = self._is_valid(action, available)
        observation, reward, done, _ = self.env.step(action)
        return {
            "observation": observation,
            "available_actions": self.env.get_available_actions(),
            "reward": float(reward),
            "done": bool(done),
            "valid_action": valid_action,
        }

    @staticmethod
    def _is_valid(action: str, available: dict[str, Any]) -> bool:
        if action.startswith("search[") and action.endswith("]"):
            return bool(available.get("has_search_bar")) and len(action) > 8
        if action.startswith("click[") and action.endswith("]"):
            clickables = {str(value).lower() for value in available.get("clickables", [])}
            return action[6:-1].lower() in clickables
        return False


class WebShopWorkflow(RolloutWorkflow):
    """Run one WebShop episode and export all model turns as one trajectory."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        max_turns: int = 10,
        webshop_root: str | None = None,
        num_products: int = 1000,
        max_observation_chars: int = 2000,
        reward_mode: str = "dense",
    ):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if reward_mode not in {"dense", "success"}:
            raise ValueError("reward_mode must be 'dense' or 'success'")
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.max_turns = max_turns
        self.webshop_root = webshop_root or os.environ.get(
            "WEBSHOP_ROOT", "data/webshop/source"
        )
        self.num_products = num_products
        self.max_observation_chars = max_observation_chars
        self.reward_mode = reward_mode

    def _initial_prompt(self, state: dict[str, Any]) -> list[int]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are controlling the WebShop text environment. "
                    "At each turn choose one listed action. Search actions use "
                    "search[keywords], and navigation actions use click[label]."
                ),
            },
            {
                "role": "user",
                "content": format_webshop_observation(
                    state, self.max_observation_chars
                ),
            },
        ]
        return apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _followup_prompt(self, state: dict[str, Any]) -> list[int]:
        assistant = {"role": "assistant", "content": "placeholder"}
        prefix = apply_chat_template(
            self.tokenizer, [assistant], tokenize=True, enable_thinking=False
        )
        messages = [
            assistant,
            {
                "role": "user",
                "content": format_webshop_observation(
                    state, self.max_observation_chars
                ),
            },
        ]
        full = apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return full[len(prefix) :]

    async def _run_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
        environment: Any,
    ) -> dict[str, torch.Tensor]:
        state = environment.state()
        input_ids = self._initial_prompt(state)
        seq: list[int] = []
        logprobs: list[float] = []
        loss_mask: list[int] = []
        versions: list[int] = []
        reward = 0.0
        invalid_actions = 0
        turns = 0
        context_exhausted = False

        for turn in range(self.max_turns):
            max_tokens = getattr(self.gconfig, "max_tokens", None)
            max_new_tokens = getattr(self.gconfig, "max_new_tokens", 0)
            if max_tokens and len(input_ids) + max_new_tokens > max_tokens:
                context_exhausted = True
                break
            request_seed = workflow_context.derive_request_seed(self.gconfig.seed)
            response = await engine.agenerate(
                ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1, seed=request_seed),
                    metadata={"is_eval": workflow_context.get().is_eval},
                    tokenizer=self.tokenizer,
                )
            )
            if response.input_tokens[: len(seq)] != seq:
                raise RuntimeError("WebShop turn is not a continuation of the trajectory")
            new_input = response.input_tokens[len(seq) :]
            input_len = len(new_input)
            seq += new_input + response.output_tokens
            logprobs += [0.0] * input_len + response.output_logprobs
            loss_mask += [0] * input_len + [1] * response.output_len
            versions += [-1] * input_len + response.output_versions

            action = extract_webshop_action(
                self.tokenizer.decode(
                    response.output_tokens, skip_special_tokens=True
                )
            )
            state = await asyncio.to_thread(environment.step, action)
            reward = float(state["reward"])
            invalid_actions += int(not state.get("valid_action", True))
            turns = turn + 1
            if state["done"]:
                break

            input_ids = seq.copy()
            if input_ids and input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
            input_ids += self._followup_prompt(state)

        success = float(reward >= 1.0 - 1e-8)
        train_reward = success if self.reward_mode == "success" else reward
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward,
            webshop_success=success,
            webshop_turns=turns,
            webshop_response_tokens_per_turn=sum(loss_mask) / max(turns, 1),
            webshop_invalid_action_rate=invalid_actions / max(turns, 1),
            webshop_context_exhausted=float(context_exhausted),
        )
        result = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "rewards": torch.tensor(train_reward, dtype=torch.float32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
        }
        return {key: value.unsqueeze(0) for key, value in result.items()}

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        environment = await asyncio.to_thread(
            WebShopEpisode,
            self.webshop_root,
            self.num_products,
            int(data["goal_index"]),
        )
        return await self._run_episode(engine, data, environment)
