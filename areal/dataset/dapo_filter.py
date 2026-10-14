# SPDX-License-Identifier: Apache-2.0
"""DAPO-filter dataset loader for AReaL RL training.

Reads the parquet at `path` (already on local Lustre, no remote download) and
emits the same {messages, answer} schema that AReaL's RLVRWorkflow expects,
mirroring `torl_data.get_torl_data_rl_dataset` minus the GitHub download step.

Source format (HF: Jiawei415/DPAO_filter):
    data_source: str
    prompt: list[{'role': ..., 'content': ...}]  -- chat messages
    ability: str
    reward_model: {'ground_truth': str, 'style': 'rule'}
    extra_info: dict
"""
from datasets import load_dataset

from areal.utils.hf_utils import apply_chat_template


def get_dapo_filter_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset("parquet", data_files=path, split="train")

    def process(sample):
        answer = sample["reward_model"]["ground_truth"]
        answer = f"\\boxed{{{answer}}}"
        return {"messages": sample["prompt"], "answer": answer}

    dataset = dataset.map(process).remove_columns(["prompt", "reward_model"])

    if max_length is not None:

        def filter_length(sample):
            tokens = apply_chat_template(
                tokenizer,
                sample["messages"],
                tokenize=True,
                add_generation_prompt=True,
            )
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
