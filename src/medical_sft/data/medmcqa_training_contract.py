"""Historical Qwen3.5-0.8B tokenization helper plus shared prompt encoding.

load_local_processor() is pinned to the old 0.8B snapshot. Do not use it as
the tokenizer preflight for the current Qwen2.5-3B or Qwen3.5-4B releases.
The caller supplies the processor for encode_training_sequence().
"""
from __future__ import annotations

from pathlib import Path
from medical_sft.paths import ROOT

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MAX_LENGTH = 512
FORMAL_MODEL_ID = "Qwen/Qwen3.5-4B"
FORMAL_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
INSTRUCTION = (
    "Answer the question and provide a concise explanation in English. "
    "Use exactly this format:\nAnswer: X\nExplanation: <brief explanation>"
)
TEMPLATE_KWARGS = dict(tokenize=True, return_dict=True, return_tensors="pt")


def training_record(record):
    c = record["content"]
    if record["answer_parse_status"] != "valid":
        raise ValueError("Cannot render an invalid/missing gold answer")
    prompt = c["question"] + "\n" + "\n".join(
        f"{letter}. {c['options'][letter]}" for letter in "ABCD"
    ) + "\n" + INSTRUCTION
    return dict(id=record["id"], messages=[
        dict(role="user", content=prompt),
        dict(role="assistant", content=f"Answer: {record['gold_answer']}\nExplanation: {c['explanation']}"),
    ])


def encode_training_sequence(processor, messages, *, enable_thinking=None):
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("The final message must be an assistant completion")
    kwargs = dict(TEMPLATE_KWARGS)
    if enable_thinking is not None:
        kwargs.update(enable_thinking=enable_thinking,
                      processor_kwargs=dict(padding=False, truncation=False))
    full = processor.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    prompt = processor.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    full_ids, prompt_ids = full["input_ids"][0], prompt["input_ids"][0]
    n, p = len(full_ids), len(prompt_ids)
    if n <= p or full_ids[:p].tolist() != prompt_ids.tolist():
        raise ValueError("Missing completion tokens or inconsistent chat-template prefix")
    return full, prompt


def load_local_processor():
    from transformers import AutoProcessor
    cache = ROOT.parent / ".hf-cache" / "hub"
    snapshot = cache / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / MODEL_REVISION
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Required offline tokenizer snapshot is missing: {snapshot}")
    return AutoProcessor.from_pretrained(str(snapshot), local_files_only=True), snapshot


def check_length(processor, record):
    full, prompt = encode_training_sequence(processor, training_record(record)["messages"])
    n, p = full["input_ids"].shape[-1], prompt["input_ids"].shape[-1]
    return dict(full_token_length=n, prompt_token_length=p, supervision_tokens=n-p,
                length_status="within_limit" if n <= MAX_LENGTH else "over_limit")
