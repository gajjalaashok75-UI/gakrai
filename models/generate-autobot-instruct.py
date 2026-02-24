"""Autobot instruct generation utilities (generation only)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

SPECIAL_TOKENS = [
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|startoftext|>",
    "<|tool_list_start|>",
    "<|tool_list_end|>",
]


def _template_token_count(tokenized: Any) -> int:
    """Normalize token count for different tokenizer return shapes."""
    if hasattr(tokenized, "shape") and len(tokenized.shape) >= 2:
        return int(tokenized.shape[1])

    if isinstance(tokenized, list):
        if tokenized and isinstance(tokenized[0], list):
            return len(tokenized[0])
        return len(tokenized)

    return 0


def _strip_special_tokens(text: str) -> str:
    cleaned = text
    for token in SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def generate_autobot_instruct(
    model,
    tokenizer,
    system_message: str,
    user_prompt: str,
    device: str,
    max_context_length: int,
    max_tokens: int,
    max_tokens_hard_limit: int,
    temperature: float,
    tools_json: Optional[List[Dict[str, Any]]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate one response and return generated text payload only."""
    print(
        "[GEN] Starting generation "
        f"(device={device}, max_tokens={max_tokens}, temp={temperature})"
    )
    chat_messages = messages or [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    print(f"[GEN] Message count: {len(chat_messages)}")

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    token_count_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }

    if tools_json:
        template_kwargs["tools"] = tools_json
        token_count_kwargs["tools"] = tools_json
        print(f"[GEN] Tools provided: {len(tools_json)}")

    try:
        template_tokens = tokenizer.apply_chat_template(chat_messages, **token_count_kwargs)
        template_token_count = _template_token_count(template_tokens)
        print("[GEN] Chat template tokenization used tools argument")
    except TypeError:
        token_count_kwargs.pop("tools", None)
        template_tokens = tokenizer.apply_chat_template(chat_messages, **token_count_kwargs)
        template_token_count = _template_token_count(template_tokens)
        print("[GEN] Chat template tokenization fallback without tools argument")

    try:
        formatted_prompt = tokenizer.apply_chat_template(chat_messages, **template_kwargs)
        print("[GEN] Chat template formatting used tools argument")
    except TypeError:
        template_kwargs.pop("tools", None)
        formatted_prompt = tokenizer.apply_chat_template(chat_messages, **template_kwargs)
        print("[GEN] Chat template formatting fallback without tools argument")

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_context_length - max_tokens,
    ).to(device)
    print(
        "[GEN] Prompt prepared "
        f"(chars={len(formatted_prompt)}, input_tokens={inputs['input_ids'].shape[1]})"
    )

    generation_config = {
        **inputs,
        "max_new_tokens": min(max_tokens, max_tokens_hard_limit),
        "do_sample": temperature > 0,
        "temperature": max(0.1, min(temperature, 1.0)),
        "top_p": 0.1,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if temperature <= 0.1:
        generation_config["do_sample"] = False
        generation_config.pop("temperature", None)
        generation_config.pop("top_p", None)
        generation_config.pop("top_k", None)
        print("[GEN] Deterministic decoding enabled")

    print("[GEN] Running model.generate")
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**generation_config)
    gen_elapsed = time.time() - gen_start
    print(f"[GEN] model.generate completed in {gen_elapsed:.2f}s")

    input_len = int(inputs["input_ids"].shape[1])
    generated_ids = output_ids[0][input_len:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    cleaned_text = _strip_special_tokens(raw_text)

    print(
        f"[GEN] Decoded output (generated_tokens={generated_ids.shape[0]}, "
        f"preview={raw_text[:160]!r})"
    )
    print("[GEN] Returning generated response payload")

    return {
        "text": cleaned_text,
        "raw_text": raw_text,
        "template_token_count": template_token_count,
        "formatted_prompt": formatted_prompt,
        "input_length": input_len,
        "generated_tokens": int(generated_ids.shape[0]),
    }
