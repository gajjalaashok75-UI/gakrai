"""Utilities to load the local Autobot Instruct model."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_model_dir(base_dir: Optional[str] = None) -> str:
    """Resolve the model directory by checking common local locations."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        base_dir = os.path.abspath(base_dir)
    
    print(f"[LOAD] Resolving model directory from base: {base_dir}")

    candidates = [
        base_dir,  # Direct path (may already point to model dir)
        os.path.join(base_dir, "autobot-instruct"),  # If base is models dir
        os.path.join(os.path.dirname(__file__), "..", "models", "autobot-instruct"),  # Fallback to root/models
        os.path.join(os.path.dirname(__file__), "autobot-instruct"),  # Fallback to same dir
    ]

    for model_dir in candidates:
        config_path = os.path.join(model_dir, "config.json")
        print(f"[LOAD] Checking candidate: {model_dir}")
        if os.path.isfile(config_path):
            print(f"[LOAD] Using model directory: {model_dir}")
            return model_dir

    checked = "\n  - ".join(["", *candidates])
    raise FileNotFoundError(
        "Could not find autobot-instruct model. Checked:" + checked
    )


def load_autobot_instruct(
    base_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Load tokenizer + model and return (tokenizer, model, resolved_model_dir)."""
    print("[LOAD] Starting autobot-instruct load")
    model_dir = _resolve_model_dir(base_dir=base_dir)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOAD] Target device: {selected_device}")

    try:
        print("[LOAD] Loading tokenizer (primary config)")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side="left",
            trust_remote_code=False,
            use_fast=True,
        )
        print(
            f"[LOAD] Tokenizer loaded. vocab_size={tokenizer.vocab_size}, "
            f"has_chat_template={bool(getattr(tokenizer, 'chat_template', None))}"
        )

        load_kwargs = {
            "torch_dtype": torch.bfloat16 if selected_device == "cuda" else torch.float32,
            "trust_remote_code": False,
            "device_map": "auto" if selected_device == "cuda" else None,
        }
        print(f"[LOAD] Loading model with kwargs: {load_kwargs}")

        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        print("[LOAD] Model weights loaded")

        if selected_device != "cuda" or load_kwargs.get("device_map") is None:
            print(f"[LOAD] Moving model to device: {selected_device}")
            model = model.to(selected_device)

        model.eval()
        model.config.use_cache = True
        print(
            f"[LOAD] Model ready. device={model.device}, dtype={model.dtype}, "
            "use_cache=True"
        )

        return tokenizer, model, model_dir
    except Exception as first_error:
        print(f"[LOAD] Primary load failed: {first_error}")
        print("[LOAD] Attempting fallback load path")
        try:
            print("[LOAD] Loading tokenizer (fallback config)")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            print("[LOAD] Loading model (fallback config)")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,
            )
            model = model.to(selected_device)
            model.eval()
            model.config.use_cache = True
            print(
                f"[LOAD] Fallback load succeeded. device={model.device}, "
                f"dtype={model.dtype}"
            )
            return tokenizer, model, model_dir
        except Exception as second_error:
            print(f"[LOAD] Fallback load failed: {second_error}")
            raise RuntimeError(
                "Failed to load autobot-instruct model. "
                f"Primary error: {first_error}. Fallback error: {second_error}"
            ) from second_error
