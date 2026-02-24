"""
LLM interface for local model inference.

This interface uses only the model loader/generator scripts in models/:
- load-autobot-instruct.py
- generate-autobot-instruct.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_TOKENS_HARD_LIMIT = 4096
DEFAULT_MAX_CONTEXT_LENGTH = 32768
DEFAULT_TEMPERATURE = 0.3


@dataclass
class LoadedModel:
    name: str
    path: str
    tokenizer: Any
    model: Any


class LLMInterface:
    """Abstraction for local model loading and generation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_context_length = int(
            config.get("agentic", {}).get("max_context_length", DEFAULT_MAX_CONTEXT_LENGTH)
        )
        self.max_tokens_hard_limit = int(
            config.get("agentic", {}).get("max_tokens_hard_limit", DEFAULT_MAX_TOKENS_HARD_LIMIT)
        )

        self.project_root = Path(__file__).resolve().parents[1]
        self.models_root = self.project_root / "models"
        self.model_paths = self._resolve_model_paths()

        self.loaded_models: Dict[str, LoadedModel] = {}
        self.model_locks = {
            "autobot_instruct": asyncio.Lock(),
        }
        self.model_available = False

        self._modules: Dict[str, Any] = {}

    def _resolve_model_paths(self) -> Dict[str, str]:
        llm_cfg = self.config.get("llm", {})
        return {
            "autobot_instruct": llm_cfg.get("intent_model", {}).get(
                "local_path", "./models/autobot-instruct"
            ),
        }

    async def initialize(self) -> bool:
        """Load configured local models once at startup."""
        self._load_runtime_modules()

        loaded_any = False
        for model_name in ("autobot_instruct",):
            try:
                bundle = await self._load_single_model(model_name)
                self.loaded_models[model_name] = bundle
                loaded_any = True
                self.logger.info("Loaded model %s from %s", model_name, bundle.path)
            except Exception as exc:
                self.logger.exception("Failed loading model %s: %s", model_name, exc)

        self.model_available = loaded_any
        if not loaded_any:
            self.logger.error("No local models were loaded successfully.")
        return loaded_any

    def _load_runtime_modules(self):
        """Load loader/generator scripts from models/ only."""
        module_files = {
            "load_instruct": "load-autobot-instruct.py",
            "gen_instruct": "generate-autobot-instruct.py",
        }

        for alias, filename in module_files.items():
            script_path = self.models_root / filename
            if not script_path.exists():
                raise FileNotFoundError(f"Required model script not found: {script_path}")
            self._modules[alias] = self._load_python_module(alias, script_path)

    def _load_python_module(self, alias: str, script_path: Path):
        spec = importlib.util.spec_from_file_location(f"autobot_{alias}", str(script_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module spec for {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    async def _load_single_model(self, model_name: str) -> LoadedModel:
        if model_name != "autobot_instruct":
            raise ValueError(f"Unsupported model: {model_name}")

        loader = getattr(self._modules["load_instruct"], "load_autobot_instruct")
        tokenizer, model, model_dir = await asyncio.to_thread(
            loader,
            self.model_paths["autobot_instruct"],
            self.device,
        )
        if tokenizer is None or model is None:
            raise RuntimeError("load_autobot_instruct failed to load model")
        return LoadedModel(
            name=model_name,
            path=model_dir,
            tokenizer=tokenizer,
            model=model,
        )

    def get_loaded_models(self) -> List[str]:
        return list(self.loaded_models.keys())

    def _select_model(self, preferred: str) -> LoadedModel:
        # All known aliases resolve to the instruct model.
        alias_map = {
            "autobot-instruct": "autobot_instruct",
            "autobot_instruct": "autobot_instruct",
            "autobot-thinking": "autobot_instruct",
        }

        normalized = str(preferred or "").strip().lower()
        target = alias_map.get(normalized, "autobot_instruct")

        if target in self.loaded_models:
            return self.loaded_models[target]

        if "autobot_instruct" in self.loaded_models:
            return self.loaded_models["autobot_instruct"]

        raise RuntimeError("No loaded models are available.")

    async def generate_with_model(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tools_json: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if not self.model_available:
            raise RuntimeError("LLMInterface is not initialized with any model.")

        bundle = self._select_model(model_name)
        lock = self.model_locks.get(bundle.name)
        if lock is None:
            return await self._generate_for_bundle(
                bundle=bundle,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools_json=tools_json,
                messages=messages,
            )

        async with lock:
            return await self._generate_for_bundle(
                bundle=bundle,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools_json=tools_json,
                messages=messages,
            )

    async def generate_with_model_raw(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tools_json: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        """Return the raw generator payload for the selected model."""
        if not self.model_available:
            raise RuntimeError("LLMInterface is not initialized with any model.")

        bundle = self._select_model(model_name)
        lock = self.model_locks.get(bundle.name)

        async def _run_raw():
            safe_max_tokens = max(64, min(int(max_tokens), self.max_tokens_hard_limit))
            system_msg, user_msg = self._resolve_generation_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
            )

            if bundle.name != "autobot_instruct":
                raise ValueError(f"Unsupported model bundle: {bundle.name}")

            generator = getattr(self._modules["gen_instruct"], "generate_autobot_instruct")
            return await asyncio.to_thread(
                generator,
                bundle.model,
                bundle.tokenizer,
                system_msg,
                user_msg,
                self.device,
                self.max_context_length,
                safe_max_tokens,
                self.max_tokens_hard_limit,
                float(temperature),
                tools_json,
                messages,
            )

        if lock is None:
            return await _run_raw()

        async with lock:
            return await _run_raw()

    async def _generate_for_bundle(
        self,
        bundle: LoadedModel,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        tools_json: Optional[List[Dict[str, Any]]],
        messages: Optional[List[Dict[str, str]]],
    ) -> str:
        safe_max_tokens = max(64, min(int(max_tokens), self.max_tokens_hard_limit))
        system_msg, user_msg = self._resolve_generation_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
        )

        if bundle.name != "autobot_instruct":
            raise ValueError(f"Unsupported model bundle: {bundle.name}")

        generator = getattr(self._modules["gen_instruct"], "generate_autobot_instruct")
        result = await asyncio.to_thread(
            generator,
            bundle.model,
            bundle.tokenizer,
            system_msg,
            user_msg,
            self.device,
            self.max_context_length,
            safe_max_tokens,
            self.max_tokens_hard_limit,
            float(temperature),
            tools_json,
            messages,
        )
        return self._extract_text_from_generation(result)

    def _resolve_generation_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
    ) -> Tuple[str, str]:
        if not messages:
            return system_prompt or "You are AutoBot, a helpful assistant.", prompt

        system_msg = system_prompt or "You are AutoBot, a helpful assistant."
        user_msg = prompt

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", ""))
            if role == "system" and content:
                system_msg = content
            if role == "user" and content:
                user_msg = content

        return system_msg, user_msg

    def _extract_text_from_generation(self, result: Any) -> str:
        if isinstance(result, dict):
            for key in ("text", "final_answer", "answer", "output"):
                value = result.get(key)
                if value:
                    return str(value).strip()
            raw_text = result.get("raw_text")
            if raw_text:
                return str(raw_text).strip()
            # Never expose internal reasoning dicts when model returned no user-facing text.
            if any(key in result for key in ("think", "raw_think", "thought")):
                return ""
            return json.dumps(result, ensure_ascii=False, default=str).strip()
        if isinstance(result, str):
            return result.strip()
        return str(result).strip()

    async def generate_with_intent_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        intent_cfg = self.config.get("llm", {}).get("intent_model", {})
        max_tokens = int(intent_cfg.get("max_tokens", 512))
        temperature = float(intent_cfg.get("temperature", 0.2))
        return await self.generate_with_model(
            "autobot_instruct",
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_reasoning_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Reasoning uses the instruct model config.
        return await self.generate_with_intent_model(
            prompt,
            system_prompt=system_prompt,
        )

    async def generate_response(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        use_reasoning: bool = True,
    ) -> str:
        del context
        if use_reasoning:
            return await self.generate_with_reasoning_model(prompt)
        return await self.generate_with_intent_model(prompt)
