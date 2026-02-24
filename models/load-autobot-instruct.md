# load-autobot-instruct.py — Documentation

## Purpose
Utilities to discover and load the local `autobot-instruct` model directory and return a ready-to-use tokenizer and causal LM model instance.

## Public API
- `load_autobot_instruct(base_dir: Optional[str] = None, device: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]`

## Description
`load_autobot_instruct` resolves a local model directory (checks common candidate paths), loads the tokenizer and model via Hugging Face Transformers, ensures the model is placed on the requested device, and returns `(tokenizer, model, resolved_model_dir)`.

## Inputs
- `base_dir` (Optional[str]) — Optional root path used to resolve where `autobot-instruct` is stored. If not provided, the code uses the repo root relative to the `models` package.
- `device` (Optional[str]) — Target device string. Common values: `'cuda'`, `'cpu'`. If omitted, function prefers `'cuda'` when `torch.cuda.is_available()`.

## Outputs / Return value
- `tokenizer` (`transformers.AutoTokenizer`) — tokenizer loaded from the model directory.
- `model` (`transformers.AutoModelForCausalLM`) — model loaded and moved to the target device. Model is set to `.eval()` and `model.config.use_cache = True`.
- `resolved_model_dir` (str) — absolute path to the model directory discovered and used.

## Behavior and details
- The loader looks for a `config.json` file inside these candidate directories (in order):
  - `.../<root>/models/autobot-instruct`
  - `.../<root>/models`
- Primary load path uses `AutoTokenizer.from_pretrained(model_dir, padding_side='left', use_fast=True, trust_remote_code=False)` and `AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=..., device_map=...)`.
- If `device` is `'cuda'`, loader uses `torch.bfloat16` for `torch_dtype` and sets `device_map='auto'` to leverage model parallelism where available. For non-CUDA it uses `torch.float32` and moves the model to the selected device explicitly.
- On failure of the primary load path, a fallback path attempts simpler `from_pretrained` calls with `torch.float32` and no special kwargs.

## Errors and side effects
- Raises `FileNotFoundError` if no candidate model directory containing `config.json` is found.
- Raises `RuntimeError` when both primary and fallback load attempts fail; the exception message includes both underlying errors.
- Emits `print()` logging messages describing each phase (resolve, tokenizer load, model load, fallback attempts).

## Usage example
```python
from models.load_autobot_instruct import load_autobot_instruct

tokenizer, model, model_dir = load_autobot_instruct(base_dir=None, device=None)
# `tokenizer` is a Transformers tokenizer; `model` is a Causal LM ready for `.generate()`
```

## Notes
- After loading, `model.eval()` is called and `model.config.use_cache = True` is set.
- Tokenizer objects in this repository may include helper functionality such as `apply_chat_template` used by the generator utilities.
