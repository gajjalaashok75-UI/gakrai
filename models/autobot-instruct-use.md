# `autobot-instruct-use.md` - End-to-End Usage Guide

## Scope
This file is an independent usage guide for this model pipeline:
- load model/tokenizer
- run first pass with 2 messages
- branch for no-tool vs tool-call flow
- run second pass with 4 messages when tool output exists

## Why `apply_chat_template(...)` Is the Core
`apply_chat_template(...)` converts structured messages into the exact ChatML prompt expected by the model.
For this tokenizer/template:
- messages become `<|im_start|>role\ncontent<|im_end|>`
- `add_generation_prompt=True` appends `<|im_start|>assistant\n`
- when `tools` is provided, tool schemas are injected into the system prompt as `List of tools: [...]`

## Step 1: Load Functions from Local Files
The filenames contain `-`, so regular Python import syntax is not valid.
Use `importlib.util`:

```python
from pathlib import Path
import importlib.util

ROOT = Path.cwd()

def load_function(py_path: Path, fn_name: str):
    spec = importlib.util.spec_from_file_location(f"{fn_name}_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec from: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn_name)

load_autobot_instruct = load_function(
    ROOT / "models" / "load-autobot-instruct.py",
    "load_autobot_instruct",
)
generate_autobot_instruct = load_function(
    ROOT / "models" / "generate-autobot-instruct.py",
    "generate_autobot_instruct",
)
```

## Step 2: Load Tokenizer and Model
```python
tokenizer, model, model_dir = load_autobot_instruct()
device = str(model.device)
```

## Step 3: Define Tools and Messages
```python
tools_json = [
    {
        "name": "web_search",
        "description": "Search the web for current facts and updates.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    }
]

system_message_content = (
    "You are a precise assistant. "
    "If fresh external data is required, output only a tool call."
)
user_content = "What are the latest EV charging connector updates?"
```

## Step 4: First Pass (2 Messages)

### Two Messages Sent
```python
messages = [
    {"role": "system", "content": system_message_content},
    {"role": "user", "content": user_content},
]
```

### How `apply_chat_template` Is Applied
```python
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools_json,
)
```

### Representation
```text
<|im_start|>system
<system_message_content>
List of tools: [{"name":"web_search",...}]<|im_end|>
<|im_start|>user
<user_content><|im_end|>
<|im_start|>assistant
```

## Step 5: Generate First Response
```python
first = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message=system_message_content,
    user_prompt=user_content,
    device=device,
    max_context_length=32768,
    max_tokens=512,
    max_tokens_hard_limit=24000,
    temperature=0.7,
    tools_json=tools_json,
    messages=messages,
)
```

## Step 6A: No Tool Call Path
If first response is normal assistant text, return:
```python
final_answer = first["text"]
```

## Step 6B: Tool Call Path
If first response contains a tool call (example):
```text
<|tool_call_start|>web_search(query="latest EV charging connector updates")<|tool_call_end|>
```

execute tool, then append two more messages.

### Four Messages Sent in Second Pass
```python
messages = [
    {"role": "system", "content": system_message_content},
    {"role": "user", "content": user_content},
    {
        "role": "assistant",
        "content": "<|tool_call_start|>web_search(query=\"latest EV charging connector updates\")<|tool_call_end|>",
    },
    {"role": "tool", "content": tool_output_text},
]
```

### `apply_chat_template` Representation (Second Pass)
```text
<|im_start|>system
<system_message_content>
List of tools: [{"name":"web_search",...}]<|im_end|>
<|im_start|>user
<user_content><|im_end|>
<|im_start|>assistant
<|tool_call_start|>web_search(query="latest EV charging connector updates")<|tool_call_end|><|im_end|>
<|im_start|>tool
<tool_output_text><|im_end|>
<|im_start|>assistant
```

### Why the 4 Messages Matter
- preserves exact model decision (`assistant` tool call)
- injects external evidence (`tool` result)
- produces grounded final assistant answer in the same turn context

## Complete Runnable Example
```python
import re

def parse_tool_call(raw_text: str):
    m = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", raw_text, re.DOTALL)
    if not m:
        return None
    payload = m.group(1).strip()
    fn = re.match(r"^([a-zA-Z_]\w*)\((.*)\)$", payload)
    if not fn:
        return None
    tool_name = fn.group(1)
    args_text = fn.group(2)
    q = re.search(r'query\s*=\s*"([^"]*)"', args_text)
    args = {"query": q.group(1)} if q else {}
    return {"tool_name": tool_name, "args": args, "raw_tool_call": m.group(0)}

def run_tool(tool_name: str, args: dict) -> str:
    if tool_name == "web_search":
        query = args.get("query", "")
        # Replace with your real tool implementation.
        return f"Mock result for query: {query}"
    return "Unsupported tool."

messages = [
    {"role": "system", "content": system_message_content},
    {"role": "user", "content": user_content},
]

first = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message=system_message_content,
    user_prompt=user_content,
    device=device,
    max_context_length=32768,
    max_tokens=512,
    max_tokens_hard_limit=24000,
    temperature=0.7,
    tools_json=tools_json,
    messages=messages,
)

decision = parse_tool_call(first["raw_text"])
if decision is None:
    final_answer = first["text"]
else:
    tool_output_text = run_tool(decision["tool_name"], decision["args"])
    messages.append({"role": "assistant", "content": decision["raw_tool_call"]})
    messages.append({"role": "tool", "content": tool_output_text})

    second = generate_autobot_instruct(
        model=model,
        tokenizer=tokenizer,
        system_message=system_message_content,
        user_prompt=user_content,
        device=device,
        max_context_length=32768,
        max_tokens=512,
        max_tokens_hard_limit=24000,
        temperature=0.7,
        tools_json=tools_json,
        messages=messages,
    )
    final_answer = second["text"]

print(final_answer)
```

## Debug and Verification
Use `formatted_prompt` from generator output to verify exact structure:
```python
print(first["formatted_prompt"])
print(first["template_token_count"])
print(first["raw_text"])
```

## Practical Summary
- first turn always starts with 2 messages: `system`, `user`
- if no tool call appears, return first response
- if tool call appears, append 2 messages (`assistant`, `tool`) and regenerate with 4-message history
- `apply_chat_template(..., add_generation_prompt=True)` controls the exact prompt shape in both paths

## Runtime Working (Message Set Notation)
1. Initial pass uses `messages = {1,2}`.
`1` = `{"role": "system", "content": system_message_content}`
`2` = `{"role": "user", "content": user_content}`

2. If the model answers directly, stop there.
No-tool route remains `messages = {1,2}`.

3. If the model requests a tool, execute it and append two messages, then run second pass with `messages = {1,2,3,4}`.
`3` = `{"role": "assistant", "content": raw_tool_call_text}`
`4` = `{"role": "tool", "content": tool_output_text}`

4. The final grounded response is generated only after `messages = {1,2,3,4}` is templated with `add_generation_prompt=True`.

## References
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
- https://docs.liquid.ai/lfm/key-concepts/chat-template
- https://docs.liquid.ai/lfm/key-concepts/tool-use
- https://huggingface.co/docs/transformers/en/chat_templating#using-applychattemplate
