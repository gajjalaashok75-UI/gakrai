# `generate-autobot-instruct.py` - Complete Reference

## Scope
This document is only about `models/generate-autobot-instruct.py`.
It explains:
- exact function inputs and outputs
- exact `apply_chat_template(...)` usage
- first pass with 2 messages
- second pass with 4 messages after tool execution

## Function Signature
```python
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
```

## Input Contract
- `messages`:
  - if provided, this is the exact conversation history used for templating
  - if `None`, the function builds:
    ```python
    [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    ```
- `tools_json`:
  - optional list of tool schemas
  - if provided, function passes it to `apply_chat_template(..., tools=tools_json)`
  - if tokenizer does not accept `tools`, function catches `TypeError` and retries without `tools`

## Internal Flow
1. Build `chat_messages`.
2. Call `tokenizer.apply_chat_template(...)` with `tokenize=True` to get token count.
3. Call `tokenizer.apply_chat_template(...)` with `tokenize=False` to get the exact prompt string.
4. Tokenize the prompt with truncation (`max_length=max_context_length - max_tokens`).
5. Run `model.generate(...)`.
6. Decode only the newly generated tokens and return payload.

## Exact `apply_chat_template` Calls Used
```python
token_count_kwargs = {"tokenize": True, "add_generation_prompt": True}
template_kwargs = {"tokenize": False, "add_generation_prompt": True}

if tools_json:
    token_count_kwargs["tools"] = tools_json
    template_kwargs["tools"] = tools_json

template_tokens = tokenizer.apply_chat_template(chat_messages, **token_count_kwargs)
formatted_prompt = tokenizer.apply_chat_template(chat_messages, **template_kwargs)
```

`add_generation_prompt=True` is critical: it appends the assistant header so generation starts in assistant role.

## Chat Template Behavior Used by This Model
From `models/autobot-instruct/chat_template.jinja`:
- takes first `system` message out of `messages`
- if `tools` is provided, appends `List of tools: [...]` to system prompt
- renders each message as:
  - `<|im_start|>{role}\n{content}<|im_end|>\n`
- if `add_generation_prompt=True`, appends:
  - `<|im_start|>assistant\n`

## First Generation Pass: 2 Messages

### Messages Sent
```python
messages = [
    {"role": "system", "content": system_message_content},
    {"role": "user", "content": user_content},
]
```

### Prompt Representation (when `tools_json` is provided)
```text
<|im_start|>system
<system_message_content>
List of tools: [{"name":"web_search",...}]<|im_end|>
<|im_start|>user
<user_content><|im_end|>
<|im_start|>assistant
```

### Prompt Representation (when `tools_json` is not provided)
```text
<|im_start|>system
<system_message_content><|im_end|>
<|im_start|>user
<user_content><|im_end|>
<|im_start|>assistant
```

### Possible Outcomes
- No tool needed: model returns normal answer text.
- Tool needed: model can emit tool-call text (for example, your pipeline may enforce markers like `<|tool_call_start|>...<|tool_call_end|>`).

## Second Generation Pass After Tool Execution: 4 Messages
When first pass asks for a tool, caller executes the tool and calls this function again with:

```python
messages = [
    {"role": "system", "content": system_message_content},
    {"role": "user", "content": user_content},
    {"role": "assistant", "content": raw_tool_call_text},
    {"role": "tool", "content": tool_output_text},
]
```

### Prompt Representation
```text
<|im_start|>system
<system_message_content>
List of tools: [{"name":"web_search",...}]<|im_end|>
<|im_start|>user
<user_content><|im_end|>
<|im_start|>assistant
<raw_tool_call_text><|im_end|>
<|im_start|>tool
<tool_output_text><|im_end|>
<|im_start|>assistant
```

## Why the 4-Message Form Is Required
- message 1 (`system`): policy and tool instructions
- message 2 (`user`): original request
- message 3 (`assistant`): exact tool decision made by model
- message 4 (`tool`): external evidence returned by the tool

With this history, the final generation is grounded and reproducible.

## Return Payload
The function returns:
- `text`: decoded output after removing selected special tokens
- `raw_text`: raw decoded model output
- `template_token_count`: token count from `apply_chat_template(..., tokenize=True)`
- `formatted_prompt`: exact template string sent to tokenizer/model
- `input_length`: prompt token length after tokenization
- `generated_tokens`: number of newly generated tokens

## Minimal Debug Example
```python
result = generate_autobot_instruct(
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

print(result["formatted_prompt"])
print(result["template_token_count"])
print(result["raw_text"])
print(result["text"])
```

## Runtime Working (Message Set Notation)
1. First generation always starts with `messages = {1,2}`.
`1` = `{"role": "system", "content": system_message_content}`
`2` = `{"role": "user", "content": user_content}`

2. If no tool call is produced, generation ends on this first pass.
No-tool path stays `messages = {1,2}`.

3. If a tool call is produced, the runtime appends the assistant tool-call text and the tool result, then generates again with `messages = {1,2,3,4}`.
`3` = `{"role": "assistant", "content": raw_tool_call_text}`
`4` = `{"role": "tool", "content": tool_output_text}`

4. Second pass with `messages = {1,2,3,4}` is the grounded-answer pass because tool evidence is now inside conversation history.

## Important Boundaries
- This function does not execute tools.
- This function does not detect tool calls.
- Tool orchestration is done by the caller; this file only formats prompt, generates, and returns outputs.
