"""Tool-call detection and parsing for Autobot Instruct outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

SPECIAL_TOKENS = [
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|startoftext|>",
    "<|tool_list_start|>",
    "<|tool_list_end|>",
]


def _strip_special_tokens(text: str) -> str:
    cleaned = text
    for token in SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _normalize_tool_name(name: str) -> str:
    if not name:
        return ""
    # Convert to lowercase, replace spaces/dashes with underscores
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    
    # Handle all variations of web_search and search
    if "websearch" in normalized:
        return "web_search"
    if "web" in normalized and "search" in normalized:
        return "web_search"
    if "search" in normalized and "web" in normalized:
        return "web_search"
    if normalized == "search":  # Plain 'search' -> 'web_search'
        return "web_search"
    
    return normalized


def _extract_tool_payload(text: str) -> Optional[Tuple[str, str]]:
    start_token = "<|tool_call_start|>"
    end_token = "<|tool_call_end|>"

    if start_token in text:
        print("[TOOL] Tool call marker detected in generated text")
        start_pos = text.find(start_token) + len(start_token)
        end_pos = text.find(end_token, start_pos)
        if end_pos != -1:
            payload = text[start_pos:end_pos].strip()
        else:
            payload = text[start_pos:].strip()
        
        # Remove outer brackets if present: [web_search(...)] → web_search(...)
        # Also handle: [{"name": "search", ...}] → {"name": "search", ...}
        while (payload.startswith("[") and payload.endswith("]")) or \
              (payload.startswith("(") and payload.endswith(")")):
            payload = payload[1:-1].strip()
        
        raw_tool_call = f"{start_token}{payload}{end_token}"
        return payload, raw_tool_call
    
    # Fallback heuristics for payloads without explicit markers
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, f"{start_token}{stripped}{end_token}"

    if "(" in stripped and ")" in stripped:
        return stripped, f"{start_token}{stripped}{end_token}"

    return None


def _parse_args(args_text: str) -> Dict[str, Any]:
    """Parse function call arguments like: query="value" or query='value' or just "value" """
    args: Dict[str, Any] = {}
    
    if not args_text.strip():
        return args

    # Try named arguments: query="value", max=5, etc.
    # Handles both single and double quotes
    for key, quoted in re.findall(r'([a-zA-Z_]\w*)\s*=\s*(\'[^\']*\'|"[^"]*")', args_text):
        # Remove quotes
        args[key] = quoted[1:-1]

    if args:
        return args

    # Try positional argument (single quoted string)
    positional = re.match(r'^\s*(\'[^\']*\'|"[^"]*")\s*$', args_text)
    if positional:
        # Remove quotes
        args["query"] = positional.group(1)[1:-1]
        return args
    
    # Fallback: treat entire args_text as query if it looks like a value
    args_text = args_text.strip()
    if args_text:
        args["query"] = args_text

    return args


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    extracted = _extract_tool_payload(text)
    if not extracted:
        return None

    payload, raw_tool_call = extracted
    payload = payload.strip()
    print(f"[TOOL] Parsing payload preview: {payload[:120]}")

    # If we have <|tool_call_start|> marker, we're 100% sure this is a tool call
    # So use aggressive extraction strategies
    has_marker = "<|tool_call_start|>" in raw_tool_call
    
    if has_marker:
        print("[TOOL] Tool call marker found - using aggressive extraction")
        
        # Step 1: Try to extract query using regex (handles both quoted and unquoted values)
        # Matches: query="value" or query=value (with value ending at comma/paren/space)
        query_match = re.search(r'query\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|([^,)\s]+(?:\s+[^,)]+)*))', payload, re.IGNORECASE)
        if query_match:
            query = query_match.group(1) or query_match.group(2) or query_match.group(3)
            if query:
                query = query.strip()
                print(f"[TOOL] Extracted query: {query}")
                
                # Step 2: Extract or infer tool name
                # Look for tool name patterns: web_search, websearch, search, search_web, etc.
                tool_name = _extract_tool_name_from_payload(payload)
                
                args = {"query": query}
                
                # Also extract max_results if present
                max_results_match = re.search(r'max_results?\s*=\s*(\d+)', payload, re.IGNORECASE)
                if max_results_match:
                    args["max_results"] = int(max_results_match.group(1))
                
                if tool_name:
                    print(f"[TOOL] Aggressive extraction successful: tool={tool_name}, query={query}")
                    return {
                        "tool_name": tool_name,
                        "args": args,
                        "raw_tool_call": raw_tool_call,
                    }

    # Fall back to JSON parsing if no marker or aggressive extraction failed
    if payload.startswith("{") and payload.endswith("}"):
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict) and "tool_call" in obj:
                tool_call = obj.get("tool_call") or {}
                name = _normalize_tool_name(str(tool_call.get("name", "")))
                args = tool_call.get("args") or {}
                if isinstance(args, dict) and name:
                    return {
                        "tool_name": name,
                        "args": args,
                        "raw_tool_call": raw_tool_call,
                    }
            if isinstance(obj, dict) and "name" in obj:
                name = _normalize_tool_name(str(obj.get("name", "")))
                args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
                if name:
                    return {
                        "tool_name": name,
                        "args": args,
                        "raw_tool_call": raw_tool_call,
                    }
        except Exception as json_err:
            print(f"[TOOL] JSON parse failed: {json_err}, trying regex fallbacks")
            # Continue to regex-based extraction if JSON fails

    # Fallback: Try to extract tool_name and query using regex for malformed JSON
    # Look for "tool_name": "value" pattern
    tool_name_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', payload)
    if not tool_name_match:
        # Try alternate pattern: "search": "value" as tool name
        search_match = re.search(r'"search"\s*:\s*"([^"]+)"', payload)
        if search_match:
            # If we find a search pattern with a value, it might be the query
            # but we'll mark tool as web_search
            tool_name = "web_search"
        else:
            tool_name = None
    else:
        tool_name = _normalize_tool_name(tool_name_match.group(1))
    
    if tool_name:
        # Look for "query": "value" or similar arguments
        query_match = re.search(r'"query"\s*:\s*"([^"]+)"', payload)
        args = {}
        if query_match:
            args["query"] = query_match.group(1)
        else:
            # Try to find any string value as fallback
            first_string_match = re.search(r':\s*"([^"]+)"', payload)
            if first_string_match:
                args["query"] = first_string_match.group(1)
        
        # Look for max_results/max_Results patterns
        max_results_match = re.search(r'"max_[Rr]esults?"\s*:\s*(\d+)', payload)
        if max_results_match:
            args["max_results"] = int(max_results_match.group(1))
        
        # Look for any other key-value pairs
        for key, value in re.findall(r'"([a-zA-Z_]+)"\s*:\s*"([^"]*)"', payload):
            if key.lower() not in ["tool_name", "name", "tool_cale", "search"]:  # Skip redundant keys
                args[key] = value
        
        if tool_name or args:
            if not tool_name:
                tool_name = "web_search"  # Default to web_search
            print(f"[TOOL] Extracted via regex: tool_name={tool_name}, args={args}")
            return {
                "tool_name": tool_name,
                "args": args,
                "raw_tool_call": raw_tool_call,
            }

    # Try function call syntax: tool_name(args) or "web search"(query="...")
    # Handle tool names with spaces by matching up to the first (
    paren_pos = payload.find("(")
    if paren_pos != -1:
        tool_name_raw = payload[:paren_pos].strip()
        args_text = payload[paren_pos+1:]
        
        # Find matching closing paren
        paren_close = args_text.rfind(")")
        if paren_close != -1:
            args_text = args_text[:paren_close].strip()
        
        name = _normalize_tool_name(tool_name_raw)
        args = _parse_args(args_text)
        
        if name:
            return {
                "tool_name": name,
                "args": args,
                "raw_tool_call": raw_tool_call,
            }
    
    return None


def _extract_tool_name_from_payload(text: str) -> Optional[str]:
    """Extract tool name from payload text, handling various naming conventions."""
    # Handle function call syntax: web_search(...) or search(...) etc.
    paren_pos = text.find("(")
    if paren_pos > 0:
        potential_name = text[:paren_pos].strip()
        # Remove brackets/quotes
        potential_name = potential_name.strip("[]\"'")
        normalized = _normalize_tool_name(potential_name)
        if normalized:
            return normalized
    
    # Look for specific tool name patterns
    # Check for variants: web_search, websearch, search_web, web search, etc.
    variants = [
        (r'\bweb[_\s]search\b', 'web_search'),
        (r'\bwebsearch\b', 'web_search'),
        (r'\bsearch[_\s]web\b', 'web_search'),
        (r'\bsearch\b', 'web_search'),  # Default 'search' to 'web_search'
    ]
    
    for pattern, tool in variants:
        if re.search(pattern, text, re.IGNORECASE):
            return tool
    
    return None


def detect_tool_call(generation_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return {'type':'tool'|'no_tool', 'args':...} from generator output."""
    # STRICTLY use raw_text (the final response). Tool calls should appear in
    # the final answer, NOT in internal thinking. Do not check raw_think.
    raw_text = str(generation_result.get("raw_text", "")).strip()
    if not raw_text:
        # Fallback to text if raw_text is missing
        raw_text = str(generation_result.get("text", "")).strip()
    
    print(f"[TOOL] Detecting tool call from raw_text preview: {raw_text[:120]!r}")

    parsed_tool = _parse_tool_call(raw_text)
    if parsed_tool:
        print(
            f"[TOOL] Tool detected: name={parsed_tool['tool_name']}, "
            f"args={parsed_tool.get('args', {})}"
        )
        return {
            "type": "tool",
            "args": {
                "tool_name": parsed_tool["tool_name"],
                "args": parsed_tool.get("args", {}),
                "raw_tool_call": parsed_tool["raw_tool_call"],
                "raw_text": raw_text,
                "template_token_count": generation_result.get("template_token_count", 0),
                "formatted_prompt": generation_result.get("formatted_prompt", ""),
                "input_length": generation_result.get("input_length", 0),
                "generated_tokens": generation_result.get("generated_tokens", 0),
            },
        }

    print("[TOOL] No tool call detected; returning normal response payload")
    no_tool_payload = {
        "text": str(generation_result.get("text", _strip_special_tokens(raw_text))).strip(),
        "raw_text": raw_text,
        "template_token_count": generation_result.get("template_token_count", 0),
        "formatted_prompt": generation_result.get("formatted_prompt", ""),
        "input_length": generation_result.get("input_length", 0),
        "generated_tokens": generation_result.get("generated_tokens", 0),
    }
    return {
        "type": "no_tool",
        "args": no_tool_payload,
    }
