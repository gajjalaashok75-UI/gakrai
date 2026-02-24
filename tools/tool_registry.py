"""
Tool Registry for AutoBot.
Manages and executes tool integrations.
"""

import asyncio
import importlib
import json
import logging
import re
from typing import Any, Callable, Dict, Optional

tools_json = [
    {
        "name": "web_search",
        "description": "Search the web for current information, facts, or recent updates.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results to fetch",
                    "default": 5,
                    "minimum": 1,
                },
                "workers": {
                    "type": "integer",
                    "description": "Parallel workers used by the search pipeline",
                    "default": 6,
                    "minimum": 1,
                },
            },
            "required": ["query"],
        },
    },
]


class ToolRegistry:
    """Registry for currently supported tools."""

    _TOOL_ALIASES = {
        "websearch": "web_search",
        "web-search": "web_search",
        "search_web": "web_search",
        "search": "web_search",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, Any] = {}
        self._web_search_runner: Optional[Callable[..., Any]] = None

    @classmethod
    def _normalize_tool_name(cls, tool_name: str) -> str:
        normalized = (tool_name or "").strip().lower().replace("-", "_").replace(" ", "_")
        return cls._TOOL_ALIASES.get(normalized, normalized)

    def _load_web_search_runner(self) -> Callable[..., Any]:
        """Load web search runner lazily to avoid import-time hard failures."""
        if self._web_search_runner is not None:
            return self._web_search_runner

        candidates = ("tools.web_search.search", "web_search.search", "search")
        import_errors = []

        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
                runner = getattr(module, "run_search", None)
                if callable(runner):
                    self._web_search_runner = runner
                    self.logger.info("Web search runner loaded from %s", module_name)
                    return runner
                import_errors.append(f"{module_name}: run_search missing")
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                import_errors.append(f"{module_name}: {exc}")

        joined_errors = "; ".join(import_errors) if import_errors else "No candidates attempted"
        raise ImportError(f"Unable to import web search runner. Attempts: {joined_errors}")

    @staticmethod
    def _coerce_positive_int(value: Any, default: int) -> int:
        try:
            coerced = int(value)
            return coerced if coerced > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_query(query_value: Any) -> str:
        query = str(query_value or "").strip()
        if not query:
            return ""

        # Handle malformed payloads like: query=hot news in India today, max_results=5
        if query.lower().startswith("query="):
            query = query.split("=", 1)[1].strip()
        query = re.split(r",\s*max_results?\s*=\s*\d+", query, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        return query.strip("\"'")

    async def initialize(self):
        """Load enabled tools."""
        enabled_tools = self.config.get("tools", {}).get("enabled", ["web_search"])
        enabled_normalized = {self._normalize_tool_name(name) for name in enabled_tools}

        try:
            if "web_search" in enabled_normalized:
                self.tools["web_search"] = {
                    "type": "web_search",
                    "runner": self._load_web_search_runner,  # Lazy loader wrapper
                    "initialized": True,
                    "lazy": True,
                }
                self.logger.info("Loaded tool: web_search with lazy wrapper")

            self.logger.info("Tool registry initialized with tools: %s", sorted(self.tools.keys()))

        except Exception as exc:
            self.logger.error("Error initializing tools: %s", exc)

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters."""
        tool_name_normalized = self._normalize_tool_name(tool_name)

        if tool_name_normalized not in self.tools or self.tools[tool_name_normalized] is None:
            raise ValueError(f"Tool {tool_name_normalized} not found or not initialized")

        tool_config = self.tools[tool_name_normalized]
        tool_type = tool_config.get("type")

        try:
            if tool_type == "web_search":
                return await self._execute_web_search(tool_config, params)
            raise ValueError(f"Unknown tool type: {tool_type}")

        except Exception as exc:
            self.logger.error("Tool execution failed (%s): %s", tool_name, exc)
            return json.dumps(
                {"status": "error", "tool": tool_name_normalized, "error": str(exc)},
                indent=2,
            )

    async def _execute_web_search(self, tool_config: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Execute web search using search engine."""
        try:
            query = self._normalize_query(params.get("query", ""))
            max_results = self._coerce_positive_int(params.get("max_results"), default=1)
            workers = self._coerce_positive_int(params.get("workers"), default=6)

            if not query:
                return json.dumps(
                    {"status": "error", "tool": "web_search", "error": "Missing query parameter"},
                    indent=2,
                )

            runner_factory = tool_config.get("runner")
            if not callable(runner_factory):
                raise RuntimeError("web_search runner is not callable")
            runner = runner_factory() if tool_config.get("lazy") else runner_factory
            if not callable(runner):
                raise RuntimeError("web_search runner factory did not return a callable")

            print(f"Executing tool: 'web_search' with args: {{'query': '{query}'}}")

            # Run search in thread pool to avoid blocking (search is synchronous)
            loop = asyncio.get_event_loop()
            run_result = await loop.run_in_executor(
                None,
                runner,
                query,
                max_results,
                workers,
            )

            if isinstance(run_result, tuple) and len(run_result) == 2:
                results, stats = run_result
            else:
                results, stats = run_result, {}

            result_count = len(results) if isinstance(results, list) else 0
            print(
                "📈 Total chars in results from search pipeline Before formated "
                f"( from tools.py ): {result_count}"
            )
            print(f"✅ Got {result_count} results from search pipeline")
            print(
                "📈 Total chars in results from search pipeline after formated"
                f"( from tools.py): {result_count}"
            )
            print("Tool 'web_search' executed successfully")

            return json.dumps(
                {
                    "status": "success",
                    "query": query,
                    "final_query_used": query,
                    "results_count": len(results) if isinstance(results, list) else 0,
                    "stats": stats,
                    "results": results if isinstance(results, list) else [],
                },
                indent=2,
                default=str,
            )

        except Exception as exc:
            self.logger.error("Web search execution error: %s", exc)
            return json.dumps(
                {"status": "error", "tool": "web_search", "error": str(exc)},
                indent=2,
            )

    async def shutdown(self):
        """Shutdown all tools."""
        self.logger.info("Shutting down tool registry")
        self.tools.clear()
        self._web_search_runner = None
