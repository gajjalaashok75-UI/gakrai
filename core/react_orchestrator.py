"""
ReAct Orchestrator for AutoBot
Implements the Reason + Act pattern with iterative reasoning and action loops.

Architecture:
- Agent Node: Reasons about the problem and decides on next action
- Tool Node: Executes tools (web search, etc.)
- Loop: Tool results feed back to agent for next reasoning step
- Summarizer: Uses autobot-instruct to summarize tool results

Models:
- autobot-instruct: For reasoning and action decisions

Features:
- Tool detection using tool-detector for parsing outputs
- Dynamic tool selection based on available tools
- Enhanced ReAct system prompt with clear instructions
- Multi-step reasoning with tool integration
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Annotated

from pydantic import BaseModel, Field

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import AnyMessage, add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    AnyMessage = Any
    add_messages = None

from core.llm_interface import LLMInterface
from memory.memory_manager import MemoryManager
from tools.tool_registry import ToolRegistry, tools_json as TOOL_SCHEMA
from tools.tool_detector import detect_tool_call


@dataclass
class ReActResult:
    """Result from ReAct agent execution."""
    response: str
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    execution_time: float = 0.0


class AgentState(BaseModel):
    """State structure for the ReAct graph."""
    messages: list = Field(default_factory=list)  # Simplified without Annotated
    user_input: str = ""
    step_count: int = 0
    max_steps: int = 10


class ReActOrchestrator:
    """ReAct (Reason + Act) agent orchestrator."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.llm = LLMInterface(config)
        self.memory = MemoryManager(config)
        self.tool_registry = ToolRegistry(config)
        
        self.max_steps = int(
            config.get("agentic", {}).get("react_max_steps", 8)
        )
        self.debug_enabled = bool(config.get("debug", {}).get("enabled", True))
        
        self.execution_trace: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, str]] = []
        
        self.project_root = Path(__file__).resolve().parents[1]
        self.react_graph = None
        self.react_ready = False

    async def initialize(self) -> bool:
        """Initialize ReAct orchestrator."""
        self.logger.info("Initializing ReAct Orchestrator...")
        
        try:
            await self.llm.initialize()
            await self.tool_registry.initialize()
            await self.memory.initialize()
            
            if LANGGRAPH_AVAILABLE:
                self._build_react_graph()
                self.react_ready = True
                self.logger.info("ReAct graph compiled successfully")
            else:
                self.logger.warning("LangGraph not available - using fallback ReAct")
                self.react_ready = True
            
            return True
        except Exception as exc:
            self.logger.exception("Failed to initialize ReAct: %s", exc)
            return False

    @staticmethod
    def _strip_tool_markup(text: str) -> str:
        """Remove tool call tokens so failed tool calls are not shown to end users."""
        if not text:
            return ""
        cleaned = re.sub(
            r"<\|tool_call_start\|>.*?<\|tool_call_end\|>",
            "",
            text,
            flags=re.DOTALL,
        )
        return cleaned.strip()

    async def _generate_tool_server_error_response(
        self,
        user_input: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        attempts: int,
        error_detail: str = "",
    ) -> str:
        """Generate a user-facing response when a tool repeatedly fails."""
        query = str(tool_args.get("query", "")).strip()
        detail = (error_detail or "Tool returned no usable data.").strip()

        prompt = (
            "Tool server error occurred while handling the user request.\n"
            f"- User request: {user_input}\n"
            f"- Tool: {tool_name}\n"
            f"- Query: {query}\n"
            f"- Attempts: {attempts}\n"
            f"- Error detail: {detail}\n\n"
            "Write a short response to the user that:\n"
            "1. Explains the tool is temporarily unavailable.\n"
            "2. States the request could not be completed right now.\n"
            "3. Asks them to retry shortly.\n"
            "Do not output tool calls."
        )

        try:
            fallback = await self.llm.generate_with_model(
                "autobot-instruct",
                prompt,
                system_prompt=(
                    "You are an assistant handling tool-failure responses. "
                    "Be concise, factual, and do not fabricate search results."
                ),
                temperature=0.1,
                max_tokens=180,
            )
            cleaned = self._strip_tool_markup(str(fallback or ""))
            if cleaned:
                return cleaned
        except Exception as exc:
            self.logger.exception("Failed to generate tool server error response: %s", exc)

        return (
            "Tool server error: web search is temporarily unavailable after multiple attempts. "
            "Please try again in a moment."
        )

    def _build_react_graph(self):
        """Build the ReAct graph with agent and tool nodes."""
        if not LANGGRAPH_AVAILABLE:
            return
        
        # Define agent node
        def agent_node(state: Any) -> Dict[str, Any]:
            # Support both plain dict state (fallback) and Pydantic AgentState used by LangGraph
            if isinstance(state, dict):
                messages = state.get("messages", [])
                user_input = state.get("user_input", "")
                step_count = state.get("step_count", 0)
            else:
                messages = getattr(state, "messages", []) or []
                user_input = getattr(state, "user_input", "") or ""
                step_count = getattr(state, "step_count", 0) or 0
            
            # Enhanced ReAct system prompt with comprehensive instructions
            available_tools = ", ".join(sorted(self.tool_registry.tools.keys())) if self.tool_registry.tools else "none"
            
            system_prompt = (
                "You are AutoBot, a sophisticated ReAct (Reason + Act) agent powered by advanced language understanding.\n\n"
                "## Your Core Responsibilities\n"
                "1. **Understand Intent**: Deeply analyze the user's query to identify their true goal and context.\n"
                "2. **Reason Step-by-Step**: Break down complex problems into logical, clear steps.\n"
                "3. **Take Deliberate Actions**: Use available tools only when necessary to fulfill the user's needs.\n"
                "4. **Synthesize Knowledge**: Integrate information from tools and your training to provide comprehensive answers.\n"
                "5. **Deliver Clear Responses**: Present final answers in a clear, well-structured, and user-focused manner.\n\n"
                "## Available Actions\n"
                f"- **Tools**: {available_tools}\n"
                "- **Direct Response**: Provide answers without tools when you have sufficient knowledge.\n"
                "- **Clarification**: Ask for clarification if the user's intent is ambiguous.\n\n"
                "## Decision Framework\n"
                "**Use tools when:**\n"
                "- Current information is needed (weather, stock prices, news)\n"
                "- Verification of facts is required\n"
                "- The user explicitly requests external data\n"
                "- Your knowledge is insufficient or outdated\n\n"
                "**Provide direct answers when:**\n"
                "- You have sufficient knowledge to provide a complete answer\n"
                "- The query is conceptual or requires reasoning\n"
                "- Tool use would add unnecessary latency without benefit\n\n"
                "## Output Format\n"
                "- **Thought**: Internal reasoning about the approach (not shown to user)\n"
                "- **Action** (if needed): Tool invocation with parameters\n"
                "- **Final Answer**: Clear, comprehensive response addressing user intent\n\n"
                "## Guidelines for Tool Calls\n"
                "When calling tools, use this format:\n"
                "<|tool_call_start|>[web_search(query=\"optimized search query written by model\")]<|tool_call_end|>\n"
                "Replace 'optimized search query written by model' with your actual search query.\n\n"
                "## Core Principles\n"
                "- Prioritize accuracy and completeness over speed\n"
                "- Never fabricate tool results or data\n"
                "- Always cite sources or tool results when used\n"
                "- Be transparent about limitations and uncertainties\n"
                "- Focus on user value, not process documentation\n\n"
                "Remember: Your goal is to provide intelligent, helpful, and accurate assistance that directly addresses the user's needs."
            )
            
            full_messages = [("system", system_prompt)]
            if isinstance(messages, list):
                full_messages.extend(messages)
            else:
                full_messages.append(("user", str(messages)))

            # Generate thought/decision using the instruct model alias (autobot-instruct)
            # Use raw generator payload so we can detect tool calls inside raw_text
            response_payload = asyncio.run(
                self.llm.generate_with_model_raw(
                    "autobot-instruct",
                    user_input if step_count == 0 else "",
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=1024,
                    tools_json=TOOL_SCHEMA,
                )
            )

            # Normalize to assistant content string if generator returned a dict
            if isinstance(response_payload, dict):
                response = response_payload.get("raw_text") or response_payload.get("text") or ""
            else:
                response = str(response_payload or "")
            
            return {
                "messages": [{"role": "assistant", "content": response}],
                "step_count": step_count + 1,
            }
        
        # Build the graph
        react_graph_builder = StateGraph(AgentState)
        react_graph_builder.add_node("agent", agent_node)
        
        # Add tool node if tools are available
        if self.tool_registry.tools:
            def tool_node_fn(state: Any) -> Dict[str, Any]:
                # Support dict or AgentState
                if isinstance(state, dict):
                    last_msg = state.get("messages", [{}])[-1]
                else:
                    last_msg = (getattr(state, "messages", []) or [{}])[-1]

                # last_msg may be a dict or a simple string
                if isinstance(last_msg, dict):
                    tool_name = last_msg.get("tool_name", "")
                    tool_input = last_msg.get("tool_input", {})
                else:
                    tool_name = ""
                    tool_input = {}

                if tool_name and tool_name in self.tool_registry.tools:
                    result = asyncio.run(
                        self.tool_registry.execute_tool(tool_name, tool_input)
                    )
                    return {"messages": [{"role": "tool", "content": str(result)}]}
                return {"messages": [{"role": "tool", "content": "Tool not found"}]}
            
            react_graph_builder.add_node("tools", tool_node_fn)
        
        react_graph_builder.set_entry_point("agent")
        
        # Add routing logic
        def should_continue(state: Any) -> str:
            if isinstance(state, dict):
                messages = state.get("messages", [])
                step_count = state.get("step_count", 0)
                max_steps = state.get("max_steps", self.max_steps)
            else:
                messages = getattr(state, "messages", []) or []
                step_count = getattr(state, "step_count", 0) or 0
                max_steps = getattr(state, "max_steps", self.max_steps) or self.max_steps
            
            if step_count >= max_steps:
                return "__end__"
            
            if messages and isinstance(messages[-1], dict):
                last_msg = messages[-1]
                if last_msg.get("tool_name"):
                    return "tools"
            
            return "__end__"
        
        react_graph_builder.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "__end__": "__end__"}
        )
        
        if self.tool_registry.tools:
            react_graph_builder.add_edge("tools", "agent")
        
        self.react_graph = react_graph_builder.compile()
        self.logger.info("ReAct graph with loop compiled")

    async def handle_input(self, user_input: str) -> str:
        """Process user input through ReAct agent."""
        start_time = time.time()
        trace_id = f"react_{int(start_time * 1000)}"
        
        try:
            reasoning_steps = []
            tool_calls = []
            
            # Use LangGraph if available
            if self.react_ready and self.react_graph:
                response, steps, calls = await self._run_langgraph_react(
                    user_input, reasoning_steps, tool_calls
                )
            else:
                response, steps, calls = await self._run_fallback_react(
                    user_input, reasoning_steps, tool_calls
                )

            # Persist only the user query and final response to short-term memory
            try:
                await self.memory.add_short_term_interaction(user_input, response)
            except Exception:
                self.logger.exception("Failed to store short-term interaction")
            
            # Store in history
            self.chat_history.append({"user": user_input, "assistant": response})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
            # Store trace
            self.execution_trace.append({
                "trace_id": trace_id,
                "flow": "react",
                "reasoning_steps": steps,
                "tool_calls": calls,
                "total_steps": len(steps),
                "duration_sec": round(time.time() - start_time, 3),
                "timestamp": time.time(),
            })
            if len(self.execution_trace) > 100:
                self.execution_trace = self.execution_trace[-100:]
            
            return response
        
        except Exception as exc:
            self.logger.exception("ReAct handle_input failed: %s", exc)
            return f"Error processing request: {str(exc)}"

    async def _run_langgraph_react(
        self,
        user_input: str,
        reasoning_steps: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run ReAct using fallback execution to preserve 2/4 message tool flow."""
        return await self._run_fallback_react(user_input, reasoning_steps, tool_calls)

    async def _run_fallback_react(
        self,
        user_input: str,
        reasoning_steps: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fallback ReAct implementation without LangGraph.
        
        Flow:
        1. First pass with 2 messages (system, user)
        2. If instruct answers directly -> return response
        3. If instruct calls tool (web_search):
           - Execute web search once
           - Second pass with 4 messages:
             (system, user, assistant tool call, tool result)
           - Return grounded final answer
        """
        available_tools = ", ".join(sorted(self.tool_registry.tools.keys())) if self.tool_registry.tools else "none"

        # Step 1: Send query to instruct model
        system_prompt = (
            "You are AutoBot, a sophisticated ReAct (Reason + Act) agent.\n\n"
            "## Your Core Responsibilities\n"
            "1. Understand Intent: Deeply analyze the user's goal and context.\n"
            "2. Reason Step-by-Step: Break down complex problems into logical steps.\n"
            "3. Take Deliberate Actions: Use tools when necessary.\n"
            "4. Synthesize Knowledge: Integrate tool results with your training.\n"
            "5. Deliver Clear Responses: Present answers in a user-focused manner.\n\n"
            f"## Available Tools: {available_tools}\n\n"
            "## Decision Framework\n"
            "**Use tools for:**\n"
            "- Current, real-time information (news, prices, weather)\n"
            "- Fact verification and validation\n"
            "- User explicitly requests external data\n\n"
            "**Provide direct answers for:**\n"
            "- You have sufficient knowledge in training data\n"
            "- Conceptual or theoretical questions\n"
            "- When tool use would add unnecessary latency\n\n"
            "## Tool Call Format\n"
            "When calling tools, use this format:\n"
            "<|tool_call_start|>[web_search(query=\"optimized search query written by model\")]<|tool_call_end|>\n"
            "Replace 'optimized search query written by model' with your actual search query.\n\n"
            "Focus on providing intelligent, helpful, and accurate assistance."
        )
        first_pass_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        # Get instruct model's response
        agent_payload = await self.llm.generate_with_model_raw(
            "autobot-instruct",
            user_input,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1024,
            tools_json=TOOL_SCHEMA,
            messages=first_pass_messages,
        )

        # Extract response text
        if isinstance(agent_payload, dict):
            agent_response = agent_payload.get("text") or agent_payload.get("final_answer") or agent_payload.get("raw_text") or ""
        else:
            agent_response = str(agent_payload or "")
        cleaned_agent_response = self._strip_tool_markup(agent_response)
        
        reasoning_steps.append({
            "step": 1,
            "type": "thought",
            "content": agent_response,
        })

        # Step 2: Detect if instruct model wants to use a tool
        if isinstance(agent_payload, dict):
            detection_result = detect_tool_call(agent_payload)
        else:
            detection_result = detect_tool_call({"raw_text": agent_response, "text": agent_response})
        
        tool_detected = detection_result.get("type") == "tool"
        
        # If no tool is called, return instruct model's answer directly
        if not tool_detected:
            self.logger.info("Instruct model answered directly - no tool call needed")
            return (cleaned_agent_response or agent_response.strip()), reasoning_steps, tool_calls
        
        # Step 3: Tool was called - extract tool info and execute
        tool_info = detection_result.get("args", {})
        tool_name = tool_info.get("tool_name", "")
        tool_args = tool_info.get("args", {})
        
        self.logger.info(f"Tool detected: {tool_name} with args: {tool_args}")
        
        # Verify tool exists in registry
        if tool_name not in self.tool_registry.tools:
            self.logger.warning(f"Detected tool '{tool_name}' not in registry")
            reasoning_steps.append({
                "step": 1,
                "type": "error",
                "content": f"Tool '{tool_name}' not available",
            })
            fallback_response = cleaned_agent_response or (
                "I could not access the requested tool right now. Please try again."
            )
            return fallback_response, reasoning_steps, tool_calls
        
        # Step 4: Execute web search
        try:
            self.logger.info(f"Executing web search for: {tool_args.get('query', '')}")
            
            search_result = await self.tool_registry.execute_tool(
                tool_name,
                tool_args
            )

            tool_calls.append({
                "step": 1,
                "tool": tool_name,
                "args": tool_args,
            })

            # Parse result
            try:
                result_data = json.loads(search_result)
                if result_data.get("status") == "error":
                    error_msg = str(result_data.get("error", "unknown error"))
                    self.logger.error(f"Web search error: {error_msg}")
                    reasoning_steps.append({
                        "step": 2,
                        "type": "error",
                        "content": f"Web search failed: {error_msg}",
                    })
                    fallback_response = await self._generate_tool_server_error_response(
                        user_input=user_input,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        attempts=1,
                        error_detail=error_msg,
                    )
                    return fallback_response, reasoning_steps, tool_calls

                results_count = result_data.get("results_count", 0)
                self.logger.info(f"Web search returned {results_count} results")
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Could not parse tool result: {str(e)}")
                reasoning_steps.append({
                    "step": 2,
                    "type": "error",
                    "content": f"Failed to parse search results: {str(e)}",
                })
                fallback_response = await self._generate_tool_server_error_response(
                    user_input=user_input,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    attempts=1,
                    error_detail=f"Parse error: {str(e)}",
                )
                return fallback_response, reasoning_steps, tool_calls
        
        except Exception as exc:
            self.logger.exception(f"Tool execution failed: {tool_name} - {str(exc)}")
            reasoning_steps.append({
                "step": 2,
                "type": "error",
                "content": f"Web search execution failed: {str(exc)}",
            })
            fallback_response = await self._generate_tool_server_error_response(
                user_input=user_input,
                tool_name=tool_name,
                tool_args=tool_args,
                attempts=1,
                error_detail=str(exc),
            )
            return fallback_response, reasoning_steps, tool_calls
        
        # Step 5: Second pass with 4 messages (system, user, assistant tool call, tool result)
        try:
            assistant_tool_call = str(tool_info.get("raw_tool_call", "")).strip()
            if not assistant_tool_call:
                assistant_tool_call = str(tool_info.get("raw_text", "")).strip()
            if not assistant_tool_call:
                assistant_tool_call = str(agent_response).strip()

            second_pass_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_tool_call},
                {"role": "tool", "content": str(search_result)},
            ]
            
            self.logger.info("Generating grounded final response with 4-message tool context")
            final_answer = await self.llm.generate_with_model(
                "autobot-instruct",
                user_input,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1024,
                tools_json=TOOL_SCHEMA,
                messages=second_pass_messages,
            )

            reasoning_steps.append({
                "step": 2,
                "type": "grounded_answer",
                "content": (
                    "Instruct model generated final answer from tool context "
                    f"(results={results_count})"
                ),
            })
            
            self.logger.info("Grounded final response generated")
            return self._strip_tool_markup(final_answer).strip(), reasoning_steps, tool_calls
        
        except Exception as exc:
            self.logger.exception(f"Instruct second-pass generation failed: {str(exc)}")
            # Fallback to instruct answer
            fallback_response = cleaned_agent_response or (
                "I retrieved results but could not complete the final response right now. Please retry."
            )
            return fallback_response, reasoning_steps, tool_calls

    async def _synthesize_response(
        self,
        user_input: str,
        reasoning_steps: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
    ) -> str:
        """Synthesize final response from reasoning steps."""
        context = f"User Query: {user_input}\n\n"
        context += f"Reasoning Steps: {len(reasoning_steps)}\n"
        context += f"Tool Calls: {len(tool_calls)}\n\n"
        
        for step in reasoning_steps[-3:]:  # Last 3 steps for context
            context += f"Step {step['step']}: {step['content'][:200]}\n"
        
        synthesis_prompt = (
            "Provide a clear, concise, and well-structured final answer to the user's query. "
            "Do not reveal internal reasoning, chain-of-thought, or mention tools or observations. "
            "Answer directly and professionally."
        )

        final_response = await self.llm.generate_with_model(
            "autobot-instruct",
            synthesis_prompt,
            system_prompt=context,
            temperature=0.2,
            max_tokens=1000,
        )
        
        return final_response.strip()

    def _extract_search_query(self, thought: str, user_input: str) -> str:
        """Extract search query from agent thought."""
        # Try to find quoted query
        if '"' in thought:
            parts = thought.split('"')
            if len(parts) > 1:
                return parts[1].strip()
        
        # Fallback to user input
        return user_input

    async def get_debug_status(self, limit: int = 10) -> Dict[str, Any]:
        """Get debug information about ReAct execution."""
        return {
            "flow": "react",
            "react_ready": self.react_ready,
            "loaded_models": self.llm.get_loaded_models(),
            "tools": sorted(self.tool_registry.tools.keys()),
            "max_steps": self.max_steps,
            "recent_trace": self.execution_trace[-max(1, limit):],
        }
