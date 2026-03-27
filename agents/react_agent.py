#!/usr/bin/env python3
"""
Reactive Echo Agent

Implements a LangGraph-based ReAct loop where the LLM decides whether to
invoke tools and receives their results before continuing the conversation.
"""

from __future__ import annotations

import json
import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated

# Set UTF-8 encoding for Windows console to handle Unicode characters
if sys.platform == 'win32':
    import io
    # Set stdout and stderr to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Also set environment variable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph


def _safe_str(obj: Any) -> str:
    """Convert object to string safely, handling Unicode encoding errors."""
    try:
        return str(obj)
    except UnicodeEncodeError:
        # If str() fails, try encoding to ASCII with replacement
        try:
            return obj.encode('ascii', 'replace').decode('ascii')
        except (AttributeError, TypeError):
            # If it's not a string-like object, convert to repr and then safe encode
            try:
                return repr(obj).encode('ascii', 'replace').decode('ascii')
            except Exception:
                return "<unable to convert to string>"


class ToolCallLog(TypedDict):
    """Structured record of an executed tool call."""

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class EchoAgentState(TypedDict):
    """State carried through the LangGraph execution."""

    messages: Annotated[List[AnyMessage], operator.add]


class ReactiveEchoAgent:
    """
    Minimal ReAct-style agent.

    The agent delegates decision making to the bound language model. Whenever
    the model emits tool calls, the specified LangChain tools are executed and
    their `ToolMessage` responses are appended to the conversation history
    before handing control back to the model.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        *,
        system_prompt: str = "",
        checkpointer: Any = None,
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        tool_cache: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._log_tools = log_tools
        self._log_dir = Path(log_dir or "logs")
        if self._log_tools:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tool result cache (shared across perspectives)
        self._tool_cache = tool_cache if tool_cache is not None else {}

        # Prepare LangGraph workflow
        workflow = StateGraph(EchoAgentState)
        workflow.add_node("process", self._process_request)
        workflow.add_node("execute", self._execute_tools)
        workflow.add_conditional_edges("process", self._has_tool_calls, {True: "execute", False: END})
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(list(self.tools.values()))

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def update_system_prompt(self, prompt: str) -> None:
        """Set a new system prompt for subsequent runs."""
        self._system_prompt = prompt

    # -- LangGraph node implementations -------------------------------------------------
    def _process_request(self, state: Dict[str, Any]) -> Dict[str, List[AnyMessage]]:
        messages: List[AnyMessage] = list(state.get("messages", []))
        if self._system_prompt:
            messages = [SystemMessage(content=self._system_prompt)] + messages

        response = self.model.invoke(messages)
        
        # Debug: Log if response has tool calls
        tool_calls = getattr(response, "tool_calls", [])
        if tool_calls:
            print(f"[DEBUG] Model made {len(tool_calls)} tool call(s): {[tc.get('name') for tc in tool_calls]}")
        else:
            content_preview = str(getattr(response, "content", ""))[:200]
            print(f"[DEBUG] Model response has no tool calls. Content preview: {content_preview}...")
        
        return {"messages": [response]}

    def _has_tool_calls(self, state: Dict[str, Any]) -> bool:
        last_message = state["messages"][-1]
        return bool(getattr(last_message, "tool_calls", []))

    def _execute_tools(self, state: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        tool_messages: List[ToolMessage] = []
        for call in state["messages"][-1].tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})
            tool_id = call.get("id", "")

            if tool_name not in self.tools:
                result_content = json.dumps(
                    {"status": "error", "error": f"Unknown tool '{tool_name}'"}, ensure_ascii=False
                )
            else:
                # Create cache key from tool name and sorted args
                cache_key = self._make_cache_key(tool_name, tool_args)
                
                # Check cache first (for deterministic tools)
                if cache_key in self._tool_cache:
                    print(f"[CACHE HIT] {tool_name} with args {tool_args}")
                    result = self._tool_cache[cache_key]
                    try:
                        result_content = json.dumps(result, ensure_ascii=False, default=str)
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        result_content = json.dumps(result, ensure_ascii=True, default=_safe_str)
                else:
                    # Cache miss - invoke tool
                    try:
                        result = self.tools[tool_name].invoke(tool_args)
                        # Store in cache (for deterministic tools)
                        if self._is_cacheable_tool(tool_name):
                            self._tool_cache[cache_key] = result
                            print(f"[CACHE STORE] {tool_name} with args {tool_args}")
                        
                        # Tool results can be complex objects; coerce to JSON string if possible.
                        try:
                            result_content = json.dumps(result, ensure_ascii=False, default=str)
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            # If JSON encoding fails due to Unicode, use ASCII-safe encoding
                            result_content = json.dumps(result, ensure_ascii=True, default=_safe_str)
                    except Exception as exc:  # noqa: BLE001
                        # Safely convert exception to string
                        try:
                            error_msg = f"{type(exc).__name__}: {exc}"
                        except UnicodeEncodeError:
                            error_msg = f"{type(exc).__name__}: {_safe_str(exc)}"
                        result_content = json.dumps(
                            {"status": "error", "error": error_msg}, ensure_ascii=True
                        )

            message = ToolMessage(
                tool_call_id=tool_id,
                name=tool_name or "unknown_tool",
                content=result_content,
                additional_kwargs={"args": tool_args},
            )
            tool_messages.append(message)

        self._log_tool_messages(tool_messages)
        return {"messages": tool_messages}
    
    def _make_cache_key(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Create a cache key from tool name and arguments."""
        # Sort args to ensure consistent keys
        sorted_args = json.dumps(tool_args, sort_keys=True, default=str)
        return f"{tool_name}:{sorted_args}"
    
    def _is_cacheable_tool(self, tool_name: str) -> bool:
        """Determine if a tool's results should be cached."""
        # Cache deterministic tools that produce the same output for same inputs
        cacheable_tools = {
            "echo_knowledge_graph",  # Same question = same guidance
            "echo_rag_guidelines",    # Same query = same retrieval
            "echo_view_classification",  # Same video dir = same classification
            "echo_measurement_prediction",  # Same video = same measurements
            "echo_disease_prediction",  # Same video = same disease predictions
        }
        return tool_name in cacheable_tools

    # -- Helpers ------------------------------------------------------------------------
    def _log_tool_messages(self, tool_messages: List[ToolMessage]) -> None:
        if not self._log_tools or not tool_messages:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = self._log_dir / f"tool_calls_{timestamp}.json"
        logs: List[ToolCallLog] = []
        for message in tool_messages:
            logs.append(ToolCallLog(
                tool_call_id=message.tool_call_id,
                name=message.name,
                args=message.additional_kwargs.get("args", {}),
                content=message.content,
                timestamp=datetime.utcnow().isoformat(),
            ))

        log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
