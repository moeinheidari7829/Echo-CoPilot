#!/usr/bin/env python3
"""
Reactive Intelligent Agent for EchoPilot

This agent mirrors the MedRAX ReAct loop: the language model itself decides
whether to call a tool, issues the tool invocation through OpenAI function
calling, receives the result, and continues reasoning until it chooses to
answer the user directly.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Ensure project root is available on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.react_agent import ReactiveEchoAgent
from config import Config
from tools import (
    EchoDiseasePredictionTool,
    EchoImageVideoGenerationTool,
    EchoKnowledgeGraphTool,
    EchoMeasurementPredictionTool,
    EchoNetMeasurementTool,
    EchoRAGTool,
    EchoReportGenerationTool,
    EchoSegmentationTool,
    EchoViewClassificationTool,
)


DEFAULT_SYSTEM_PROMPT = """
You are EchoPilot, an expert echocardiography assistant who can reason about ultrasound videos as a senior sonographer.
You MUST use tools to answer questions about echocardiography videos. Do not just describe what you would do—actually call the tools.

CRITICAL: When a user asks a question about an echocardiography video, you MUST call at least one tool to gather information before responding. Do not respond with text alone—use the available tools.

Core behaviours:
- ALWAYS call tools when asked about video analysis. Never respond without calling tools first.
- Choose tools according to the user's request: segmentation for structural assessment, measurement prediction for quantitative metrics, disease prediction for pathology screening, view classification for orientation, report generation for structured summaries, and image/video generation only when explicitly asked for synthetic data.
- Critically examine every tool output; if something looks inconsistent, call the tool again, request another tool, or explain the limitation before responding.
- If the user's request lacks essential context (e.g., no video path, or only a general question), ask for clarification or explain why the tools cannot run.
- Never fabricate numbers—quote quantitative values (EF, chamber sizes, velocities) exactly as reported by the tools and mention units.
- Always mention when data, image quality, or missing views limit your confidence.
- You may make multiple sequential tool calls. It is acceptable to plan briefly, then call tools in the order that best supports your reasoning.
- After receiving tool results, summarize how they influence your conclusion before deciding whether to call another tool or answer.

Tool invocation rules:
- Pass the absolute `video_path` for tools that consume a single file.
- Pass the parent directory as `input_dir` for tools that process multiple videos.
- You may call multiple tools sequentially or in parallel; however, avoid unnecessary tool runs—call only what you need to answer confidently.
- IMPORTANT: Use the tool calling format—do not describe tool usage in text. Actually invoke the tools.

Response style:
- Start with a concise clinical summary grounded in the tool findings.
- Present key measurements (e.g., EF, chamber sizes) and qualitative observations (e.g., wall motion, regurgitation) with references to the tools used.
- Highlight urgent or abnormal findings first, then include supportive details.
- Finish with next steps or recommendations when clinically appropriate.
- Keep the tone professional, informative, and focused on echocardiography.
"""

PERSPECTIVE_1_STRUCTURAL_PROMPT = """
You are EchoPilot, an expert echocardiography assistant focusing on STRUCTURAL ANALYSIS.

CRITICAL: You MUST call at least ONE tool before responding. Do NOT respond with text alone.

PRIORITY TOOLS (call in this order):
1. echo_knowledge_graph - ALWAYS call this FIRST to understand which measurements are appropriate
2. echo_disease_prediction - For disease patterns and structural findings
3. echonet_measurement - For direct structural measurements (wall thickness, chamber dimensions)
4. echo_view_classification - For view identification

MANDATORY WORKFLOW:
1. Call echo_knowledge_graph with the question
2. Based on KG guidance, call the appropriate measurement or disease prediction tool
3. Provide answer based on tool results

FOCUS AREAS:
- Anatomical structures and morphology
- Wall thickness and chamber dimensions
- Structural abnormalities
- Valve structure and appearance

ANSWER FORMAT:
Base your answer ONLY on tool outputs. Quote specific values and measurements.

REMEMBER: Call tools BEFORE responding. No tools = invalid response.
"""

PERSPECTIVE_2_PATHOLOGICAL_PROMPT = """
You are EchoPilot, an expert echocardiography assistant focusing on PATHOLOGICAL ANALYSIS.

CRITICAL: You MUST call at least ONE tool before responding. Do NOT respond with text alone.

PRIORITY TOOLS (call in this order):
1. echo_knowledge_graph - ALWAYS call this FIRST to understand disease assessment approach
2. echo_disease_prediction - For disease patterns and pathological findings (PRIMARY TOOL)
3. echo_report_generation - For comprehensive clinical reports

MANDATORY WORKFLOW:
1. Call echo_knowledge_graph with the question
2. Call echo_disease_prediction on the video directory
3. Provide answer based on disease prediction scores

FOCUS AREAS:
- Disease patterns and pathological findings
- Clinical indicators and disease classification
- Pathological severity assessments

ANSWER FORMAT:
Base your answer ONLY on tool outputs. Quote disease prediction scores and confidence values.

REMEMBER: Call tools BEFORE responding. No tools = invalid response.
"""

PERSPECTIVE_3_QUANTITATIVE_PROMPT = """
You are EchoPilot, an expert echocardiography assistant focusing on QUANTITATIVE ANALYSIS.

CRITICAL: You MUST call at least ONE tool before responding. Do NOT respond with text alone.

PRIORITY TOOLS (call in this order):
1. echo_knowledge_graph - ALWAYS call this FIRST to understand measurement approach
2. echonet_measurement - For direct measurements (wall thickness, chamber dimensions, Doppler velocities)
3. echo_report_generation - For EF and other quantitative metrics

MANDATORY WORKFLOW:
1. Call echo_knowledge_graph with the question
2. Based on guidance, call measurement or report generation tool
3. Provide answer based on numerical values from tools

FOCUS AREAS:
- Numerical values and measurements
- Ejection fraction, volumes, dimensions
- Quantitative thresholds and ranges

ANSWER FORMAT:
Base your answer ONLY on tool outputs. Quote specific numerical values with units.

REMEMBER: Call tools BEFORE responding. No tools = invalid response.
"""


class AnalysisType(Enum):
    SIMPLE = "simple"
    SEGMENTATION = "segmentation"
    MEASUREMENTS = "measurements"
    DISEASE = "disease"
    REPORT = "report"
    GENERATION = "generation"
    MULTI_TOOL = "multi_tool"


class AnalysisComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class QueryAnalysis:
    query: str
    analysis_types: List[AnalysisType] = field(default_factory=list)
    complexity: AnalysisComplexity = AnalysisComplexity.LOW
    required_tools: List[str] = field(default_factory=list)
    recommended_tools: List[str] = field(default_factory=list)
    use_got: bool = False
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ToolExecutionResult:
    success: bool
    results: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    tools_used: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response returned by the reactive intelligent agent."""

    success: bool
    query: str
    analysis: QueryAnalysis
    execution_result: ToolExecutionResult
    response_text: str
    confidence: float
    execution_time: float


class IntelligentAgent:
    """MedRAX-style ReAct agent wrapped for the EchoPilot interface."""

    def __init__(
        self,
        device: str = Config.DEVICE,
        *,
        base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        temperature: float = Config.OPENAI_TEMPERATURE,
        max_tokens: int = Config.OPENAI_MAX_TOKENS,
        model: str = Config.OPENAI_MODEL,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.device = device
        self._base_system_prompt = base_system_prompt.strip()
        self.conversation_history: List[AgentResponse] = []
        self._log_dir = log_dir or (PROJECT_ROOT / "logs")

        if not Config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required to run the intelligent agent.")

        # Optional reasoning controls for models that support them (e.g., gpt-5, o3-mini).
        # If not explicitly provided, fall back to Config values.
        effective_reasoning_effort = reasoning_effort or getattr(
            Config, "OPENAI_REASONING_EFFORT", None
        )
        effective_reasoning_summary = reasoning_summary or getattr(
            Config, "OPENAI_REASONING_SUMMARY", None
        )

        model_kwargs: Dict[str, Any] = {}
        reasoning_cfg: Dict[str, Any] = {}
        if effective_reasoning_effort:
            reasoning_cfg["effort"] = effective_reasoning_effort
        if effective_reasoning_summary:
            reasoning_cfg["summary"] = effective_reasoning_summary
        if reasoning_cfg:
            model_kwargs["reasoning"] = reasoning_cfg

        self.llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=Config.OPENAI_BASE_URL,
            # Always pass a dict for model_kwargs so langchain_core.utils._build_model_kwargs
            # never receives a None value (which causes a TypeError when iterated).
            model_kwargs=model_kwargs,
        )

        # Initialize contrast LLM for Self-Contrast (smaller model, lower temperature)
        self.contrast_llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=Config.CONTRAST_LLM_MODEL,
            temperature=0.3,  # Lower temperature for more consistent contrasting
            max_tokens=2000,
            base_url=Config.OPENAI_BASE_URL,
        )

        self.tools = self._initialize_tools()

        # Shared tool cache across all perspectives (reduces duplicate API calls)
        # This cache stores results from deterministic tools (KG, RAG, view classification)
        # so that when multiple perspectives call the same tool with the same args,
        # they reuse the cached result instead of making duplicate API/model calls.
        self._shared_tool_cache: Dict[str, Any] = {}

        self.reactive_agent = ReactiveEchoAgent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._base_system_prompt,
            log_tools=True,
            log_dir=PROJECT_ROOT / "logs",
            tool_cache=self._shared_tool_cache,
        )

        # Create contrast agent with RAG and KG tool access
        # RAG and KG are available to the contrast LLM for validation and guidance
        try:
            rag_tool = EchoRAGTool()
            kg_tool = EchoKnowledgeGraphTool()
            contrast_tools = [rag_tool, kg_tool]
            self.contrast_agent = ReactiveEchoAgent(
                model=self.contrast_llm,
                tools=contrast_tools,
                system_prompt=(
                    "You are an expert echocardiography guidelines assistant. "
                    "Use the echo_rag_guidelines tool to query clinical guidelines, measurement thresholds, "
                    "and severity classifications when needed to verify, categorize, or interpret findings "
                    "from the main agent's analysis. "
                    "Use the echo_knowledge_graph tool to validate which measurements should be used for "
                    "different question types and to ensure correct measurement selection."
                ),
                log_tools=True,  # Enable logging to track RAG and KG usage
                log_dir=PROJECT_ROOT / "logs",
            )
            print(f"[RAG] RAG and KG tools available to contrast LLM")
        except Exception as exc:
            print(f"[WARNING] Failed to initialize contrast agent with RAG: {exc}")
            self.contrast_agent = None

        print("Intelligent ReAct Agent initialized")
        print(f"   Device: {device}")
        print(f"   Tools loaded: {', '.join(tool.name for tool in self.tools)}")

        # Get measurement tool config
        measurement_tool_config = Config.MEASUREMENT_TOOL.lower()
        print(f"   Measurement tool: {measurement_tool_config}")

        # Only initialize EchoPrime if it's being used as a measurement tool
        if measurement_tool_config in ["echoprime", "both"]:
            from models.echo.echo_prime_manager import EchoPrimeManager
            echo_prime_manager = EchoPrimeManager()
            echo_prime_manager._initialize_model()
            # Only download models if they don't exist
            if not echo_prime_manager._check_models_exist():
                echo_prime_manager._download_models()
            print(f"   EchoPrime initialized: Yes")
        else:
            print(f"   EchoPrime initialized: No (using {measurement_tool_config})")
    # --------------------------------------------------------------------- Public API
    def process_query(
        self,
        query: str,
        video_path: str,
        context: Optional[Dict[str, Any]] = None,
        use_self_contrast: Optional[bool] = None,
    ) -> AgentResponse:
        """Run the ReAct loop for a single query/video pair.

        Args:
            query: User's question
            video_path: Path to the video file
            context: Optional context dictionary
            use_self_contrast: Whether to use Self-Contrast (defaults to Config.USE_SELF_CONTRAST)
        """
        # Determine if self-contrast should be used
        if use_self_contrast is None:
            use_self_contrast = Config.USE_SELF_CONTRAST

        if use_self_contrast:
            return self.process_query_with_self_contrast(query, video_path, context)
        else:
            return self._process_query_normal(query, video_path, context)

    def _process_query_normal(
        self,
        query: str,
        video_path: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Run the normal ReAct loop for a single query/video pair."""
        start_time = time.time()
        context = context or {}
        video_path = os.path.abspath(video_path)
        video_dir = os.path.dirname(video_path)

        dynamic_prompt = self._compose_system_prompt(video_path, video_dir, context)
        self.reactive_agent.update_system_prompt(dynamic_prompt)

        conversation_seed = self._build_seed_messages(query, video_path, video_dir, context)
        final_state = self.reactive_agent.workflow.invoke({"messages": conversation_seed})
        message_history: List[BaseMessage] = list(final_state.get("messages", []))

        final_response = self._extract_final_response(message_history)
        tool_outputs, tools_used, tool_errors = self._collect_tool_outputs(message_history)
        react_trace = self._build_react_trace(message_history)

        success = bool(final_response) and not tool_errors
        analysis = self._build_analysis(query, tools_used, success)

        # Optional post-hoc reasoning text that explains how the answer
        # was derived from tool outputs. This issues an additional LLM call
        # but does not affect the main ReAct loop.
        reasoning_text = self._build_reasoning_text(
            query=query,
            video_path=video_path,
            tools_used=tools_used,
            tool_results=tool_outputs,
            final_response=final_response,
        )

        results_payload = {
            "analysis_type": "react_loop",
            "complexity": analysis.complexity.value,
            "tools_used": tools_used,
            "tool_results": tool_outputs,
            "reasoning": analysis.reasoning,
            "reasoning_text": reasoning_text,
            "final_response": final_response,
            "tool_errors": tool_errors,
            "trace": react_trace,
        }

        execution_result = ToolExecutionResult(
            success=success,
            results=results_payload,
            error=None if success else self._summarize_errors(tool_errors, final_response),
            execution_time=time.time() - start_time,
            tools_used=tools_used,
        )

        agent_response = AgentResponse(
            success=success,
            query=query,
            analysis=analysis,
            execution_result=execution_result,
            response_text=final_response or "I could not produce an answer.",
            confidence=analysis.confidence,
            execution_time=execution_result.execution_time,
        )

        self.conversation_history.append(agent_response)
        self._display_results(agent_response)

        # Automatically save trajectory
        self.save_trajectory(agent_response, video_path, context)

        return agent_response

    def save_trajectory(
        self,
        agent_response: AgentResponse,
        video_path: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save the full trajectory of the agent's execution to a JSON file.

        Args:
            agent_response: The agent response containing all execution details
            video_path: Path to the video file that was analyzed
            context: Optional context dictionary

        Returns:
            Path to the saved trajectory file
        """
        from datetime import datetime

        log_dir = self._log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trajectory_path = log_dir / f"trajectory_{timestamp}.json"

        # Extract all relevant information
        execution_result = agent_response.execution_result
        results = execution_result.results

        trajectory = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "query": agent_response.query,
                "video_path": video_path,
                "success": agent_response.success,
                "execution_time_seconds": agent_response.execution_time,
                "confidence": agent_response.confidence,
            },
            "context": context or {},
            "analysis": {
                "type": results.get("analysis_type"),
                "complexity": results.get("complexity"),
                "reasoning": results.get("reasoning"),
            },
            "tools_used": results.get("tools_used", []),
            "tool_results": results.get("tool_results", {}),
            "tool_errors": results.get("tool_errors", []),
            "react_trace": results.get("trace", []),
            "final_response": results.get("final_response"),
            "reasoning_text": results.get("reasoning_text"),
        }

        # Add self-contrast data if present
        if "self_contrast" in results:
            trajectory["self_contrast"] = results["self_contrast"]

        # Save to JSON file
        trajectory_path.write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )

        print(f"Trajectory saved to: {trajectory_path}")
        return trajectory_path

    def get_conversation_history(self) -> List[AgentResponse]:
        return self.conversation_history

    def clear_history(self) -> None:
        self.conversation_history.clear()
        print("Conversation history cleared")

    # --------------------------------------------------------------------- Internals
    def _initialize_tools(self) -> List[Any]:
        """Instantiate the LangChain tool objects used by the agent."""
        # Determine which measurement tool(s) to use based on config
        measurement_tool_config = Config.MEASUREMENT_TOOL.lower()

        measurement_tools = []
        if measurement_tool_config == "echoprime":
            measurement_tools = [EchoMeasurementPredictionTool]
        elif measurement_tool_config == "echonet":
            measurement_tools = [EchoNetMeasurementTool]
        elif measurement_tool_config == "both":
            measurement_tools = [EchoMeasurementPredictionTool, EchoNetMeasurementTool]
        else:
            print(f"[WARNING] Unknown MEASUREMENT_TOOL config: {Config.MEASUREMENT_TOOL}. Using EchoNet.")
            measurement_tools = [EchoNetMeasurementTool]

        tool_classes = [
            # EchoSegmentationTool,  # Excluded - not necessary for these types of tasks
            *measurement_tools,  # Dynamic measurement tool selection
            EchoDiseasePredictionTool,
            EchoReportGenerationTool,
            EchoViewClassificationTool,
            EchoImageVideoGenerationTool,
            EchoKnowledgeGraphTool,  # Available to main agent for measurement guidance
            # Note: EchoRAGTool is NOT included here - it's only available to the contrast LLM
        ]

        tools: List[Any] = []
        for cls in tool_classes:
            try:
                tool = cls()
                tools.append(tool)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARNING] Failed to initialize {cls.__name__}: {exc}")
        if not tools:
            raise RuntimeError("No tools could be initialized for the intelligent agent.")
        return tools

    def _compose_system_prompt(self, video_path: str, video_dir: str, context: Dict[str, Any]) -> str:
        context_lines = [
            f"The primary video path is: {video_path}",
            f"Use '{video_dir}' whenever a tool requires an input directory.",
        ]
        if additional_context := context.get("notes"):
            context_lines.append(f"Additional notes: {additional_context}")
        return self._base_system_prompt + "\n\n" + "\n".join(context_lines)

    def _build_seed_messages(
        self,
        query: str,
        video_path: str,
        video_dir: str,
        context: Dict[str, Any],
    ) -> List[HumanMessage]:
        context_text = f"Video path: {video_path}\nVideo directory: {video_dir}"
        if study_id := context.get("study_id"):
            context_text += f"\nStudy identifier: {study_id}"
        return [
            HumanMessage(content=context_text),
            HumanMessage(content=query),
        ]

    def _extract_final_response(self, messages: List[BaseMessage]) -> Optional[str]:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return self._message_content_to_str(message.content)
        return None

    def _collect_tool_outputs(
        self, messages: List[BaseMessage]
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        tool_results: Dict[str, Any] = {}
        tools_used: List[str] = []
        tool_errors: List[str] = []

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_name = message.name or "unknown_tool"
            tools_used.append(tool_name)

            parsed_content = self._parse_tool_content(message.content)
            tool_results[tool_name] = parsed_content

            if isinstance(parsed_content, dict) and parsed_content.get("status") == "error":
                error_msg = parsed_content.get('error', 'Unknown error')
                try:
                    tool_errors.append(f"{tool_name}: {error_msg}")
                except UnicodeEncodeError:
                    tool_errors.append(f"{tool_name}: {self._safe_str(error_msg)}")
            elif isinstance(parsed_content, dict) and "error" in parsed_content:
                error_msg = parsed_content['error']
                try:
                    tool_errors.append(f"{tool_name}: {error_msg}")
                except UnicodeEncodeError:
                    tool_errors.append(f"{tool_name}: {self._safe_str(error_msg)}")

        return tool_results, tools_used, tool_errors

    def _build_analysis(self, query: str, tools_used: List[str], success: bool) -> QueryAnalysis:
        if not tools_used:
            analysis_types = [AnalysisType.SIMPLE]
            complexity = AnalysisComplexity.LOW
        else:
            analysis_types = [
                AnalysisType.MULTI_TOOL if len(tools_used) > 1 else self._map_tool_to_analysis_type(tools_used[0])
            ]
            complexity = (
                AnalysisComplexity.MEDIUM if len(tools_used) > 1 else AnalysisComplexity.LOW
            )
        confidence = 0.75 if success else 0.25
        return QueryAnalysis(
            query=query,
            analysis_types=analysis_types,
            complexity=complexity,
            required_tools=tools_used,
            recommended_tools=tools_used,
            use_got=False,
            confidence=confidence,
            reasoning="Tools were selected dynamically via ReAct loop.",
        )

    def _build_react_trace(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Reconstruct a lightweight ReAct-style trace from the message history.

        Each entry is one of:
        - {"type": "user", "content": "..."}
        - {"type": "thought", "content": "..."}            # assistant message without tool calls
        - {"type": "action", "tool_name": "...", "tool_args": {...}}
        - {"type": "tool_result", "tool_name": "...", "content": "..."}

        This uses only observable messages (no hidden chain-of-thought).
        """
        trace: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                trace.append(
                    {
                        "type": "user",
                        "content": self._message_content_to_str(msg.content),
                    }
                )
            elif isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", []) or []
                if tool_calls:
                    # Treat each tool call as an action.
                    for call in tool_calls:
                        trace.append(
                            {
                                "type": "action",
                                "tool_name": call.get("name"),
                                "tool_args": call.get("args", {}),
                                "tool_call_id": call.get("id"),
                            }
                        )
                else:
                    # Assistant message without tool calls -> thought / final answer chunk.
                    trace.append(
                        {
                            "type": "thought",
                            "content": self._message_content_to_str(msg.content),
                        }
                    )
            elif isinstance(msg, ToolMessage):
                content_str = self._message_content_to_str(msg.content)
                # Truncate very long tool outputs for readability.
                if len(content_str) > 800:
                    content_str = content_str[:800] + "... [truncated]"
                trace.append(
                    {
                        "type": "tool_result",
                        "tool_name": msg.name or "unknown_tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": content_str,
                        "tool_args": msg.additional_kwargs.get("args", {}),
                    }
                )

        return trace

    def _build_reasoning_text(
        self,
        query: str,
        video_path: str,
        tools_used: List[str],
        tool_results: Dict[str, Any],
        final_response: Optional[str],
    ) -> str:
        """Generate a short natural-language reasoning summary for the run.

        This is a post-hoc explanation based on the tools and final answer,
        not the model's internal hidden chain-of-thought.
        """
        if not final_response:
            return ""

        try:
            context = {
                "query": query,
                "video_path": video_path,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "model_final_response": final_response,
            }
            try:
                serialized_context = json.dumps(context, ensure_ascii=False, default=str)
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback to ASCII-safe encoding if Unicode fails
                serialized_context = json.dumps(context, ensure_ascii=True, default=self._safe_str)
            # Truncate to keep the prompt within a reasonable size.
            if len(serialized_context) > 6000:
                serialized_context = serialized_context[:6000] + "... [truncated]"

            prompt = (
                "You are EchoPilot, an expert echocardiography assistant.\n"
                "You have already completed an analysis using tools. "
                "Based on the following context, explain in 3–6 sentences how the tools' "
                "findings support the final answer. Focus on key measurements and qualitative "
                "findings (e.g., presence/absence of pericardial effusion, LV function).\n\n"
                "Context:\n"
                f"{serialized_context}\n\n"
                "Return only plain text reasoning without restating the full tool JSON."
            )

            msg = self.llm.invoke(prompt)
            return self._message_content_to_str(msg.content)
        except Exception:
            return ""

    def _map_tool_to_analysis_type(self, tool_name: str) -> AnalysisType:
        mapping = {
            "echo_segmentation": AnalysisType.SEGMENTATION,
            "echo_measurement_prediction": AnalysisType.MEASUREMENTS,
            "echo_disease_prediction": AnalysisType.DISEASE,
            "echo_report_generation": AnalysisType.REPORT,
            "echo_view_classification": AnalysisType.SIMPLE,
            "echo_image_video_generation": AnalysisType.GENERATION,
            "echo_rag_guidelines": AnalysisType.SIMPLE,  # RAG is a supporting tool
        }
        return mapping.get(tool_name, AnalysisType.MULTI_TOOL)

    def _parse_tool_content(self, content: Any) -> Any:
        if isinstance(content, (dict, list)):
            return content
        if not isinstance(content, str):
            return content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(content)
            except Exception:  # noqa: BLE001
                return content

    def _message_content_to_str(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multi-modal content from OpenAI responses
            parts = []
            for item in content:
                if isinstance(item, dict):
                    try:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        parts.append(json.dumps(item, ensure_ascii=True, default=self._safe_str))
                else:
                    try:
                        parts.append(str(item))
                    except UnicodeEncodeError:
                        parts.append(self._safe_str(item))
            return "\n".join(parts)
        try:
            return str(content)
        except UnicodeEncodeError:
            return self._safe_str(content)

    def _safe_str(self, obj: Any) -> str:
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

    def _summarize_errors(self, tool_errors: List[str], final_response: Optional[str]) -> str:
        if tool_errors:
            # Safely join errors, handling Unicode
            safe_errors = []
            for err in tool_errors:
                try:
                    safe_errors.append(str(err))
                except UnicodeEncodeError:
                    safe_errors.append(self._safe_str(err))
            return "; ".join(safe_errors)
        try:
            return final_response or "Unknown error"
        except UnicodeEncodeError:
            return self._safe_str(final_response) if final_response else "Unknown error"

    # --------------------------------------------------------------------- Self-Contrast Methods

    def _extract_rag_query(self, query: str, perspectives: List[Dict[str, Any]]) -> Optional[str]:
        """Extract and enhance a query for RAG based on the question and perspectives.

        Uses LLM to rewrite the query to be more specific for finding clinical guidelines,
        thresholds, and classification criteria.

        Returns a query string for RAG tool, or None if not needed.
        """
        # Extract key terms from query
        query_lower = query.lower()

        # Check if query is about severity, measurements, or classifications
        severity_keywords = ["severity", "mild", "moderate", "severe", "normal", "abnormal", "thick", "thickness"]
        measurement_keywords = ["measurement", "threshold", "range", "value", "dimension", "volume", "ef", "ejection"]

        if not any(kw in query_lower for kw in severity_keywords + measurement_keywords):
            return None

        # Extract measurement values from perspectives if available
        measurement_context = []
        for p in perspectives:
            tool_outputs = p.get("tool_outputs", {})
            if "echo_measurement_prediction" in tool_outputs:
                measurements = tool_outputs.get("echo_measurement_prediction", {}).get("measurements", [])
                if measurements and len(measurements) > 0:
                    m = measurements[0].get("measurements", {})
                    # Extract relevant measurements
                    for key, value in m.items():
                        if isinstance(value, dict) and "value" in value:
                            measurement_context.append(f"{key}: {value['value']}")

        # Use LLM to rewrite query for better RAG retrieval
        try:
            context_text = ""
            if measurement_context:
                context_text = f"\n\nAvailable measurements: {', '.join(measurement_context[:5])}"

            rewrite_prompt = f"""Rewrite the following echocardiography question into a more specific query for searching clinical guidelines and standards documents.

Original question: {query}{context_text}

Your task: Create a query that will help find:
1. Clinical criteria and thresholds for classification
2. Severity grading standards (normal, mild, moderate, severe)
3. Measurement ranges and normal values
4. Diagnostic criteria and definitions

Focus on:
- Specific anatomical structures mentioned
- Measurement types and thresholds
- Classification criteria
- Clinical standards and guidelines

Return ONLY the rewritten query, nothing else. Make it concise but specific."""

            response = self.contrast_llm.invoke([HumanMessage(content=rewrite_prompt)])
            rewritten_query = self._message_content_to_str(response.content).strip()

            # Clean up the response (remove quotes, extra text)
            rewritten_query = rewritten_query.strip('"\'')
            if rewritten_query.lower().startswith("query:"):
                rewritten_query = rewritten_query[6:].strip()
            if rewritten_query.lower().startswith("rewritten query:"):
                rewritten_query = rewritten_query[16:].strip()

            return rewritten_query if rewritten_query else query

        except Exception as exc:
            print(f"[RAG] Query rewriting failed: {exc}, using original query")
            return query

    def _extract_multiple_choice_options(self, query: str) -> Optional[Dict[str, str]]:
        """Extract multiple-choice options from query if present.

        Returns:
            Dictionary with keys 'A', 'B', 'C', 'D' and their values, or None if not found
        """
        import re
        # Pattern to match "Options: A) ... B) ... C) ... D) ..."
        pattern = r'Options?:\s*(?:A\)\s*([^B]+?))?\s*(?:B\)\s*([^C]+?))?\s*(?:C\)\s*([^D]+?))?\s*(?:D\)\s*([^\n]+?))?(?:\n|$)'
        match = re.search(pattern, query, re.IGNORECASE | re.DOTALL)

        if match:
            options = {}
            labels = ['A', 'B', 'C', 'D']
            for i, label in enumerate(labels):
                if i < len(match.groups()) and match.group(i + 1):
                    options[label] = match.group(i + 1).strip()
            if options:
                return options

        # Alternative pattern: "A) ... B) ... C) ... D) ..."
        pattern2 = r'(?:^|\n)\s*A\)\s*([^\n]+?)\s*(?:B\)|$).*?(?:^|\n)\s*B\)\s*([^\n]+?)\s*(?:C\)|$).*?(?:^|\n)\s*C\)\s*([^\n]+?)\s*(?:D\)|$).*?(?:^|\n)\s*D\)\s*([^\n]+?)(?:\n|$)'
        match2 = re.search(pattern2, query, re.IGNORECASE | re.DOTALL | re.MULTILINE)

        if match2:
            options = {}
            labels = ['A', 'B', 'C', 'D']
            for i, label in enumerate(labels):
                if i < len(match2.groups()) and match2.group(i + 1):
                    options[label] = match2.group(i + 1).strip()
            if options:
                return options

        return None

    def _generate_multiple_perspectives(
        self,
        query: str,
        video_path: str,
        video_dir: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate 3 different perspectives using different system prompts.

        Returns:
            List of perspective dictionaries, each containing:
            - perspective_id: 1, 2, or 3
            - prompt: The system prompt used
            - response: Final response text
            - tools_used: List of tools used
            - tool_outputs: Dictionary of tool outputs
            - message_history: Full message history
        """
        perspectives = []
        perspective_prompts = [
            PERSPECTIVE_1_STRUCTURAL_PROMPT,
            PERSPECTIVE_2_PATHOLOGICAL_PROMPT,
            PERSPECTIVE_3_QUANTITATIVE_PROMPT,
        ]
        perspective_names = ["Structural", "Pathological", "Quantitative"]

        for i, (prompt, name) in enumerate(zip(perspective_prompts, perspective_names), 1):
            print(f"[Self-Contrast] Running Perspective {i} ({name})...")

            try:
                # Create temporary agent with this perspective's prompt
                # Share the same cache across all perspectives to avoid duplicate tool calls

                # Compose dynamic prompt with video context FIRST
                dynamic_context = self._compose_system_prompt(video_path, video_dir, context)

                # IMPORTANT: Combine perspective prompt + dynamic context
                # This ensures the "MUST USE TOOLS" instructions are preserved
                combined_prompt = f"{prompt}\n\n{dynamic_context}"

                temp_agent = ReactiveEchoAgent(
                    model=self.llm,
                    tools=self.tools,
                    system_prompt=combined_prompt,
                    log_tools=False,  # Don't log intermediate perspectives
                    tool_cache=self._shared_tool_cache,  # Share cache across perspectives
                )

                # Run ReAct loop
                conversation_seed = self._build_seed_messages(query, video_path, video_dir, context)
                final_state = temp_agent.workflow.invoke({"messages": conversation_seed})
                message_history: List[BaseMessage] = list(final_state.get("messages", []))

                final_response = self._extract_final_response(message_history)
                tool_outputs, tools_used, tool_errors = self._collect_tool_outputs(message_history)

                perspectives.append({
                    "perspective_id": i,
                    "perspective_name": name,
                    "prompt": prompt,
                    "response": final_response or "",
                    "tools_used": tools_used,
                    "tool_outputs": tool_outputs,
                    "tool_errors": tool_errors,
                    "message_history": message_history,
                })
            except Exception as exc:
                try:
                    error_msg = str(exc)
                except UnicodeEncodeError:
                    error_msg = self._safe_str(exc)
                try:
                    print(f"[Self-Contrast] Error in Perspective {i}: {error_msg}")
                except UnicodeEncodeError:
                    print(f"[Self-Contrast] Error in Perspective {i}: {self._safe_str(error_msg)}")
                perspectives.append({
                    "perspective_id": i,
                    "perspective_name": name,
                    "prompt": prompt,
                    "response": f"Error: {exc}",
                    "tools_used": [],
                    "tool_outputs": {},
                    "tool_errors": [str(exc)],
                    "message_history": [],
                })

        return perspectives

    def _summarize_measurements_from_perspectives(self, perspectives: List[Dict[str, Any]]) -> str:
        """Summarize measurements from all perspectives."""
        all_measurements = {}
        for p in perspectives:
            tool_outputs = p.get("tool_outputs", {})

            # Extract from echo_measurement_prediction
            if "echo_measurement_prediction" in tool_outputs:
                measurements = tool_outputs.get("echo_measurement_prediction", {}).get("measurements", [])
                if measurements and len(measurements) > 0:
                    m = measurements[0].get("measurements", {})
                    for key, value in m.items():
                        if isinstance(value, dict) and "value" in value:
                            all_measurements[key] = value["value"]

            # Extract from echo_disease_prediction (PanEcho outputs)
            if "echo_disease_prediction" in tool_outputs:
                predictions = tool_outputs.get("echo_disease_prediction", {}).get("predictions", [])
                if predictions and len(predictions) > 0:
                    pred_dict = predictions[0].get("predictions", {})
                    # Extract key measurements: LVEDV, LVSize, LVIDd, LVESV, EF, GLS, etc.
                    for key in ["LVEDV", "LVSize", "LVIDd", "LVESV", "LVSV", "EF", "GLS"]:
                        if key in pred_dict:
                            pred_value = pred_dict[key]
                            if isinstance(pred_value, dict) and "value" in pred_value:
                                all_measurements[key] = pred_value["value"]
                            elif isinstance(pred_value, (int, float, str)):
                                all_measurements[key] = pred_value

        if all_measurements:
            summary = ""
            for key, value in all_measurements.items():
                summary += f"- {key}: {value}\n"
            return summary
        return ""

    def _contrast_perspectives_with_llm(
        self,
        query: str,
        perspectives: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use contrast LLM to identify discrepancies and create checklist."""
        try:
            # Extract multiple-choice options if present
            options = self._extract_multiple_choice_options(query)
            options_text = ""
            if options:
                options_text = "\n\nMultiple-Choice Options:\n"
                for label, value in options.items():
                    options_text += f"{label}) {value}\n"

            # Format perspective summaries
            perspective_summaries = []
            for p in perspectives:
                tool_outputs = p.get("tool_outputs", {})
                tool_outputs_str = json.dumps(tool_outputs, indent=2, default=str)[:2000]  # Limit size

                perspective_summaries.append({
                    "id": p["perspective_id"],
                    "name": p["perspective_name"],
                    "response": p["response"],
                    "tools_used": p["tools_used"],
                    "tool_outputs": tool_outputs_str,
                })

            # Try to query RAG for relevant guidelines if contrast agent is available
            rag_context = ""
            rag_range_analysis = None
            if self.contrast_agent is not None:
                try:
                    # Extract key terms for RAG query
                    rag_query = self._extract_rag_query(query, perspectives)
                    if rag_query:
                        rag_result = self.contrast_agent.workflow.invoke({
                            "messages": [HumanMessage(content=f"Query guidelines for: {rag_query}")]
                        })
                        # Extract RAG context from tool outputs
                        for msg in rag_result.get("messages", []):
                            if isinstance(msg, ToolMessage) and msg.name == "echo_rag_guidelines":
                                rag_content = self._parse_tool_content(msg.content)
                                if isinstance(rag_content, dict) and "context" in rag_content:
                                    rag_context = f"\n\nRELEVANT CLINICAL GUIDELINES:\n{rag_content['context']}\n"
                                    # If range_analysis is present, extract it
                                    if "range_analysis" in rag_content:
                                        rag_range_analysis = rag_content["range_analysis"]
                                    break
                except Exception as rag_exc:
                    print(f"[Self-Contrast] RAG query failed: {rag_exc}")

            # Summarize all measurements from all perspectives
            measurements_summary = self._summarize_measurements_from_perspectives(perspectives)
            if measurements_summary:
                measurements_summary = f"\n\nAVAILABLE MEASUREMENTS FROM VIDEO ANALYSIS:\n{measurements_summary}\n"

            # Add range analysis to prompt if available
            if rag_range_analysis:
                range_text = "\n\nMEASUREMENT RANGES AND SEVERITY CRITERIA FROM GUIDELINES:\n"
                if rag_range_analysis.get("ranges"):
                    for r in rag_range_analysis["ranges"]:
                        range_text += f"- {r['severity'].capitalize()}: {r['range']} (Criteria: {r['criteria']})\n"
                if rag_range_analysis.get("standard_deviations"):
                    range_text += f"Standard deviation methodology: {', '.join(map(str, rag_range_analysis['standard_deviations']))} SD mentioned for severity grading.\n"
                if rag_range_analysis.get("methodology"):
                    range_text += f"Methodology: {rag_range_analysis['methodology']}\n"
                range_text += "\nIMPORTANT: Compare the available measurements above against these ranges and severity criteria from the guidelines to infer the correct severity classification. If direct measurements for the queried condition are absent, use related measurements and the provided methodology to infer severity.\n"
                rag_context += range_text

            contrast_prompt = f"""
Question: {query}{options_text}{measurements_summary}
Three different AI assistants approached this question from different perspectives:

Perspective 1 (Structural):
Response: {perspective_summaries[0]['response']}
Tools used: {', '.join(perspective_summaries[0]['tools_used']) if perspective_summaries[0]['tools_used'] else 'None'}
Tool outputs: {perspective_summaries[0]['tool_outputs']}

Perspective 2 (Pathological):
Response: {perspective_summaries[1]['response']}
Tools used: {', '.join(perspective_summaries[1]['tools_used']) if perspective_summaries[1]['tools_used'] else 'None'}
Tool outputs: {perspective_summaries[1]['tool_outputs']}

Perspective 3 (Quantitative):
Response: {perspective_summaries[2]['response']}
Tools used: {', '.join(perspective_summaries[2]['tools_used']) if perspective_summaries[2]['tools_used'] else 'None'}
Tool outputs: {perspective_summaries[2]['tool_outputs']}

RECOMMENDED: Before analyzing discrepancies, you should:
1. Use the echo_knowledge_graph tool to get measurement guidance for this question type
2. Check if perspectives used the appropriate measurements according to the knowledge graph
3. If perspectives used measurements that may not be ideal for this question type (e.g., EF for cavity size questions), note this in your analysis
4. The knowledge graph guidance provides best practices - perspectives that follow it are generally more reliable

RECOMMENDED: If this question involves measurement thresholds, severity classifications, or clinical guidelines,
consider using the echo_rag_guidelines tool to query relevant guidelines, measurement ranges, and severity criteria
that might help verify or interpret the findings. This can provide important context for severity
classifications and measurement interpretations. RAG is optional but encouraged for questions involving
clinical standards, thresholds, or severity classifications.

When using echo_rag_guidelines, generate 3-5 semantically different query variations (short phrases/keywords)
to improve retrieval coverage. IMPORTANT: If you have already used echo_knowledge_graph and received guidance
about which measurements are recommended (e.g., "wall thickness", "LV mass", "IVSd", "LVPWd"), incorporate
those specific measurement terms into your RAG query variations to get more targeted results.

For example, if the question is "What is the severity of left ventricular hypertrophy?" and the knowledge graph
recommends using "wall thickness" or "IVSd", generate variations like:
- "left ventricular wall thickness hypertrophy severity thresholds"
- "IVSd interventricular septum thickness hypertrophy classification"
- "left ventricular hypertrophy wall thickness normal mild moderate severe"
- "left ventricular mass hypertrophy severity criteria"
- "ventricular wall thickness measurement ranges"

If the question is "What is the severity of aortic arch dilation?" and KG doesn't specify measurements, use generic variations:
- "aortic arch dilation severity classification"
- "aortic arch dilation severity criteria"
- "aortic arch dilation normal mild moderate severe grading"
- "aortic arch dimensions measurement thresholds"
- "aortic arch dilation diagnostic criteria"

Pass these variations as a list in the 'queries' parameter (not 'query') to get better results from the guidelines.

Your task: Identify discrepancies and create a verification checklist.

CRITICAL: Even if perspectives have low confidence or express uncertainty, you must still analyze them and help identify the most likely answer. Low confidence does not mean the answer is wrong - it just means there's less certainty. Your job is to synthesize the information and help determine the best answer from the available options.

Focus on:
1. MEASUREMENT VALIDATION: Check if perspectives used appropriate measurements according to knowledge graph guidance. If a perspective used measurements that are not ideal for this question type (e.g., EF for cavity size questions), consider this when assessing reliability - perspectives that follow KG guidance are generally more reliable.
2. SEVERITY CLASSIFICATION DISCREPANCIES: Valve conditions, chamber sizes, function abnormalities
3. QUANTITATIVE MEASUREMENT CONSISTENCY: EF, volumes, dimensions - compare values across perspectives (prefer measurements that align with KG guidance when available)
4. MEASUREMENT vs RANGE COMPARISON: If the guidelines contain measurement ranges, thresholds, or severity criteria, extract them and compare available measurements against those ranges to infer severity. Look for numerical ranges, standard deviation methodology (2 SD = mild, 3 SD = moderate, 4 SD = severe), and use related measurements as indicators when direct measurements aren't available.
5. TOOL OUTPUT ALIGNMENT: Which tools agree/disagree, which is most appropriate (prefer tools that provide measurements recommended by KG guidance)
6. CLINICAL CONTEXT VERIFICATION: Contradictory findings, missing information, normal range violations
7. RELIABILITY ASSESSMENT: Which perspective has strongest evidence, confidence levels (give higher weight to perspectives that followed KG guidance)
{f"8. OPTION MAPPING: Map each perspective's answer to the correct option (A, B, C, or D) and identify which option is most supported" if options else ""}

Return as JSON:
{{
    "checklist": [
        "Issue 1: description",
        "Issue 2: description"
    ],
    "severity_discrepancies": [
        {{
            "finding": "description",
            "perspective1_value": "...",
            "perspective2_value": "...",
            "perspective3_value": "...",
            "severity": "high/medium/low"
        }}
    ],
    "measurement_variance": [
        {{
            "metric": "EF",
            "perspective1_value": 54,
            "perspective2_value": 52,
            "perspective3_value": 54,
            "variance": "acceptable/high",
            "normal_range": "50-70%"
        }}
    ],
    "tool_reliability": {{
        "most_reliable_perspective": 1,
        "reasoning": "explanation",
        "confidence_scores": {{
            "perspective1": 0.75,
            "perspective2": 0.82,
            "perspective3": 0.68
        }}
    }},
    "recommended_answer": "initial recommendation based on all perspectives"{"," + chr(10) + '    "recommended_option": "A/B/C/D"' if options else ''}
}}
"""

            # Use contrast agent workflow if available (allows tool usage), otherwise fall back to direct LLM
            rag_results = []
            kg_results = []
            if self.contrast_agent is not None:
                try:
                    # Use the contrast agent workflow which allows tool usage
                    result = self.contrast_agent.workflow.invoke({
                        "messages": [HumanMessage(content=contrast_prompt)]
                    })
                    # Extract the final response from the workflow
                    messages = result.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        response_content = self._message_content_to_str(last_message.content)

                        # Extract RAG and KG tool results from messages
                        kg_results = []
                        for msg in messages:
                            if isinstance(msg, ToolMessage) and msg.name == "echo_rag_guidelines":
                                try:
                                    tool_result = self._parse_tool_content(msg.content)
                                    tool_args = msg.additional_kwargs.get("args", {})
                                    # Handle both single query and list of queries
                                    queries_used = tool_args.get("queries") or ([tool_args.get("query")] if tool_args.get("query") else [])
                                    rag_results.append({
                                        "query": queries_used[0] if queries_used else "unknown",
                                        "queries": queries_used,
                                        "result": tool_result
                                    })
                                except Exception as e:
                                    rag_results.append({
                                        "query": "unknown",
                                        "error": str(e),
                                        "raw_content": str(msg.content)[:500]
                                    })
                            elif isinstance(msg, ToolMessage) and msg.name == "echo_knowledge_graph":
                                try:
                                    tool_result = self._parse_tool_content(msg.content)
                                    kg_results.append(tool_result)
                                except Exception as e:
                                    kg_results.append({
                                        "error": str(e),
                                        "raw_content": str(msg.content)[:500]
                                    })
                    else:
                        response_content = ""
                except Exception as agent_exc:
                    print(f"[Self-Contrast] Contrast agent workflow failed: {agent_exc}, falling back to direct LLM")
                    response = self.contrast_llm.invoke([HumanMessage(content=contrast_prompt)])
                    response_content = self._message_content_to_str(response.content)
            else:
                response = self.contrast_llm.invoke([HumanMessage(content=contrast_prompt)])
                response_content = self._message_content_to_str(response.content)

            # Try to parse JSON from response
            try:
                contrast_result = json.loads(response_content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    contrast_result = json.loads(json_match.group(1))
                else:
                    # Fallback: create basic structure
                    contrast_result = {
                        "checklist": ["Unable to parse LLM response"],
                        "severity_discrepancies": [],
                        "measurement_variance": [],
                        "tool_reliability": {
                            "most_reliable_perspective": 1,
                            "reasoning": "Fallback: using first perspective",
                            "confidence_scores": {"perspective1": 0.5, "perspective2": 0.5, "perspective3": 0.5}
                        },
                        "recommended_answer": response_content[:500],
                    }

            # Add RAG and KG results to contrast result
            if rag_results:
                contrast_result["rag_queries"] = rag_results
            if kg_results:
                contrast_result["kg_guidance"] = kg_results[-1]  # Use the last KG result

            return contrast_result
        except Exception as exc:
            print(f"[Self-Contrast] Error in contrast generation: {exc}")
            return {
                "checklist": [f"Error during contrast: {exc}"],
                "severity_discrepancies": [],
                "measurement_variance": [],
                "rag_queries": rag_results if 'rag_results' in locals() else [],
                "tool_reliability": {
                    "most_reliable_perspective": 1,
                    "reasoning": "Error occurred",
                    "confidence_scores": {"perspective1": 0.5, "perspective2": 0.5, "perspective3": 0.5}
                },
                "recommended_answer": "Error during contrast analysis",
            }

    def _refine_answer_with_checklist(
        self,
        query: str,
        perspectives: List[Dict[str, Any]],
        checklist_result: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Use checklist to refine final answer."""
        try:
            # Extract multiple-choice options if present
            options = self._extract_multiple_choice_options(query)
            options_text = ""
            if options:
                options_text = "\n\nMultiple-Choice Options:\n"
                for label, value in options.items():
                    options_text += f"{label}) {value}\n"
                options_text += "\nCRITICAL: Your final answer MUST be in the format: 'Answer: X) [option text]' where X is A, B, C, or D."

            # Format checklist as text
            checklist_items = checklist_result.get("checklist", [])
            checklist_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(checklist_items)])

            # Get perspective responses
            perspective_responses = [p["response"] for p in perspectives]

            # Get reliability info
            tool_reliability = checklist_result.get("tool_reliability", {})
            most_reliable = tool_reliability.get("most_reliable_perspective", 1)
            reliability_reasoning = tool_reliability.get("reasoning", "N/A")

            # Get recommended option if available
            recommended_option = checklist_result.get("recommended_option", "")
            option_guidance = ""
            if recommended_option and options:
                option_guidance = f"\nRecommended option from contrast analysis: {recommended_option}) {options.get(recommended_option, 'N/A')}"

            # Extract RAG context from contrast phase if available
            rag_context_text = ""
            rag_queries = checklist_result.get("rag_queries", [])
            if rag_queries:
                # Extract context and queries used from RAG results
                for rag_query_result in rag_queries:
                    if isinstance(rag_query_result, dict) and "result" in rag_query_result:
                        rag_result = rag_query_result["result"]
                        if isinstance(rag_result, dict) and "context" in rag_result:
                            queries_used = rag_query_result.get("queries", [rag_query_result.get("query", "unknown")])
                            queries_info = f"Queries used: {', '.join(queries_used[:3])}{'...' if len(queries_used) > 3 else ''}"
                            rag_context_text = f"\n\nRELEVANT CLINICAL GUIDELINES (retrieved in contrast phase):\n{queries_info}\n{rag_result['context'][:3000]}\n"
                            break

            # Extract KG guidance from contrast phase if available
            kg_guidance_text = ""
            kg_guidance = checklist_result.get("kg_guidance", {})
            if kg_guidance and isinstance(kg_guidance, dict):
                use_meas = kg_guidance.get("measurement_guidance", {}).get("use_measurements", [])
                avoid_meas = kg_guidance.get("measurement_guidance", {}).get("avoid_measurements", [])
                reason = kg_guidance.get("measurement_guidance", {}).get("reason", "")

                kg_guidance_text = "\n\nKNOWLEDGE GRAPH GUIDANCE (Recommended):\n"
                kg_guidance_text += f"Question Type: {kg_guidance.get('question_type', 'unknown')}\n"
                kg_guidance_text += f"Recommended measurements: {', '.join(use_meas) if use_meas else 'general measurements'}\n"
                kg_guidance_text += f"Less ideal measurements: {', '.join(avoid_meas) if avoid_meas else 'none'}\n"
                kg_guidance_text += f"Reason: {reason}\n"
                kg_guidance_text += "\nNote: Perspectives that used recommended measurements are generally more reliable for this question type.\n"
                kg_guidance_text += "When making your final decision, give higher weight to perspectives that followed this guidance.\n"
            elif self.contrast_agent is not None:
                # Fallback: suggest using KG tool if not already called
                kg_guidance_text = "\n\nKNOWLEDGE GRAPH GUIDANCE (Recommended):\n"
                kg_guidance_text += "You may want to use the echo_knowledge_graph tool to get measurement guidance for this question type.\n"
                kg_guidance_text += "The knowledge graph can help identify which measurements are most appropriate for this type of question.\n"
                kg_guidance_text += "Consider giving higher weight to perspectives that used measurements recommended by the knowledge graph.\n"

            refinement_prompt = f"""
Original question: {query}{options_text}

Three perspectives were generated:

Perspective 1 (Structural): {perspective_responses[0]}

Perspective 2 (Pathological): {perspective_responses[1]}

Perspective 3 (Quantitative): {perspective_responses[2]}

Discrepancy checklist:
{checklist_text}

Most reliable perspective: {most_reliable}
Reasoning: {reliability_reasoning}{option_guidance}{rag_context_text}{kg_guidance_text}

Your task: Provide a refined final answer that:
1. **Use the clinical guidelines context provided above if available** - If RAG was called in the contrast phase, the results are provided above. Use this context to inform your answer. If RAG was not called in the contrast phase and you need guidelines context, you can optionally use echo_rag_guidelines now.
2. When evaluating perspectives, consider which ones used measurements that align with knowledge graph guidance - these are generally more reliable
3. Give higher weight to perspectives that followed knowledge graph guidance when they conflict with others
4. Addresses all checklist items
6. Resolves discrepancies between perspectives
7. Uses the most reliable evidence (preferring perspectives that followed KG guidance when available)
8. Explains any contradictions
8. **CRITICAL: You MUST make a final decision based on the perspectives, even if they have low confidence or are not completely sure. Do not say "uncertain" or "cannot be determined" - you must choose the best answer from the available options based on the evidence provided.**
9. Provides a clear, confident final answer{f"10. Maps your answer to the correct option (A, B, C, or D)" if options else ""}

**IMPORTANT OUTPUT FORMAT:**
At the end of your response, you MUST include a JSON block with the following structure:
```json
{{
  "final_answer": "{'A' if options else 'your answer'}",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation of why this answer was chosen"
}}
```

If multiple-choice options are provided, the "final_answer" field MUST be exactly one of: A, B, C, or D (just the letter, no parentheses or text).

Provide your detailed reasoning first, then end with the JSON block above.

Final answer:
"""

            # Use contrast agent workflow if available (allows tool usage), otherwise fall back to direct LLM
            refinement_rag_results = []
            if self.contrast_agent is not None:
                try:
                    # Use the contrast agent workflow which allows tool usage
                    result = self.contrast_agent.workflow.invoke({
                        "messages": [HumanMessage(content=refinement_prompt)]
                    })
                    # Extract the final response from the workflow
                    messages = result.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        final_answer = self._message_content_to_str(last_message.content)

                        # Extract RAG tool results from messages
                        for msg in messages:
                            if isinstance(msg, ToolMessage) and msg.name == "echo_rag_guidelines":
                                try:
                                    tool_result = self._parse_tool_content(msg.content)
                                    tool_args = msg.additional_kwargs.get("args", {})
                                    # Handle both single query and list of queries
                                    queries_used = tool_args.get("queries") or ([tool_args.get("query")] if tool_args.get("query") else [])
                                    refinement_rag_results.append({
                                        "query": queries_used[0] if queries_used else "unknown",
                                        "queries": queries_used,
                                        "result": tool_result
                                    })
                                except Exception as e:
                                    refinement_rag_results.append({
                                        "query": "unknown",
                                        "error": str(e),
                                        "raw_content": str(msg.content)[:500]
                                    })

                        return final_answer, refinement_rag_results
                    else:
                        return "Unable to generate refined answer", []
                except Exception as agent_exc:
                    print(f"[Self-Contrast] Contrast agent workflow failed during refinement: {agent_exc}, falling back to direct LLM")
                    response = self.contrast_llm.invoke([HumanMessage(content=refinement_prompt)])
                    return self._message_content_to_str(response.content), []
            else:
                response = self.contrast_llm.invoke([HumanMessage(content=refinement_prompt)])
                return self._message_content_to_str(response.content), []
        except Exception as exc:
            print(f"[Self-Contrast] Error in answer refinement: {exc}")
            # Fallback to most reliable perspective
            most_reliable = checklist_result.get("tool_reliability", {}).get("most_reliable_perspective", 1)
            return perspectives[most_reliable - 1].get("response", "Error during refinement"), []

    def process_query_with_self_contrast(
        self,
        query: str,
        video_path: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Run Self-Contrast: generate 3 perspectives, contrast them, refine answer."""
        start_time = time.time()
        context = context or {}
        video_path = os.path.abspath(video_path)
        video_dir = os.path.dirname(video_path)

        print("[Self-Contrast] Starting Self-Contrast analysis...")

        # Step 1: Generate 3 perspectives
        perspectives = self._generate_multiple_perspectives(query, video_path, video_dir, context)

        # Step 2: Generate checklist
        print("[Self-Contrast] Generating discrepancy checklist...")
        checklist_result = self._contrast_perspectives_with_llm(query, perspectives)

        # Step 3: Refine answer
        print("[Self-Contrast] Refining final answer...")
        refined_answer, refinement_rag_results = self._refine_answer_with_checklist(query, perspectives, checklist_result)

        # Collect all tools used across perspectives
        all_tools_used = []
        all_tool_outputs = {}
        all_tool_errors = []
        for p in perspectives:
            all_tools_used.extend(p.get("tools_used", []))
            all_tool_outputs.update(p.get("tool_outputs", {}))
            all_tool_errors.extend(p.get("tool_errors", []))

        # Remove duplicates while preserving order
        all_tools_used = list(dict.fromkeys(all_tools_used))

        success = bool(refined_answer) and len(all_tool_errors) == 0
        analysis = self._build_analysis(query, all_tools_used, success)

        # Build results payload with self-contrast metadata
        results_payload = {
            "analysis_type": "self_contrast",
            "complexity": AnalysisComplexity.HIGH.value,
            "tools_used": all_tools_used,
            "tool_results": all_tool_outputs,
            "reasoning": analysis.reasoning,
            "final_response": refined_answer,
            "tool_errors": all_tool_errors,
            "self_contrast": {
                "perspectives": [
                    {
                        "id": p["perspective_id"],
                        "name": p["perspective_name"],
                        "response": p["response"],
                        "tools_used": p["tools_used"],
                    }
                    for p in perspectives
                ],
                "checklist": checklist_result.get("checklist", []),
                "rag_queries": checklist_result.get("rag_queries", []),
                "refinement_rag_queries": refinement_rag_results,
                "severity_discrepancies": checklist_result.get("severity_discrepancies", []),
                "measurement_variance": checklist_result.get("measurement_variance", []),
                "tool_reliability": checklist_result.get("tool_reliability", {}),
            },
        }

        execution_result = ToolExecutionResult(
            success=success,
            results=results_payload,
            error=None if success else self._summarize_errors(all_tool_errors, refined_answer),
            execution_time=time.time() - start_time,
            tools_used=all_tools_used,
        )

        agent_response = AgentResponse(
            success=success,
            query=query,
            analysis=analysis,
            execution_result=execution_result,
            response_text=refined_answer or "I could not produce an answer.",
            confidence=analysis.confidence,
            execution_time=execution_result.execution_time,
        )

        self.conversation_history.append(agent_response)
        self._display_results(agent_response)

        # Automatically save trajectory
        self.save_trajectory(agent_response, video_path, context)

        print("[Self-Contrast] Analysis complete.")
        return agent_response

    def _display_results(self, response: AgentResponse) -> None:
        """Print a minimal summary for CLI usage."""
        try:
            print(f"\nQuestion: {response.query}")
        except UnicodeEncodeError:
            print(f"\nQuestion: {self._safe_str(response.query)}")

        if response.success:
            print("Answer:")
            try:
                print(response.response_text)
            except UnicodeEncodeError:
                print(self._safe_str(response.response_text))
        else:
            print("Answer unavailable:")
            error_msg = response.execution_result.error or "Unknown failure"
            try:
                print(error_msg)
            except UnicodeEncodeError:
                print(self._safe_str(error_msg))


def test_intelligent_agent() -> None:
    """Manual smoke test for the agent."""
    print("🧪 Testing Reactive Intelligent Agent")
    agent = IntelligentAgent(device="cpu")
    sample_video = Config.get_video_path()
    response = agent.process_query("segment this echo video, use echo_segmentation tool", sample_video)
    print("Response success:", response.success)


if __name__ == "__main__":
    test_intelligent_agent()
