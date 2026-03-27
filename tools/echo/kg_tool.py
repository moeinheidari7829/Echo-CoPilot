"""
Echo Knowledge Graph Tool
Allows the agent to query the knowledge graph for measurement guidance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.echo_knowledge_graph import build_echo_kg, query_kg_for_question, validate_measurement_usage


class EchoKGInput(BaseModel):
    question: str = Field(
        ...,
        description="The echocardiography question to get measurement guidance for. "
        "Examples: 'What is the severity of left ventricular cavity dilation?', "
        "'What is the ejection fraction?', 'What is the severity of mitral regurgitation?'"
    )


class EchoKnowledgeGraphTool(BaseTool):
    """
    Knowledge Graph tool for querying measurement guidance based on medical knowledge.
    
    This tool provides guidance on which measurements to use or avoid for different
    types of echocardiography questions, based on general medical principles from
    textbooks and guidelines (NOT from test data).
    
    Use this tool to:
    - Determine which measurements are appropriate for a question type
    - Understand which measurements to avoid for specific assessments
    - Get medical reasoning for measurement selection
    - Validate measurement usage
    """
    
    name: str = "echo_knowledge_graph"
    description: str = (
        "Query the echocardiography knowledge graph to get measurement guidance for questions. "
        "This tool tells you which measurements to USE and which to AVOID for different question types, "
        "based on general medical knowledge (anatomy, physiology, measurement principles). "
        "Use this tool when you need to know: "
        "- Which measurements are appropriate for cavity size questions (use LVEDV, avoid EF) "
        "- Which measurements are appropriate for function questions (use EF, avoid volumes) "
        "- Which measurements are appropriate for valvular questions (use disease scores, avoid chamber measurements) "
        "- Medical reasoning behind measurement selection"
    )
    args_schema: Type[BaseModel] = EchoKGInput
    
    _kg = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Lazy load KG on first use
        self._kg = None
    
    def _get_kg(self):
        """Lazy load knowledge graph."""
        if self._kg is None:
            self._kg = build_echo_kg()
        return self._kg
    
    def _run(
        self,
        question: str,
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph for measurement guidance.
        
        Args:
            question: The echocardiography question
            
        Returns:
            Dictionary containing:
            - question_type: Type of question (e.g., "cavity_dilation")
            - structure: Relevant cardiac structure
            - use_measurements: List of measurements to use
            - avoid_measurements: List of measurements to avoid
            - guidance: Human-readable guidance text
            - reason: Medical reasoning
            - special_rules: Any special interpretation rules
        """
        kg = self._get_kg()
        guidance = query_kg_for_question(question, kg)
        
        # Format response for agent
        response = {
            "status": "success",
            "question": question,
            "question_type": guidance["question_type"],
            "structure": guidance["structure"],
            "measurement_guidance": {
                "use_measurements": guidance["use_measurements"],
                "avoid_measurements": guidance["avoid_measurements"],
                "reason": guidance["reason"],
            },
            "guidance_text": guidance["guidance"],
            "special_rules": guidance.get("special_rules", {}),
            "message": (
                f"For this question type ({guidance['question_type']}), "
                f"use measurements: {', '.join(guidance['use_measurements']) if guidance['use_measurements'] else 'general measurements'}, "
                f"avoid measurements: {', '.join(guidance['avoid_measurements']) if guidance['avoid_measurements'] else 'none'}. "
                f"Reason: {guidance['reason']}"
            )
        }
        
        return response


# For backward compatibility and easier imports
EchoKGTool = EchoKnowledgeGraphTool

