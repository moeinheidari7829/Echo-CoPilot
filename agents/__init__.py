"""Agent access helpers."""


def get_intelligent_agent():
    from .intelligent_agent import IntelligentAgent, AgentResponse

    return IntelligentAgent, AgentResponse


__all__ = ["get_intelligent_agent"]
