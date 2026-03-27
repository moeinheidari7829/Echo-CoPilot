"""
BioMCP Tool for EchoPilot

This tool integrates BioMCP (Biomedical Model Context Protocol) to provide
biomedical knowledge access for the contrast LLM.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class BioMCPInput(BaseModel):
    """Input schema for BioMCP tool."""
    
    query: str = Field(
        description=(
            "The biomedical query. Can be a gene name, disease name, variant, "
            "or clinical trial search term. Examples: 'TP53', 'lung cancer', "
            "'BRAF V600E', 'melanoma clinical trials'"
        )
    )
    query_type: str = Field(
        default="auto",
        description=(
            "Type of query: 'auto' (detect automatically), 'gene', 'disease', "
            "'variant', 'trial', 'article'. Default: 'auto'"
        )
    )


class BioMCPTool(BaseTool):
    """
    Tool for querying BioMCP (Biomedical Model Context Protocol) server.
    
    BioMCP provides access to biomedical databases including:
    - Gene information (NCBI, Ensembl)
    - Disease information and synonyms
    - Clinical trials (ClinicalTrials.gov)
    - Variant annotations
    - PubMed articles
    
    This tool is designed for the contrast LLM to access biomedical knowledge
    when verifying or interpreting echocardiography findings.
    """
    
    name: str = "biomcp_query"
    description: str = (
        "Query biomedical databases for gene information, disease information, "
        "clinical trials, variant annotations, and research articles. Use this tool "
        "when you need to: "
        "- Look up gene functions and pathways related to cardiac conditions "
        "- Find disease synonyms or related conditions "
        "- Search for clinical trials related to cardiac diseases or treatments "
        "- Get variant annotations for genetic cardiac conditions "
        "- Find research articles about specific cardiac conditions or measurements"
    )
    args_schema: type[BaseModel] = BioMCPInput
    
    _mcp_server_process: Optional[subprocess.Popen] = None
    _mcp_available: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_biomcp_availability()
    
    def _check_biomcp_availability(self) -> None:
        """Check if BioMCP is available and can be used."""
        try:
            # Try to import biomcp or check if it's installed
            result = subprocess.run(
                ["python", "-c", "import biomcp; print('OK')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self._mcp_available = True
                print("[BioMCP] BioMCP package detected")
            else:
                # Try with uv
                result = subprocess.run(
                    ["uv", "run", "--with", "biomcp-python", "python", "-c", "import biomcp; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self._mcp_available = True
                    print("[BioMCP] BioMCP available via uv")
        except Exception as e:
            print(f"[BioMCP] Warning: BioMCP not available: {e}")
            print("[BioMCP] Install with: pip install biomcp-python or uv pip install biomcp-python")
            self._mcp_available = False
    
    def _run_biomcp_command(
        self, 
        command: List[str], 
        input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a BioMCP command and return the result."""
        if not self._mcp_available:
            return {
                "status": "error",
                "error": "BioMCP is not available. Install with: pip install biomcp-python",
                "suggestion": "BioMCP provides biomedical knowledge access. Install it to enable this feature."
            }
        
        try:
            # Use CLI directly (more reliable than Python API)
            return self._query_biomcp_cli(command, input_data)
        except Exception as e:
            return {
                "status": "error",
                "error": f"BioMCP query failed: {str(e)}",
                "query": input_data or " ".join(command)
            }
    
    def _query_biomcp_python(
        self, 
        command: List[str], 
        input_data: Optional[str]
    ) -> Dict[str, Any]:
        """Query BioMCP using Python API."""
        # This is a placeholder - actual implementation would use BioMCP's Python API
        # For now, return a message indicating BioMCP integration is in progress
        return {
            "status": "info",
            "message": "BioMCP Python API integration in progress",
            "query": input_data or " ".join(command),
            "note": "BioMCP provides access to biomedical databases. Full integration requires BioMCP Python package."
        }
    
    def _query_biomcp_cli(
        self, 
        command: List[str], 
        input_data: Optional[str]
    ) -> Dict[str, Any]:
        """Query BioMCP using CLI."""
        try:
            # Try direct biomcp command first
            cmd = ["biomcp"] + command
            try:
                result = subprocess.run(
                    cmd,
                    input=input_data if input_data else None,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='replace'
                )
            except FileNotFoundError:
                # Fall back to uv run
                cmd = ["uv", "run", "--with", "biomcp-python", "biomcp"] + command
                result = subprocess.run(
                    cmd,
                    input=input_data if input_data else None,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='replace'
                )
            
            stdout_clean = (result.stdout or "").strip()
            stderr_clean = (result.stderr or "").strip()
            
            if result.returncode == 0:
                # Try to parse as JSON first
                if stdout_clean:
                    try:
                        parsed = json.loads(stdout_clean)
                        return {
                            "status": "success",
                            "data": parsed,
                            "query": input_data or " ".join(command)
                        }
                    except json.JSONDecodeError:
                        # Return as text if not JSON
                        return {
                            "status": "success",
                            "output": stdout_clean,
                            "query": input_data or " ".join(command)
                        }
                else:
                    return {
                        "status": "success",
                        "output": "No output from BioMCP command",
                        "query": input_data or " ".join(command)
                    }
            else:
                # Check if it's a "not found" type error that's still useful
                # Some BioMCP commands return non-zero but still have useful output
                if stdout_clean:
                    try:
                        parsed = json.loads(stdout_clean)
                        return {
                            "status": "success",
                            "data": parsed,
                            "query": input_data or " ".join(command),
                            "warning": stderr_clean if stderr_clean else None
                        }
                    except json.JSONDecodeError:
                        # Check if it's a "not found" message
                        if "not found" in stderr_clean.lower() or "not found" in stdout_clean.lower():
                            return {
                                "status": "not_found",
                                "message": stdout_clean or stderr_clean,
                                "query": input_data or " ".join(command)
                            }
                        return {
                            "status": "partial",
                            "output": stdout_clean,
                            "error": stderr_clean if stderr_clean else "Command returned non-zero exit code",
                            "query": input_data or " ".join(command)
                        }
                else:
                    # Check if it's a "not found" message
                    if "not found" in stderr_clean.lower():
                        return {
                            "status": "not_found",
                            "message": stderr_clean,
                            "query": input_data or " ".join(command)
                        }
                    return {
                        "status": "error",
                        "error": stderr_clean or "BioMCP command failed",
                        "query": input_data or " ".join(command)
                    }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "BioMCP query timed out",
                "query": input_data or " ".join(command)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": input_data or " ".join(command)
            }
    
    def _run(
        self,
        query: str,
        query_type: str = "auto",
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the BioMCP query."""
        if not self._mcp_available:
            return {
                "status": "error",
                "error": "BioMCP is not available",
                "message": "Install BioMCP with: pip install biomcp-python or uv pip install biomcp-python",
                "query": query
            }
        
        # Map query type to BioMCP command
        query_lower = query.lower()
        
        # Auto-detect query type if needed
        if query_type == "auto":
            if any(term in query_lower for term in ["trial", "clinical trial", "nct"]):
                query_type = "trial"
            elif any(term in query_lower for term in ["variant", "mutation", "rs", "hgvs"]):
                query_type = "variant"
            elif any(term in query_lower for term in ["article", "pubmed", "publication"]):
                query_type = "article"
            elif any(term in query_lower for term in ["gene", "protein", "pathway"]):
                query_type = "gene"
            else:
                query_type = "disease"  # Default to disease search
        
        # Build command based on query type
        if query_type == "trial":
            # Extract condition from query
            condition = query.replace("clinical trial", "").replace("trial", "").replace("clinical trials", "").strip()
            command = ["trial", "search", "--condition", condition, "--json"]
        elif query_type == "variant":
            command = ["variant", "search", "--query", query]
        elif query_type == "article":
            # Extract search terms and use --disease or --keyword
            search_terms = query.replace("research articles", "").replace("articles", "").replace("article", "").strip()
            # Check if it's a disease-related query
            if any(term in search_terms.lower() for term in ["disease", "stenosis", "valve", "cardiac", "heart"]):
                command = ["article", "search", "--disease", search_terms, "--json"]
            else:
                command = ["article", "search", "--keyword", search_terms, "--json"]
        elif query_type == "gene":
            command = ["gene", "get", query]
        else:  # disease
            # Use search instead of get for better results
            command = ["disease", "search", query]
        
        result = self._run_biomcp_command(command, query)
        
        return {
            "status": result.get("status", "unknown"),
            "query": query,
            "query_type": query_type,
            "result": result,
            "message": f"BioMCP query executed for: {query} (type: {query_type})"
        }

