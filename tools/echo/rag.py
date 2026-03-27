"""Echo RAG tool for querying echocardiography guidelines and standards from PDF."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_community.vectorstores import FAISS
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

try:
    import pypdf
except ImportError:
    pypdf = None


class EchoRAGInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description="Single query about echocardiography guidelines, measurement thresholds, severity classifications, or clinical standards. Either 'query' or 'queries' must be provided."
    )
    queries: Optional[List[str]] = Field(
        None,
        description="List of query variations to search. If provided, this takes precedence over 'query'. Generate 3-5 semantically different variations of the same question to improve retrieval coverage. Either 'query' or 'queries' must be provided."
    )
    top_k: int = Field(
        3,
        description="Number of relevant document chunks to retrieve per query (default: 3)"
    )


class EchoRAGTool(BaseTool):
    """
    RAG tool for querying echocardiography guidelines and standards.

    This tool loads a PDF document containing echocardiography guidelines,
    creates embeddings, and provides semantic search capabilities to help
    categorize measurements, interpret severity classifications, and reference
    clinical standards.
    """

    name: str = "echo_rag_guidelines"
    description: str = (
        "Query echocardiography guidelines, measurement thresholds, severity classifications, "
        "and clinical standards from authoritative sources. Use this tool to: "
        "- Categorize measurements (e.g., normal vs abnormal ranges) "
        "- Interpret severity classifications (e.g., mild, moderate, severe) "
        "- Reference clinical standards and thresholds "
        "- Understand measurement criteria and definitions. "
        "RECOMMENDED: For better retrieval, pass 3-5 semantically different query variations as a list "
        "in the 'queries' parameter instead of a single 'query'. This improves coverage of relevant guidelines."
    )
    args_schema: Type[BaseModel] = EchoRAGInput

    _vector_store: Optional[FAISS] = None
    _embeddings: Optional[OpenAIEmbeddings] = None
    _pdf_paths: List[Path] = []

    def __init__(self, pdf_paths: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        # Default to all PDFs in RAG_clinical_Data folder
        if pdf_paths is None:
            # Go up from tools/echo/rag.py to project root (2 levels up)
            project_root = Path(__file__).resolve().parents[2]
            # Look for PDFs in RAG_Clinical_Data folder
            rag_data_dir = project_root / "RAG_Clinical_Data"
            if rag_data_dir.exists():
                pdf_files = list(rag_data_dir.glob("*.pdf"))
                if pdf_files:
                    pdf_paths = [str(p) for p in pdf_files]
                    print(f"[RAG] Found {len(pdf_files)} PDF file(s) in RAG_Clinical_Data: {[p.name for p in pdf_files]}")
                else:
                    print(f"[RAG] No PDF files found in RAG_Clinical_Data folder")
            else:
                print(f"[RAG] RAG_Clinical_Data folder not found at {rag_data_dir}")

        self._pdf_paths = [Path(p) for p in pdf_paths] if pdf_paths else []
        self._initialized = False
        try:
            self._initialize_rag()
            self._initialized = True
        except Exception as exc:
            print(f"[RAG] Warning: Failed to initialize RAG system: {exc}")
            print(f"[RAG] RAG tool will not be available. Agent will continue without it.")
            self._initialized = False

    def _initialize_rag(self) -> None:
        """Initialize the RAG system by loading PDFs and creating vector store."""
        if pypdf is None:
            raise ImportError(
                "pypdf is required for RAG functionality. "
                "Install it with: pip install pypdf"
            )

        # Validate all PDF paths exist
        missing_pdfs = [p for p in self._pdf_paths if not p.exists()]
        if missing_pdfs:
            raise FileNotFoundError(
                f"PDF file(s) not found: {[str(p) for p in missing_pdfs]}. "
                "Please ensure the PDF files exist."
            )

        if not self._pdf_paths:
            raise ValueError("No PDF files specified for RAG initialization.")

        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for RAG embeddings. "
                "Set it as an environment variable."
            )

        # Initialize embeddings
        self._embeddings = OpenAIEmbeddings(api_key=api_key)

        # Create a combined name for the vector store based on all PDFs
        # Sanitize PDF names to avoid filesystem issues
        def sanitize_name(name: str) -> str:
            # Replace spaces and special characters with underscores
            import re
            name = re.sub(r'[^\w\-_]', '_', name)
            name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
            return name[:50]  # Limit length

        # Save vector store in RAG_VectorDB folder in project root
        project_root = Path(__file__).resolve().parents[2]
        vector_store_path = project_root / "RAG_VectorDB"

        if vector_store_path.exists() and (vector_store_path / "index.faiss").exists():
            # Load existing vector store
            print(f"[RAG] Loading existing vector store from {vector_store_path}")
            self._vector_store = FAISS.load_local(
                str(vector_store_path),
                self._embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Create new vector store from all PDFs
            print(f"[RAG] Creating vector store from {len(self._pdf_paths)} PDF file(s): {[p.name for p in self._pdf_paths]}")
            all_documents = []
            for pdf_path in self._pdf_paths:
                pdf_docs = self._load_pdf(pdf_path)
                if pdf_docs:
                    # Add source metadata to each document
                    for doc in pdf_docs:
                        all_documents.append(f"[Source: {pdf_path.name}]\n{doc}")
                    print(f"[RAG] Extracted {len(pdf_docs)} pages from {pdf_path.name}")

            if not all_documents:
                raise ValueError(f"No text extracted from any PDF: {[str(p) for p in self._pdf_paths]}")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.create_documents(all_documents)

            # Create vector store
            self._vector_store = FAISS.from_documents(chunks, self._embeddings)

            # Save vector store for future use
            vector_store_path.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(vector_store_path))
            print(f"[RAG] Vector store saved to {vector_store_path}")

    def _load_pdf(self, pdf_path: Path) -> list[str]:
        """Extract text from a PDF file."""
        if pypdf is None:
            return []

        texts = []
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF {pdf_path}: {e}") from e

        return texts

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple query variations for better retrieval."""
        variations = [query]  # Always include original

        # Extract key terms
        query_lower = query.lower()

        # Add variations focusing on different aspects
        if "severity" in query_lower or "mild" in query_lower or "moderate" in query_lower:
            # Add variations for severity classification
            variations.append(f"{query} classification criteria thresholds")
            variations.append(f"{query} normal mild moderate severe grading")
            variations.append(f"{query} diagnostic criteria")

        if "thickness" in query_lower or "thick" in query_lower:
            # Add variations for thickness measurements
            variations.append(f"{query} measurement normal range")
            variations.append(f"{query} criteria definition")
            variations.append(f"{query} structural abnormality")

        if "mitral" in query_lower or "valve" in query_lower:
            # Add variations for valve-specific queries
            variations.append(f"{query} echocardiography guidelines")
            variations.append(f"{query} ASE recommendations")

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            v_lower = v.lower()
            if v_lower not in seen:
                seen.add(v_lower)
                unique_variations.append(v)

        return unique_variations[:5]  # Limit to 5 variations

    def _run(
        self,
        query: Optional[str] = None,
        queries: Optional[List[str]] = None,
        top_k: int = 3,
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Query the RAG system for relevant guidelines and standards."""
        if not self._initialized or self._vector_store is None:
            return {
                "status": "error",
                "query": query or (queries[0] if queries else "unknown"),
                "message": "RAG system not initialized. Please ensure PDF file exists and dependencies are installed.",
                "context": "",
            }

        # Determine which queries to use
        if queries:
            # Use provided queries list (from LLM)
            query_variations = queries
            primary_query = queries[0] if queries else query
        elif query:
            # Use single query and generate keyword-based variations as fallback
            query_variations = self._generate_query_variations(query)
            primary_query = query
        else:
            return {
                "status": "error",
                "query": "unknown",
                "message": "Either 'query' or 'queries' parameter must be provided.",
                "context": "",
            }

        # Detect if this is a measurement comparison query (needs thresholds/ranges)
        # Use a balanced approach: prioritize thresholds but don't exclude other relevant info
        query_lower = primary_query.lower()
        is_measurement_comparison = any(term in query_lower for term in [
            "severity", "mild", "moderate", "severe", "normal", "abnormal",
            "threshold", "range", "criteria", "classification", "grading",
            "compare", "measurement", "value", "diameter", "volume", "thickness"
        ])

        # If measurement comparison, gently enhance retrieval for thresholds
        if is_measurement_comparison:
            # Add a few threshold-specific query variations (not too many)
            threshold_queries = [
                f"{primary_query} threshold criteria",
                f"{primary_query} normal range classification",
            ]
            query_variations.extend(threshold_queries)
            # Slightly increase top_k to get more comprehensive results
            effective_top_k = max(top_k, 4)  # Only slightly more, not too aggressive
        else:
            effective_top_k = top_k

        # Collect results from all query variations
        all_docs = []
        seen_content = set()

        for q_var in query_variations:
            docs = self._vector_store.similarity_search(q_var, k=effective_top_k)
            for doc in docs:
                # Deduplicate by content
                content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)

        # Gently prioritize chunks containing threshold/range information for measurement comparisons
        # Balance: prefer thresholds when available, but don't exclude other relevant content
        if is_measurement_comparison:
            # Score chunks by presence of threshold/range indicators (gentle scoring)
            threshold_keywords = ["mm", "cm", "%", "ml", "ml/m2", "normal", "mild", "moderate", "severe",
                                 "range", "threshold", "criteria", "classification", "grading",
                                 "≤", "≥", "<", ">", "-", "to", "between"]
            scored_docs = []
            for doc in all_docs:
                content_lower = doc.page_content.lower()
                # Gentle scoring: count keywords but don't over-weight
                score = sum(0.5 for keyword in threshold_keywords if keyword in content_lower)
                # Moderate bonus for chunks with numeric ranges (e.g., "3.0-3.5 cm", "< 40 mm")
                has_numeric_range = bool(re.search(r'\d+\.?\d*\s*[-–—]\s*\d+\.?\d*', doc.page_content))
                has_comparison = bool(re.search(r'[<>≤≥]\s*\d+\.?\d*', doc.page_content))
                if has_numeric_range or has_comparison:
                    score += 2  # Reduced from 5 to 2 - gentle boost, not dominant
                scored_docs.append((score, doc))

            # Sort by score but keep some diversity - take top results with threshold preference
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            # Take top results, but ensure we get a mix if scores are similar
            selected_docs = [doc for _, doc in scored_docs[:effective_top_k]]
        else:
            # For non-measurement queries, just take top_k unique results
            selected_docs = all_docs[:top_k] if len(all_docs) > top_k else all_docs

        # If we have fewer results, try the primary query with higher k
        if len(selected_docs) < effective_top_k:
            additional_docs = self._vector_store.similarity_search(primary_query, k=effective_top_k * 2)
            for doc in additional_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    selected_docs.append(doc)
                    if len(selected_docs) >= effective_top_k:
                        break

        # Extract relevant text chunks
        relevant_chunks = [doc.page_content for doc in selected_docs]
        sources = [doc.metadata.get("source", "unknown") for doc in selected_docs]

        # For measurement comparisons, gently remind about thresholds
        if is_measurement_comparison:
            # Add a gentle note to help LLM notice thresholds when present
            threshold_note = "[Note: When available, threshold values, ranges, and classification criteria in the chunks below are particularly useful for measurement comparison]\n\n"
        else:
            threshold_note = ""

        # Combine chunks into a single context
        context = threshold_note + "\n\n---\n\n".join(
            [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(relevant_chunks)]
        )

        # Don't force range extraction - just return the context
        # LLMs can extract and use ranges from the context themselves if needed
        # Build result message
        if is_measurement_comparison:
            message = f"Retrieved {len(selected_docs)} relevant document chunks (prioritized for thresholds/ranges) using {len(query_variations)} query variation(s) for: {primary_query[:100]}..."
        else:
            message = f"Retrieved {len(selected_docs)} relevant document chunks using {len(query_variations)} query variation(s) for: {primary_query[:100]}..."

        result = {
            "status": "success",
            "query": primary_query,
            "queries_used": query_variations,
            "queries_used_count": len(query_variations),
            "top_k": top_k,
            "effective_top_k": effective_top_k if is_measurement_comparison else top_k,
            "num_results": len(selected_docs),
            "is_measurement_comparison": is_measurement_comparison,
            "context": context,
            "sources": sources,
            "message": message,
        }

        return result

    def _extract_ranges_and_classify(
        self,
        context: str,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """Extract measurement ranges from context and attempt to classify severity.

        This method uses LLM to intelligently extract ranges from the PDF text
        and structure them for comparison with measurements.
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            # Use LLM to extract ranges from the context
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None

            llm = ChatOpenAI(api_key=api_key, model=Config.OPENAI_MODEL, temperature=0)

            extract_prompt = f"""Extract measurement ranges and severity criteria from the following echocardiography guidelines text.

Query: {query}

Guidelines text:
{context[:3000]}  # Limit context to avoid token limits

Your task:
1. Identify any numerical ranges (e.g., "X-Y mm", ">X cm", "<X%") related to the query
2. Extract severity classifications (normal, mild, moderate, severe) and their associated ranges or criteria
3. Note any standard deviation methodology (2 SD, 3 SD, 4 SD) mentioned
4. Structure the information clearly

Return as JSON:
{{
    "measurement_name": "name of measurement from query",
    "ranges": [
        {{
            "severity": "normal/mild/moderate/severe",
            "range": "X-Y units or description",
            "criteria": "description of criteria"
        }}
    ],
    "standard_deviations": [2, 3, 4] if mentioned,
    "methodology": "description of how severity is determined",
    "note": "any additional relevant information"
}}

If no specific ranges are found for the exact measurement in the query, return ranges for related measurements that might be relevant."""

            response = llm.invoke([HumanMessage(content=extract_prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Try to parse JSON from response
            import json
            try:
                # Remove markdown code blocks if present
                response_text = re.sub(r'```json\n?', '', response_text)
                response_text = re.sub(r'```\n?', '', response_text)
                range_data = json.loads(response_text.strip())
                return range_data
            except json.JSONDecodeError:
                # Fallback to regex-based extraction
                return self._extract_ranges_regex(context, query)

        except Exception as e:
            print(f"[RAG] LLM range extraction failed: {e}, falling back to regex")
            return self._extract_ranges_regex(context, query)

    def _extract_ranges_regex(
        self,
        context: str,
        query: str
    ) -> Dict[str, Any]:
        """Fallback regex-based range extraction."""
        # Look for numerical ranges in the context
        range_patterns = [
            r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',  # X - Y
            r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)',  # X to Y
            r'[<>≤≥]\s*(\d+\.?\d*)',  # >X, <X, ≤X, ≥X
        ]

        ranges_found = []
        for pattern in range_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                ranges_found.append({
                    "text": match.group(0),
                    "span": match.span(),
                })

        # Look for severity-related text
        severity_keywords = {
            "normal": ["normal", "reference", "expected"],
            "mild": ["mild", "slight", "minimal"],
            "moderate": ["moderate"],
            "severe": ["severe", "marked", "significant"],
        }

        severity_mentions = {}
        for severity, keywords in severity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in context.lower():
                    severity_mentions[severity] = severity_mentions.get(severity, 0) + 1

        # Look for standard deviation methodology
        sd_pattern = r'(\d+)\s*standard\s*deviation[s]?|(\d+)\s*SD'
        sd_matches = re.finditer(sd_pattern, context, re.IGNORECASE)
        sd_values = []
        for match in sd_matches:
            sd_val = match.group(1) or match.group(2)
            if sd_val:
                sd_values.append(int(sd_val))

        # Extract measurement name from query
        measurement_name = None
        query_lower = query.lower()
        if "mitral" in query_lower and "leaflet" in query_lower and "thickness" in query_lower:
            measurement_name = "mitral_valve_leaflet_thickness"
        elif "wall thickness" in query_lower:
            measurement_name = "wall_thickness"
        elif "ef" in query_lower or "ejection" in query_lower:
            measurement_name = "ejection_fraction"

        return {
            "measurement_name": measurement_name,
            "ranges_found": ranges_found[:10],
            "severity_mentions": severity_mentions,
            "standard_deviations": sd_values,
            "has_range_information": len(ranges_found) > 0,
            "note": "Ranges extracted using regex. For better results, use LLM extraction.",
        }

