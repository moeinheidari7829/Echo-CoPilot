# EchoPilot Agent

An echocardiography ReAct agent focusing on EchoPilot’s specialised tools.

## Features

- **Self-Contrast Mechanism**: 3-perspective voting for error correction
- **Medical Knowledge Integration**: Knowledge graph + RAG for clinical guidelines
- **Multi-tool Analysis**: Disease prediction, measurements, segmentation
- **Comprehensive Evaluation**: Tested on MIMICEchoQA benchmark

## Results (MIMICEchoQA Benchmark - 622 examples)

| System | Accuracy |
|--------|----------|
| LLM + Tools | **54.0%** |
| Agent + Self-Contrast | **52.3%** |
| Baseline LLM | 48.3% |

Self-contrast provides error correction through 3-perspective voting.

## Quick Start

```bash
# Install dependencies (using uv)
uv pip install -r requirements.txt
# Or with pip
pip install -r requirements.txt

# Setup API key
cp .env.example .env
# Edit .env with your API key

# Run evaluation
uv run python experiment/test_llm_with_tools.py --num-examples 10
```

## Tools

- **Disease Prediction**: PanEcho (40 cardiac tasks)
- **Measurement Prediction**: EchoPrime measurements
- **View Classification**: Echocardiogram view identification
- **Knowledge Graph**: Medical measurement guidance
- **RAG**: Clinical guideline retrieval
- **Segmentation**: MedSAM2 cardiac structures
- **Report Generation**: Structured clinical reports

## Evaluation

Run comparison of all system configurations:

```bash
./experiment/run_full_accuracy_comparison.sh 622
```

This tests 4 modes: Baseline LLM, LLM+Tools, Agent, Agent+Self-Contrast.

## Configuration

Set in `.env`:
- `OPENAI_API_KEY`: Your API key
- `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)
- `MEASUREMENT_TOOL`: echoprime/echonet/both
- `USE_SELF_CONTRAST`: Enable 3-perspective voting



