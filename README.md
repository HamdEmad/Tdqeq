# Tdqeq

> An enterprise-grade, high-throughput pipeline for unstructured document table extraction and normalization.

Tdqeq orchestrates a multi-stage vision and NLP pipeline to accurately detect, extract, and reconstruct complex tabular data from PDF documents. By decoupling layout detection (YOLOv10), table parsing (SlaNet-Plus / UniTable), and schema normalization (LLMs), Tdqeq provides a highly resilient and scalable solution for transforming unstructured documents into structured `JSON` or `Pandas DataFrames`.

## Core Capabilities

- **Adaptive Model Routing**: Dynamically analyzes structural complexity and classification confidence to route tables to the optimal parsing model (e.g., fast processing via SlaNet-Plus vs. high-fidelity reasoning via UniTable).
- **High-Throughput Inference**: Built from the ground up for batched GPU inference, enabling parallel processing of multi-page documents to maximize hardware utilization.
- **Agentic Integration (MCP)**: Features a native Model Context Protocol (MCP) server, allowing persistent, in-memory execution and seamless integration into agentic workflows (e.g., Claude Desktop, Cursor, Hermes).
- **Finetuning & Data Curation**: Ships with comprehensive utilities and Colab notebooks for generating supervised fine-tuning (SFT) datasets, facilitating domain adaptation for models like Qwen2.5-3B-Instruct.

## System Architecture

The extraction pipeline operates in five distinct, loosely coupled stages:

1. **Rasterization & Extraction (`PDFLoader`)**: Rasterizes PDF pages at configurable DPIs while simultaneously extracting word-level bounding boxes and text blocks via PyMuPDF.
2. **Layout Detection (`TableDetector`)**: Utilizes `doclayout_yolo` (YOLOv10) to accurately identify table boundaries and associated caption regions.
3. **Spatial Intersection (`TextClipper`)**: Computes intersections between page-level text bounding boxes and table regions to isolate precise cellular text content.
4. **Structural Parsing (`TableParser`)**: Classifies the table topology (Wired vs. Wireless) and reconstructs the HTML structure and cellular grid using `RapidTable`. Text is deterministically mapped to cells via center-point geometric matching.

## Installation

The package is fully configured via `pyproject.toml` and supports direct installation from the repository.

### Direct Installation
```bash
pip install git+https://github.com/HamdEmad/Tdqeq.git
```

### Local Development
```bash
git clone https://github.com/HamdEmad/Tdqeq.git
cd Tdqeq
pip install -e .
```

## Quick Start

The core `Pipeline` abstraction handles all model instantiation, weight management, and memory allocation internally.

```python
from tdqeq.pipeline import Pipeline
import json

# 1. Initialize the Pipeline
# Weights are automatically resolved and cached from the Hugging Face Hub.
pipeline = Pipeline(
    dpi=150,           # Rasterization resolution
    device="cuda",     # Hardware acceleration ('cuda' or 'cpu')
    batch_size=4,      # Batched inference for optimal VRAM utilization
    mode="auto"        # Smart routing: auto, tdqeq (fast), tdqeq+ (accurate)
)

# 2. Execute Extraction
pdf_path = "path/to/your/document.pdf"
tables = pipeline.run(pdf_path)
print(f"Successfully extracted {len(tables)} tables.")

# 3. Downstream Processing
for table in tables:
    # Convert to standard Pandas DataFrame for data science workflows
    df = table.to_pandas()
    
    # Export to strict JSON for API responses
    payload = json.dumps(table.to_dict(), indent=2)
```

## MCP Server Integration

To avoid the latency overhead of repeatedly loading heavy vision models into memory, Tdqeq provides a persistent Model Context Protocol (MCP) server.

### Execution
Start the server using the provided console script:
```bash
tdqeq-mcp
```
*Alternatively, run the module directly: `python -m tdqeq.mcp_server`*

### Client Configuration

**Claude Desktop / Cursor**
Add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "tdqeq": {
      "command": "python",
      "args": ["-u", "-m", "tdqeq.mcp_server"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

> [!NOTE]
> Ensure the Python environment containing `tdqeq` is accessible via your system PATH, or provide the absolute path to the Python executable.

### API Surface

#### `extract_tables`
Extracts and reconstructs tables from a local PDF document.

**Parameters**:
- `pdf_path` *(string, optional)*: Absolute path to the source PDF.
- `pdf_bytes` *(string, optional)*: Base64-encoded PDF document bytes.
- `pdf_url` *(string, optional)*: URL to download the PDF document from.
- `mode` *(string, optional, default: "auto")*: Routing mode (`"auto"`, `"tdqeq"`, `"tdqeq+"`, `"tdqeq++"`).
- `start_page` *(integer, optional, default: null)*: 0-indexed start page (inclusive).
- `end_page` *(integer, optional, default: null)*: 0-indexed end page (inclusive).

> [!IMPORTANT]
> At least one of `pdf_path`, `pdf_bytes`, or `pdf_url` must be provided.

**Returns**:
A JSON payload containing the extracted tables, spatial bounding boxes, HTML DOM representations, and structured cellular data.

## License

This software is distributed under the [MIT License](LICENSE).
