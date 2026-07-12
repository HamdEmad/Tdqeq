# Tdqeq

A powerful, highly optimized pipeline for Table Detection and Extraction from PDF documents. `Tdqeq` orchestrates state-of-the-art vision models (YOLO, SlaNet-Plus, UniTable) to accurately identify tables, extract their text and layout, and seamlessly reconstruct them into structured data (JSON, Pandas DataFrames).

## Features

- **End-to-End Pipeline**: Handles everything from raw PDFs to structured tabular data.
- **Smart Model Routing**: Dynamically routes tables to the most efficient parsing model based on structural complexity and classification confidence.
- **Batched GPU Inference**: Detects and parses multiple pages/regions in parallel for maximum hardware utilization.
- **Automated Weight Management**: Automatically downloads and caches required YOLO weights from the Hugging Face Hub.
- **Rich Export Options**: Easily convert extracted tables to JSON or Pandas DataFrames.
- **Intelligent Caption Matching**: Seamlessly combines AI-detected caption regions with dynamic stylistic heuristics to guarantee high-accuracy table caption extraction.

## Installation

Since `pyproject.toml` is fully configured, users can install `Tdqeq` directly from GitHub using `pip`, or by cloning the repository.

### Option 1: Direct Install from GitHub (Easiest)
```bash
pip install git+https://github.com/HamdEmad/Tdqeq.git
```

### Option 2: Clone and Install
```bash
git clone https://github.com/HamdEmad/Tdqeq.git
cd Tdqeq
pip install .
```

## Quick Start

```python
from tdqeq.loader.pdf_loader import PDFLoader
from tdqeq.detector.table_detector import TableDetector
from tdqeq.extractor.text_clipper import TextClipper
from tdqeq.extractor.table_parser import TableParser
from tdqeq.pipeline import Pipeline
from tdqeq.types import RawTable

# 1. Initialize components (Weights will auto-download if not provided)
loader = PDFLoader(dpi=150)
detector = TableDetector(device="cpu") # Use "cuda" for GPU
clipper = TextClipper()
parser = TableParser(device="cpu", batch_size=4)

# 2. Build the Pipeline
pipeline = Pipeline(
    loader=loader,
    detector=detector,
    clipper=clipper,
    parser=parser,
    batch_size=4
)

# 3. Extract Tables
pdf_path = "path/to/your/document.pdf"
tables = pipeline.run(pdf_path)

print(f"Extracted {len(tables)} tables!")

import json

# 4. Export to Pandas DataFrame and JSON
for i, table in enumerate(tables):
    # Pandas DataFrame
    df = table.to_pandas()
    print(df)
    
    # JSON String
    table_json = json.dumps(table.to_dict(), indent=2)
    print(table_json)
```

## MCP Server Integration

`Tdqeq` includes a built-in Model Context Protocol (MCP) server that runs the table extraction pipeline persistently. This prevents loading the heavy OCR and table parsing models into memory on every run.

### Running the MCP Server

You can run the MCP server directly using its console script entry point:

```bash
tdqeq-mcp
```

Or run the module via Python:

```bash
python -m tdqeq.mcp_server
```

The server uses the `stdio` transport to communicate.

### Connecting to Claude Desktop / Cursor

To connect this MCP server to **Claude Desktop**, add the following configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tdqeq": {
      "command": "python",
      "args": [
        "-u",
        "-m",
        "tdqeq.mcp_server"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

> [!NOTE]
> Make sure the active Python environment where `tdqeq` is installed is on your system PATH, or specify the absolute path to your Python executable (e.g. `C:\Users\<username>\.venv\Scripts\python.exe`).

### Connecting to Hermes Agent

To connect this MCP server to **Hermes Agent**, manually edit your configuration file:

* **Windows**: `C:\Users\<username>\.hermes\config.yaml`
* **macOS / Linux**: `~/.hermes/config.yaml`

Add the `tdqeq` server configuration under the `mcp_servers` section:

```yaml
mcp_servers:
  tdqeq:
    command: "python"
    args:
      - "-u"
      - "-m"
      - "tdqeq.mcp_server"
    env:
      PYTHONUNBUFFERED: "1"
```

Restart Hermes Agent or run the `/reload-mcp` command inside a chat session to load the new server.

### Available Tools

#### `extract_tables`

Extracts tables from a local PDF document.

* **Arguments**:
  * `pdf_path` (string, required): The absolute path to the local PDF file.
  * `accelerate` (boolean, optional, default: `false`): If `true`, forces the pipeline to use the faster SlaNet-Plus model instead of UniTable.
  * `start_page` (integer, optional, default: `null`): The 0-indexed start page number to process (inclusive).
  * `end_page` (integer, optional, default: `null`): The 0-indexed end page number to process (inclusive).
* **Returns**:
  A JSON string containing the list of extracted tables, their page numbers, bounding boxes, HTML structures, and individual cell details.

## Architecture

1. **PDFLoader**: Rasterizes pages and extracts word-level bounding boxes (CPU bound).
2. **TableDetector**: Uses `doclayout_yolo` to find table and caption bounding boxes.
3. **TextClipper**: Intersects page-level words with table bounding boxes.
4. **TableParser**: Classifies the table (Wired vs. Wireless) and routes it to `RapidTable` (SlaNet-Plus or UniTable) to predict HTML structure and cell bounds. Words are mapped to cells via center-point geometry.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
