"""
MCP Server for Tdqeq.
Runs the Tdqeq pipeline persistently so that heavy models only load once.
"""

import json
import base64
import urllib.request
from pathlib import Path
from typing import Optional
from loguru import logger

# Import FastMCP
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.error("The 'mcp' package is not installed. Run `pip install mcp`.")
    raise

from tdqeq.loader.pdf_loader import PDFLoader
from tdqeq.detector.table_detector import TableDetector
from tdqeq.extractor.text_clipper import TextClipper
from tdqeq.extractor.table_parser import TableParser
from tdqeq.pipeline import Pipeline
from tdqeq.config import settings

# Initialize FastMCP Server
mcp = FastMCP("TdqeqServer")

# ---------------------------------------------------------------------------
# Global Pipeline Initialization
# ---------------------------------------------------------------------------
# We initialize the models BEFORE the server starts listening.
# This keeps the models hot in memory.


logger.info("Starting Tdqeq MCP Server...")
logger.info("Loading heavy models into memory (this happens only once)...")

try:
    _loader = PDFLoader(dpi=settings.DEFAULT_DPI)
    _detector = TableDetector(device="cpu")  # Modify to 'cuda' if user has GPU
    _clipper = TextClipper()
    _parser = TableParser(device="cpu", batch_size=settings.DEFAULT_BATCH_SIZE)

    _pipeline = Pipeline(
        loader=_loader,
        detector=_detector,
        clipper=_clipper,
        parser=_parser,
        batch_size=settings.DEFAULT_BATCH_SIZE,
        accelerate=False
    )
    logger.info("Models loaded successfully. Server is ready.")
except Exception as e:
    logger.exception("Failed to initialize Tdqeq models.")
    raise

def _download_pdf(url: str) -> bytes:
    """Download PDF bytes from a URL with a standard User-Agent header."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response:
        return response.read()

# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def extract_tables(
    pdf_path: Optional[str] = None,
    pdf_bytes: Optional[str] = None,
    pdf_url: Optional[str] = None,
    accelerate: bool = False,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> str:
    """
    Extracts tables from a PDF document (provided via file path, Base64 bytes, or URL) using the Tdqeq pipeline.
    
    Args:
        pdf_path: The absolute path to the local PDF file.
        pdf_bytes: Base64-encoded PDF document bytes.
        pdf_url: The URL to download the PDF document from.
        accelerate: If true, forces the pipeline to use the faster SlaNet-Plus model instead of UniTable.
        start_page: The 0-indexed start page number to process (inclusive). If None, defaults to the first page.
        end_page: The 0-indexed end page number to process (inclusive). If None, defaults to the last page.
    
    Returns:
        A JSON string containing the extracted tables, their HTML structures, and cell text.
    """
    pdf_source = None
    log_source = ""
    
    if pdf_url is not None:
        try:
            pdf_source = _download_pdf(pdf_url)
            log_source = f"URL: {pdf_url}"
        except Exception as e:
            return json.dumps({"error": f"Failed to download PDF from URL: {e}"})
    elif pdf_bytes is not None:
        try:
            pdf_source = base64.b64decode(pdf_bytes)
            log_source = "Base64 bytes"
        except Exception as e:
            return json.dumps({"error": f"Failed to decode base64 pdf_bytes: {e}"})
    elif pdf_path is not None:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return json.dumps({"error": f"File not found: {pdf_path}"})
        pdf_source = pdf_file
        log_source = f"path: {pdf_path}"
    else:
        return json.dumps({"error": "Either pdf_path, pdf_bytes, or pdf_url must be provided."})
        
    try:
        logger.info(f"MCP Request: Extracting tables from {log_source} (pages: {start_page} to {end_page})")
        
        # Build page_range tuple if start_page or end_page is provided
        page_range = None
        if start_page is not None or end_page is not None:
            import fitz
            try:
                if isinstance(pdf_source, bytes):
                    with fitz.open(stream=pdf_source, filetype="pdf") as doc:
                        total_pages = len(doc)
                else:
                    with fitz.open(str(pdf_source)) as doc:
                        total_pages = len(doc)
            except Exception as e:
                return json.dumps({"error": f"Failed to open PDF to resolve page count: {e}"})
            
            s = start_page if start_page is not None else 0
            e = end_page if end_page is not None else (total_pages - 1)
            page_range = (s, e)

        # Temporarily apply accelerate flag if needed
        original_accelerate = _pipeline._accelerate
        if accelerate != original_accelerate:
            _pipeline._accelerate = accelerate
            _pipeline._parser._accelerate = accelerate
            
        tables = _pipeline.run(pdf_path=pdf_source, page_range=page_range)
        
        # Restore original accelerate flag
        if accelerate != original_accelerate:
            _pipeline._accelerate = original_accelerate
            _pipeline._parser._accelerate = original_accelerate
        
        payload = [t.to_dict() for t in tables]
        
        return json.dumps(payload, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.exception(f"Error extracting tables from {log_source}")
        return json.dumps({"error": str(e)})


def main():
    """Entry point for the CLI script."""
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
