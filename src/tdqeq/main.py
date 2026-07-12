"""
Command Line Interface for the Tdqeq Pipeline.
Allows running table extraction directly from the terminal.
"""

import argparse
import sys
import json
from pathlib import Path
from loguru import logger

from tdqeq.loader.pdf_loader import PDFLoader
from tdqeq.detector.table_detector import TableDetector
from tdqeq.extractor.text_clipper import TextClipper
from tdqeq.extractor.table_parser import TableParser
from tdqeq.pipeline import Pipeline
from tdqeq.config import settings

def main():
    parser = argparse.ArgumentParser(
        description="Tdqeq: A powerful PDF table detection and extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF document to process."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.json",
        help="Path to save the resulting JSON file."
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=settings.DEFAULT_BATCH_SIZE,
        help="Batch size for YOLO detection and rapid_table parsing."
    )
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Force the parser to use the faster SlaNet-Plus model always."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on ('cpu' or 'cuda')."
    )

    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        logger.error(f"Error: The file {pdf_file} does not exist.")
        sys.exit(1)

    logger.info("Initializing Tdqeq pipeline components...")
    
    try:
        loader = PDFLoader(dpi=settings.DEFAULT_DPI)
        detector = TableDetector(device=args.device)
        clipper = TextClipper()
        table_parser = TableParser(device=args.device, batch_size=args.batch_size)

        pipeline = Pipeline(
            loader=loader,
            detector=detector,
            clipper=clipper,
            parser=table_parser,
            batch_size=args.batch_size,
            accelerate=args.accelerate
        )

        logger.info(f"Running pipeline on {pdf_file.name}...")
        
        tables = pipeline.run(pdf_path=pdf_file)
        
        payload = [t.to_dict() for t in tables]
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Extraction complete! Results saved to {args.output}")

    except Exception as e:
        logger.exception("An error occurred during pipeline execution.")
        sys.exit(1)

if __name__ == "__main__":
    main()
