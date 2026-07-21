"""
Command Line Interface for the Tdqeq Pipeline.
Allows running table extraction directly from the terminal.
"""

import argparse
import sys
import json
from pathlib import Path
from loguru import logger

from tdqeq.pipeline import Pipeline
from tdqeq.config import settings
from tdqeq.normalizer.llm_normalizer import LLMNormalizer


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
        "--dpi",
        type=int,
        default=settings.DEFAULT_DPI,
        help="Resolution (DPI) used when rasterizing PDF pages."
    )
    parser.add_argument(
        "--table-mode",
        type=str,
        default="auto",
        choices=["auto", "tdqeq", "tdqeq+"],
        help=(
            "Routing mode for model selection: "
            "'auto' (auto select between faster mode and more accuracy mode based on the hardness of the table), "
            "'tdqeq' (faster but lower accuracy), "
            "or 'tdqeq+' (high accuracy but slowest)."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--normalize", "-n",
        action="store_true",
        help="Normalize the extracted tables using OpenAI LLM."
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="OpenAI API Key (overrides env var)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI Model to use (overrides config)."
    )

    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        logger.error(f"Error: The file {pdf_file} does not exist.")
        sys.exit(1)

    logger.info("Initializing Tdqeq pipeline...")

    try:
        pipeline = Pipeline(
            dpi=args.dpi,
            device=args.device,
            batch_size=args.batch_size,
            mode=args.table_mode,
        )

        logger.info(f"Running pipeline on {pdf_file.name}...")

        tables = pipeline.run(pdf_path=pdf_file)

        if args.normalize:
            logger.info("Running LLM normalization layer...")
            normalizer = LLMNormalizer(
                api_key=args.openai_key,
                model=args.model
            )
            payload = normalizer.normalize(tables)
        else:
            payload = [t.to_dict() for t in tables]

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Extraction complete! Results saved to {args.output}")

    except Exception:
        logger.exception("An error occurred during pipeline execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
