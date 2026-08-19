import json
from tdqeq.pipeline import Pipeline
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

def run():
    print("Initializing Pipeline...")
    pipeline = Pipeline(
        dpi=150,
        device="cpu", # use cpu for local testing to avoid cuda issues if they don't have it
        batch_size=2,
        mode="auto"
    )
    
    pdf_path = "test_input.pdf"
    print(f"Running pipeline on {pdf_path}...")
    tables = pipeline.run(pdf_path)
    
    print(f"\nExtracted {len(tables)} tables!\n")
    
    for i, table in enumerate(tables):
        print(f"--- Table {i+1} ---")
        df = table.to_pandas()
        print(df)
        print("\n")
        
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in tables], f, ensure_ascii=False, indent=2)
    print("Saved output to result.json")

if __name__ == "__main__":
    run()
