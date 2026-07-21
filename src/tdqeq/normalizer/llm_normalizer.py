"""
LLM-based table normalization layer.
Converts raw table HTML outputs into a flat key-values JSON structure,
merges split/truncated tables, and fixes OCR/detection errors.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
from loguru import logger
import openai
from pydantic import BaseModel, Field, AliasChoices

from tdqeq.config import settings
from tdqeq.exceptions import TdqeqError
from tdqeq.types import RawTable


class NormalizedEntry(BaseModel):
    caption: str = Field(
        description="The normalized caption/name of the table. Inherited from a preceding table if split/uncaptioned. Use empty string '' if no caption exists."
    )
    table_confidence: float = Field(
        description="The original detection confidence of the table or segment.",
        validation_alias=AliasChoices("table_confidence", "confidence", "detection_confidence")
    )
    features: str = Field(
        description="The name of the feature or property (e.g. column header or left-hand key label). Trailing colons should be stripped."
    )
    values: str = Field(
        description="The value of the feature. For grid tables with multiple rows, join the cell values for this column using a slash '/'."
    )


class NormalizationResponse(BaseModel):
    entries: List[NormalizedEntry]


SYSTEM_PROMPT = """You are an expert data normalizer. You are given a list of extracted tables from a PDF document in HTML format along with their detection confidence and page number.

Your job is to normalize these tables into a flat list of key-value records.

Follow these strict rules:
1. Determine the structure of each table:
   - "Row-Oriented" (Key-Value): Typically 2 columns, where each row contains a label/key in the first column and a value in the second (e.g., "Suitable Applications: Security System...").
   - "Column-Oriented" (Grid): Multiple columns where the first row contains headers and subsequent rows contain data values (e.g., "Variants" table with headers "Item #", "Color", "UPC").
2. For Row-Oriented tables (Key-Value): Emit one entry per row. "features" is the key in the first column, "values" is the value in the second column. Strip any trailing colons (e.g., change "Suitable Applications:" to "Suitable Applications").
3. For Column-Oriented tables (Grid):
   - If there is only one data row: Emit one entry per column. "features" is the column header, "values" is the data cell value.
   - If there are multiple data rows: Emit one entry per column header. For each column, join the values from all data rows in order using a forward slash "/" (e.g., "6522UE 8771000/6522UE 877U1000/6522UE 877U500").
4. Handle split or truncated tables (e.g. tables split across pages or broken by detection):
   - If a table segment has no caption (caption is null/empty) but is a continuation of a captioned table (e.g. same column structure, or immediately follows it and represents similar content like "Standards and Compliance"), merge their rows/cells under the caption of the parent table.
   - When merging tables with different detection confidences, keep the original detection confidence of the segment each record was derived from.
5. Correct any OCR or formatting issues in the HTML:
   - Handle empty cells, shifted cells, or missing headers using context.
   - For example, if a table has an empty column name or shifted columns, align them logically.
   - Clean up garbled text if it was caused by layout issues.
6. The "caption" field should contain the table caption. If the table class or caption is missing and cannot be inferred or inherited from a preceding table, use empty string "".

Respond ONLY with a JSON object containing a key "entries" which maps to the list of normalized records. Do not include markdown code blocks or explanation outside the JSON.
"""


class LLMNormalizer:
    """
    LLM-based normalizer for tdqeq extraction outputs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.base_url = base_url or settings.OPENAI_BASE_URL

        # Fallback to standard OpenAI environment variable
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API Key is missing. Please set TDQEQ_OPENAI_API_KEY or OPENAI_API_KEY."
            )

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def normalize(self, tables: List[RawTable]) -> List[Dict[str, Any]]:
        """
        Normalize raw tables into flat key-value structures.
        """
        if not tables:
            logger.info("No tables to normalize.")
            return []

        # Step 1: Preprocess to save tokens
        preprocessed_data = self._preprocess(tables)

        # Step 2: Call the LLM
        logger.info(f"Sending {len(preprocessed_data)} tables to OpenAI ({self.model}) for normalization...")
        raw_response = self._call_llm(preprocessed_data)

        # Step 3: Validate and parse output
        try:
            clean_response = self._extract_json(raw_response)
            parsed_response = NormalizationResponse.model_validate_json(clean_response)
            result = [entry.model_dump() for entry in parsed_response.entries]
            logger.info(f"Successfully normalized into {len(result)} key-value entries.")
            return result
        except Exception as e:
            logger.error(f"Failed to validate LLM response: {e}")
            logger.debug(f"Raw response was: {raw_response}")
            raise TdqeqError(f"LLM normalization validation failed: {e}") from e

    def _preprocess(self, tables: List[RawTable]) -> List[Dict[str, Any]]:
        """
        Extract only necessary fields for normalization to minimize prompt size.
        """
        simplified = []
        for t in tables:
            simplified.append({
                "page_number": t.page_number,
                "caption": t.caption,
                "detection_confidence": t.detection_confidence,
                "html": t.html,
            })
        return simplified

    def _extract_json(self, text: str) -> str:
        """
        Extract the first JSON block from text, handling markdown fences.
        """
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start_idx = -1
        for i, c in enumerate(text):
            if c in ("{", "["):
                start_idx = i
                break

        end_idx = -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ("}", "]"):
                end_idx = i
                break

        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            return text[start_idx:end_idx + 1]

        return text

    def _call_llm(self, data: List[Dict[str, Any]]) -> str:
        """
        Execute the API call to OpenAI with transient failure retry logic.
        """
        user_message = f"Please normalize the following table data:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
        
        max_retries = 3
        delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise TdqeqError("OpenAI returned an empty response.")
                return content
            except openai.APIConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise TdqeqError(f"Failed to connect to OpenAI API: {e}") from e
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise TdqeqError(f"OpenAI API rate limit exceeded: {e}") from e
            except openai.APIStatusError as e:
                logger.error(f"OpenAI API returned status error {e.status_code}: {e.response}")
                raise TdqeqError(f"OpenAI API error: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {e}")
                raise TdqeqError(f"LLM normalization failed due to API error: {e}") from e
            
            time.sleep(delay)
            delay *= 2.0
            
        raise TdqeqError("Failed to get response from OpenAI API after retries.")
