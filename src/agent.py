import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from jsonschema import validate
from jsonschema.exceptions import ValidationError

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
PROMPT_RULES_PATH = CONFIG_DIR / "prompt_rules.txt"
INPUT_SCHEMA_PATH = CONFIG_DIR / "input_schema.json"
OUTPUT_SCHEMA_PATH = CONFIG_DIR / "output_schema.json"

# Environment load api key
load_dotenv(BASE_DIR / ".env")

# Load Configuration
# Load Prompt Rules from prompt_rules.txt.
# Load Input JSON Schema.
# Load Output JSON Schema.
def load_prompt_rules():

    if not PROMPT_RULES_PATH.exists():
        raise FileNotFoundError(
            f"Prompt Rules file not found: {PROMPT_RULES_PATH}"
        )
    with open(PROMPT_RULES_PATH, "r", encoding="utf-8") as file:
        return file.read()

def load_input_schema():

    if not INPUT_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Input Schema file not found: {INPUT_SCHEMA_PATH}"
        )
    with open(INPUT_SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def load_output_schema():

    if not OUTPUT_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Output Schema file not found: {OUTPUT_SCHEMA_PATH}"
        )
    with open(OUTPUT_SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

# JSON Schema Validation
# Validate input_data against input_schema.json.
# Validate Gemini's output against output_schema.json.
def validate_input(input_data):

    schema = load_input_schema()
    try:
        validate(
            instance=input_data,
            schema=schema
        )
    except ValidationError as error:
        raise ValueError(
            f"Input JSON validation failed: {error.message}"
        ) from error

def validate_output(output_data):

    output_schema = load_output_schema()
    try:
        validate(
            instance=output_data,
            schema=output_schema
        )
    except ValidationError as error:
        raise ValueError(
            f"Output JSON validation failed: {error.message}"
        ) from error
    return output_data

def build_gemini_schema(schema):
    """
    Create a Gemini-compatible response schema
    from the full JSON Schema.

    Fields unsupported by Gemini Structured Output
    are removed while keeping the original schema
    unchanged for local validation.
    """

    if isinstance(schema, dict):
        return {
            key: build_gemini_schema(value)
            for key, value in schema.items()
            if key not in {
                "$schema",
                "uniqueItems",
                "additionalProperties"
            }
        }

    if isinstance(schema, list):
        return [
            build_gemini_schema(item)
            for item in schema
        ]

    return schema

# Gemini Client
def create_client():
   
    # Create Gemini API client.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Please check your .env file."
        )
    return genai.Client(api_key=api_key)

# Prompt Construction
def build_prompt(input_data):
    """
    Build the prompt that will be sent to Gemini.
    """

    prompt_rules = load_prompt_rules()
    input_json = json.dumps(
        input_data,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
    {prompt_rules}

    以下是本次需要分析的 Input JSON：

    ```json
    {input_json}
    ```
    請依照上述規則分析每一筆資料。

    請注意：

    必須分析所有輸入資料。
    每一筆輸入資料必須對應一筆輸出結果。
    不得自行建立或修改 sensor threshold。
    必須使用 Input JSON 中提供的 rules。
    detect 只能是 "normal" 或 "abnormal"。
    warning 狀態必須記錄在 warning_sensors。
    abnormal 狀態必須記錄在 anomaly_sensors。
    anomaly score 必須符合 Prompt Rules 中定義的計算方式。
    請提供 reason。
    對 abnormal 或 warning 狀況提供實際可行的 suggestion。
    請嚴格遵守 Output JSON Schema。
    只輸出 JSON。
    """
    return prompt

def call_gemini(client, prompt, output_schema):

    gemini_schema = build_gemini_schema(
        output_schema
    )

    # Send the prompt to Gemini and request structured JSON output.
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema
        )
    )
    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )
    return response.text

def parse_response(response_text):

    # Convert Gemini response text into a Python dictionary.
    try:
        output_data = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Gemini response is not valid JSON."
        ) from error
    return output_data

def run_agent(input_data):

    validate_input(input_data)
    client = create_client()
    prompt = build_prompt(input_data)
    output_schema = load_output_schema()

    response_text = call_gemini(
        client,
        prompt,
        output_schema
    )
    output_data = parse_response(
        response_text
    )
    validate_output(output_data)

    return output_data