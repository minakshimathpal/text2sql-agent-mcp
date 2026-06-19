from typing import List, Optional, Literal
from pydantic import BaseModel
import os
import re
import json
from dotenv import load_dotenv
from Agent.modules.model_manager import ModelManager
from Agent.modules.tools import summarize_tools
import pandas as pd

model = ModelManager()
tool_context = summarize_tools(model.get_all_tools()) if hasattr(model, "get_all_tools") else ""


class ProblemPerceptionResult(BaseModel):
    user_input: str
    intent: str
    problem_type: Literal["mathematical", "general", "sql-generation", "visualization"] = "general"


async def extract_problem_perception(user_input: str) -> ProblemPerceptionResult:
    """
    Uses LLMs to extract structured info:
    - problem_type: Anything in ["mathematical", "general", "sql-generation", "visualization"]
    - intent: (brief phrase about what the user wants)

    """

    prompt = f"""
You are an AI that tries to identify the exact problem type the user-query belongs to.

Available tools: {tool_context}

Input: "{user_input}"

Return the response as a Python dictionary with keys:
- problem_type: Exactly identifying the problem type of the user-query by taking the hints into consideration through the capabalities offered by the tools mentioned.
   Mention any of the following classes only. Anything in ["mathematical", "general", "sql-generation", "visualization"]
- intent: (brief phrase about what the user wants)

Output only the dictionary on a single line. Do NOT wrap it in ```json or other formatting. Ensure `entities` is a list of strings, not a dictionary.
"""

    try:
        response = await model.generate_text(prompt)

        # Clean up raw if wrapped in markdown-style ```json
        raw = response.strip()
        if not raw or raw.lower() in ["none", "null", "undefined"]:
            raise ValueError("Empty or null model output")

        # Clean and parse
        clean = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        import json

        try:
            parsed = json.loads(clean.replace("null", "null"))  # Clean up non-Python nulls
            print("parsed_output", parsed)
        except Exception as json_error:
            print(f"[perception] JSON parsing failed: {json_error}")
            parsed = {}

        # Ensure Keys
        if not isinstance(parsed, dict):
            raise ValueError("Parsed LLM output is not a dict")
        if "user_input" not in parsed:
            parsed["user_input"] = user_input

        if "intent" not in parsed:
            parsed['intent'] = None

        if "problem_type" not in parsed:
            parsed["problem_type"] = None

        parsed["user_input"] = user_input  # overwrite or insert safely
        print("final_parsed", parsed)
        return ProblemPerceptionResult(**parsed)


    except Exception as e:
        print(f"[perception] ⚠️ LLM perception failed: {e}")
        return ProblemPerceptionResult(user_input=user_input)
