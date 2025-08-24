from typing import List, Optional, Union
from pydantic import BaseModel
import os
import re
import json
from dotenv import load_dotenv
from Agent.modules.model_manager import ModelManager
from Agent.modules.tools import summarize_tools
from Agent.modules.problem_perception import ProblemPerceptionResult
import pandas as pd

model = ModelManager()
tool_context = summarize_tools(model.get_all_tools()) if hasattr(model, "get_all_tools") else ""


class TaskPerceptionResult(BaseModel):
    user_input: str
    current_step_perception: Optional[str]
    entities: List[str] = []
    tool_hint: Optional[str] = None


class SQLPerceptionResult(BaseModel):
    user_input: str
    current_step_perception: Optional[str]
    entities: List[str] = []
    tool_hint: Optional[str] = None
    prev_df_result: Optional[bool] = None



async def extract_sql_perception(user_input: ProblemPerceptionResult) -> SQLPerceptionResult:

    """
    Uses LLMs to extract structured info:
    - current_step_perception: current step high-level goal
    - entities: keywords or values
    - tool_hint: likely MCP tool name (optional)
    """

    user_quer = user_input.user_input

    prompt = f"""
You are an AI Expert that works on predominantly two databases [Employee and IMDB database]. Your primary role is to extract key information
and facts regarding sql query operations from the given user-input.

Available tools: {tool_context}

Input: "{user_quer}"

Return the response as a Python dictionary with keys:
- current_step_perception: (brief phrase about what to do in this iteration/step)
- entities: a list of strings representing the database info, table information, and column information. [column name, table name, db name]
- tool_hint: (name of the MCP tool that might be useful, if any or 'None')
- previous_tool_output: "Yes" or "No"

Output only the dictionary on a single line. Do NOT wrap it in ```json or other formatting. Ensure `entities` is a list of strings, not a dictionary.
"""

    try:
        with open(r"C:\Users\aniru\Downloads\agentic_framework_mcp\text2sql_agent_mcp\perception_prompt.txt", "w") as f:
            f.write(prompt)
        response = await model.generate_text(prompt)
        print("SQL PERCEPTION RESPONSE", response)

        # Clean up raw if wrapped in markdown-style ```json
        raw = response.strip()
        if not raw or raw.lower() in ["none", "null", "undefined"]:
            raise ValueError("Empty or null model output")

        # Clean and parse
        clean = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        with open(r"C:\Users\aniru\Downloads\agentic_framework_mcp\res.txt", "w") as f:
            f.write(clean)

        try:
            parsed = json.loads(clean.replace("null", "null"))  # Clean up non-Python nulls
        except Exception as json_error:
            print(f"[perception] JSON parsing failed: {json_error}")
            parsed = {}

        # Ensure Keys
        if not isinstance(parsed, dict):
            raise ValueError("Parsed LLM output is not a dict")
        if "user_input" not in parsed:
            parsed["user_input"] = user_input.user_input
        if "current_step_perception" not in parsed:
            parsed['current_step_perception'] = None
        # Fix common issues
        if isinstance(parsed.get("entities"), dict):
            parsed["entities"] = list(parsed["entities"].values())

        if "prev_df_result" not in parsed:
            parsed["prev_df_result"] = None

        parsed["user_input"] = user_input.user_input  # overwrite or insert safely
        print("Sql_parse", parsed)
        return SQLPerceptionResult(**parsed)

    except Exception as e:
        print(f"[perception] ⚠️ LLM perception failed: {e}\n\n raw_response: {raw}")
        return SQLPerceptionResult(user_input=user_input.user_input)


async def extract_task_perception(user_input: ProblemPerceptionResult) -> TaskPerceptionResult:
    
    """
    Uses LLMs to extract structured info:
    - intent: user’s high-level goal
    - entities: keywords or values
    - tool_hint: likely MCP tool name (optional)
    """

    user_quer = user_input.user_input

    prompt = f"""
You are an AI Expert that extracts structured facts from user input.

Available tools: {tool_context}

Input: "{user_quer}"

Return the response as a Python dictionary with keys:
- intent: (brief phrase about what the user wants)
- entities: a list of strings representing keywords or values (e.g., ["INDIA", "ASCII"])
- tool_hint: (name of the MCP tool that might be useful, if any)
- user_input: same as above

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
        # Fix common issues
        if isinstance(parsed.get("entities"), dict):
            parsed["entities"] = list(parsed["entities"].values())

        parsed["user_input"] = user_input  # overwrite or insert safely
        return TaskPerceptionResult(**parsed)


    except Exception as e:

        print(f"[perception] ⚠️ LLM perception failed: {e}")
        return TaskPerceptionResult(user_input=user_input)
