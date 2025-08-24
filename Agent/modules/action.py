# modules/action.py

from typing import Dict, Any, Union
from pydantic import BaseModel
import ast
import traceback

# Optional logging fallback
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")


class ToolCallResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Union[str, list, dict]
    raw_response: Any

def validate_schema(tool_name, param_parts, tools_schema):
    
    bool_ = False
    tool_name_mcp = [tool for tool in tools_schema if tool.name == tool_name]
    if tool_name_mcp == []:
        return "Invalid Function name. Please try to fetch the right tool"
    else:
        tool_name_mcp = tool_name_mcp[0]
    
    param_parts_cleaned = [part for part in param_parts if part != ""]

    schema_properties = tool_name_mcp.inputSchema.get('properties', {})
    print(f"DEBUG: Schema properties: {schema_properties}")
    print(f"[Param Parts]: {param_parts}")

    if schema_properties == {} and param_parts_cleaned == []:
        bool_ = True
    
    elif schema_properties != {} and param_parts_cleaned != []:
        bool_ = True
    
    return bool_

def parse_function_call(response: str, context, all_tools: dict) -> tuple[str, Dict[str, Any]]:
    """
    Parses a FUNCTION_CALL string like:
    "FUNCTION_CALL: add|a=5|b=7"
    Into a tool name and a dictionary of arguments.
    """
    try:
        if "FUNCTION_CALL:" not in response:
            raise ValueError("Invalid function call format.")

        _, raw = response.split(":", 1)
        parts = [p.strip() for p in raw.split("|")]
        tool_name, param_parts = parts[0], parts[1:]
        print("[TOOL_REQUIREMENTS]", tool_name, param_parts)
        param_parts_cleaned = [p for p in param_parts if p != ""]
        print(f"[Param_Parts_Cleaned]: {param_parts_cleaned}")
        schema_validity = validate_schema(tool_name=tool_name, tools_schema=all_tools, param_parts=param_parts)
        print("[Schema Validity]", schema_validity)
        args = {}
        if schema_validity and param_parts_cleaned != []:
            for part in param_parts_cleaned:
                if "=" not in part and not schema_validity:
                    raise ValueError(f"Invalid parameter: {part}")
                
                key, val = part.split("=", 1)

                if "PREVIOUS_TOOL_RESULT" in val:

                    parsed_val = context.memory_trace[-1].raw_result
                    print(key,val,type(parsed_val))
                # Try parsing as literal, fallback to string
                else:
                    try:
                        parsed_val = ast.literal_eval(val)
                    except Exception:
                        parsed_val = val.strip()

                # Support nested keys (e.g., input.value)
                keys = key.split(".")
                current = args
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = parsed_val

            log("parser", f"Parsed: {tool_name} → {args}")
        return tool_name, args

    except Exception as e:
        print("[TRACEBACK]", traceback.format_exc(e))
        log("parser", f"❌ Parse failed: {traceback.format_exc(e)}")
        raise
