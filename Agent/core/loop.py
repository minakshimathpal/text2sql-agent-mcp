# core/loop.py

import asyncio
from Agent.core.context import AgentContext
from Agent.core.session import MultiMCP
from Agent.core.strategy import decide_next_action
from Agent.modules.problem_perception import extract_problem_perception, ProblemPerceptionResult
from Agent.modules.agent_perception import extract_task_perception, extract_sql_perception, TaskPerceptionResult, SQLPerceptionResult
from Agent.modules.action import ToolCallResult, parse_function_call
from Agent.modules.memory import MemoryItem
import pandas as pd
import traceback
import re
import json

def get_next_query(context, perceptions):
    
    query = f"""Original User Query: {context.user_input}\n\n"""
    query += "Your previous Execution Trace is given below: \n\n"

    if perceptions != [] and context.memory_trace != []:

        for index, intent in enumerate(perceptions):
            memory_item = context.memory_trace[index]
            raw_result = memory_item.raw_result
            query += f"Iteration:{index} \n\n" f"Perception: \n\n {intent} \n\n"
            query += f"Tool_Name: \n\n{memory_item.tool_name}\n\n"
            if raw_result is not None and isinstance(raw_result, pd.DataFrame):
                query += f"Resultant Output: \n\n {raw_result.head(2).to_markdown(index=False)}" + "\n\n"
            else:
                query += f"Result: \n\n"
                if isinstance(raw_result, dict):
                    for db_name in raw_result:
                        for table_name in raw_result[db_name]:
                            schema = raw_result[db_name][table_name]
                            query += f" {table_name} \n\n {schema} \n\n"
                
                else:
                    query += f"{raw_result}\n\n"
        
        query += "If this fully answers the task, return: \
                    FINAL_ANSWER: your answer Otherwise, return the next FUNCTION_CALL." + "\n\n"

    return query

from io import StringIO
def markdown_to_df(markdown_table):
    data_io = StringIO(markdown_table)
    df = pd.read_csv(data_io, sep='|', skiprows=[2], skipinitialspace=True)
    df = df.iloc[1:, 1:-1]  # Remove first and last empty columns
    df.columns = [col.strip() for col in df.columns]  # Clean column names


class AgentLoop:
    def __init__(self, user_input: str, dispatcher: MultiMCP):
        self.context = AgentContext(user_input)
        self.mcp = dispatcher
        self.tools = dispatcher.get_all_tools()

    def tool_expects_input(self, tool_name: str) -> bool:
        tool = next((t for t in self.tools if getattr(t, "name", None) == tool_name), None)
        if not tool:
            return False
        parameters = getattr(tool, "parameters", {})
        return list(parameters.keys()) == ["input"]

    async def run(self) -> str:
        print(f"[agent] Starting session: {self.context.session_id}")

        try:
            query = self.context.user_input
            problem_perception = await extract_problem_perception(query)
            print("Problem perception", problem_perception)
            prev_intents = []
            if isinstance(problem_perception, ProblemPerceptionResult):
                problem_type = problem_perception.problem_type
                max_steps = self.context.agent_profile.max_steps
                for step in range(max_steps):
                    self.context.step = step
                    print(f"[loop] Step {step + 1} of {max_steps}")

                    if problem_type == "sql-generation":
                        # 🧠 Perception
                        perception_raw = await extract_sql_perception(problem_perception)
                    
                    else:
                        perception_raw = await extract_task_perception(problem_perception)


                    # ✅ Exit cleanly on FINAL_ANSWER
                    # ✅ Handle string outputs safely before trying to parse
                    if isinstance(perception_raw, str):
                        pr_str = perception_raw.strip()
                        
                        # Clean exit if it's a FINAL_ANSWER
                        if pr_str.startswith("FINAL_ANSWER:"):
                            self.context.final_answer = pr_str
                            break

                        # Detect LLM echoing the prompt
                        if "Your last tool produced this result" in pr_str or "Original user task:" in pr_str:
                            print("[perception] ⚠️ LLM likely echoed prompt. No actionable plan.")
                            self.context.final_answer = "FINAL_ANSWER: [no result]"
                            break

                        # Try to decode stringified JSON if it looks valid
                        try:
                            perception_raw = json.loads(pr_str)
                        except json.JSONDecodeError:
                            print("[perception] ⚠️ LLM response was neither valid JSON nor actionable text.")
                            self.context.final_answer = "FINAL_ANSWER: [no result]"
                            break


                    # ✅ Try parsing PerceptionResult
                    if isinstance(perception_raw, TaskPerceptionResult) or isinstance(perception_raw, SQLPerceptionResult):
                        perception = perception_raw
                    else:
                        try:
                            # Attempt to parse stringified JSON if needed
                            if isinstance(perception_raw, str):
                                perception_raw = json.loads(perception_raw)
                            
                            if problem_type == "sql-generation":
                                perception = SQLPerceptionResult(**perception_raw)
                            
                            else:
                                perception = TaskPerceptionResult(**perception_raw)
                        except Exception as e:
                            print(f"[perception] ⚠️ LLM perception failed: {e}")
                            print(f"[perception] Raw output: {perception_raw}")
                            break

                    print(f"[perception] Intent: {perception.current_step_perception}, Hint: {perception.tool_hint}")
                    prev_intents.append(perception.current_step_perception)

                    # 💾 Memory Retrieval
                    retrieved = self.context.memory.retrieve(
                        query=query,
                        top_k=self.context.agent_profile.memory_config["top_k"],
                        type_filter=self.context.agent_profile.memory_config.get("type_filter", None),
                        session_filter=self.context.session_id
                    )
                    print(f"[memory] Retrieved {len(retrieved)} memories")

                    # 📊 Planning (via strategy)
                    plan = await decide_next_action(
                        context=self.context,
                        perception=perception,
                        problem_type=problem_type,
                        memory_items=retrieved,
                        all_tools=self.tools
                    )
                    print(f"[plan] {plan}")

                    if "FINAL_ANSWER:" in plan:
                        # Optionally extract the final answer portion
                        final_lines = [line for line in plan.splitlines() if "FINAL_ANSWER:" in line.strip()]
                        if final_lines:
                            self.context.final_answer = final_lines[-1].strip()
                        else:
                            self.context.final_answer = "FINAL_ANSWER: [result found, but could not extract]"
                        break


                    # ⚙️ Tool Execution
                    try:
                        result_obj = None
                        tool_name, arguments = parse_function_call(plan, context=self.context, all_tools=self.tools)


                        if self.tool_expects_input(tool_name):
                            tool_input = {'input': arguments} if not (isinstance(arguments, dict) and 'input' in arguments) else arguments
                        else:
                            tool_input = arguments

                        try:
                            response = await self.mcp.call_tool(tool_name, tool_input)
                            # Your existing logic here
                        except Exception as e:
                            print(f"Exception type: {type(e)}")
                            print(f"Exception value: {e}")
                            print(f"Traceback: {traceback.format_exc()}")
                            raise

                        # ✅ Safe TextContent parsing
                        raw = getattr(response.content[0], 'text', str(response.content[0]))
                        if raw not in ["None", None]: 
                            if len(re.findall(r"\|", raw)) > 1:
                                if "schema" in tool_name:
                                    result_str = raw
                                else: 
                                    result_obj = markdown_to_df(raw)
                                    result_str = result_obj.head(2).to_markdown(index=False)
                                #print(f"[action] {tool_name} → {result_str}")   
                            try:
                                result_obj = json.loads(raw) if raw.strip().startswith("{") else raw
                        
                            except json.JSONDecodeError:
                                result_obj = raw
                        
                        else:
                            print(f"raw tool calling output is {raw} for iteration {step}")

                        print("[Processed Tool Call Response]", result_obj)
                        result_str = result_obj.get("markdown") if isinstance(result_obj, dict) else str(result_obj)

                        if problem_type == "sql-generation":

                            # 🧠 Add memory
                            memory_item = MemoryItem(
                                text=f"{tool_name}({arguments}) → {result_str}",
                                raw_result = result_obj,
                                type="tool_output",
                                tool_name=tool_name,
                                user_query=query,
                                tags=[tool_name],
                                session_id=self.context.session_id
                            )
                        
                        else:
                            # 🧠 Add memory
                            memory_item = MemoryItem(
                                text=f"{tool_name}({arguments}) → {result_str}",
                                raw_result = result_str,
                                type="tool_output",
                                tool_name=tool_name,
                                user_query=query,
                                tags=[tool_name],
                                session_id=self.context.session_id
                            )

                        self.context.add_memory(memory_item)
                        tool = None
                        tool_filter = [tool for tool in self.tools if tool.name == tool_name]
                        if tool_filter != []:
                            tool = tool_filter[0]

                        if tool is not None:  
                            tool_schema = tool.inputSchema
                            desc = getattr(tool_schema, 'description', 'No description available')

                        # 🔁 Next query
                        query = get_next_query(self.context, prev_intents)
                        next_quer_dict = {"user_input": query, "intent": problem_perception.intent, "problem_type": problem_type}
                        problem_perception = ProblemPerceptionResult(**next_quer_dict)
                    except Exception as e:
                        print(f"[error] Tool execution failed: {traceback.format_exc(e)}")
                        break
            else:
                print(f"[agent]: Problem Perception Module Failed: {e}")


        except Exception as e:
            print(f"[agent] Session failed: {traceback.format_exc(e)}")

        return self.context.final_answer or "FINAL_ANSWER: [no result]"


