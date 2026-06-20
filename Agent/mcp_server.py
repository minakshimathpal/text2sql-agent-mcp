from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import asyncio
import importlib
import uuid
import os
import re

# Attempt to import agent workflow for schema/inspector & utilities (optional)
try:
    import Agent.agentic_workflow as aw
    print(f"[MCP STARTUP] Successfully imported Agent.agentic_workflow")
except Exception as e:
    print(f"[MCP STARTUP] Failed to import Agent.agentic_workflow: {e}")
    aw = None

app = FastAPI(title="Minimal MCP Server (dev)")

# Simple tool registry
TOOLS: Dict[str, Any] = {}


class CompleteRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = 30.0


def register_tool(name: str, fn: Any):
    TOOLS[name] = fn


async def _call_tool(name: str, args: Dict[str, Any]) -> Any:
    if name not in TOOLS:
        raise KeyError(f"Tool '{name}' not registered")
    fn = TOOLS[name]
    if asyncio.iscoroutinefunction(fn):
        return await fn(**(args or {}))
    else:
        # run sync in threadpool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(**(args or {})))


@app.on_event("startup")
async def _startup():
    # Register built-in/mock tools lazily to avoid importing heavy modules at import time
    # Try to register a real document scanner implementation; otherwise register a mock
    ds_registered = None
    try:
        from .tools import document_scanner as ds
        register_tool("document_scanner.process", ds.process_image)
        ds_registered = f"Agent.tools.document_scanner.process_image (relative import)"
    except Exception:
        # If relative import fails (running as script), try absolute import
        try:
            import Agent.tools.document_scanner as ds2
            register_tool("document_scanner.process", ds2.process_image)
            ds_registered = f"Agent.tools.document_scanner.process_image (absolute import)"
        except Exception:
            # Register a fallback mock tool
            def _mock_doc(**kwargs):
                # Accept arbitrary keyword args (image_url, image_bytes, etc.) to be tolerant
                return {"doc_id": str(uuid.uuid4()), "text": "(mock OCR output)", "metadata": {"pages": 1, "confidence": 0.0}, "received_args": kwargs}
            register_tool("document_scanner.process", _mock_doc)
            ds_registered = "mock(document_scanner.process)"

    # Log which document scanner implementation was registered for easier debugging
    try:
        print(f"[MCP STARTUP] Registered document_scanner.process -> {ds_registered}")
    except Exception:
        pass

    # Register Granite Vision direct visual Q&A tool
    gv_registered = None
    try:
        from .tools import granite_vision as gv
        register_tool("granite_vision.qa", gv.qa)
        gv_registered = "Agent.tools.granite_vision.qa (relative import)"
    except Exception:
        try:
            import Agent.tools.granite_vision as gv2
            register_tool("granite_vision.qa", gv2.qa)
            gv_registered = "Agent.tools.granite_vision.qa (absolute import)"
        except Exception:
            def _mock_gv(**kwargs):
                return {"answer": "(granite_vision unavailable - mock)", "model": "mock", "received_args": kwargs}
            register_tool("granite_vision.qa", _mock_gv)
            gv_registered = "mock(granite_vision.qa)"
    print(f"[MCP STARTUP] Registered granite_vision.qa -> {gv_registered}")

    # Register OCR QA composite tool: runs OCR then runs the agent OCR QA helper
    try:
        # Use the global aw variable instead of re-importing
        if not aw:
            raise ImportError("Agent workflow not available")

        async def _ocr_qa_tool(image_url: Optional[str] = None, image_bytes: Optional[bytes] = None, question: Optional[str] = None, **kwargs):
            # Call the document scanner tool first (use the registered one to allow mocks)
            ds_fn = TOOLS.get("document_scanner.process")
            if not ds_fn:
                raise RuntimeError("document_scanner.process not registered")

            loop = asyncio.get_running_loop()
            try:
                if asyncio.iscoroutinefunction(ds_fn):
                    tool_res = await ds_fn(image_url=image_url, image_bytes=image_bytes, options=kwargs)
                else:
                    tool_res = await loop.run_in_executor(None, lambda: ds_fn(image_url=image_url, image_bytes=image_bytes, options=kwargs))
            except Exception as e:
                return {"error": f"OCR tool failed: {e}"}

            doc_id = tool_res.get("doc_id") if isinstance(tool_res, dict) else None
            ocr_text = (tool_res.get("text") if isinstance(tool_res, dict) else str(tool_res)) or ""

            # Persist OCR text so downstream callers can retrieve it via doc_id
            try:
                if doc_id and ocr_text:
                    docs_dir = os.path.join("chat_store", "docs")
                    os.makedirs(docs_dir, exist_ok=True)
                    doc_path = os.path.join(docs_dir, f"{doc_id}.txt")
                    # Write file (overwrite if exists)
                    with open(doc_path, "w", encoding="utf-8") as df:
                        df.write(ocr_text)
                    # Also add a short history entry if agent chat store available
                    try:
                        if hasattr(aw, 'chat_store_private') and hasattr(aw, 'ChatMessage') and hasattr(aw, 'MessageRole'):
                            user_msg = aw.ChatMessage(role=aw.MessageRole.USER, content=f"Uploaded document {doc_id}")
                            assistant_msg = aw.ChatMessage(role=aw.MessageRole.ASSISTANT, content=f"OCR stored: {ocr_text[:400]}")
                            aw.chat_store_private.add_message(key="conversation", message=user_msg)
                            aw.chat_store_private.add_message(key="conversation", message=assistant_msg)
                            try:
                                aw.chat_store_private.persist(str(aw.private_store_path))
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                # Non-fatal: continue even if persistence fails
                pass

            # If a question was provided, answer it using the agent helper (run in executor to avoid blocking)
            if question:
                try:
                    # Use executor if helper is sync to avoid blocking event loop
                    if callable(getattr(aw, 'ocr_agent_qa', None)):
                        answer = await loop.run_in_executor(None, lambda: aw.ocr_agent_qa(question, doc_id))
                        return {"doc_id": doc_id, "answer": answer}
                    else:
                        return {"doc_id": doc_id, "answer": "ocr_agent_qa not available"}
                except Exception as e:
                    return {"error": f"ocr_agent_qa failed: {e}", "tool_result": tool_res}

            # No question: return the OCR result and doc_id
            return {"doc_id": doc_id, "ocr_text": ocr_text}

        register_tool("ocr_qa", _ocr_qa_tool)
        print(f"[MCP STARTUP] Registered ocr_qa -> real _ocr_qa_tool (agent available)")
    except Exception:
        # If agent import fails, register a tolerant mock tool
        def _ocr_qa_mock(**kwargs):
            return {"doc_id": str(uuid.uuid4()), "answer": "(ocr_qa unavailable - mock response)", "received_args": kwargs}
        register_tool("ocr_qa", _ocr_qa_mock)
        print(f"[MCP STARTUP] Registered ocr_qa -> mock(ocr_qa)")

    # ------------------------------------------------------------------
    # Deterministic schema-aware SQL helper tools (agentic compliant)
    # ------------------------------------------------------------------
    ENABLE_FAST_SQL = os.environ.get("ENABLE_MCP_FAST_TOOLS", "1") in ("1","true","True")
    if not ENABLE_FAST_SQL:
        print("[MCP STARTUP] Fast SQL tools disabled via ENABLE_MCP_FAST_TOOLS=0")
        return

    # Initialize database connection if not already done
    if not aw:
        print("[MCP STARTUP][FAST] Agent workflow not available; skipping fast SQL tool registration.")
        return
        
    if not getattr(aw, 'inspector', None) or not getattr(aw, 'available_tables', None):
        print("[MCP STARTUP][FAST] Database not initialized; attempting to initialize...")
        try:
            # Initialize database connection - use await since we're in async context
            db_uri = os.environ.get("DB_CONNECTION_URL")
            if not db_uri:
                print("[MCP STARTUP][FAST] DB_CONNECTION_URL not set; skipping fast SQL tool registration.")
                return
            db_result = await aw.initialize_database(db_uri)
            if db_result.get("tables"):
                print(f"[MCP STARTUP][FAST] Database initialized with {len(db_result['tables'])} tables")
            else:
                print("[MCP STARTUP][FAST] Database initialization failed; skipping fast SQL tool registration.")
                return
        except Exception as db_err:
            print(f"[MCP STARTUP][FAST] Database initialization error: {db_err}; skipping fast SQL tool registration.")
            return

    inspector = aw.inspector
    available_tables = aw.available_tables

    def _colnames(table: str) -> List[str]:
        try:
            return [c['name'] for c in inspector.get_columns(table_name=table)]
        except Exception:
            return []

    def _has_tables(*tables: str) -> bool:
        at_lower = {t.lower() for t in available_tables}
        return all(t.lower() in at_lower for t in tables)

    # Simple validation & sanitization re-use
    validate_sql_query = getattr(aw, 'validate_sql_query', lambda q: {"valid": True})
    fix_common_sql_errors = getattr(aw, 'fix_common_sql_errors', lambda q: q)

    async def _verify_sql(sql: str, question: str, tag: str) -> str:
        """Optional light LLM verification (short prompt) if LLM available."""
        if not aw or not getattr(aw, 'Settings', None) or not getattr(aw.Settings, 'llm', None):
            return sql
        try:
            prompt = (
                f"{aw.EMPLOYEE_DB_SCHEMA}\n\nYou are an SQL verifier. Tag={tag}."
                " If the candidate SQL is valid and uses only existing tables/columns keep it."
                " If SMALL fixes (alias, LIMIT 100, semicolon) needed, fix them. Never invent tables/columns."
                f"\nQuestion: {question}\nCandidate SQL:\n{sql}\nReturn ONLY final SQL." )
            resp = aw.Settings.llm.complete(prompt=prompt, timeout=12.0)
            txt = getattr(resp, 'text', str(resp)).strip()
            if '```' in txt:
                parts = re.split(r"```(?:sql)?", txt, flags=re.IGNORECASE)
                txt = '\n'.join(p for p in parts if 'select' in p.lower() or 'with' in p.lower())
            if txt.lower().startswith('sql'):
                txt = txt[3:].strip()
            txt = fix_common_sql_errors(txt)
            v = validate_sql_query(txt)
            if v.get('valid'):
                return txt
        except Exception as ve:
            print(f"[FAST][VERIFY][WARN] Verification failed: {ve}; using original SQL")
        return sql

    def _build_sql(select_cols: List[str], base_table: str, joins: Optional[List[str]] = None,
                   where: Optional[List[str]] = None, group_by: Optional[List[str]] = None,
                   order_by: Optional[str] = None, limit: Optional[int] = 100) -> str:
        parts = ["SELECT " + ", ".join(select_cols), "FROM " + base_table]
        for j in (joins or []):
            parts.append(j)
        if where:
            parts.append("WHERE " + " AND ".join(where))
        if group_by:
            parts.append("GROUP BY " + ", ".join(group_by))
        if order_by:
            parts.append("ORDER BY " + order_by)
        if limit is not None:
            parts.append(f"LIMIT {limit}")
        sql = " ".join(parts)
        if not sql.strip().endswith(';'):
            sql += ';'
        return sql

    async def tool_list_departments(question: str = ""):
        if not _has_tables('department'):
            return {"error": "department table not available"}
        cols = _colnames('department')
        name_col = 'dept_name' if 'dept_name' in cols else (cols[0] if cols else 'dept_name')
        sql = _build_sql([f"d.{name_col} AS department_name"], 'department d', order_by='department_name ASC')
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'List departments', 'list_departments')
        return {"sql": sql}

    async def tool_count_employees(question: str = ""):
        if not _has_tables('employee'):
            return {"error": "employee table not available"}
        sql = _build_sql(["COUNT(*) AS total_employees"], 'employee e', limit=None)
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Total employees', 'count_employees')
        return {"sql": sql}

    async def tool_employee_count_by_department(question: str = ""):
        if not _has_tables('department','dept_emp'):
            return {"error": "required tables missing"}
        dcols = _colnames('department'); decols = _colnames('dept_emp')
        dept_name = 'dept_name' if 'dept_name' in dcols else (dcols[0] if dcols else 'dept_name')
        dept_no = 'dept_no' if 'dept_no' in dcols or 'dept_no' in decols else 'dept_no'
        emp_no = 'emp_no' if 'emp_no' in decols else 'emp_no'
        where = []
        if 'to_date' in decols:
            where.append("de.to_date = '9999-01-01'")
        sql = _build_sql([f"d.{dept_name} AS department_name", f"COUNT(DISTINCT de.{emp_no}) AS employee_count"],
                         'department d', joins=[f"JOIN dept_emp de ON d.{dept_no} = de.{dept_no}"],
                         where=where, group_by=[f"d.{dept_name}"], order_by='employee_count DESC')
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Employees per department', 'employee_count_by_department')
        return {"sql": sql}

    async def tool_gender_pay_gap(question: str = ""):
        if not _has_tables('department','dept_emp','employee','salary'):
            return {"error": "required tables missing"}
        scolumns = _colnames('salary')
        sval = 'salary' if 'salary' in scolumns else ('amount' if 'amount' in scolumns else (scolumns[0] if scolumns else 'salary'))
        where = ["de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'"]
        sql = _build_sql([
            "d.dept_name",
            f"AVG(CASE WHEN e.gender='M' THEN s.{sval} END) AS avg_male_salary",
            f"AVG(CASE WHEN e.gender='F' THEN s.{sval} END) AS avg_female_salary",
            f"(AVG(CASE WHEN e.gender='M' THEN s.{sval} END) - AVG(CASE WHEN e.gender='F' THEN s.{sval} END)) AS gap",
            f"CASE WHEN AVG(CASE WHEN e.gender='M' THEN s.{sval} END)=0 THEN NULL ELSE ((AVG(CASE WHEN e.gender='M' THEN s.{sval} END) - AVG(CASE WHEN e.gender='F' THEN s.{sval} END))/NULLIF(AVG(CASE WHEN e.gender='M' THEN s.{sval} END),0))*100 END AS pct_gap"
        ], 'department d',
            joins=['JOIN dept_emp de ON d.dept_no = de.dept_no','JOIN employee e ON de.emp_no = e.emp_no','JOIN salary s ON e.emp_no = s.emp_no'],
            where=where, group_by=['d.dept_name'], order_by='gap DESC')
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Gender pay gap', 'gender_pay_gap')
        return {"sql": sql}

    async def tool_salary_extremes(question: str = ""):
        if not _has_tables('salary'):
            return {"error": "salary table missing"}
        scolumns = _colnames('salary'); sval = 'salary' if 'salary' in scolumns else ('amount' if 'amount' in scolumns else (scolumns[0] if scolumns else 'salary'))
        where = ["s.to_date = '9999-01-01'"] if 'to_date' in scolumns else []
        sql = _build_sql([f"MAX(s.{sval}) AS highest_salary", f"MIN(s.{sval}) AS lowest_salary"], 'salary s', where=where, limit=None)
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Salary extremes', 'salary_extremes')
        return {"sql": sql}

    async def tool_salary_range_department(department_name: str = "Development", question: str = ""):
        if not _has_tables('department','dept_emp','salary'):
            return {"error": "required tables missing"}
        scolumns = _colnames('salary'); sval = 'salary' if 'salary' in scolumns else ('amount' if 'amount' in scolumns else (scolumns[0] if scolumns else 'salary'))
        # Escape single quotes in department_name for SQL safety (basic)
        escaped_dept = department_name.replace("'", "''")
        where = ["de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'", f"d.dept_name = '{escaped_dept}'"]
        sql = _build_sql(["d.dept_name", f"MAX(s.{sval}) AS max_salary", f"MIN(s.{sval}) AS min_salary"], 'department d',
                         joins=['JOIN dept_emp de ON d.dept_no = de.dept_no','JOIN salary s ON de.emp_no = s.emp_no'],
                         where=where, group_by=['d.dept_name'], limit=None)
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Department salary range', 'salary_range_department')
        return {"sql": sql, "department": department_name}

    async def tool_top_paid_employees(top_n: int = 10, question: str = ""):
        if not _has_tables('employee','salary'):
            return {"error": "required tables missing"}
        scolumns = _colnames('salary'); sval = 'salary' if 'salary' in scolumns else (scolumns[0] if scolumns else 'salary')
        where = ["s.to_date = '9999-01-01'"] if 'to_date' in scolumns else []
        sql = _build_sql(["e.emp_no","e.first_name","e.last_name", f"s.{sval} AS salary"], 'employee e',
                         joins=['JOIN salary s ON e.emp_no = s.emp_no'], where=where,
                         order_by=f"s.{sval} DESC", limit=top_n)
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Top paid employees', 'top_paid_employees')
        return {"sql": sql, "top_n": top_n}

    async def tool_department_highest_avg_salary(question: str = ""):
        if not _has_tables('department','dept_emp','salary'):
            return {"error": "required tables missing"}
        scolumns = _colnames('salary'); sval = 'salary' if 'salary' in scolumns else (scolumns[0] if scolumns else 'salary')
        where = ["de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'"]
        sql = _build_sql(["d.dept_name", f"AVG(s.{sval}) AS avg_salary"], 'department d',
                         joins=['JOIN dept_emp de ON d.dept_no = de.dept_no','JOIN salary s ON de.emp_no = s.emp_no'],
                         where=where, group_by=['d.dept_name'], order_by='avg_salary DESC', limit=1)
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Department highest average salary', 'dept_highest_avg_salary')
        return {"sql": sql}

    async def tool_department_manager_listing(question: str = ""):
        if not _has_tables('department','dept_manager','employee'):
            return {"error": "required tables missing"}
        where = ["dm.to_date = '9999-01-01'"]
        sql = _build_sql(["d.dept_name AS department_name","e.first_name","e.last_name"], 'department d',
                         joins=['JOIN dept_manager dm ON d.dept_no = dm.dept_no','JOIN employee e ON dm.emp_no = e.emp_no'],
                         where=where, order_by='department_name ASC')
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Department managers', 'department_manager_listing')
        return {"sql": sql}

    # NEW: Complex analytical tools for business intelligence
    async def tool_gender_distribution_by_dept(question: str = ""):
        if not _has_tables('department','dept_emp','employee'):
            return {"error": "required tables missing"}
        where = ["de.to_date = '9999-01-01'"]
        sql = _build_sql([
            "d.dept_name AS department_name",
            "e.gender",
            "COUNT(*) AS count",
            "ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY d.dept_name), 2) AS percentage"
        ], 'department d',
            joins=['JOIN dept_emp de ON d.dept_no = de.dept_no','JOIN employee e ON de.emp_no = e.emp_no'],
            where=where, group_by=['d.dept_name', 'e.gender'], order_by='department_name, gender')
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Gender distribution by department', 'gender_distribution_by_dept')
        return {"sql": sql}

    async def tool_hiring_trend_by_year(question: str = ""):
        if not _has_tables('employee'):
            return {"error": "required table employee missing"}
        sql = _build_sql([
            "EXTRACT(YEAR FROM e.hire_date) AS hire_year",
            "COUNT(*) AS employees_hired",
            "COUNT(CASE WHEN e.gender='M' THEN 1 END) AS male_hires",
            "COUNT(CASE WHEN e.gender='F' THEN 1 END) AS female_hires"
        ], 'employee e', group_by=['EXTRACT(YEAR FROM e.hire_date)'], order_by='hire_year ASC')
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Hiring trends by year', 'hiring_trend_by_year')
        return {"sql": sql}

    async def tool_employee_tenure_by_dept(question: str = ""):
        if not _has_tables('department','dept_emp','employee'):
            return {"error": "required tables missing"}
        where = ["de.to_date = '9999-01-01'"]
        sql = _build_sql([
            "d.dept_name AS department_name",
            "AVG(EXTRACT(DAYS FROM (CURRENT_DATE - e.hire_date))/365.25) AS avg_tenure_years",
            "MIN(EXTRACT(DAYS FROM (CURRENT_DATE - e.hire_date))/365.25) AS min_tenure_years", 
            "MAX(EXTRACT(DAYS FROM (CURRENT_DATE - e.hire_date))/365.25) AS max_tenure_years"
        ], 'department d',
            joins=['JOIN dept_emp de ON d.dept_no = de.dept_no','JOIN employee e ON de.emp_no = e.emp_no'],
            where=where, group_by=['d.dept_name'], order_by='avg_tenure_years DESC')
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Employee tenure by department', 'employee_tenure_by_dept')
        return {"sql": sql}

    async def tool_female_managers_count(question: str = ""):
        if not _has_tables('dept_manager','employee'):
            return {"error": "required tables missing"}
        where = ["dm.to_date = '9999-01-01'", "e.gender = 'F'"]
        sql = _build_sql([
            "COUNT(*) AS female_managers_count",
            "COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dept_manager dm2 WHERE dm2.to_date = '9999-01-01') AS percentage_female_managers"
        ], 'dept_manager dm',
            joins=['JOIN employee e ON dm.emp_no = e.emp_no'],
            where=where, limit=None)
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Female managers count', 'female_managers_count')
        return {"sql": sql}

    async def tool_avg_salary_by_title_dept(question: str = ""):
        """Get average salary by title and department."""
        if not _has_tables('department', 'dept_emp', 'salary', 'title', 'employee'):
            return {"error": "required tables missing"}
        where = ["de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'", "t.to_date = '9999-01-01'"]
        sql = _build_sql(["d.dept_name AS department_name", "t.title", "AVG(s.salary) AS avg_salary"], 'department d',
                         joins=['JOIN dept_emp de ON d.dept_no = de.dept_no',
                               'JOIN employee e ON de.emp_no = e.emp_no',
                               'JOIN salary s ON e.emp_no = s.emp_no',
                               'JOIN title t ON e.emp_no = t.emp_no'],
                         where=where, group_by=['d.dept_name', 't.title'], 
                         order_by='d.dept_name, avg_salary DESC', limit=100)
        sql = fix_common_sql_errors(sql)
        sql = await _verify_sql(sql, question or 'Average salary by title and department', 'avg_salary_by_title_dept')
        return {"sql": sql}

    register_tool('fast_sql.list_departments', tool_list_departments)
    register_tool('fast_sql.count_employees', tool_count_employees)
    register_tool('fast_sql.employee_count_by_department', tool_employee_count_by_department)
    register_tool('fast_sql.gender_pay_gap', tool_gender_pay_gap)  # RE-ENABLED: Complex analytical queries
    register_tool('fast_sql.salary_extremes', tool_salary_extremes)
    register_tool('fast_sql.salary_range_department', tool_salary_range_department)
    register_tool('fast_sql.top_paid_employees', tool_top_paid_employees)
    register_tool('fast_sql.department_highest_avg_salary', tool_department_highest_avg_salary)
    register_tool('fast_sql.department_manager_listing', tool_department_manager_listing)
    
    # Register new complex analytical tools
    register_tool('fast_sql.gender_distribution_by_dept', tool_gender_distribution_by_dept)
    register_tool('fast_sql.hiring_trend_by_year', tool_hiring_trend_by_year)
    register_tool('fast_sql.employee_tenure_by_dept', tool_employee_tenure_by_dept)
    register_tool('fast_sql.female_managers_count', tool_female_managers_count)
    register_tool('fast_sql.avg_salary_by_title_dept', tool_avg_salary_by_title_dept)
    print('[MCP STARTUP] Registered fast SQL tools: ' + ', '.join([k for k in TOOLS if k.startswith('fast_sql.')]))


@app.post("/complete")
async def complete(req: CompleteRequest):
    """Minimal completion endpoint:
    - If context contains {'tool_call': {'name': ..., 'args': {...}}}, the server will run the tool and return its result.
    - Otherwise it returns a simple echo of the prompt under 'text'.
    This scaffold is purposely small so you can extend tool orchestration later.
    """
    ctx = req.context or {}
    # Tool invocation flow
    if 'tool_call' in ctx:
        tc = ctx['tool_call']
        name = tc.get('name')
        args = tc.get('args', {})
        try:
            tool_res = await _call_tool(name, args)
            return {"text": f"Tool '{name}' executed.", "tool_result": tool_res}
        except KeyError as ke:
            raise HTTPException(status_code=404, detail=str(ke))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Tool call failed: {e}")

    # No tool requested: return a simple completion echo for now
    # Future: integrate a local model here and tool loop orchestration
    return {"text": req.prompt}


@app.get("/tools")
def list_tools():
    return {"tools": list(TOOLS.keys())}


@app.get("/health")
def health():
    return {"status": "ok", "tools_registered": len(TOOLS)}
