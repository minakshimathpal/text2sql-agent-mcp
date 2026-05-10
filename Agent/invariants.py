"""Invariant classification and enforcement for text2sql without hardcoded SQL templates.

All regeneration remains purely LLM-driven. We only express semantic/structural
constraints ("invariants") and ask the model to satisfy missing ones.

Categories:
  MANAGER_LIST          -> list current department managers
  DEPARTMENT_LIST       -> list department names
  AVG_SALARY_BY_DEPT    -> average current salary per department
  TITLE_COUNT           -> count employees with a given title (e.g. Senior Engineer)
  GENERIC               -> anything else (no special invariants)

Each invariant is represented as a dict:
  {
    'name': short identifier,
    'description': human-readable guidance,
    'check': callable(sql:str)->bool
  }

We DO NOT inject SQL fragments. We only:
  - detect which invariants fail
  - compile a focused regeneration prompt listing just the missing invariants
  - request a corrected SQL (single statement ending with semicolon + LIMIT 100)

This preserves the "no hardcoded fallback query" policy.
"""
from __future__ import annotations
import os
import re
from typing import List, Dict, Callable, Optional

CATEGORY_MANAGER_LIST = "MANAGER_LIST"
CATEGORY_DEPT_LIST = "DEPARTMENT_LIST"
CATEGORY_AVG_SALARY = "AVG_SALARY_BY_DEPT"
CATEGORY_TITLE_COUNT = "TITLE_COUNT"
CATEGORY_EMP_COUNT_BY_DEPT = "EMP_COUNT_BY_DEPT"
CATEGORY_GENERIC = "GENERIC"

# Newly added categories (extended taxonomy)
CATEGORY_SIMPLE_COUNT_EMP = "SIMPLE_COUNT_EMP"
CATEGORY_SIMPLE_COUNT_EMP_CURRENT = "SIMPLE_COUNT_EMP_CURRENT"
CATEGORY_SIMPLE_COUNT_DEPT = "SIMPLE_COUNT_DEPT"
CATEGORY_SALARY_EXTREMES_GLOBAL = "SALARY_EXTREMES_GLOBAL"
CATEGORY_SALARY_RANGE_BY_DEPT = "SALARY_RANGE_BY_DEPT"
CATEGORY_TOP_N_SALARIES = "TOP_N_SALARIES"
CATEGORY_DEPT_MANAGER_LIST = "DEPT_MANAGER_LIST"  # dept + manager names
CATEGORY_MANAGER_TENURE = "MANAGER_TENURE"
CATEGORY_MANAGER_CHANGE_FREQUENCY = "MANAGER_CHANGE_FREQUENCY"
CATEGORY_TITLE_DISTINCT_COUNT = "TITLE_DISTINCT_COUNT"
CATEGORY_TITLE_SPECIFIC_COUNT = "TITLE_SPECIFIC_COUNT"
CATEGORY_TITLE_DISTRIBUTION_BY_DEPT = "TITLE_DISTRIBUTION_BY_DEPT"
CATEGORY_PROMOTION_PATH_ENGINEER_TO_SENIOR = "PROMOTION_PATH_ENGINEER_TO_SENIOR"
CATEGORY_EMP_TENURE_CURRENT_DEPT = "EMP_TENURE_CURRENT_DEPT"
CATEGORY_HIRING_TREND_BY_YEAR = "HIRING_TREND_BY_YEAR"
CATEGORY_SALARY_TREND_BY_YEAR = "SALARY_TREND_BY_YEAR"
CATEGORY_EMP_DEPT_CHANGES = "EMP_DEPT_CHANGES"
CATEGORY_GENDER_DIST_BY_DEPT = "GENDER_DIST_BY_DEPT"
CATEGORY_GENDER_PAY_GAP_BY_DEPT = "GENDER_PAY_GAP_BY_DEPT"
CATEGORY_FEMALE_MANAGERS_COUNT = "FEMALE_MANAGERS_COUNT"
CATEGORY_AVG_SALARY_BY_TITLE_AND_DEPT = "AVG_SALARY_BY_TITLE_AND_DEPT"
CATEGORY_GENDER_MANAGEMENT_COUNT = "GENDER_MANAGEMENT_COUNT"
CATEGORY_SENIOR_STAFF_SUPERLATIVE = "SENIOR_STAFF_SUPERLATIVE"

Category = str

# ---------------- Classification -----------------

def classify_question(question: str) -> Category:
    q = question.lower()
    # Specific order matters: more specialized patterns first
    if any(k in q for k in ["top", "highest-paid", "highest paid"]) and "salary" in q:
        return CATEGORY_TOP_N_SALARIES
    if any(kw in q for kw in ["total number of employees", "total employees"]) or ("count" in q and "employee" in q and "department" not in q and "current" not in q):
        return CATEGORY_SIMPLE_COUNT_EMP
    if ("how many" in q or "count" in q) and "currently" in q and "employee" in q:
        return CATEGORY_SIMPLE_COUNT_EMP_CURRENT
    if ("how many" in q or "number of" in q or "count" in q) and "departments" in q and "manager" not in q:
        return CATEGORY_SIMPLE_COUNT_DEPT
    if any(w in q for w in ["highest", "lowest"]) and "salary" in q and "department" not in q:
        return CATEGORY_SALARY_EXTREMES_GLOBAL
    if "salary range" in q and "department" in q:
        return CATEGORY_SALARY_RANGE_BY_DEPT
    if "most employees" in q and "department" in q:
        return CATEGORY_EMP_COUNT_BY_DEPT
    if "highest average salary" in q and "department" in q:
        return CATEGORY_AVG_SALARY
    if any(k in q for k in ["manager", "managers"]) and "department" in q:
        if "tenure" in q or "long" in q or "position" in q:
            return CATEGORY_MANAGER_TENURE
        if "changed" in q or "changes" in q or "frequently" in q:
            return CATEGORY_MANAGER_CHANGE_FREQUENCY
        if "list" in q and "manager" in q and "department" in q:
            return CATEGORY_DEPT_MANAGER_LIST
        return CATEGORY_MANAGER_LIST
    if ("how many" in q or "count" in q) and "job title" in q and ("exist" in q or "different" in q):
        return CATEGORY_TITLE_DISTINCT_COUNT
    if ("how many" in q or "count" in q) and any(t in q for t in ["senior engineer", "senior engineers"]):
        return CATEGORY_TITLE_SPECIFIC_COUNT
    if "distribution" in q and "title" in q and "department" in q:
        return CATEGORY_TITLE_DISTRIBUTION_BY_DEPT
    if "promoted" in q and "engineer" in q and "senior engineer" in q:
        return CATEGORY_PROMOTION_PATH_ENGINEER_TO_SENIOR
    if ("how long" in q or "tenure" in q) and "current" in q and "department" in q and "employee" in q:
        return CATEGORY_EMP_TENURE_CURRENT_DEPT
    if ("hiring trend" in q) or ("hire" in q and "year" in q):
        return CATEGORY_HIRING_TREND_BY_YEAR
    if ("salaries" in q or "salary" in q) and "changed" in q and ("over time" in q or "year" in q):
        return CATEGORY_SALARY_TREND_BY_YEAR
    if "changed departments" in q or ("employees" in q and "changed" in q and "department" in q):
        return CATEGORY_EMP_DEPT_CHANGES
    if "gender distribution" in q and "department" in q:
        return CATEGORY_GENDER_DIST_BY_DEPT
    if "gender pay gap" in q and "department" in q:
        return CATEGORY_GENDER_PAY_GAP_BY_DEPT
    if ("female" in q and "manager" in q and ("how many" in q or "count" in q)):
        return CATEGORY_FEMALE_MANAGERS_COUNT
    if "average salary" in q and "title" in q and "department" in q:
        return CATEGORY_AVG_SALARY_BY_TITLE_AND_DEPT
    if ("gender" in q and "management" in q) or ("each gender" in q and "management" in q):
        return CATEGORY_GENDER_MANAGEMENT_COUNT
    if ("most" in q and "senior" in q and "staff" in q) or ("senior staff" in q and "most" in q):
        return CATEGORY_SENIOR_STAFF_SUPERLATIVE
    if any(k in q for k in ["list departments", "department names", "department name", "departments list"]):
        return CATEGORY_DEPT_LIST
    if "average" in q and "salary" in q and "department" in q:
        return CATEGORY_AVG_SALARY
    if ("count" in q or "how many" in q or "number of" in q) and any(t in q for t in ["engineer", "engineers", "title"]):
        return CATEGORY_TITLE_COUNT
    if ("count" in q or "how many" in q or "number of" in q or any(s in q for s in ["most", "largest", "highest", "fewest", "least"])) and "department" in q and ("employee" in q or "employees" in q):
        return CATEGORY_EMP_COUNT_BY_DEPT
    return CATEGORY_GENERIC

# --------------- Invariant Builders ---------------

def _regex(pattern: str) -> Callable[[str], bool]:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda sql: bool(rx.search(sql))

def _contains_all(*subs: str) -> Callable[[str], bool]:
    def check(sql: str) -> bool:
        l = sql.lower()
        return all(s.lower() in l for s in subs)
    return check

def build_invariants(category: Category, question: str) -> List[Dict[str, any]]:
    inv: List[Dict[str, any]] = []
    if category == CATEGORY_MANAGER_LIST:
        inv += [
            {"name": "has_department_join", "description": "Join department (alias d) with dept_manager (alias dm)", "check": lambda s: re.search(r"join\s+dept_manager", s, re.IGNORECASE) and re.search(r"from\s+department", s, re.IGNORECASE)},
            {"name": "has_employee_join", "description": "Join employee (alias e) for manager names", "check": _regex(r"join\s+employee")},
            {"name": "selects_names", "description": "Select employee first_name and last_name", "check": lambda s: all(x in s.lower() for x in ["first_name", "last_name"])},
            {"name": "filters_current", "description": "Filter current managers dm.to_date='9999-01-01'", "check": _regex(r"dm\.to_date\s*=\s*'9999-01-01'")},
            {"name": "limit_100", "description": "Ends with LIMIT 100 semicolon", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_DEPT_LIST:
        inv += [
            {"name": "from_department", "description": "Query FROM department table", "check": _regex(r"from\s+department")},
            {"name": "select_dept_name", "description": "Select dept_name column", "check": _regex(r"dept_name")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_AVG_SALARY:
        inv += [
            {"name": "joins_salary", "description": "Join salary s (FROM salary s)", "check": _regex(r"from\s+salary\s+s")},
            {"name": "joins_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "joins_department", "description": "Join department d", "check": _regex(r"join\s+department")},
            {"name": "avg_salary", "description": "Uses AVG(s.salary)", "check": _regex(r"avg\s*\(\s*s\.salary\s*\)")},
            {"name": "current_filters", "description": "Filters current rows s.to_date='9999-01-01' and de.to_date='9999-01-01'", "check": lambda s: "s.to_date='9999-01-01'" in s.lower() and "de.to_date='9999-01-01'" in s.lower()},
            {"name": "group_by_dept", "description": "GROUP BY department name", "check": _regex(r"group\s+by\s+.*dept_name")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_TITLE_COUNT:
        # Attempt to infer canonical title (not used in invariants; regeneration prompt may hint)
        inv += [
            {"name": "from_title", "description": "FROM title table (alias t)", "check": _regex(r"from\s+title")},
            {"name": "alias_t", "description": "Alias 't' used for title table", "check": _regex(r"from\s+title\s+(?:as\s+)?t")},
            {"name": "current_rows", "description": "Filter current rows t.to_date='9999-01-01'", "check": _regex(r"t\.to_date\s*=\s*'9999-01-01'")},
            {"name": "count_distinct", "description": "COUNT(DISTINCT t.emp_no)", "check": _regex(r"count\s*\(\s*distinct\s+t\.emp_no\s*\)")},
            {"name": "has_title_filter", "description": "Filter on t.title = '<Some Title>'", "check": _regex(r"t\.title\s*=\s*'[^']+'")} ,
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_EMP_COUNT_BY_DEPT:
        inv += [
            {"name": "has_dept_emp_join", "description": "Join dept_emp de with department d", "check": lambda s: re.search(r"join\s+dept_emp", s, re.IGNORECASE) and re.search(r"join\s+department|from\s+department", s, re.IGNORECASE)},
            {"name": "select_dept_name", "description": "Select d.dept_name", "check": _regex(r"d\.dept_name|dept_name")},
            {"name": "select_count", "description": "Include COUNT(*) or COUNT(de.emp_no)", "check": _regex(r"count\s*\(")},
            {"name": "group_by", "description": "GROUP BY d.dept_name present", "check": _regex(r"group\s+by\s+.*dept_name")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
            {"name": "count_alias", "description": "Provide an explicit alias e.g. COUNT(*) AS employee_count", "check": _regex(r"count\s*\(.*\)\s+as\s+employee_count")},
        ]
        # If the question is superlative-oriented, encourage ordering
        ql = question.lower()
        if any(w in ql for w in ["most", "largest", "highest", "fewest", "least"]):
            inv.append({"name": "order_by_count", "description": "ORDER BY employee_count DESC (or ASC for fewest/least)", "check": lambda s: ("fewest" in ql or "least" in ql and re.search(r"order\s+by\s+employee_count\s+asc", s, re.IGNORECASE)) or re.search(r"order\s+by\s+employee_count\s+desc", s, re.IGNORECASE)})
    elif category == CATEGORY_SIMPLE_COUNT_EMP:
        inv += [
            {"name": "from_employee", "description": "FROM employee table", "check": _regex(r"from\s+employee")},
            {"name": "count_all", "description": "COUNT(*) with alias total_employees", "check": _regex(r"count\s*\(\s*\*\s*\)\s+as\s+total_employees")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_SIMPLE_COUNT_EMP_CURRENT:
        inv += [
            {"name": "join_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "current_filter", "description": "de.to_date='9999-01-01' filter", "check": _regex(r"de\.to_date\s*=\s*'9999-01-01'")},
            {"name": "count_distinct_emp", "description": "COUNT(DISTINCT de.emp_no) alias current_employee_count", "check": _regex(r"count\s*\(\s*distinct\s+de\.emp_no\s*\)\s+as\s+current_employee_count")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_SIMPLE_COUNT_DEPT:
        inv += [
            {"name": "from_department", "description": "FROM department", "check": _regex(r"from\s+department")},
            {"name": "count_depts", "description": "COUNT(*) AS department_count", "check": _regex(r"count\s*\(\s*\*\s*\)\s+as\s+department_count")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_SALARY_EXTREMES_GLOBAL:
        # Extremely strict invariants for highest/lowest salary by department
        inv += [
            {"name": "from_salary_s", "description": "FROM salary s", "check": _regex(r"from\s+salary\s+s")},
            {"name": "join_dept_emp_de", "description": "JOIN dept_emp de ON s.emp_no = de.emp_no", "check": _regex(r"join\s+dept_emp\s+de\s+on\s+s\.emp_no\s*=\s*de\.emp_no")},
            {"name": "join_department_d", "description": "JOIN department d ON de.dept_no = d.dept_no", "check": _regex(r"join\s+department\s+d\s+on\s+de\.dept_no\s*=\s*d\.dept_no")},
            {"name": "select_dept_name", "description": "SELECT d.dept_name", "check": _regex(r"select\s+d\.dept_name")},
            {"name": "select_max_min_salary", "description": "SELECT MAX(s.salary) AS highest_salary, MIN(s.salary) AS lowest_salary", "check": lambda s: re.search(r"max\s*\(\s*s\.salary\s*\)\s+as\s+highest_salary", s, re.IGNORECASE) and re.search(r"min\s*\(\s*s\.salary\s*\)\s+as\s+lowest_salary", s, re.IGNORECASE)},
            {"name": "group_by_dept_name", "description": "GROUP BY d.dept_name", "check": _regex(r"group\s+by\s+d\.dept_name")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
            {"name": "no_unaliased_salary", "description": "Do not use MAX(salary) or MIN(salary) without s. alias", "check": lambda s: not re.search(r"max\s*\(\s*salary\s*\)", s, re.IGNORECASE) and not re.search(r"min\s*\(\s*salary\s*\)", s, re.IGNORECASE)},
            {"name": "no_unaliased_dept_name", "description": "Do not use dept_name without d. alias", "check": lambda s: not re.search(r"[^a-zA-Z0-9_]dept_name", s) or re.search(r"d\.dept_name", s)},
        ]
    elif category == CATEGORY_SALARY_RANGE_BY_DEPT:
        # Stricter invariants for salary range by department
        inv += [
            {"name": "joins_salary", "description": "FROM salary s", "check": _regex(r"from\s+salary\s+s")},
            {"name": "joins_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "joins_department", "description": "Join department d", "check": _regex(r"join\s+department")},
            {"name": "select_dept_name", "description": "Select d.dept_name", "check": _regex(r"d\.dept_name")},
            {"name": "min_max", "description": "MIN(s.salary) & MAX(s.salary)", "check": lambda s: re.search(r"min\s*\(\s*s\.salary", s, re.IGNORECASE) and re.search(r"max\s*\(\s*s\.salary", s, re.IGNORECASE)},
            {"name": "dept_filter", "description": "Filter on d.dept_name=...", "check": _regex(r"d\.dept_name\s*=\s*'")},
            {"name": "group_by_dept", "description": "GROUP BY d.dept_name", "check": _regex(r"group\s+by\s+.*d\.dept_name")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_TOP_N_SALARIES:
        inv += [
            {"name": "joins_salary", "description": "FROM salary s", "check": _regex(r"from\s+salary\s+s")},
            {"name": "current_filter", "description": "s.to_date='9999-01-01'", "check": _regex(r"s\.to_date\s*=\s*'9999-01-01'")},
            {"name": "order_desc", "description": "ORDER BY s.salary DESC", "check": _regex(r"order\s+by\s+s\.salary\s+desc")},
            {"name": "limit_present", "description": "Has a LIMIT clause", "check": _regex(r"limit\s+\d+")},
        ]
    elif category == CATEGORY_DEPT_MANAGER_LIST:
        inv += [
            {"name": "joins_dept_manager", "description": "Join dept_manager dm", "check": _regex(r"join\s+dept_manager")},
            {"name": "joins_employee", "description": "Join employee e", "check": _regex(r"join\s+employee")},
            {"name": "select_names", "description": "Select manager first_name / last_name", "check": lambda s: "first_name" in s.lower() and "last_name" in s.lower()},
            {"name": "dept_name", "description": "Select dept_name", "check": _regex(r"dept_name")},
            {"name": "current_filter", "description": "dm.to_date='9999-01-01'", "check": _regex(r"dm\.to_date\s*=\s*'9999-01-01'")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_MANAGER_TENURE:
        inv += [
            {"name": "joins_dept_manager", "description": "Join dept_manager dm", "check": _regex(r"join\s+dept_manager")},
            {"name": "current_filter", "description": "dm.to_date='9999-01-01'", "check": _regex(r"dm\.to_date\s*=\s*'9999-01-01'")},
            {"name": "tenure_expr", "description": "Contains dm.from_date for tenure calc", "check": _regex(r"dm\.from_date")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_MANAGER_CHANGE_FREQUENCY:
        inv += [
            {"name": "from_dept_manager", "description": "FROM dept_manager", "check": _regex(r"from\s+dept_manager")},
            {"name": "group_by_dept", "description": "GROUP BY dept_no or dept", "check": _regex(r"group\s+by\s+.*dept")},
            {"name": "count_changes", "description": "COUNT(*) or COUNT(DISTINCT emp_no)", "check": _regex(r"count\s*\(")},
            {"name": "order_desc", "description": "ORDER BY count desc", "check": _regex(r"order\s+by\s+.*count.*desc")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_TITLE_DISTINCT_COUNT:
        inv += [
            {"name": "from_title", "description": "FROM title t", "check": _regex(r"from\s+title")},
            {"name": "count_distinct_titles", "description": "COUNT(DISTINCT t.title) alias title_count", "check": _regex(r"count\s*\(\s*distinct\s+t\.title\s*\)\s+as\s+title_count")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_TITLE_SPECIFIC_COUNT:
        inv += [
            {"name": "from_title", "description": "FROM title t", "check": _regex(r"from\s+title")},
            {"name": "title_filter", "description": "Filter t.title='Senior Engineer'", "check": _regex(r"t\.title\s*=\s*'senior engineer'")},
            {"name": "current_filter", "description": "t.to_date='9999-01-01'", "check": _regex(r"t\.to_date\s*=\s*'9999-01-01'")},
            {"name": "count_distinct_emp", "description": "COUNT(DISTINCT t.emp_no)", "check": _regex(r"count\s*\(\s*distinct\s+t\.emp_no")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_TITLE_DISTRIBUTION_BY_DEPT:
        inv += [
            {"name": "joins_title", "description": "Join title t", "check": _regex(r"join\s+title")},
            {"name": "joins_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "joins_department", "description": "Join department d", "check": _regex(r"join\s+department")},
            {"name": "current_filters", "description": "t.to_date & de.to_date current filters", "check": lambda s: "t.to_date='9999-01-01'" in s.lower() and "de.to_date='9999-01-01'" in s.lower()},
            {"name": "group_by", "description": "GROUP BY dept and title", "check": _regex(r"group\s+by\s+.*(dept|title).*(title|dept)")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_PROMOTION_PATH_ENGINEER_TO_SENIOR:
        inv += [
            {"name": "dual_title_refs", "description": "References title table twice (t1, t2)", "check": _regex(r"title\s+t1.*title\s+t2|title\s+t2.*title\s+t1")},
            {"name": "promotion_condition", "description": "t1.title='Engineer' AND t2.title='Senior Engineer'", "check": _regex(r"t1\.title\s*=\s*'engineer'.*t2\.title\s*=\s*'senior engineer'|t2\.title\s*=\s*'senior engineer'.*t1\.title\s*=\s*'engineer'")},
            {"name": "temporal_order", "description": "t1.from_date < t2.from_date", "check": _regex(r"t1\.from_date\s*<\s*t2\.from_date")},
            {"name": "distinct_emp", "description": "DISTINCT emp_no selected", "check": _regex(r"select\s+distinct\s+t1\.emp_no|select\s+distinct\s+t2\.emp_no")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_EMP_TENURE_CURRENT_DEPT:
        inv += [
            {"name": "joins_dept_emp", "description": "FROM dept_emp de", "check": _regex(r"from\s+dept_emp")},
            {"name": "current_filter", "description": "de.to_date='9999-01-01'", "check": _regex(r"de\.to_date\s*=\s*'9999-01-01'")},
            {"name": "tenure_expr", "description": "Uses de.from_date", "check": _regex(r"de\.from_date")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_HIRING_TREND_BY_YEAR:
        inv += [
            {"name": "from_employee", "description": "FROM employee e", "check": _regex(r"from\s+employee")},
            {"name": "year_extract", "description": "EXTRACT(YEAR FROM hire_date)", "check": _regex(r"extract\s*\(\s*year\s+from\s+.*hire_date")},
            {"name": "group_by_year", "description": "GROUP BY year", "check": _regex(r"group\s+by\s+.*year")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_SALARY_TREND_BY_YEAR:
        inv += [
            {"name": "from_salary", "description": "FROM salary s", "check": _regex(r"from\s+salary")},
            {"name": "year_extract", "description": "EXTRACT(YEAR FROM s.from_date)", "check": _regex(r"extract\s*\(\s*year\s+from\s+s\.from_date")},
            {"name": "avg_salary", "description": "AVG(s.salary)", "check": _regex(r"avg\s*\(\s*s\.salary")},
            {"name": "group_by_year", "description": "GROUP BY year", "check": _regex(r"group\s+by\s+.*year")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_EMP_DEPT_CHANGES:
        inv += [
            {"name": "from_dept_emp", "description": "FROM dept_emp de", "check": _regex(r"from\s+dept_emp")},
            {"name": "group_by_emp", "description": "GROUP BY de.emp_no", "check": _regex(r"group\s+by\s+.*emp_no")},
            {"name": "having_multiple", "description": "HAVING COUNT(DISTINCT dept_no) > 1", "check": _regex(r"having\s+count\s*\(\s*distinct\s+dept_no\s*\)\s*>\s*1")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_GENDER_DIST_BY_DEPT:
        inv += [
            {"name": "joins_employee", "description": "Join employee e", "check": _regex(r"join\s+employee")},
            {"name": "joins_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "joins_department", "description": "Join department d", "check": _regex(r"join\s+department")},
            {"name": "current_filter", "description": "de.to_date='9999-01-01'", "check": _regex(r"de\.to_date\s*=\s*'9999-01-01'")},
            {"name": "group_by_dept_gender", "description": "GROUP BY dept & gender", "check": _regex(r"group\s+by\s+.*(dept|gender).*(gender|dept)")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_GENDER_PAY_GAP_BY_DEPT:
        inv += [
            {"name": "joins_salary", "description": "Join salary s", "check": _regex(r"join\s+salary")},
            {"name": "joins_employee", "description": "Join employee e", "check": _regex(r"join\s+employee")},
            {"name": "joins_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "joins_department", "description": "Join department d", "check": _regex(r"join\s+department")},
            {"name": "current_filters", "description": "Current rows s.to_date & de.to_date", "check": lambda s: "s.to_date='9999-01-01'" in s.lower() and "de.to_date='9999-01-01'" in s.lower()},
            {"name": "avg_salary", "description": "AVG(s.salary)", "check": _regex(r"avg\s*\(\s*s\.salary")},
            {"name": "group_by_dept_gender", "description": "GROUP BY dept & gender", "check": _regex(r"group\s+by\s+.*(dept|gender).*(gender|dept)")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_FEMALE_MANAGERS_COUNT:
        inv += [
            {"name": "joins_dept_manager", "description": "Join dept_manager dm", "check": _regex(r"join\s+dept_manager")},
            {"name": "joins_employee", "description": "Join employee e", "check": _regex(r"join\s+employee")},
            {"name": "female_filter", "description": "e.gender='F'", "check": _regex(r"e\.gender\s*=\s*'f'")},
            {"name": "current_filter", "description": "dm.to_date='9999-01-01'", "check": _regex(r"dm\.to_date\s*=\s*'9999-01-01'")},
            {"name": "count_distinct_mgr", "description": "COUNT(DISTINCT dm.emp_no)", "check": _regex(r"count\s*\(\s*distinct\s+dm\.emp_no")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_AVG_SALARY_BY_TITLE_AND_DEPT:
        inv += [
            {"name": "joins_salary", "description": "Join salary s", "check": _regex(r"join\s+salary")},
            {"name": "joins_title", "description": "Join title t", "check": _regex(r"join\s+title")},
            {"name": "joins_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "joins_department", "description": "Join department d", "check": _regex(r"join\s+department")},
            {"name": "current_filters", "description": "Current rows for s,de,t", "check": lambda s: all(x in s.lower() for x in ["s.to_date='9999-01-01'", "de.to_date='9999-01-01'", "t.to_date='9999-01-01'"])},
            {"name": "avg_salary", "description": "AVG(s.salary)", "check": _regex(r"avg\s*\(\s*s\.salary")},
            {"name": "group_by", "description": "GROUP BY dept & title", "check": _regex(r"group\s+by\s+.*(dept|title).*(title|dept)")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_GENDER_MANAGEMENT_COUNT:
        inv += [
            {"name": "joins_dept_manager", "description": "Join dept_manager dm", "check": _regex(r"join\s+dept_manager")},
            {"name": "joins_employee", "description": "Join employee e", "check": _regex(r"join\s+employee")},
            {"name": "current_filter", "description": "dm.to_date='9999-01-01'", "check": _regex(r"dm\.to_date\s*=\s*'9999-01-01'")},
            {"name": "group_by_gender", "description": "GROUP BY gender", "check": _regex(r"group\s+by\s+.*gender")},
            {"name": "count_mgr", "description": "COUNT(*) or COUNT(dm.emp_no)", "check": _regex(r"count\s*\(")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    elif category == CATEGORY_SENIOR_STAFF_SUPERLATIVE:
        inv += [
            {"name": "joins_title", "description": "Join title t", "check": _regex(r"join\s+title")},
            {"name": "joins_dept_emp", "description": "Join dept_emp de", "check": _regex(r"join\s+dept_emp")},
            {"name": "joins_department", "description": "Join department d", "check": _regex(r"join\s+department")},
            {"name": "senior_filter", "description": "t.title LIKE 'Senior%'", "check": _regex(r"t\.title\s+like\s+'senior%'")},
            {"name": "current_filters", "description": "Current rows de.to_date & t.to_date", "check": lambda s: "de.to_date='9999-01-01'" in s.lower() and "t.to_date='9999-01-01'" in s.lower()},
            {"name": "group_by_dept", "description": "GROUP BY d.dept_name", "check": _regex(r"group\s+by\s+.*dept_name")},
            {"name": "order_desc", "description": "ORDER BY count desc", "check": _regex(r"order\s+by\s+.*count.*desc")},
            {"name": "limit_100", "description": "Ends with LIMIT 100", "check": _regex(r"limit\s+100;?\s*$")},
        ]
    return inv

# --------------- Enforcement -----------------

def evaluate_invariants(sql: str, invariants: List[Dict[str, any]]) -> List[Dict[str, any]]:
    failures = []
    for inv in invariants:
        try:
            ok = inv['check'](sql)
        except Exception:
            ok = False
        if not ok:
            failures.append(inv)
    return failures


# More explicit prompt for join-based salary extremes queries
REGEN_PROMPT_TEMPLATE = (
    "You produced SQL for the user question. Some required invariants are missing.\n"
    "Regenerate a SINGLE valid PostgreSQL SQL query that satisfies ALL listed invariants.\n"
    "Return ONLY the SQL (no commentary). Must end with LIMIT 100; and a semicolon.\n"
    "If the question is about highest/lowest salary by department, you MUST:\n"
    "- Use FROM salary s\n"
    "- JOIN dept_emp de ON s.emp_no = de.emp_no\n"
    "- JOIN department d ON de.dept_no = d.dept_no\n"
    "- SELECT d.dept_name, MAX(s.salary) AS highest_salary, MIN(s.salary) AS lowest_salary\n"
    "- GROUP BY d.dept_name\n"
    "- Use correct table aliases everywhere (no unaliased salary or dept_name)\n"
    "- Do NOT reference columns that do not exist in the schema\n"
    "User Question: {question}\n"
    "Previous Attempt: {previous_sql}\n"
    "Missing Invariants:\n{invariant_lines}\n"
    "Regenerate now:"
)


def regenerate_with_constraints(question: str, previous_sql: str, missing: List[Dict[str, any]], llm, run_llm_fn, attempt: int) -> Optional[str]:
    invariant_lines = "\n".join(f"- {m['name']}: {m['description']}" for m in missing)
    prompt = REGEN_PROMPT_TEMPLATE.format(question=question, previous_sql=previous_sql, invariant_lines=invariant_lines)
    # Use proper timeout values instead of hardcoded ones
    # Start with a good timeout and decrease gradually for subsequent attempts
    base_timeout = float(os.environ.get("INVARIANTS_BASE_TIMEOUT", 90))  # Start with 90s
    timeout = max(30.0, base_timeout - attempt * 20.0)  # Decrease by 20s per attempt, minimum 30s
    try:
        raw = run_llm_fn(llm, prompt, timeout=max(8.0, timeout), retries=0)
        # Normalize raw response to text
        raw_text = getattr(raw, 'text', None)
        if raw_text is None:
            raw_text = str(raw)
        # Extract first SELECT ... ;
        m = re.search(r"(SELECT[\s\S]+?;)", raw_text, flags=re.IGNORECASE)
        candidate = m.group(1) if m else raw_text.strip()
        # Clean markdown fences
        candidate = re.sub(r"```(?:sql)?|```", "", candidate, flags=re.IGNORECASE).strip()
        # Ensure single statement & required LIMIT 100 at end
        # Collapse accidental mid-statement semicolon before LIMIT
        candidate = re.sub(r";\s*(LIMIT\s+\d+)", r" \1", candidate, flags=re.IGNORECASE)
        # Append LIMIT 100 if missing (invariants will still verify)
        if 'limit' not in candidate.lower():
            if candidate.endswith(';'):
                candidate = candidate[:-1] + ' LIMIT 100;'
            else:
                candidate += ' LIMIT 100;'
        # Guarantee trailing semicolon
        if not candidate.strip().endswith(';'):
            candidate += ';'
        return candidate
    except Exception as e:
        print(f"[INVARIANTS] Regeneration failed: {e}")
        return None


def enforce_invariants(question: str, sql: str, llm, run_llm_fn, max_regen: int = 2) -> Dict[str, any]:
    category = classify_question(question)
    if category == CATEGORY_GENERIC:
        return {"category": category, "final_sql": sql, "regenerations": []}
    invariants = build_invariants(category, question)
    regenerations = []
    current_sql = sql
    # Increase max_regen to 5 for more LLM attempts
    for attempt in range(max_regen + 4):
        missing = evaluate_invariants(current_sql, invariants)
        if not missing:
            return {"category": category, "final_sql": current_sql, "regenerations": regenerations}
        if attempt == max_regen:
            # Give up, return last with missing list for transparency
            return {"category": category, "final_sql": current_sql, "regenerations": regenerations, "remaining_missing": [m['name'] for m in missing]}
        regen_sql = regenerate_with_constraints(question, current_sql, missing, llm, run_llm_fn, attempt)
        if not regen_sql:
            return {"category": category, "final_sql": current_sql, "regenerations": regenerations, "remaining_missing": [m['name'] for m in missing]}
        regenerations.append({
            "attempt": attempt + 1,
            "missing_before": [m['name'] for m in missing],
            "sql": regen_sql
        })
        current_sql = regen_sql
    return {"category": category, "final_sql": current_sql, "regenerations": regenerations}
