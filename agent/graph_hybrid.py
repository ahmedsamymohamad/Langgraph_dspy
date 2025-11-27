# agent/graph_hybrid.py

from __future__ import annotations
import json
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from agent.rag.retrieval import LocalDocRetriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import (
    router,
    planner,
    nl2sql,
    synth,
)


# ============================================================
# State definition
# ============================================================

class AgentState(TypedDict):
    """
    Shared state object for LangGraph. Keys include:
    question, format_hint, route, retrieved_docs, plan,
    sql, sql_result, rows, columns, error, citations,
    final_answer, confidence
    """
    question: str
    format_hint: str
    route: str
    retrieved_docs: list
    plan: dict
    sql: str
    sql_result: dict
    rows: list
    columns: list
    tables_used: list
    error: Optional[str]
    citations: list
    final_answer: str
    explanation: str
    repairs: int


# ============================================================
# Core Agent Components
# ============================================================

retriever = LocalDocRetriever("docs/")
sql_tool = SQLiteTool("data/northwind.sqlite")


# ============================================================
# Node: Router
# ============================================================

def node_router(state: AgentState) -> AgentState:
    q = state["question"]
    try:
        out = router(question=q)
        route = getattr(out, "route", "hybrid")
        if route:
            state["route"] = str(route).strip().lower()
        else:
            state["route"] = "hybrid"
    except Exception as e:
        # If router fails, default to hybrid approach
        state["route"] = "hybrid"
        state["error"] = f"Router failed: {str(e)[:100]}"
    return state


# ============================================================
# Node: RAG Retriever
# ============================================================

def node_retrieve(state: AgentState) -> AgentState:
    k = 6
    q = state["question"]
    docs = retriever.retrieve(q, k=k)
    state["retrieved_docs"] = docs
    return state


# ============================================================
# Node: Planner (extract entities & SQL constraints)
# ============================================================

def node_planner(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("retrieved_docs", [])
    plan_out = planner(question=q, retrieved_docs=docs)
    state["plan"] = plan_out.plan
    return state


# ============================================================
# Node: NL → SQL
# ============================================================

def node_sqlgen(state: AgentState) -> AgentState:
    q = state["question"]
    plan = state.get("plan", {})
    schema = sql_tool.get_schema_snapshot()
    try:
        sql_out = nl2sql(question=q, plan=plan, schema=schema)
    except Exception as e:
        state["sql"] = ""
        state["sql_result"] = {"error": f"NL2SQL call failed: {e}"}
        state["error"] = str(e)
        return state

    # Guard against DSPy adapters returning None or missing fields
    if sql_out is None:
        state["sql"] = ""
        state["sql_result"] = {"error": "NL2SQL produced no output"}
        state["error"] = "NL2SQL produced no output"
        return state

    sql_text = getattr(sql_out, "sql", None)
    if not sql_text:
        state["sql"] = ""
        state["sql_result"] = {"error": "NL2SQL returned empty SQL"}
        state["error"] = "NL2SQL returned empty SQL"
        return state

    sql_query = sql_text.strip()
    state["sql"] = sql_query
    return state


# ============================================================
# Node: SQL Executor
# ============================================================

def node_sqlexec(state: AgentState) -> AgentState:
    query = state.get("sql", "")
    if query == "":
        state["sql_result"] = {"error": "Empty SQL."}
        return state

    result = sql_tool.run_sql(query)
    state["sql_result"] = result
    state["rows"] = result.get("rows", [])
    state["columns"] = result.get("columns", [])
    state["tables_used"] = result.get("tables_used", [])
    state["error"] = result.get("error")
    return state


# ============================================================
# Node: Synthesizer
# ============================================================

def node_synthesize(state: AgentState) -> AgentState:
    q = state["question"]
    fmt = state["format_hint"]
    docs = state.get("retrieved_docs", [])
    plan = state.get("plan", {})
    sql = state.get("sql", "")
    rows = state.get("rows", [])
    cols = state.get("columns", [])
    tables = state.get("tables_used", [])

    try:
        out = synth(
            question=q,
            format_hint=fmt,
            plan=plan,
            retrieved_docs=docs,
            sql=sql,
            rows=rows,
            columns=cols,
            tables_used=tables,
        )
    except Exception as e:
        # If synthesis fails (e.g., timeout, parse error), create a fallback answer
        state["final_answer"] = f"Unable to synthesize answer: {str(e)[:100]}"
        state["explanation"] = "The synthesis step encountered an error."
        state["citations"] = []
        state["error"] = str(e)
        return state

    if out is None:
        state["final_answer"] = "No synthesis output"
        state["explanation"] = ""
        state["citations"] = []
        return state

    state["final_answer"] = getattr(out, "final_answer", "")
    state["explanation"] = getattr(out, "explanation", "")
    state["citations"] = getattr(out, "citations", [])
    return state


# ============================================================
# Node: Repair & Validation
# ============================================================

def node_validate_or_repair(state: AgentState) -> AgentState:
    error = state.get("error")

    # SQL failed → attempt repair
    if error:
        state.setdefault("repairs", 0)
        if state["repairs"] >= 2:
            return state

        # Force a retry: regenerate SQL
        state["repairs"] += 1
        new_state = state.copy()
        new_state["sql"] = ""
        return new_state  # signals loop
    return state


def should_repair(state: AgentState) -> str:
    """Routing decision after Synthesizer."""
    if state.get("error"):
        repairs = state.get("repairs", 0)
        if repairs < 2:
            return "repair"
    return "end"


# ============================================================
# Confidence Scoring
# ============================================================

def compute_confidence(state: AgentState) -> float:
    conf = 0.1

    if not state.get("error"):
        conf += 0.4

    repairs = state.get("repairs", 0)
    if repairs == 0:
        conf += 0.3

    docs = state.get("retrieved_docs", [])
    if docs:
        conf += 0.3

    return min(conf, 1.0)


# ============================================================
# Build the LangGraph
# ============================================================

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", node_router)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("planner", node_planner)
    graph.add_node("sqlgen", node_sqlgen)
    graph.add_node("sqlexec", node_sqlexec)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("repair", node_validate_or_repair)

    graph.set_entry_point("router")

    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "planner")
    graph.add_edge("planner", "sqlgen")
    graph.add_edge("sqlgen", "sqlexec")
    graph.add_edge("sqlexec", "synthesize")

    graph.add_conditional_edges(
        "synthesize",
        should_repair,
        {
            "repair": "repair",
            "end": END
        }
    )

    graph.add_edge("repair", "sqlgen")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# API for run_agent_hybrid.py
# ============================================================

agent_graph = build_graph()


def run_hybrid_agent(question: str, format_hint: str, checkpoint_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Entrypoint used by run_agent_hybrid.py
    Returns the final JSON output contract.
    """
    if checkpoint_config is None:
        checkpoint_config = {"thread_id": "main_thread"}

    # Ensure the input is a proper AgentState with ALL required keys
    inputs: AgentState = {
        "question": question,
        "format_hint": format_hint,
        "route": "",
        "retrieved_docs": [],
        "plan": {},
        "sql": "",
        "sql_result": {},
        "rows": [],
        "columns": [],
        "tables_used": [],
        "error": None,
        "citations": [],
        "final_answer": "",
        "explanation": "",
        "repairs": 0,
    }

    # invoke the LangGraph
    out = agent_graph.invoke(inputs, config=checkpoint_config)

    final = {
        "id": "",
        "final_answer": out.get("final_answer", ""),
        "sql": out.get("sql", ""),
        "confidence": compute_confidence(out),
        "explanation": out.get("explanation", ""),
        "citations": out.get("citations", []),
    }
    return final
