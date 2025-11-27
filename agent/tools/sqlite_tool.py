# agent/tools/sqlite_tool.py

import sqlite3
import re
from typing import Dict, Any, List, Tuple


class SQLiteTool:
    """
    SQLite execution + schema introspection + table extraction for citations.
    """

    def __init__(self, db_path: str = "../../data/northwind.sqlite"):
        self.db_path = db_path

    # -------------------------------------------
    # Connection manager
    # -------------------------------------------
    def _connect(self):
        return sqlite3.connect(self.db_path)

    # -------------------------------------------
    # Schema Introspection
    # -------------------------------------------
    def get_tables(self) -> List[str]:
        """
        Return all table names from SQLite.
        """
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cur.fetchall()]
            conn.close()
            return tables
        except Exception:
            return []

    def get_table_columns(self, table: str) -> List[str]:
        """
        Returns list of column names for a given table.
        """
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info('{table}')")
            cols = [row[1] for row in cur.fetchall()]
            conn.close()
            return cols
        except Exception:
            return []

    def get_schema_snapshot(self) -> Dict[str, List[str]]:
        """
        Returns the full schema in dict form: {table: [cols]}
        """
        schema = {}
        tables = self.get_tables()
        for t in tables:
            schema[t] = self.get_table_columns(t)
        return schema

    # -------------------------------------------
    # SQL Execution
    # -------------------------------------------
    def run_sql(self, query: str) -> Dict[str, Any]:
        """
        Executes SQL safely and returns:
        {
            'columns': [...],
            'rows': [...],
            'tables_used': [...],
            'error': <str or None>
        }
        """
        result = {
            "columns": [],
            "rows": [],
            "tables_used": self.extract_tables_used(query),
            "error": None,
        }

        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(query)

            # SELECT → fetch
            if query.strip().lower().startswith("select"):
                result["columns"] = [desc[0] for desc in cur.description]
                result["rows"] = cur.fetchall()

            conn.commit()
            conn.close()

        except Exception as e:
            result["error"] = str(e)

        return result

    # -------------------------------------------
    # Table Extraction (for citations)
    # -------------------------------------------
    def extract_tables_used(self, query: str) -> List[str]:
        """
        Very simple parser: extracts table names after FROM / JOIN.
        """
        pattern = r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_ ]*)"
        matches = re.findall(pattern, query, flags=re.IGNORECASE)

        # Normalize whitespace / quotes
        cleaned = []
        for m in matches:
            name = m.strip().replace('"', "")
            cleaned.append(name)

        return list(set(cleaned))  # unique


# Quick test
if __name__ == "__main__":
    tool = SQLiteTool()

    print("Tables:", tool.get_tables())
    q = """
        SELECT ProductName, UnitPrice
        FROM Products
        LIMIT 3;
    """
    out = tool.run_sql(q)
    print(out)
