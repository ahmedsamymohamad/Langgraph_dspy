# agent/dspy_signatures.py

import dspy
from typing import List, Dict, Any


# ============================================================
# Ollama Configuration
# ============================================================

# Make sure Ollama is running locally at http://localhost:11434
# Model: phi3.5:3.8b-mini-instruct-q4_K_M (or any other model you've pulled)

import requests
import json
import re
import ast

class OllamaDSPyWrapper(dspy.BaseLM):
    """Wrapper to make Ollama compatible with DSPy"""
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        super().__init__(model=model)
        self.model = model
        self.base_url = base_url
        self.history = []
        self.kwargs = {}  # Required by DSPy BaseLM
    
    def basic_request(self, prompt: str, **kwargs):
        """Make a basic request to Ollama and normalize the returned text.

        This handles different possible response shapes, strips markdown
        fences, and (when the text looks like SQL) prefixes it with
        "sql: " so DSPy's JSON/Text adapters can detect the field.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": kwargs.get("temperature", 0.7),
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

            # Ollama responses vary; try common locations for the text
            text = ""
            print("ollama result:",result)
            if isinstance(result, dict):
                # new Ollama shape: maybe 'response' or 'choices' or 'results'
                text = result.get("response") or ""
                if not text:
                    choices = result.get("choices") or result.get("results") or []
                    if isinstance(choices, list) and choices:
                        # choices may be dicts or strings
                        first = choices[0]
                        if isinstance(first, dict):
                            # common key is 'text' or 'content'
                            text = first.get("text") or first.get("content") or ""
                        else:
                            text = str(first)
            else:
                text = str(result)

            # Fallback: if empty, try raw text
            if not text:
                text = "".join(response.iter_lines(decode_unicode=True))

            # If the LM returned a Python-dict-like string (single quotes), try to parse it
            parsed = None
            try:
                if isinstance(text, str) and text.strip().startswith("{"):
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        # try python literal (single quotes)
                        try:
                            parsed = ast.literal_eval(text)
                        except Exception:
                            parsed = None
            except Exception:
                parsed = None

            # If parsed is a dict and contains typical fields, extract
            if isinstance(parsed, dict):
                # If it contains a nested 'text' field, use that
                if "text" in parsed and isinstance(parsed["text"], str):
                    text = parsed["text"]
                else:
                    # If parsed already contains the expected output fields (final_answer, explanation, citations, sql), validate and return it as JSON
                    keys = set(parsed.keys())
                    wanted = {"final_answer", "explanation", "citations", "sql"}
                    if keys & wanted:
                        # Ensure all typical fields exist before returning
                        if "final_answer" not in parsed:
                            parsed["final_answer"] = parsed.get("sql", "")
                        if "explanation" not in parsed:
                            parsed["explanation"] = ""
                        if "citations" not in parsed:
                            parsed["citations"] = []
                        # Validate JSON serialization
                        try:
                            return json.dumps(parsed)
                        except Exception:
                            # If serialization fails, continue to heuristic extraction
                            pass

            # Strip markdown code fences ```sql or ```
            if text.startswith("```") and "```" in text[3:]:
                parts = text.split("```", 2)
                if len(parts) >= 3:
                    inner = parts[2]
                    inner = inner.rsplit("```", 1)[0]
                    text = inner.strip()

            # Heuristics: extract structured fields using markers or headings
            out = {}

            # 1) Marker-style: [[ ## final_answer ## ]] ... [ ## explanation ## ] ... [ ## citations ## ]
            def extract_marker(field_name, txt):
                pattern = re.compile(r"\[+\s*##\s*" + re.escape(field_name) + r"\s*##\s*\]+\s*(.*?)(?=(\n\[|\Z))", re.S | re.I)
                m = pattern.search(txt)
                if m:
                    return m.group(1).strip()
                return None

            for fld in ("final_answer", "explanation", "citations", "sql"):
                val = extract_marker(fld, text)
                if val:
                    out[fld] = val

            # 2) Heading-style: 'Final Answer:' 'Explanation:' 'Citations:'
            if "final_answer" not in out:
                m = re.search(r"Final Answer:\s*(.*?)($|\n\n|\n\[)", text, re.I | re.S)
                if m:
                    out["final_answer"] = m.group(1).strip()

            if "explanation" not in out:
                m = re.search(r"Explanation[:\-]?\s*(.*?)($|\n\n|\n\[)", text, re.I | re.S)
                if m:
                    out["explanation"] = m.group(1).strip()

            if "citations" not in out:
                m = re.search(r"Citations?:\s*(.*?)($|\n\n|\n\[)", text, re.I | re.S)
                if m:
                    ctext = m.group(1).strip()
                    # split by newlines or semicolons
                    items = [ln.strip() for ln in re.split(r"\n|;|,", ctext) if ln.strip()]
                    out["citations"] = items if items else [ctext]

            # 3) If nothing structured found, but the text is SQL-like, prefix as before
            sql_like = False
            first_token = text.strip().split(None, 1)[:1]
            if first_token:
                tok = first_token[0].upper()
                if tok in {"SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE"}:
                    sql_like = True

            if sql_like and "sql" not in out:
                out["sql"] = text.strip()

            # If we extracted some fields, populate defaults for missing fields and return JSON
            if out:
                # Map common field names to expected output names
                mapped = dict(out)
                
                # Smart mapping: if we have final_answer but need route, use it
                if "route" not in mapped:
                    for candidate_key in ["explanation", "final_answer", "plan"]:
                        if candidate_key in mapped:
                            val = str(mapped[candidate_key]).lower()
                            for route_type in ["rag", "sql", "hybrid"]:
                                if route_type in val:
                                    mapped["route"] = route_type
                                    break
                            if "route" in mapped:
                                break
                
                if "sql" not in mapped:
                    for candidate_key in ["explanation", "final_answer"]:
                        if candidate_key in mapped and "select" in str(mapped[candidate_key]).lower():
                            mapped["sql"] = mapped[candidate_key]
                            break
                
                # Ensure all common fields have at least an empty value
                if "final_answer" not in mapped:
                    mapped["final_answer"] = mapped.get("sql", mapped.get("explanation", text.strip()[:200]))
                if "explanation" not in mapped:
                    mapped["explanation"] = ""
                if "citations" not in mapped:
                    mapped["citations"] = []
                if "route" not in mapped:
                    mapped["route"] = "hybrid"
                if "sql" not in mapped:
                    mapped["sql"] = ""
                if "plan" not in mapped:
                    mapped["plan"] = {}
                
                return json.dumps(mapped)

            # As a last resort return the raw text
            return text
        except requests.exceptions.ReadTimeout as e:
            raise RuntimeError(f"Ollama read timeout: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Ollama connection failed: {str(e)}")
    
    def __call__(self, prompt: str = None, messages: list = None, **kwargs):
        """Handle both prompt-based and message-based calls.

        DSPy sometimes calls LMs with `messages=`; support that shape by
        converting to a single prompt. Return a list of outputs as DSPy
        expects.
        """
        if messages:
            # Convert messages to a single prompt string
            prompt_text = ""
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "").upper()
                    content = msg.get("content", "")
                    prompt_text += f"{role}: {content}\n"
                else:
                    prompt_text += str(msg) + "\n"
            prompt = prompt_text

        if not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        response_text = self.basic_request(prompt, **kwargs)

        # DSPy expects a list of strings (one completion per returned item)
        return [response_text]
    
    def forward(self, prompt: str, **kwargs) -> str:
        """Forward pass using Ollama API"""
        return self.basic_request(prompt, **kwargs)

ollama_lm = OllamaDSPyWrapper(model="phi3.5:3.8b-mini-instruct-q4_K_M")
dspy.settings.configure(lm=ollama_lm)


# ============================================================
# Router — classify question into: rag | sql | hybrid
# ============================================================

class RouteQuestion(dspy.Signature):
    question = dspy.InputField()
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")


# ============================================================
# Planner
# ============================================================

class PlanQuery(dspy.Signature):
    question = dspy.InputField()
    retrieved_docs = dspy.InputField()
    plan = dspy.OutputField(desc="Dictionary of extracted constraints")


# ============================================================
# NL → SQL Generator
# ============================================================

class GenerateSQL(dspy.Signature):
    question = dspy.InputField()
    plan = dspy.InputField()
    schema = dspy.InputField()
    sql = dspy.OutputField(desc="Generated SQLite query")


# ============================================================
# Synthesizer
# ============================================================

class SynthesizeAnswer(dspy.Signature):
    question = dspy.InputField()
    format_hint = dspy.InputField()
    plan = dspy.InputField()
    retrieved_docs = dspy.InputField()
    sql = dspy.InputField()
    rows = dspy.InputField()
    columns = dspy.InputField()
    tables_used = dspy.InputField()

    final_answer = dspy.OutputField()
    explanation = dspy.OutputField()
    citations = dspy.OutputField()


# ============================================================
# DSPy Modules
# ============================================================

class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RouteQuestion)

    def forward(self, question: str):
        return self.predict(question=question)


class PlannerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PlanQuery)

    def forward(self, question: str, retrieved_docs):
        return self.predict(question=question, retrieved_docs=retrieved_docs)


class NL2SQLModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GenerateSQL)

    def forward(self, question: str, plan, schema):
        return self.predict(question=question, plan=plan, schema=schema)


class SynthesizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SynthesizeAnswer)

    def forward(
        self,
        question,
        format_hint,
        plan,
        retrieved_docs,
        sql,
        rows,
        columns,
        tables_used,
    ):
        return self.predict(
            question=question,
            format_hint=format_hint,
            plan=plan,
            retrieved_docs=retrieved_docs,
            sql=sql,
            rows=rows,
            columns=columns,
            tables_used=tables_used,
        )


# ============================================================
# Module Instances (Initialized after Ollama config)
# ============================================================

router = RouterModule()
planner = PlannerModule()
nl2sql = NL2SQLModule()
synth = SynthesizerModule()
