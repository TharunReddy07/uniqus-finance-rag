import os, json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from .utils.constants import DEFAULT_LLM_MODEL

def _compact_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "company": r.get("company"),
            "year": r.get("year"),
            "page": r.get("page"),
            "metric_key": r.get("metric_key"),
            "value": r.get("value"),
            "excerpt": (r.get("excerpt") or ""),
        })
    return out

class GeminiLLM:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY for LLM synthesis.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(DEFAULT_LLM_MODEL)
        print(f"[LLM] Initialized Gemini LLM with model: {DEFAULT_LLM_MODEL}")


    def decompose_query(self, query: str) -> dict:
        prompt = f"""
            You are a financial query decomposition engine.

            Your task:
            1. Identify the user’s intent. Choose one from:
            [compare_metric_one_year, yoy_growth, share_of_total,
                direct_metric, collect_ai_mentions, default]

            2. Generate `sub_queries`: a list of simpler queries. 
            - If the user asks for a comparison (e.g., "highest", "lowest", "compare"),
                ALWAYS include **all three companies: Google, Microsoft, NVIDIA**,
                even if the question only names one.
            - Each sub-query must include: Company + Metric + Year (if given).
            - Keep them short and consistent.

            Format: VALID JSON ONLY.

            User query: "{query}"

            ---

            ### Example
            Input: "Which company had the highest operating margin in 2023?"
            Output:
            {{
            "intent": "compare_metric_one_year",
            "sub_queries": [
                "Google operating margin 2023",
                "Microsoft operating margin 2023",
                "NVIDIA operating margin 2023"
            ]
            }}

            Now decompose the given query.
        """

        resp = self.model.generate_content(prompt)
        text = resp.text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            plan = json.loads(text)
        except Exception as e:
            raise RuntimeError(f"LLM decomposition failed: {e}\nRaw response: {text}")
        return plan
    

    def synthesize(self, query: str, sub_queries: List[str], rows: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        context = {
            "query": query,
            "sub_queries": sub_queries,
            "evidence": _compact_rows(rows),
            "instructions": (
                "Use only the given context; do not use outside knowledge. "
                "When numbers exist, **quote exact** financial figures or statements (preserve % and units as written). "
                "If the question asks for a comparison (e.g., highest/lowest) and multiple companies/years are present with numeric values, compute it and state the winner clearly. "
                "If numeric values are NOT present but relevant qualitative statements exist (e.g., 'increased', 'declined', 'management expects...'), answer qualitatively by quoting those statements verbatim. "
                "If neither numeric values nor relevant qualitative statements exist to answer the query, reply exactly: 'Insufficient evidence in provided sources.' "
                "Never invent numbers, estimates, or ranges. "
                "Return a compact JSON object with keys: answer, reasoning."
            ),
        }

        prompt = (
            "You are a financial analysis assistant. Use ONLY the provided evidence.\n"
            "- If a metric appears with units/percent signs, interpret it correctly and quote it exactly.\n"
            "- If the user asks for a comparison and numeric data is available for the compared items, compute it.\n"
            "- If numeric data is missing but relevant qualitative statements exist, answer using those statements (quoted verbatim).\n"
            "- If nothing in the evidence lets you answer, output the exact string: 'Insufficient evidence in provided sources.'\n\n"
            f"DATA:\n{json.dumps(context, ensure_ascii=False)}\n\n"
            'Respond with JSON only: {"answer": "...", "reasoning": "..."}'
        )


        try:
            resp = self.model.generate_content(prompt)
            text = resp.text.strip()

            # Extract last JSON object if extra text
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

            obj = json.loads(text)
            if "answer" in obj and "reasoning" in obj:
                return {"answer": obj["answer"], "reasoning": obj["reasoning"]}
        except Exception as e:
            print("[LLM] Error:", e)
            return None
        return None


def get_llm() -> Optional[GeminiLLM]:
    try:
        return GeminiLLM()
    except Exception:
        print("[LLM] Gemini LLM not available.")
        return None
