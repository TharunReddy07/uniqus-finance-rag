import re, math
from typing import Dict, List, Any
from .embed_store import EmbedStore
from .utils.parser import COMPANY_ALIASES, COMPANY_LIST
from .utils.constants import RERANK_TOP_K
from .llm import get_llm
from .bge_reranker import BGEReranker


METRIC_PATTERNS = [
    # (metric_key, regex pattern, value_kind)
    ("operating margin", r"operating margin[^\d%]*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", "percent"),
    ("gross margin",     r"gross margin[^\d%]*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", "percent"),
    ("cloud revenue",    r"(?:cloud|google cloud|microsoft cloud|azure|gcp)[^\$\d%]{0,40}(\$?[\d,\.]+)\s*(billion|million|thousand|bn|m|k)?", "money"),
    ("data center revenue", r"data\s*center[^\$\d%]{0,40}(\$?[\d,\.]+)\s*(billion|million|thousand|bn|m|k)?", "money"),
    ("total revenue",    r"total\s+revenue[^\$\d%]{0,40}(\$?[\d,\.]+)\s*(billion|million|thousand|bn|m|k)?", "money"),
]

UNIT_MULT = {
    None: 1.0,
    "billion": 1e9, "bn": 1e9,
    "million": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

def _money_to_float(val: str, unit: str) -> float:
    v = val.replace("$","").replace(",","")
    try:
        x = float(v)
    except:
        return float("nan")
    mult = UNIT_MULT.get(unit.lower() if unit else None, 1.0)
    return x * mult


def _find_metric_value(text: str, metric_key: str):
    for key, pattern, kind in METRIC_PATTERNS:
        if key == metric_key:
            m = re.search(pattern, text, flags=re.I)
            if not m:
                continue
            if kind == "percent":
                value = float(m.group(1))
                return {"kind": "percent", "value": value}
            if kind == "money":
                value = _money_to_float(m.group(1), m.group(2) if len(m.groups())>1 else None)
                return {"kind": "money", "value": value}
    return None


def _build_where(company_filter: List[str] | None, years: List[str] | None):
    clauses = []
    if company_filter:
        company_filter = [str(c) for c in company_filter]
        clauses.append({"company": company_filter[0]} if len(company_filter) == 1
                       else {"company": {"$in": company_filter}})
    if years:
        years = [str(y) for y in years]
        clauses.append({"year": years[0]} if len(years) == 1
                       else {"year": {"$in": years}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def run_query(query: str, store: EmbedStore, k: int = 10) -> Dict[str, Any]:
    llm = get_llm()
    if not llm:
        raise RuntimeError("LLM not available for query decomposition.")
    plan = llm.decompose_query(query)
    subqs = plan["sub_queries"]
    intent = plan["intent"]

    reranker = BGEReranker()
    results_per_sub: List[Dict[str, Any]] = []
    for sq in subqs:
        orig_q = sq.lower()
        company_filter = []
        if "google" in orig_q or "googl" in orig_q or "alphabet" in orig_q:
            company_filter.append("GOOGL")
        elif "microsoft" in orig_q or "msft" in orig_q:
            company_filter.append("MSFT")
        elif "nvidia" in orig_q or "nvda" in orig_q:
            company_filter.append("NVDA")

        years_in_q = re.findall(r"(20\d{2})", query)
        where = _build_where(company_filter, years_in_q)

        hits = store.query(sq, k, where=where)
        metric_key = None
        sql = sq.lower()
        for key, _, _ in METRIC_PATTERNS:
            if key in sql:
                metric_key = key
                break
        
        # Rerank and keep only top 3
        top_hits = reranker.rerank(sq, hits, top_k=RERANK_TOP_K)  
        for hit in top_hits:
            company = (hit.get("company") if hit else None)
            year = (hit.get("year") if hit else None)
            page = hit.get("page_start") if hit else None
            
            value = None
            if hit and metric_key:
                value = _find_metric_value(hit["text"], metric_key)
            
            results_per_sub.append({
                "sub_query": sq,
                "company": company,
                "year": year,
                "page": page,
                "metric_key": metric_key,
                "value": value,
                "excerpt": hit["text"] if hit else None,
                "raw": hit,
            })
    
    final_answer, final_reasoning = None, None
    if llm:
        llm_out = llm.synthesize(query, subqs, results_per_sub)
        if llm_out and isinstance(llm_out, dict):
            final_answer = llm_out.get("answer")
            final_reasoning = llm_out.get("reasoning")

    sources = []
    added = set()
    for r in results_per_sub:
        key = (r["company"], r["year"], r["page"])
        sources.append({
            "company": r["company"],
            "year": str(r["year"]) if r["year"] else None,
            "excerpt": r["excerpt"],
            "page": r["page"],
        })
        added.add(key)

    return {
        "query": query,
        "answer": final_answer,
        "reasoning": final_reasoning,
        "sub_queries": subqs,
        "sources": sources
    }