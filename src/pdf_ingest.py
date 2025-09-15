from typing import List, Dict, Any, Tuple
import pdfplumber, os
from .utils.parser import clean_whitespace, dehyphenate, parse_company_year_from_filename


def _table_to_markdown(table: List[List[str]]) -> str:
    if not table or not any(any(cell for cell in row) for row in table):
        return ""

    # Use first non-empty row as header
    header = None
    body = []
    for row in table:
        row = [ (cell if cell is not None else "").strip() for cell in row ]
        if header is None and any(cell for cell in row):
            header = row
        else:
            body.append(row)
    if header is None:
        header = [" "]

    # Normalize column width
    width = max(len(header), max((len(r) for r in body), default=0))
    def pad(row): return row + [""]*(width - len(row))
    header = pad(header)
    body = [pad(r) for r in body]

    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"]*width) + " |")
    for r in body:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def extract_pdf_to_markdown(pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Returns (full_markdown_text, page_records).
    Each page_record: {page: int, text: str, tables: [markdown_str, ...]}
    And we inline tables after the page's text with headings "**Table p{page}_{i}**".
    """
    page_records: List[Dict[str, Any]] = []
    parts: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = dehyphenate(clean_whitespace(text))

            # tables
            tables_md: List[str] = []
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            tcount = 0
            for tbl in tables:
                md = _table_to_markdown(tbl)
                if md.strip():
                    tcount += 1
                    tables_md.append(md)

            # combine: text + markdown tables
            combined = text
            for idx, md in enumerate(tables_md, start=1):
                combined += f"\n\n**Table p{i}_{idx}**\n\n{md}\n"

            page_records.append({"page": i, "text": combined, "tables": tables_md})
            parts.append(f"\n\n# [Page {i}]\n\n" + combined)

    full_markdown = ("\n".join(parts)).strip()
    return full_markdown, page_records


def persist_markdown(pdf_path: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    md, pages = extract_pdf_to_markdown(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, base + ".md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    company, year = parse_company_year_from_filename(pdf_path)
    return {"markdown_path": out_path, "company": company, "year": year, "pages": pages}
