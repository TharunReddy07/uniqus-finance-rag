from typing import List, Dict, Any
from transformers import AutoTokenizer
from .utils.constants import DEFAULT_EMBEDDING_MODEL


def chunk_markdown_pages(pages: List[Dict[str, Any]],
                         model_name: str = DEFAULT_EMBEDDING_MODEL,
                         chunk_tokens: int = 900,
                         overlap_tokens: int = 100):
    """Token-aware chunker across page records.
    Returns list of dicts: {text, page_start, page_end}
    """
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    chunks = []
    buf = []
    buf_tokens = 0
    page_start = None
    current_page = None

    def flush(end_page):
        nonlocal buf, buf_tokens, page_start
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            chunks.append({"text": text, "page_start": page_start, "page_end": end_page})
        buf, buf_tokens, page_start = [], 0, None

    for rec in pages:
        current_page = rec["page"]
        part = rec["text"].strip()
        if not part:
            continue
        tokens = tok.encode(part, add_special_tokens=False)
        
        i = 0
        while i < len(tokens):
            space_left = chunk_tokens - buf_tokens
            take = min(space_left, len(tokens) - i)
            
            # Decode token slice to text and append
            piece = tok.decode(tokens[i:i+take])
            if page_start is None:
                page_start = current_page
            buf.append(piece)
            buf_tokens += take
            i += take
            
            if buf_tokens >= chunk_tokens:
                # flush and create overlap from end
                flush(current_page)
                # overlap: re-seed buffer with tail
                tail_tokens = tokens[max(0, i - overlap_tokens):i]
                if tail_tokens:
                    tail_text = tok.decode(tail_tokens)
                    buf = [tail_text]
                    buf_tokens = len(tail_tokens)
                    page_start = current_page
    
    # final flush
    flush(current_page)
    return chunks
