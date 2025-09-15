import argparse, os, glob, json
from dotenv import load_dotenv
from src.utils.constants import PDF_DIR, OUT_PATH, PERSIST_DIR, ARTIFACTS_DIR, DFEAULT_TOP_K
from huggingface_hub import login
from src.pdf_ingest import persist_markdown
from src.splitter import chunk_markdown_pages
from src.embed_store import EmbedStore
from src.query_engine import run_query
from src.utils.parser import parse_company_year_from_filename


load_dotenv()
login(token=os.getenv("HUGGINGFACE_API_KEY"))


def build_index():
    store = EmbedStore(persist_dir=PERSIST_DIR)
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found under {PDF_DIR}. Please add 10-K PDFs first.")
        return
    for pdf in pdfs:
        print(f"[INGEST] {pdf}")
        info = persist_markdown(pdf, ARTIFACTS_DIR)
        pages = info["pages"]
        company, year = info["company"], info["year"]
        base = os.path.splitext(os.path.basename(pdf))[0]
        doc_id = base
        
        chunks = chunk_markdown_pages(pages)
        meta = {
            "doc_id": doc_id,
            "company": company,
            "year": year,
            "source_pdf": os.path.basename(pdf),
        }
        store.add_chunks(doc_id, chunks, meta)
        print(f"[INDEXED] {base} -> {len(chunks)} chunks")


def ask(query: str, k: int = DFEAULT_TOP_K):
    store = EmbedStore(persist_dir=PERSIST_DIR)
    resp = run_query(query, store, k=k)
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    return resp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true", help="Ingest PDFs, extract text+tables→markdown, chunk, embed, and index into Chroma.")
    args = ap.parse_args()

    if args.build_index:
        build_index()
    else:
        print("\n--- Uniqus RAG CLI Chat ---")
        print("Type your question and press Enter. Type 'exit' to quit.\n")
        
        history = []
        
        q_num = 1
        while True:
            user_q = input(f"Question {q_num}: ")
            if user_q.strip().lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break
            print(f"\n{'='*60}\nQ{q_num}: {user_q}\n{'-'*60}")
            try:
                resp = ask(user_q)
                history.append({"question": user_q, "response": resp})
                print(f"{'='*60}\n")
                q_num += 1
            except Exception as e:
                print(f"Error: {e}\n{'='*60}\n")

        # Save history at the end of session
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
