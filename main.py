import argparse, os, glob, json

PDF_DIR = "data/pdfs"
OUT_PATH = "chat_history.json"
PERSIST_DIR = "chroma_db"
ARTIFACTS_DIR = "artifacts/processed"
DFEAULT_K = 10


def build_index():
    pass

def ask(query: str):
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true", help="Ingest PDFs, extract text+tables→markdown, chunk, embed, and index into Chroma.")
    args = ap.parse_args()

    if args.build_index:
        build_index()
    else:
        ask()