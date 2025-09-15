from typing import List, Dict, Any, Tuple
import chromadb, os, uuid
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from .utils.constants import DEFAULT_EMBEDDING_MODEL


class EmbedStore:
    def __init__(self, persist_dir: str = "chroma_db",
                 collection: str = "uniqus_rag",
                 model_name: str = DEFAULT_EMBEDDING_MODEL):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name

    def _embed_passages(self, texts: List[str]) -> List[List[float]]:
        # BGE trick: prefix with 'passage: '
        prefixed = [f"passage: {t}" for t in texts]
        embs = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        return embs.astype(np.float32).tolist()

    def _embed_query(self, q: str) -> List[float]:
        q = f"query: {q}"
        emb = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        return emb.astype(np.float32).tolist()

    def add_chunks(self, doc_id: str, chunks: List[Dict[str, Any]], metadata_base: Dict[str, Any]):
        texts = [c["text"] for c in chunks]
        embs = self._embed_passages(texts)
        ids = [f"{doc_id}:{i}:{uuid.uuid4().hex[:8]}" for i in range(len(texts))]
        metas = []
        for c in chunks:
            meta = metadata_base.copy()
            meta.update({
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
            })
            metas.append(meta)
        self.collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)

    def query(self, q: str, k: int = 10, where: dict | None = None) -> List[Dict[str, Any]]:
        emb = self._embed_query(q)
        res = self.collection.query(query_embeddings=[emb], n_results=k, include=["documents", "metadatas", "distances"], where=where or {})
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0] if "ids" in res else [None] * len(docs)

        for doc, meta, dist, _id in zip(docs, metas, dists, ids):
            r = {"text": doc, "score": float(dist)}
            r.update(meta or {})

            if _id is not None:
                r["id"] = _id

            out.append(r)
        return out
