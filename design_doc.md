# Design Document: Uniqus Finance RAG

## 1. Chunking Strategy

- **PDF Ingestion:** Each 10-K PDF is parsed and converted to markdown using `pdf_ingest.py`. This preserves tables, section headers, and page structure.
- **Chunking:** Markdown is split into chunks using `chunk_markdown_pages`. Chunks are typically paragraphs or logical sections, balancing retrieval granularity and context preservation.
- **Metadata:** Each chunk is tagged with company, year, source PDF, and page number for traceability and filtering.

## 2. Embedding Model Choice

**Model:** Google EmbeddingGemma-300M (`google/embeddinggemma-300m`)

- **Default Configuration:** Set in `src/utils/constants.py` and used for all chunk embeddings throughout the pipeline.
- **Architecture:** Transformer-based, 300M parameters, designed for high-dimensional semantic representation.
- **Domain Training:** Trained on a large, diverse corpus including financial, technical, and business documents, making it highly effective for extracting context and relationships from 10-K filings.
- **Integration:**
  - All markdown chunks are embedded using this model before being stored in ChromaDB.
  - Embedding dimension and tokenization are natively supported by ChromaDB, ensuring efficient similarity search and filtering.
- **Embedding Storage:**
  - Chunks are embedded with EmbeddingGemma-300M and stored in ChromaDB. This enables fast, accurate semantic search and supports advanced filtering by company, year, and metric.

## 3. Query Decomposition & Agent Flow

- **LLM Decomposition:**
  - User queries are parsed by `GeminiLLM.decompose_query` to identify intent (e.g., comparison, growth, direct lookup, share, AI mentions).
  - The LLM generates atomic sub-queries for each company/metric/year as needed.
- **Retrieval Pipeline:**
  1. For each sub-query, relevant chunks are retrieved from ChromaDB using semantic similarity.
  2. Chunks are reranked using BGE cross-encoder for relevance.
  3. Regex-based extraction (`query_engine.py`) pulls metrics or qualitative statements from top chunks.
  4. Results are synthesized by the LLM (`GeminiLLM.synthesize`) into a compact JSON answer and reasoning.
- **Complex Queries:**
  - Comparative and multi-step queries are decomposed into sub-queries, results are aggregated, and the LLM produces a final answer with explicit reasoning and evidence.

## 4. Data Flow & Error Handling

- **Data Flow:**
  - PDF → Markdown → Chunks (+metadata) → Embeddings → ChromaDB
  - User Query → Decomposition → Sub-queries → Retrieval → Reranking → Extraction → Synthesis
- **Error Handling:**
  - If no relevant numeric or qualitative evidence is found, the system returns "Insufficient evidence in provided sources." (see LLM instructions)

---

For implementation details, see the source code.
