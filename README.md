# Uniqus Finance RAG

## Overview

This project is a Retrieval-Augmented Generation (RAG) system for financial analysis of 10-K filings from Google, Microsoft, and NVIDIA. It supports answering direct, comparative, and complex queries using chunked document embeddings and agent-based query decomposition.

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Add your 10-K PDFs:**
   Place PDF files in `data/pdfs/` (sample files for GOOGL, MSFT, NVDA for 2022-2024 are included).
4. **Set environment variables:**
   - Create a `.env` file with your HuggingFace API key and Gemini API key:
     ```
     HUGGINGFACE_API_KEY=your_hf_token
     GEMINI_API_KEY=your_gemini_api_key
     ```

## Usage

### Build the Index

Extract, chunk, embed, and index all PDFs:

```powershell
python main.py --build-index
```

### Run the CLI Chat

Start an interactive chat session:

```powershell
python main.py
```

Type your financial question and press Enter. Type 'exit' to quit.

### Output

Session history is saved to `chat_history.json`.

## Design Doc

See the included design doc for architecture overview.
