# DocuChat: Local PDF Q&A with Basic RAG

A small, fully local Retrieval-Augmented Generation (RAG) app. Upload PDFs, ask questions in plain English, and get answers generated only from your documents. No API keys, no cloud services, and your files never leave your laptop.

This is a **learning project**: a clean, minimal implementation of the core RAG loop (load, chunk, embed, store, retrieve, generate), built step by step on modest hardware (8 GB RAM, CPU only).

<!-- Add a screenshot or GIF of the app here: ![DocuChat demo](docs/demo.png) -->

## What it does

- Extracts text from PDFs page by page (PyMuPDF)
- Splits pages into overlapping chunks that start and end on whole words
- Converts chunks to vectors with a local embedding model
- Stores vectors in a persistent local vector database (ChromaDB)
- Retrieves the top-k chunks most similar to your question (vector similarity only)
- Builds a grounded prompt and streams an answer from a local LLM via Ollama
- Shows the retrieved chunks with their source file, page number, and similarity score so you can check where an answer came from

## What it does not do (yet)

| Not included | Why it matters |
|---|---|
| Hybrid (keyword + vector) search | Pure vector search can miss exact terms |
| Reranking | The top result is not always the best one |
| Evaluation set and metrics | Quality has not been measured, only checked by hand |
| Conversation memory or query rewriting | Each question is answered on its own |
| OCR | Scanned PDFs have no extractable text |
| Agentic behaviour (routing, self-correction) | The pipeline is a single fixed pass |

## How it works

```
INDEXING (once per PDF)
PDF -> text per page -> chunks (800 chars, 150 overlap) -> embeddings (384 numbers each)
    -> ChromaDB (text + vector + source + page)

QUESTION (every time)
question -> embedding -> top-k most similar chunks -> grounded prompt -> llama3.2:3b -> streamed answer
```

## Tech stack

| Component | Choice |
|---|---|
| Language | Python 3.12 |
| LLM | `llama3.2:3b` via Ollama (runs on CPU) |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers, 384 dimensions) |
| Vector database | ChromaDB (persistent, cosine distance) |
| PDF parsing | PyMuPDF |
| UI | Streamlit |

## Project structure

```
rag-journey/
  app.py              Streamlit chat UI (upload, index, ask, view sources)
  rag_core.py         Pipeline: chunk_text, ingest_pdf, retrieve, build_prompt, answer_stream
  01 ... 08_*.py      Step-by-step learning scripts (LLM, embeddings, chunking, vector DB, mini RAG, PDF loading)
  data/               Your PDFs (optional, you can also upload in the app)
  db/                 ChromaDB files, created automatically
  requirements.txt
```

## Setup (Windows, PowerShell)

**Prerequisites:** Python 3.12 and [Ollama](https://ollama.com) installed.

```powershell
# 1. Create and activate a virtual environment
py -3.12 -m venv venv
venv\Scripts\activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers chromadb pymupdf ollama streamlit

# 3. Download the language model (about 2 GB)
ollama pull llama3.2:3b
```

The first run also downloads the embedding model (about 130 MB) from Hugging Face. After that, everything works offline.

## Usage

**Web app**

```powershell
venv\Scripts\activate
streamlit run app.py
```

1. Open `http://localhost:8501`.
2. Upload one or more text-based PDFs in the sidebar and click **Index documents**.
3. Ask a question in the chat box.
4. Expand **Sources** under an answer to see the file, page, similarity score, and text of each retrieved chunk.

**Command line**

```powershell
python .\rag_core.py
```

Type a question and the script prints the retrieved sources and streams the answer.

## Configuration

Settings are constants at the top of `rag_core.py`:

| Setting | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `llama3.2:3b` | Model used for answers. Use `llama3.2:1b` if your machine is slow |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_SIZE` / `OVERLAP` | 800 / 150 characters | Chunk length and shared text between neighbours |
| `TOP_K` | 4 | Chunks retrieved per question (also a slider in the UI, 1-8) |

If you change chunking settings or the embedding model, delete the `db` folder and re-index. Old chunks will not match the new settings.

## Design decisions

- **Page-aware chunking.** Text is split per page before chunking, so every chunk keeps its page number.
- **Word-boundary cuts with overlap.** Chunks never start or end mid-word, and neighbours share text so a sentence cut at a boundary still appears whole in one chunk.
- **Query prefix.** BGE models expect an instruction prefix on search questions but not on stored documents, so the prefix is added only at query time.
- **Normalized vectors and cosine distance.** Similarity reduces to a dot product, and the stored scores are easy to interpret.
- **Grounded prompt.** The model is told to use only the supplied context, to treat synonyms as equivalent, and to reply with a fixed refusal sentence when the answer is not in the context.
- **`temperature: 0` and `num_ctx: 4096`.** Deterministic factual answers, and a small context window that fits in 8 GB of RAM.
- **Re-index by source.** Uploading a PDF again replaces its old chunks instead of duplicating them.

## Known limitations

- **Quality has not been formally evaluated.** Behaviour was checked by hand on small examples only.
- **Retrieval always returns k chunks, even when nothing is relevant.** The refusal depends on the prompt, not a similarity threshold. In testing, an off-topic question was refused correctly, but this is not guaranteed.
- **A 3B model can be too literal.** In one test it failed to connect "vacation days" with "paid leave" even though the right chunk was retrieved. The synonym rule in the prompt targets this, but small models stay less reliable than large ones.
- **Single-turn only.** Follow-up questions like "what about its cost?" will not work, and chat history is lost when the page is refreshed.
- **Text PDFs only.** Scanned PDFs return no text. Tables flatten poorly into text.
- **CPU speed.** Answers can take several seconds to a minute on a laptop without a GPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip` or `streamlit` not recognized | Activate the environment: `venv\Scripts\activate` (needed in every new PowerShell window) |
| `ollama` not recognized | Add `%LOCALAPPDATA%\Programs\Ollama` to your PATH, then open a new PowerShell window |
| Connection error to Ollama | Make sure the Ollama app is running (system tray) |
| `WinError 10054` / `ConnectionResetError` in the terminal | Harmless on Windows. It appears when the browser closes a connection |
| `fitz` deprecation warning | Harmless. The code uses `import pymupdf` |
| "No text found" for a PDF | It is probably scanned. Try a text-based PDF |
| Slow answers or a frozen laptop | Close heavy apps, use `llama3.2:1b`, lower `TOP_K` |
| Wrong results after changing chunk settings | Delete the `db` folder and re-index |

## What I learned

- How embeddings capture meaning, and why semantic search finds "cancel subscription" from "terminate membership".
- Why chunk size and overlap matter, and how to diagnose a bad answer by reading the retrieved chunks first (retrieval problem vs generation problem).
- How a prompt can ground a model in supplied context and make it refuse when the answer is missing.
- How Streamlit re-runs the script on every interaction, and why chat state lives in `st.session_state`.

## Next steps

Planned improvements, roughly in order:

1. Build a hand-labelled evaluation set and measure recall@k and MRR.
2. Add keyword search (BM25) and combine it with vector search using Reciprocal Rank Fusion.
3. Add a cross-encoder reranker.
4. Add a similarity threshold for refusal, and conversational query rewriting.
5. Wrap the pipeline in an API and add logging, tests, and deployment.

## License

Add a license of your choice (for example MIT) before publishing.
