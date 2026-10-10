import os
import chromadb
import ollama
import pymupdf
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 4
PREFIX = "Represent this sentence for searching relevant passages: "

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("pdf_docs", metadata={"hnsw:space": "cosine"})


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
        if text[start - 1] != " ":
            nxt = text.find(" ", start, end)
            if nxt != -1:
                start = nxt + 1
    return chunks


def ingest_pdf(data, filename):
    """Chunk, embed, and store a PDF (given as bytes). Returns the chunk count."""
    collection.delete(where={"source": filename})  # remove an older version
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(pymupdf.open(stream=data, filetype="pdf"), start=1):
        for j, chunk in enumerate(chunk_text(page.get_text())):
            ids.append(f"{filename}-p{page_no}-c{j}")
            docs.append(chunk)
            metas.append({"source": filename, "page": page_no})
    if not ids:
        return 0
    embeddings = model.encode(docs, normalize_embeddings=True, batch_size=32).tolist() # convert chunks into embeddings
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings) # upsert refers to avoid duplcaition of data in vectordb and store embeddings in vectordb
    return len(ids)


def retrieve(question, k=TOP_K):
    q_vec = model.encode([PREFIX + question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"], "score": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


SYSTEM_PROMPT = (
    "Answer the question using ONLY the context provided.\n"
    "Rules:\n"
    "- Treat synonyms as the same thing (for example, 'vacation days' and 'paid leave').\n"
    "- Cite the context blocks you used like [1] or [2].\n"
    "- If the context truly does not contain the answer, reply exactly: I couldn't find that in the document.\n"
    "- Do not add outside knowledge."
)


def build_prompt(question, hits):
    blocks = [
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    ]
    return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def answer_stream(question, hits):
    """Yield the answer one piece at a time."""
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        options={"temperature": 0, "num_ctx": 4096},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]


if __name__ == "__main__":
    q = input("Question: ")
    hits = retrieve(q)
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h['source']} p.{h['page']} (similarity {h['score']:.2f})")
    print("\nAnswer: ", end="")
    for piece in answer_stream(q, hits):
        print(piece, end="", flush=True)
    print()