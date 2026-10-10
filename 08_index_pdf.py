import os
import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/mod5.pdf"  
CHUNK_SIZE, OVERLAP = 800, 150
PREFIX = "Represent this sentence for searching relevant passages: "

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./db")

# Start fresh each run so old chunks never mix with new ones
try:
    client.delete_collection("pdf_docs")
except Exception:
    pass
collection = client.create_collection("pdf_docs", metadata={"hnsw:space": "cosine"})


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


# 1. Load + chunk, keeping the page number for every chunk
source = os.path.basename(PDF_PATH)
ids, docs, metas = [], [], []
for page_no, page in enumerate(pymupdf.open(PDF_PATH), start=1):
    for j, chunk in enumerate(chunk_text(page.get_text())):
        ids.append(f"{source}-p{page_no}-c{j}")
        docs.append(chunk)
        metas.append({"source": source, "page": page_no})

# 2. Embed + store
embeddings = model.encode(docs, normalize_embeddings=True, batch_size=32).tolist()
collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
print("Chunks stored:", collection.count())

# 3. Test search
for question in ["What is a file?", "What does a directory entry consist of?"]:
    q_vec = model.encode([PREFIX + question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=3)
    print("\nQuestion:", question)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"  page {meta['page']}  similarity {1 - dist:.3f}  {doc[:90]}...")