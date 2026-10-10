import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(
    name="demo",
    metadata={"hnsw:space": "cosine"},
)

chunks = [
    "To cancel your subscription, go to Settings and click Manage Plan.",
    "The office is closed on public holidays and weekends.",
    "Employees get 24 days of paid leave per year.",
    "Refunds are processed within 7 business days.",
    "Project Falcon launches in March under the leadership of Priya.",
]

# Store: id + text + vector for each chunk
collection.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=model.encode(chunks, normalize_embeddings=True).tolist(),
)
print("Chunks stored:", collection.count())

# Search: embed the question, find the closest chunks
prefix = "Represent this sentence for searching relevant passages: "
for question in ["How do I terminate my membership?", "How many vacation days do I get?"]:
    q_vec = model.encode([prefix + question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=q_vec, n_results=2)
    print("\nQuestion:", question)
    for text, dist in zip(result["documents"][0], result["distances"][0]):
        print(f"  similarity {1 - dist:.3f}  {text}")