import chromadb
import ollama
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient(path="./db").get_collection("demo")

PREFIX = "Represent this sentence for searching relevant passages: "


def ask(question, k=2):
    # 1. Retrieve
    q_vec = model.encode([PREFIX + question], normalize_embeddings=True).tolist()
    chunks = collection.query(query_embeddings=q_vec, n_results=k)["documents"][0]
    for i, c in enumerate(chunks, 1):
        print(f"   retrieved [{i}]: {c}")

    # 2. Build the prompt
    context = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    prompt = (
        "Answer the question using ONLY the context below.\n"
        "Rules:\n"
        "- Treat synonyms as the same thing (for example, 'vacation days' and 'paid leave').\n"
        "- Cite the chunk you used like [1].\n"
        "- If the context truly does not contain the answer, reply exactly: I couldn't find that.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    # 3. Generate
    reply = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return reply["message"]["content"]


for q in [
    "How many vacation days do I get?",
    "Who leads Project Falcon?",
    "What is the capital of Australia?",
]:
    print("\nQ:", q)
    print("A:", ask(q))