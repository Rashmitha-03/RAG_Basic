
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "RAG helps a language model retrieve relevant information."

embedding = model.encode(text)

print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])