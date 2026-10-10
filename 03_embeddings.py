import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = [
    "How do I cancel my subscription?",
    "Steps to terminate your membership",
    "The mitochondria is the powerhouse of the cell",
    "Best way to bake sourdough bread",
]

vectors = model.encode(sentences, normalize_embeddings=True)
print("Vector shape:", vectors.shape)

print("\nSimilarity to:", sentences[0])
for s, v in zip(sentences, vectors):
    print(f"  {np.dot(vectors[0], v):.3f}  {s}")