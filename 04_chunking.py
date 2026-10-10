def chunk_text(text, size=200, overlap=50):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                      # end on a space, not mid-word
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
        if text[start - 1] != " ":               # start on a whole word
            nxt = text.find(" ", start, end)
            if nxt != -1:
                start = nxt + 1
    return chunks


sample = (
    "Retrieval-Augmented Generation combines search with a language model. "
    "First, documents are split into chunks. Each chunk is converted into a vector. "
    "When a user asks a question, the question is also converted into a vector. "
    "The system finds the chunks whose vectors are most similar to the question. "
    "Those chunks are placed into the prompt so the model can answer from them."
)

for i, c in enumerate(chunk_text(sample)):
    print(f"Chunk {i} ({len(c)} chars): {c}\n")