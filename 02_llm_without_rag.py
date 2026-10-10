import ollama

question = "What is the secret launch code of Project Falcon?"

# Attempt 1: no context
r1 = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": question}],
    options={"temperature": 0},
)
print("WITHOUT context:\n", r1["message"]["content"])

# Attempt 2: we provide the fact inside the prompt
context = "Internal memo: The secret launch code of Project Falcon is BLUE-HERON-42."
prompt = f"Context: {context}\n\nUsing only the context, answer: {question}"
r2 = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0},
)
print("\nWITH context:\n", r2["message"]["content"])