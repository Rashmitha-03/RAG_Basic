import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "In one sentence, what is a vector database?"}],
    options={"temperature": 0},
)
print(response["message"]["content"])