import sys
import psycopg2
import requests

OLLAMA_API = "http://localhost:11434/api"
DB_PARAMS = {"dbname": "mydb", "user": "myuser", "password": "mypassword",
             "host": "localhost", "port": "5444"}

def get_embedding(text):
    r = requests.post(f"{OLLAMA_API}/embeddings", json={"model": "all-minilm", "prompt": text})
    return r.json()["embedding"]

def generate_response(prompt):
    r = requests.post(f"{OLLAMA_API}/generate",
                      json={"model": "llama3.2", "prompt": prompt, "stream": False})
    return r.json()["response"]

def ask(query):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    emb = "[" + ",".join(map(str, get_embedding(query))) + "]"
    cur.execute("""
        SELECT title, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents ORDER BY similarity DESC LIMIT 5;
    """, (emb,))
    rows = cur.fetchall()
    context = "\n\n".join(f"Title: {r[0]}\nContent: {r[1]}" for r in rows)
    prompt = f"Query: {query}\nContext: {context}\nPlease provide a concise answer based on the given context."
    print("Response:", generate_response(prompt))
    cur.close()
    conn.close()

if __name__ == "__main__":
    ask(" ".join(sys.argv[1:]) or "Tell me about landmarks in Seoul")
