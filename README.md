# AggGuard

Deterministic disclosure control (small-group suppression + query auditing) for multi-turn RAG assistants. CS 562 project.

## Local RAG setup
1. Install Docker Desktop and Ollama.
2. `ollama pull llama3.2` and `ollama pull all-minilm`
3. `docker compose up -d`
4. `docker exec -it timescaledb psql -U myuser -d mydb`, then:
   `CREATE EXTENSION IF NOT EXISTS vector;`
   `CREATE TABLE IF NOT EXISTS documents (id SERIAL PRIMARY KEY, title TEXT, content TEXT, embedding VECTOR(384));`
5. `pip install -r requirements.txt`
6. `python rag/ingest.py` (run once), then `python rag/rag.py "your question"`

## Layout
- `rag/`: plain-RAG baseline (Postgres + pgvector + Ollama)
- `guards/`: prototype guards (R1 to R3)
- `eval/`: attacks, benign workload, results
