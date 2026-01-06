## Vector Search – Minimal Demo

# Vector Search Demo – Milk Tea Edition 🧋

## What is Vector Search?

Vector search transforms text into vectors (points in a semantic space) and uses geometric distance to measure semantic similarity.

In simple terms:
- Similar meanings → vectors are closer
- Different meanings → vectors are farther apart

---

## Project Goal

This project demonstrates an end-to-end minimal vector search pipeline:

- Natural language logs about milk tea experiences
- Text is converted into vectors (embeddings)
- A natural language query is also embedded
- Cosine similarity is used to find the most semantically similar logs

The goal is not performance, but understanding how vector search works.

## Technologies Used

- Python
- SQLite (in-memory)
- NumPy
- OpenAI Embeddings API (`text-embedding-3-small`)

This project uses OpenAI's embedding model to convert natural language
into vector representations for semantic similarity search.

---

## How It Works

1. **Logs (Data)**
   - Milk tea experiences written in natural language
   - Includes positive, negative, and neutral experiences

2. **Embedding**
   - Each log and query is converted into a vector
   - Vectors represent semantic meaning, not keywords

3. **Similarity Search**
   - Cosine similarity measures how close meanings are
   - Results are ranked by similarity score

---
## Key Takeaways

- Vector search focuses on **semantic meaning**, not exact word matching  
  (e.g., "happy", "enjoyed", and "satisfied" are understood as similar ideas)

- Similarity is measured geometrically:
  - Higher similarity score → closer meaning to the query
  - Lower similarity score → less relevant meaning

- Good vector search results depend on:
  - Well-designed and clearly differentiated data
  - Clear and specific natural language queries

- Vector search is a core building block of modern systems such as:
  - Semantic search
  - Retrieval-Augmented Generation (RAG)
---

## Example Query

```text
Which milk tea experience made me feel the happiest?

Example Output
- I drank a warm brown sugar milk tea and felt relaxed, happy, and satisfied
- The matcha milk tea was perfectly balanced, not too sweet, and I really enjoyed it
- The fruit milk tea was refreshing and made my afternoon much better
