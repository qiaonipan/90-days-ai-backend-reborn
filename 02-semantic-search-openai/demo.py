from openai import OpenAI
import numpy as np

client = OpenAI()

# ---------- Embedding helper ----------
def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Dataset 1: Milk tea experiences ----------
milk_tea_logs = [
    "I drank a warm brown sugar milk tea and felt relaxed, happy, and satisfied",
    "The matcha milk tea was perfectly balanced, not too sweet, and I really enjoyed it",
    "The fruit milk tea was refreshing and made my afternoon much better",
    "The milk tea shop was very crowded and I had to wait 30 minutes",
    "I tried a cheese foam oolong tea but didn't really like the aftertaste"
]

milk_tea_query = "Which milk tea experience made me feel the happiest?"

# ---------- Dataset 2: Coffee experiences ----------
coffee_logs = [
    "The espresso shot gave me a strong boost of energy",
    "My iced americano was too watery and bland",
    "A creamy cappuccino helped me stay focused during the meeting",
    "The coffee was too bitter and upset my stomach",
    "I enjoyed a caramel macchiato while reading my favorite book"
]

coffee_query = "Which coffee made me feel most energized?"

# ---------- Search function ----------
def vector_search(logs, query, title):
    print(f"\n📋 {title}")
    print(f"🔍 Query: {query}\n📌 Results:")

    query_vec = np.array(embed(query), dtype=np.float32)
    results = []

    for log in logs:
        vec = np.array(embed(log), dtype=np.float32)
        score = cosine(query_vec, vec)
        results.append((score, log))

    results.sort(reverse=True)
    for score, log in results[:3]:
        print(f"- {log}  (score={score:.3f})")

# ---------- Run both searches ----------
vector_search(milk_tea_logs, milk_tea_query, "Milk Tea Search")
vector_search(coffee_logs, coffee_query, "Coffee Search")

