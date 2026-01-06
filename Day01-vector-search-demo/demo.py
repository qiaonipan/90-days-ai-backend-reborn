import sqlite3
import numpy as np
from openai import OpenAI

client = OpenAI()

# -----------------------------
# 1. Embedding function
# -----------------------------

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


# -----------------------------
# 2. Qiaoni's milk tea logs 🧋
# -----------------------------
logs = [
    # ✅ Extremely positive
    "I drank a warm brown sugar milk tea and felt relaxed, happy, and satisfied",
    "The matcha milk tea was perfectly balanced, not too sweet, and I really enjoyed it",
    "The fruit milk tea was refreshing and made my afternoon much better",

    # ❌ Extremely negative
    "I waited more than 25 minutes for my milk tea and felt frustrated and tired",
    "The milk tea was overly sweet and made my throat uncomfortable",
    "The shop was crowded, the service was slow, and the experience was unpleasant",

    # ⚪ neutral/irrelevant
    "I usually drink coffee in the morning instead of milk tea",
    "I bought milk tea for my friend, but I did not drink it myself"
]


# -----------------------------
# 3. Store logs in SQLite
# -----------------------------
conn = sqlite3.connect(":memory:")
cur = conn.cursor()

cur.execute("""
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    text TEXT,
    vector BLOB
)
""")

for i, text in enumerate(logs):
    vec = np.array(embed(text), dtype=np.float32)
    cur.execute(
        "INSERT INTO logs VALUES (?, ?, ?)",
        (i, text, vec.tobytes())
    )

conn.commit()


# -----------------------------
# 4. Qiaoni asks a question 💬
# -----------------------------
query = "Which milk tea experience made me feel the happiest?"
q_vec = np.array(embed(query), dtype=np.float32)


# -----------------------------
# 5. Cosine similarity
# -----------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# 6. Vector search
# -----------------------------
results = []

for text, blob in cur.execute("SELECT text, vector FROM logs"):
    vec = np.frombuffer(blob, dtype=np.float32)
    score = cosine(q_vec, vec)
    results.append((score, text))

results.sort(reverse=True)


# -----------------------------
# 7. Print results
# -----------------------------
print("🔍 Query:")
print(query)
print("\n📌 Results:")

for score, text in results[:3]:
    print(f"- {text}  (score={score:.3f})")
