import os
import openai
import oracledb
from dotenv import load_dotenv
import array

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
username = os.getenv("ORACLE_USERNAME")
password = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_DSN")                  # required
wallet_path = os.getenv("ORACLE_WALLET_PATH")

# Connect to database
connection = oracledb.connect(
    user=username,
    password=password,
    dsn=dsn,
    config_dir=wallet_path,
    wallet_location=wallet_path,
    wallet_password=password
)
cursor = connection.cursor()

def oracle_vector_search(query, title, top_k=3):
    print(f"\n📋 {title}")
    print(f"🔍 Query: {query}\n📌 Top {top_k} most relevant results:")

    # Generate query embedding
    query_embedding_list = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Convert to the format Oracle prefers
    query_embedding = array.array('f', query_embedding_list)

    # Query database
    cursor.execute("""
        SELECT text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
        FROM docs
        ORDER BY distance ASC
        FETCH FIRST :top_k ROWS ONLY
    """, query_vec=query_embedding, top_k=top_k)

    results = cursor.fetchall()
    for i, (text, distance) in enumerate(results, 1):
        similarity = 1 - distance / 2   # rough conversion to similarity
        print(f"{i}. {text}")
        print(f"   (similarity ≈ {similarity:.3f}, distance = {distance:.4f})")

# ---------- Run three searches ----------
oracle_vector_search(
    "What makes Qiaoni a great contributor to a team project?",
    "Qiaoni - Team Contribution Search"
)

oracle_vector_search(
    "Which milk tea experience made me feel the happiest?",
    "Happiest Milk Tea Experience Search"
)

oracle_vector_search(
    "Which coffee made me feel most energized?",
    "Most Energizing Coffee Search"
)

# ---------- Close connection after all searches complete ----------
connection.close()
print("\n✅ Search complete, database connection closed.")