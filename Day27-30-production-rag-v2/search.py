import os
import openai
import oracledb
from dotenv import load_dotenv
import array

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
username = os.getenv("ORACLE_USERNAME")
password = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_DSN")  # 必需
wallet_path = os.getenv("ORACLE_WALLET_PATH")

# 连接到数据库
connection = oracledb.connect(
    user=username,
    password=password,
    dsn=dsn,
    config_dir=wallet_path,
    wallet_location=wallet_path,
    wallet_password=password,
)
cursor = connection.cursor()


def oracle_vector_search(query, title, top_k=3):
    print(f"\n📋 {title}")
    print(f"🔍 Query: {query}\n📌 Top {top_k} most relevant results:")

    # 生成查询embedding
    query_embedding_list = (
        openai.embeddings.create(model="text-embedding-3-small", input=query)
        .data[0]
        .embedding
    )

    # 转换为Oracle首选的格式
    query_embedding = array.array("f", query_embedding_list)

    # 查询数据库
    cursor.execute(
        """
        SELECT text, VECTOR_DISTANCE(embedding, :query_vec) AS distance
        FROM docs
        ORDER BY distance ASC
        FETCH FIRST :top_k ROWS ONLY
    """,
        query_vec=query_embedding,
        top_k=top_k,
    )

    results = cursor.fetchall()
    for i, (text, distance) in enumerate(results, 1):
        similarity = 1 - distance / 2  # 粗略转换为相似度
        print(f"{i}. {text}")
        print(f"   (similarity ≈ {similarity:.3f}, distance = {distance:.4f})")


# ---------- 运行示例搜索 ----------
oracle_vector_search(
    "What caused the block to be missing?", "HDFS Block Missing Search"
)

oracle_vector_search(
    "Why did the DataNode stop responding?", "DataNode Response Issue Search"
)

oracle_vector_search(
    "PacketResponder terminating", "PacketResponder Termination Search"
)

# ---------- 所有搜索完成后关闭连接 ----------
connection.close()
print("\n✅ Search complete, database connection closed.")
