import os
import openai
import oracledb
from dotenv import load_dotenv
import array

# Load secrets from .env
load_dotenv()

# Read configuration securely from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

username = os.getenv("ORACLE_USERNAME")
password = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_DSN")
wallet_path = os.getenv("ORACLE_WALLET_PATH")

# Connect to Oracle 26ai database
connection = oracledb.connect(
    user=username,
    password=password,
    dsn=dsn,
    config_dir=wallet_path,
    wallet_location=wallet_path,
    wallet_password=password
)

cursor = connection.cursor()

# Clear old data (insert fresh each run)
cursor.execute("DELETE FROM docs")
connection.commit()

# Initialize datasets list
datasets = []

# ---------- Large logs: HDFS real production logs ----------
try:
    log_path = "data/HDFS_2k.log"
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Filter out empty or too-short lines; each line is a log entry
    hdfs_logs = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]

    # Take the first 2000 entries (Oracle Always Free is sufficient; faster inserts)
    hdfs_logs = hdfs_logs[:2000]

    datasets.append(("HDFS real production logs (2000 items)", hdfs_logs))
    print(f"Successfully loaded {len(hdfs_logs)} real HDFS production logs; will insert into database")
except Exception as e:
    print(f"Failed to load large logs (small dataset unaffected): {e}")

# Embed and insert
id_counter = 1
for title, logs in datasets:
    print(f"Inserting: {title} ({len(logs)} items)")
    for i, text in enumerate(logs):
        # Generate embedding
        response = openai.embeddings.create(input=text, model="text-embedding-3-small")
        embedding_list = response.data[0].embedding

        # Key conversion: convert to array.array('f') preferred by Oracle VECTOR
        embedding = array.array('f', embedding_list)

        # Insert
        cursor.execute("""
            INSERT INTO docs (id, text, embedding)
            VALUES (:1, :2, :3)
        """, (id_counter, text, embedding))
        id_counter += 1

        if (i + 1) % 50 == 0:
            print(f"Inserted {i+1}/{len(logs)} items")

connection.commit()
connection.close()
print("\n🎉 All data has been safely inserted into the Oracle 26ai database!")