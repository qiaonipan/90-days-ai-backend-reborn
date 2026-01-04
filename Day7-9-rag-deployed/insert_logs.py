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

# ---------- Small dataset (expanded Qiaoni section) ----------
datasets = [
    ("Qiaoni's profile and preferences", [
        # Basic traits
        "Qiaoni is a motivated individual who loves to explore new technologies",
        "Qiaoni has a passion for sharing knowledge with others",
        "Qiaoni enjoys working on collaborative projects and learning from peers",
        "Qiaoni consistently takes initiative to solve problems before they escalate",
        "Qiaoni communicates clearly and adapts well to feedback",
        "Qiaoni is resilient and turns challenges into learning opportunities",
        "Qiaoni values work-life balance and enjoys quality time with family",

        # Interests and hobbies
        "Qiaoni loves reading historical novels and biographies in her free time",
        "Qiaoni enjoys watching sci-fi movies, especially the Avatar series",
        "Qiaoni is learning AI and cloud database technologies as a personal challenge",
        "Qiaoni likes long walks in the park to clear her mind",
        "Qiaoni practices meditation every morning for 10 minutes",
        "Qiaoni enjoys gardening and has several potted plants at home",

        # Food preferences
        "Qiaoni's favorite food is homemade dumplings with vinegar",
        "Qiaoni loves spicy Sichuan hotpot on cold days",
        "Qiaoni enjoys fresh seafood, especially steamed fish",
        "Qiaoni likes trying new dessert shops on weekends",
        "Qiaoni prefers light breakfasts like congee and youtiao",
        "Qiaoni's comfort food is tomato egg noodle soup",
        "Qiaoni enjoys fruit teas more than milk teas",

        # Travel and lifestyle
        "Qiaoni is planning to travel to Florida next spring",
        "Qiaoni dreams of visiting Japan during cherry blossom season",
        "Qiaoni recently tried a new ramen restaurant downtown and loved it",
        "Qiaoni prefers quiet cafes for working or reading",
        "Qiaoni enjoys family gatherings and cooking big meals together",
        "Qiaoni values lifelong learning and personal growth",

        # Daily habits
        "Qiaoni wakes up at 6:30 AM every day",
        "Qiaoni drinks two cups of green tea daily",
        "Qiaoni tracks her reading progress with a notebook",
        "Qiaoni organizes her code projects meticulously",
        "Qiaoni believes in continuous self-improvement after career setbacks"
    ])
]

# ---------- Large logs: HDFS real production logs ----------
try:
    log_path = "data/HDFS_2k.log"
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Filter out empty or too-short lines; each line is a log entry
    hdfs_logs = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]

    # Take the first 1000 entries (Oracle Always Free is sufficient; faster inserts)
    hdfs_logs = hdfs_logs[:1000]

    datasets.append(("HDFS real production logs (1000 items)", hdfs_logs))
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