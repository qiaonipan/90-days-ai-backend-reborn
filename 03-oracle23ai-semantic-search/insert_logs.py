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

# ---------- Datasets ----------
datasets = [
    ("Qiaoni's Best Qualities", [
        "Qiaoni is a motivated individual who loves to explore new technologies",
        "Qiaoni has a passion for sharing knowledge with others",
        "Qiaoni enjoys working on collaborative projects and learning from peers",
        "Qiaoni consistently takes initiative to solve problems before they escalate",
        "Qiaoni communicates clearly and adapts well to feedback",
        "Qiaoni enjoyed watching 'Avatar: Fire and Ash' last weekend",
        "Qiaoni is planning to travel to Florida next spring",
        "Qiaoni recently tried a new ramen restaurant downtown",
        "Qiaoni is currently reading a historical novel"
    ]),
    ("Milk Tea Experiences", [
        "I drank a warm brown sugar milk tea and felt relaxed, happy, and satisfied",
        "The matcha milk tea was perfectly balanced, not too sweet, and I really enjoyed it",
        "The fruit milk tea was refreshing and made my afternoon much better",
        "The milk tea shop was very crowded and I had to wait 30 minutes",
        "I tried a cheese foam oolong tea but didn't really like the aftertaste"
    ]),
    ("Coffee Experiences", [
        "The espresso shot gave me a strong boost of energy",
        "My iced americano was too watery and bland",
        "A creamy cappuccino helped me stay focused during the meeting",
        "The coffee was too bitter and upset my stomach",
        "I enjoyed a caramel macchiato while reading my favorite book"
    ])
]

# Embed and insert
id_counter = 1
for title, logs in datasets:
    print(f"Inserting: {title} ({len(logs)} items)")
    for text in logs:
        # Generate embeddings
        embedding_list = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        # Key conversion: convert to array.array('f') preferred by Oracle VECTOR
        embedding = array.array('f', embedding_list)

        # Insert into database
        cursor.execute("""
            INSERT INTO docs (id, text, embedding)
            VALUES (:1, :2, :3)
        """, (id_counter, text, embedding))
        id_counter += 1

connection.commit()
connection.close()
print("\n🎉 All data has been safely inserted into the Oracle 26ai database!")