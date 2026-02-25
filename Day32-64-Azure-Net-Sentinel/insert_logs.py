import os
import openai
import oracledb
from dotenv import load_dotenv
import array

# 从.env加载密钥
load_dotenv()

# 从环境变量安全读取配置
openai.api_key = os.getenv("OPENAI_API_KEY")

username = os.getenv("ORACLE_USERNAME")
password = os.getenv("ORACLE_PASSWORD")
dsn = os.getenv("ORACLE_DSN")
wallet_path = os.getenv("ORACLE_WALLET_PATH")

# 连接到Oracle 26ai数据库
connection = oracledb.connect(
    user=username,
    password=password,
    dsn=dsn,
    config_dir=wallet_path,
    wallet_location=wallet_path,
    wallet_password=password,
)

cursor = connection.cursor()

# 清除旧数据（每次运行都插入新数据）
cursor.execute("DELETE FROM docs")
connection.commit()

# 初始化数据集列表
datasets = []

# ---------- 大型日志：HDFS真实生产日志 ----------
try:
    log_path = "data/HDFS_2k.log"
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 过滤掉空行或过短的行；每行是一个日志条目
    hdfs_logs = [
        line.strip() for line in lines if line.strip() and len(line.strip()) > 20
    ]

    # 取前1000条（Oracle Always Free足够；插入更快）
    hdfs_logs = hdfs_logs[:1000]

    datasets.append(("HDFS real production logs (1000 items)", hdfs_logs))
    print(
        f"Successfully loaded {len(hdfs_logs)} real HDFS production logs; will insert into database"
    )
except Exception as e:
    print(f"Failed to load large logs (small dataset unaffected): {e}")

# 嵌入并插入
for title, logs in datasets:
    print(f"Inserting: {title} ({len(logs)} items)")
    for i, text in enumerate(logs):
        # 生成embedding
        response = openai.embeddings.create(input=text, model="text-embedding-3-small")
        embedding_list = response.data[0].embedding

        # 关键转换：转换为Oracle VECTOR首选的array.array('f')
        embedding = array.array("f", embedding_list)

        # 插入（ID由数据库自动生成）
        cursor.execute(
            """
            INSERT INTO docs (text, embedding)
            VALUES (:1, :2)
        """,
            (text, embedding),
        )

        if (i + 1) % 50 == 0:
            print(f"Inserted {i+1}/{len(logs)} items")

connection.commit()
connection.close()
print("\n🎉 All data has been safely inserted into the Oracle 26ai database!")
