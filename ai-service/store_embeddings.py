import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer

# MySQL connection
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="aitraning"
)
cursor = db.cursor()


model = SentenceTransformer("all-MiniLM-L6-v2")

cursor.execute("SELECT id, issue, solution FROM troubleshooting_guides")
rows = cursor.fetchall()

for row in rows:
    text = f"{row[1]} {row[2]}"
    embedding = model.encode(text)
    embedding_bytes = embedding.tobytes()
    
    cursor.execute("UPDATE troubleshooting_guides SET embedding=%s WHERE id=%s", (embedding_bytes, row[0]))

db.commit()
print("Embeddings stored successfully!")
