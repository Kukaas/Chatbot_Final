from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import mysql.connector
from sentence_transformers import SentenceTransformer
import json
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a lightweight model good for embeddings

# MySQL connection
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="aitraning"
)
cursor = db.cursor()

def get_embedding(text):
    # Generate embedding using sentence-transformers
    embedding = model.encode(text)
    return embedding.tolist()  # Convert numpy array to list for JSON serialization

class SearchQuery(BaseModel):
    query: str

@app.post("/search")
async def search(query_data: SearchQuery):
    query_embedding = get_embedding(query_data.query)
    
    # Get all embeddings from database
    cursor.execute("SELECT id, issue, solution, embedding FROM troubleshooting_guides")
    results = cursor.fetchall()
    
    # Calculate similarities
    similarities = []
    for result in results:
        db_embedding = np.array(json.loads(result[3]))
        query_embedding_np = np.array(query_embedding)
        similarity = np.dot(query_embedding_np, db_embedding) / (
            np.linalg.norm(query_embedding_np) * np.linalg.norm(db_embedding)
        )
        similarities.append((similarity, result))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    if similarities:
        best_match = similarities[0][1]
        return {
            "issue": best_match[1],
            "solution": best_match[2]
        }
    
    return {"error": "No relevant guide found"}

@app.post("/store-feedback")
def store_feedback(issue: str, user_feedback: str):
    cursor.execute("INSERT INTO feedback (issue, feedback) VALUES (%s, %s)", (issue, user_feedback))
    db.commit()
    return {"message": "Feedback stored successfully"}

@app.post("/generate-embedding")
async def generate_embedding(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required for embedding generation.")
    
    embedding = get_embedding(text)
    return {"embedding": embedding}
