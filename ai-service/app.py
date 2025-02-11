from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import mysql.connector
from sentence_transformers import SentenceTransformer
import json
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure Gemini with explicit API key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-pro')

# Configure Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", 
                  "http://127.0.0.1:8001", "http://localhost:8001"],  # Add both ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection function
@contextmanager
def get_db_connection():
    db = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="aitraning"
    )
    try:
        yield db
    finally:
        db.close()

# Get cursor function
@contextmanager
def get_db_cursor():
    with get_db_connection() as db:
        cursor = db.cursor()
        try:
            yield cursor, db
        finally:
            cursor.close()

def get_embedding(text):
    # Generate embedding using sentence-transformers
    embedding = embedding_model.encode(text)
    return embedding.tolist()

class SearchQuery(BaseModel):
    query: str

@app.post("/search")
async def search(query_data: SearchQuery):
    try:
        print(f"\n[DEBUG] Processing query: {query_data.query}")
        
        query_embedding = get_embedding(query_data.query)
        
        with get_db_cursor() as (cursor, db):
            # First check troubleshooting_guides
            cursor.execute("SELECT id, issue, solution, embedding FROM troubleshooting_guides")
            guide_results = cursor.fetchall()
            
            # Then check ai_responses
            cursor.execute("SELECT id, query, issue, solution, embedding, usage_count FROM ai_responses")
            ai_results = cursor.fetchall()
            
            # Calculate similarities for both tables
            similarities = []
            relevant_content = []
            
            print("[DEBUG] Calculating similarities with database entries...")
            
            # Process troubleshooting guides
            for result in guide_results:
                if result[3]:
                    similarity = calculate_similarity(query_embedding, result[3])
                    similarities.append(('guide', similarity, result))
                    if similarity > 0.5:
                        relevant_content.append(f"Issue: {result[1]}\nSolution: {result[2]}")
                        print(f"[DEBUG] Found relevant guide (similarity: {similarity:.2f}): {result[1]}")

            # Process AI responses
            for result in ai_results:
                if result[4]:
                    similarity = calculate_similarity(query_embedding, result[4])
                    similarities.append(('ai', similarity, result))
                    if similarity > 0.5:
                        relevant_content.append(f"Previous Query: {result[1]}\nIssue: {result[2]}\nSolution: {result[3]}")
                        print(f"[DEBUG] Found relevant AI response (similarity: {similarity:.2f}): {result[2]}")

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # If we have a good match
            if similarities and similarities[0][1] > 0.60:
                source_type, similarity, best_match = similarities[0]
                
                if source_type == 'guide':
                    print(f"\n[DEBUG] Found direct database match (similarity: {similarity:.2f})")
                    print(f"[DEBUG] Using guide response for: {best_match[1]}")
                    return {
                        "issue": best_match[1],
                        "solution": best_match[2],
                        "source": "database",
                        "similarity": float(similarity)
                    }
                else:
                    print(f"\n[DEBUG] Found AI response match (similarity: {similarity:.2f})")
                    print(f"[DEBUG] Using AI response for: {best_match[2]}")
                    # Increment usage count for this response
                    cursor.execute("UPDATE ai_responses SET usage_count = usage_count + 1 WHERE id = %s", (best_match[0],))
                    db.commit()
                    return {
                        "issue": best_match[2],
                        "solution": best_match[3],
                        "source": "ai_database",
                        "similarity": float(similarity)
                    }
            
            # Generate new response with Gemini
            print("\n[DEBUG] No exact match found, using Gemini with context")
            context = "\n\n".join(relevant_content[:3])
            
            response_prompt = f"""
            You are a tech support AI assistant. Use the following relevant information from our knowledge base to help answer the user's question. If the information isn't directly relevant, use it as context to provide a helpful response.

            Knowledge Base Information:
            {context}

            User Question: {query_data.query}

            Please provide a response that:
            1. Clearly states the issue
            2. Provides a step-by-step solution
            3. Uses relevant information from our knowledge base where applicable
            4. Adds your own expertise for a complete answer

            Format your response with a clear issue summary followed by numbered solution steps.
            """
            
            print("[DEBUG] Sending prompt to Gemini...")
            response = gemini_model.generate_content(response_prompt)
            generated_response = response.text
            
            try:
                parts = generated_response.split('\n', 1)
                issue = parts[0].strip()
                solution = parts[1].strip()
                
                # Store the new response in the database
                new_embedding = get_embedding(query_data.query + " " + issue + " " + solution)
                cursor.execute("""
                    INSERT INTO ai_responses (query, issue, solution, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (query_data.query, issue, solution, json.dumps(new_embedding)))
                db.commit()
                print("[DEBUG] Stored new AI response in database")
                
            except:
                issue = "Tech Support Response"
                solution = generated_response
            
            print("[DEBUG] Received response from Gemini")
            return {
                "issue": issue,
                "solution": solution,
                "source": "gemini",
                "context_used": bool(relevant_content),
                "num_contexts": len(relevant_content[:3])
            }
        
        return {
            "issue": issue,
            "solution": solution,
            "source": "gemini",
            "context_used": bool(relevant_content),
            "num_contexts": len(relevant_content[:3])
        }
        
    except Exception as e:
        print(f"[DEBUG] Error occurred: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    if isinstance(embedding2, str):
        embedding2 = json.loads(embedding2)
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/store-feedback")
def store_feedback(issue: str, user_feedback: str):
    with get_db_cursor() as (cursor, db):
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

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    print("Starting up FastAPI application...")

@app.on_event("shutdown")
async def shutdown():
    print("Shutting down FastAPI application...")
