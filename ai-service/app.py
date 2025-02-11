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
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

# Load environment variables
load_dotenv()

# Define the SearchQuery model
class SearchQuery(BaseModel):
    query: str
    conversation_history: list = []  # Add this field to track conversation

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

def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize text for better matching"""
    # Common misspellings dictionary
    misspellings = {
        'overhiting': 'overheating',
        'overheat': 'overheating',
        'overheated': 'overheating',
        'heating': 'overheating',
        'hot': 'overheating',
        'temperature': 'overheating',
        # Add more common misspellings as needed
    }
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Replace known misspellings
    words = text.split()
    normalized_words = []
    for word in words:
        # Use fuzzy matching to find the closest misspelling
        best_match = process.extractOne(word, misspellings.keys())
        if best_match and best_match[1] >= 80:  # 80% similarity threshold
            normalized_words.append(misspellings[best_match[0]])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

def find_closest_match(text: str, choices: list, threshold: int = 80) -> str:
    """Find the closest matching word from a list of choices"""
    # Clean the input text
    text = clean_and_normalize_text(text)
    # Find the best match
    best_match = process.extractOne(text, choices)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return text

def analyze_question(query: str) -> dict:
    """Analyze the question to determine its type and context"""
    # Clean and normalize the query
    query_lower = clean_and_normalize_text(query)
    
    # Define categories and their keywords with common misspellings
    categories = {
        'wifi': ['wifi', 'wireless', 'router', 'network', 'connection', 'internet', 
                'wifi', 'wi-fi', 'wiifi', 'wirless', 'ruter', 'conection', 'inet'],
        'data': ['data', 'mobile data', 'cellular', '4g', '5g', 'signal', 
                'mobil', 'celular', 'signals', 'mobiledata', 'cell'],
        'login': ['login', 'password', 'account', 'username', 'sign in', 
                'signin', 'loging', 'passwd', 'user', 'credentials', 'pwd'],
        'hardware': ['device', 'phone', 'computer', 'laptop', 'screen', 'battery',
                    'fone', 'compter', 'labtop', 'devise', 'batery'],
        'software': ['app', 'application', 'program', 'software', 'system',
                    'aplication', 'progam', 'sofware', 'sistm']
    }
    
    # Find matching categories using fuzzy matching
    matched_categories = []
    words = query_lower.split()
    
    for category, keywords in categories.items():
        # Check each word in the query against the keywords
        for word in words:
            # Use fuzzy matching to find similar words
            best_match = process.extractOne(word, keywords)
            if best_match and best_match[1] >= 75:  # 75% similarity threshold
                matched_categories.append(category)
                break
    
    # Common misspellings for problem indicators
    problem_indicators = [
        'issue', 'problem', 'error', 'not working', 'failed', 'help',
        'isue', 'problm', 'eror', 'notworking', 'faild', 'halp', 'hlp',
        'trouble', 'truble', 'fix', 'broken', 'brokn', 'stuck', 'stuk'
    ]
    
    question_indicators = [
        'how', 'what', 'why', 'where', 'when', 'can', 'could',
        'hw', 'wut', 'wy', 'wher', 'wen', 'cn', 'cud'
    ]
    
    return {
        'categories': matched_categories,
        'is_question': any(fuzz.partial_ratio(q, query_lower) >= 75 for q in question_indicators),
        'is_problem_statement': any(fuzz.partial_ratio(p, query_lower) >= 75 for p in problem_indicators),
        'normalized_query': query_lower
    }

def is_troubleshooting_query(query: str, conversation_history: list = None) -> bool:
    # Clean and normalize the query
    query_lower = clean_and_normalize_text(query)
    
    # If this is part of an ongoing conversation, treat it as troubleshooting
    if conversation_history and len(conversation_history) > 0:
        return True
        
    # List of troubleshooting-related keywords with common misspellings
    troubleshooting_keywords = [
        'error', 'eror', 'err',
        'issue', 'isue', 'prob',
        'problem', 'problm', 'trouble',
        'fix', 'help', 'halp',
        'not working', 'notworking', 'broken',
        'failed', 'faild', 'stuck',
        'crash', 'crashd', 'bug',
        'debug', 'cant', 'cannot',
        'wont', 'doesnt', 'frozen',
        'slow', 'connection', 'conectn'
    ]
    
    # Use fuzzy matching for each keyword
    return any(fuzz.partial_ratio(keyword, query_lower) >= 75 for keyword in troubleshooting_keywords)

def analyze_database_solutions(cursor, query_embedding, category):
    """Analyze database solutions to provide alternative approaches"""
    alternative_solutions = []
    
    # Get all relevant solutions from both tables
    cursor.execute("""
        SELECT 'guide' as source, issue, solution, embedding 
        FROM troubleshooting_guides 
        UNION ALL 
        SELECT 'ai' as source, issue, solution, embedding 
        FROM ai_responses
    """)
    results = cursor.fetchall()
    
    # Analyze and group similar solutions
    for result in results:
        if result[3]:  # if embedding exists
            similarity = calculate_similarity(query_embedding, result[3])
            if similarity > 0.4:  # Lower threshold to get more alternatives
                alternative_solutions.append({
                    'source': result[0],
                    'issue': result[1],
                    'solution': result[2],
                    'similarity': similarity
                })
    
    # Sort by similarity and remove duplicates
    alternative_solutions.sort(key=lambda x: x['similarity'], reverse=True)
    unique_solutions = []
    seen_solutions = set()
    
    for sol in alternative_solutions:
        # Create a simplified version of the solution for comparison
        simple_sol = ' '.join(sol['solution'].lower().split())
        if simple_sol not in seen_solutions:
            unique_solutions.append(sol)
            seen_solutions.add(simple_sol)
    
    return unique_solutions[:5]  # Return top 5 unique solutions

def is_follow_up_question(query: str, conversation_history: list) -> tuple:
    # Ignore the initial greeting when checking conversation history
    actual_history = [msg for msg in conversation_history if not msg.startswith("Hello")]
    
    # If there's no real history, it can't be a follow-up
    if len(actual_history) < 2:
        return False, False
        
    follow_up_indicators = [
        "still", "not working", "didn't work", "doesn't work",
        "another", "else", "other", "more", "again", "alternative",
        "what about", "how about", "then", "after that",
        "try something else", "different solution", "new solution"
    ]
    query_lower = query.lower()
    
    # Check if the user is explicitly requesting an alternative
    is_alternative_request = any(phrase in query_lower for phrase in [
        "alternative", "another solution", "something else", 
        "different", "other way", "new solution"
    ])
    
    # Check if the query contains follow-up indicators
    is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
    
    return is_follow_up, is_alternative_request

def calculate_similarity(embedding1, embedding2, query_text=None, match_text=None):
    """Calculate cosine similarity between two embeddings"""
    if isinstance(embedding2, str):
        embedding2 = json.loads(embedding2)
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate embedding similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # If we have text to compare, include text similarity
    if query_text and match_text:
        normalized_text1 = clean_and_normalize_text(query_text)
        normalized_text2 = clean_and_normalize_text(match_text)
        text_similarity = fuzz.ratio(normalized_text1, normalized_text2) / 100.0
        # Combine embedding similarity with text similarity
        final_similarity = (similarity + text_similarity) / 2
        return final_similarity
    
    return similarity

@app.post("/search")
async def search(query_data: SearchQuery):
    try:
        print(f"\n[DEBUG] Processing query: {query_data.query}")
        
        # Normalize the query first
        normalized_query = clean_and_normalize_text(query_data.query)
        print(f"[DEBUG] Normalized query: {normalized_query}")
        
        # Get query embedding using normalized query
        query_embedding = get_embedding(normalized_query)
        
        # Initialize relevant_content list and context
        relevant_content = []
        context = ""
        
        # Analyze the question first
        question_analysis = analyze_question(query_data.query)
        print(f"[DEBUG] Question analysis: {question_analysis}")
        
        # Search for relevant content in database
        best_match = None
        best_similarity = 0
        
        with get_db_cursor() as (cursor, db):
            # First check troubleshooting_guides
            cursor.execute("SELECT id, issue, solution, embedding FROM troubleshooting_guides")
            guide_results = cursor.fetchall()
            
            # Then check ai_responses
            cursor.execute("SELECT id, query, issue, solution, embedding FROM ai_responses")
            ai_results = cursor.fetchall()
            
            # Calculate similarities and find best match
            for result in guide_results:
                if result[3]:  # if embedding exists
                    similarity = calculate_similarity(
                        query_embedding, 
                        result[3],
                        query_data.query,
                        result[1]  # issue text
                    )
                    print(f"[DEBUG] Guide similarity: {similarity:.2f} for issue: {result[1]}")
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'source': 'guide',
                            'issue': result[1],
                            'solution': result[2],
                            'similarity': similarity
                        }
                    if similarity > 0.5:
                        relevant_content.append(f"Issue: {result[1]}\nSolution: {result[2]}")
                        print(f"[DEBUG] Found relevant guide (similarity: {similarity:.2f}): {result[1]}")
            
            for result in ai_results:
                if result[4]:  # if embedding exists
                    similarity = calculate_similarity(
                        query_embedding, 
                        result[4],
                        result[1],  # query text
                        result[2]  # issue text
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'source': 'ai',
                            'issue': result[2],
                            'solution': result[3],
                            'similarity': similarity
                        }
                    if similarity > 0.5:
                        relevant_content.append(f"Previous Query: {result[1]}\nIssue: {result[2]}\nSolution: {result[3]}")
                        print(f"[DEBUG] Found relevant AI response (similarity: {similarity:.2f}): {result[2]}")

        # Lower the threshold for using existing solutions
        if best_match and best_match['similarity'] > 0.6:  # Lower from 0.75 to 0.6
            print(f"[DEBUG] Using existing solution from {best_match['source']} (similarity: {best_match['similarity']:.2f})")
            return {
                "issue": best_match['issue'],
                "options": "Would you like to know more about this solution?",
                "solution": best_match['solution'],
                "source": best_match['source'],
                "similarity": float(best_match['similarity']),
                "type": "initial_response"
            }

        # Create context from relevant content
        if relevant_content:
            context = "\n\n".join(relevant_content)
            print("[DEBUG] Using relevant content in Gemini prompt:")
            for content in relevant_content:
                print(f"[DEBUG] Content: {content}")

        # Generate the response prompt
        response_prompt = f"""
        You are a friendly tech support assistant. Help the user with their technical issue.

        User's question: {query_data.query}
        {f'Previous solutions from our database:{chr(10)}{context}' if context else ''}

        If relevant solutions from our database are provided above, use them as a basis for your response.
        Adapt and expand upon these solutions rather than creating entirely new ones.

        Respond naturally like a helpful tech support person. Format your response with "|||" separators:
        1. A brief understanding of their issue
        2. The main points we'll address
        3. The detailed solution steps

        Example:
        I understand you're experiencing overheating issues|||Let's check these potential causes:
        • CPU and cooling system
        • Ventilation and airflow|||Here's what we'll do:
        1. First step...
        2. Second step...
        """
        
        print("[DEBUG] Sending prompt to Gemini...")
        response = gemini_model.generate_content(response_prompt)
        generated_response = response.text
        
        try:
            # Split the response into sections using the delimiter
            sections = generated_response.split('|||')
            if len(sections) >= 3:
                issue = sections[0].strip()
                options = sections[1].strip()
                solution = sections[2].strip()
            else:
                # Fallback if response isn't properly formatted
                issue = "Tech Support Response"
                options = "Would you like more details?"
                solution = generated_response

            # Store the new response in the database
            new_embedding = get_embedding(query_data.query + " " + issue + " " + solution)
            with get_db_cursor() as (cursor, db):
                cursor.execute("""
                    INSERT INTO ai_responses (query, issue, solution, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (query_data.query, issue, solution, json.dumps(new_embedding)))
                db.commit()
                print("[DEBUG] Stored new AI response in database")
                
            return {
                "issue": issue,
                "options": options,
                "solution": solution,
                "source": "gemini",
                "context_used": bool(relevant_content),
                "num_contexts": len(relevant_content[:3]),
                "type": "initial_response"
            }
        
        except Exception as e:
            print(f"[DEBUG] Error parsing Gemini response: {str(e)}")
            issue = "Tech Support Response"
            solution = generated_response
            
            # Store the unparsed response in the database
            new_embedding = get_embedding(query_data.query + " " + issue + " " + solution)
            with get_db_cursor() as (cursor, db):
                cursor.execute("""
                    INSERT INTO ai_responses (query, issue, solution, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (query_data.query, issue, solution, json.dumps(new_embedding)))
                db.commit()
                print("[DEBUG] Stored unparsed AI response in database")
            
            return {
                "issue": issue,
                "options": "Would you like to see the full response?",
                "solution": solution,
                "source": "gemini",
                "context_used": bool(relevant_content),
                "num_contexts": len(relevant_content[:3]),
                "type": "initial_response"
            }
        
        print("[DEBUG] Received response from Gemini")

    except Exception as e:
        print(f"[DEBUG] Error occurred: {str(e)}")
        error_response = {"error": f"An error occurred: {str(e)}"}
        
        # Store error responses too
        with get_db_cursor() as (cursor, db):
            new_embedding = get_embedding(query_data.query + " Error " + str(e))
            cursor.execute("""
                INSERT INTO ai_responses (query, issue, solution, embedding)
                VALUES (%s, %s, %s, %s)
            """, (
                query_data.query,
                "Error Response",
                f"An error occurred: {str(e)}",
                json.dumps(new_embedding)
            ))
            db.commit()
            print("[DEBUG] Stored error response in database")
        
        return error_response

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
