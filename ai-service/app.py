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

class FeedbackData(BaseModel):
    response_id: int
    was_helpful: bool
    feedback_text: str | None = None  # Make it optional with None default

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
                  "http://127.0.0.1:8001", "http://localhost:8001"],  # Both ports already configured
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
        'wifi': 'wireless network',
        'wi-fi': 'wireless network',
        'wireless': 'wireless network',
        'internet': 'network connection',
        'net': 'network connection',
        'connection': 'network connection',
        'router': 'network device',
        'modem': 'network device',
        'signal': 'network signal',
        'weak': 'poor connection',
        'slow': 'poor performance',
        'disconnect': 'connection loss',
        'dropped': 'connection loss',
        'password': 'network credentials',
        'wpa': 'network security',
        'wep': 'network security',
    }
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Replace known misspellings only if they appear as whole words
    words = text.split()
    normalized_words = []
    for word in words:
        # Only replace if it's an exact match or very close match
        best_match = process.extractOne(word, misspellings.keys())
        if best_match and best_match[1] >= 90:  # Increase threshold to 90%
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

def calculate_similarity(embedding1, embedding2, query_text=None, match_text=None, effectiveness_score=1.0):
    """Calculate cosine similarity between two embeddings, weighted by effectiveness"""
    if isinstance(embedding2, str):
        embedding2 = json.loads(embedding2)
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate embedding similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # If we have text to compare, include text similarity
    if query_text and match_text:
        normalized_query = clean_and_normalize_text(query_text)
        normalized_match = clean_and_normalize_text(match_text)
        
        # Calculate fuzzy string matching similarity
        text_similarity = fuzz.ratio(normalized_query, normalized_match) / 100.0
        
        # Calculate keyword matching
        query_keywords = set(normalized_query.split())
        match_keywords = set(normalized_match.split())
        
        # Keywords grouped by category for better matching
        important_keywords = {
            # Network related
            'wireless', 'network', 'wifi', 'connection', 'internet', 
            'router', 'modem', 'signal', 'connectivity',
            # Hardware related
            'hardware', 'device', 'computer', 'laptop', 'desktop',
            'screen', 'display', 'keyboard', 'mouse', 'battery',
            'power', 'charging', 'usb', 'port', 'cable',
            # Software related
            'software', 'program', 'application', 'app', 'windows',
            'mac', 'update', 'install', 'driver', 'system',
            # Performance related
            'slow', 'fast', 'speed', 'performance', 'memory',
            'ram', 'cpu', 'processor', 'disk', 'storage',
            # Error related
            'error', 'issue', 'problem', 'fail', 'crash',
            'freeze', 'hang', 'stop', 'break', 'bug',
            # Status words
            'not', 'working', 'broken', 'failed', 'dead',
            'stuck', 'frozen', 'crashed', 'slow', 'overheating'
        }
        
        # Calculate keyword overlap with extra weight for important keywords
        common_keywords = query_keywords & match_keywords
        important_matches = len(common_keywords & important_keywords)
        
        # Calculate category-based relevance
        keyword_similarity = (len(common_keywords) + important_matches * 2) / (len(query_keywords) + len(match_keywords))
        
        # Combine similarities with adjusted weights
        combined_similarity = (
            similarity * 0.3 +          # Embedding similarity (reduced weight)
            text_similarity * 0.3 +     # Text similarity
            keyword_similarity * 0.4    # Keyword similarity (increased weight)
        )
        
        # Weight more heavily for follow-up questions or urgent issues
        urgency_keywords = {'not', 'working', 'failed', 'error', 'help', 'urgent', 'emergency'}
        if any(word in normalized_query for word in urgency_keywords):
            similarity = combined_similarity * 1.2
        else:
            similarity = combined_similarity
    
    # Weight by effectiveness score with higher baseline
    weighted_similarity = similarity * (effectiveness_score / 5.0 + 0.8)
    
    print(f"[DEBUG] Similarity breakdown for '{match_text}':")
    print(f"  - Embedding similarity: {similarity:.3f}")
    if query_text and match_text:
        print(f"  - Text similarity: {text_similarity:.3f}")
        print(f"  - Keyword similarity: {keyword_similarity:.3f}")
        print(f"  - Combined similarity: {combined_similarity:.3f}")
        print(f"  - Final weighted similarity: {weighted_similarity:.3f}")
        print(f"  - Common keywords: {common_keywords}")
        print(f"  - Important matches: {common_keywords & important_keywords}")
    
    return weighted_similarity

def get_conversation_context(cursor, conversation_history):
    """Get context from conversation history"""
    context = []
    
    if conversation_history:
        # Get the last few messages
        recent_messages = conversation_history[-3:]  # Last 3 messages
        
        # Get related solutions for the conversation topic
        topic_text = " ".join(recent_messages)
        topic_embedding = get_embedding(topic_text)
        
        cursor.execute("""
            SELECT ar.solution, ar.effectiveness_score
            FROM ai_responses ar
            JOIN feedback f ON ar.id = f.response_id
            WHERE f.was_helpful = TRUE
            AND ar.effectiveness_score > 0.7
            ORDER BY ar.effectiveness_score DESC
            LIMIT 3
        """)
        
        related_solutions = cursor.fetchall()
        if related_solutions:
            context.extend([sol[0] for sol in related_solutions])
    
    return context

@app.post("/search")
async def search(query_data: SearchQuery):
    try:
        print(f"\n[DEBUG] Processing query: {query_data.query}")
        normalized_query = clean_and_normalize_text(query_data.query)
        query_embedding = get_embedding(normalized_query)
        
        with get_db_cursor() as (cursor, db):
            try:
                print("\n[DEBUG] 1. Searching troubleshooting guides...")
                cursor.execute("""
                    SELECT id, title, content, embedding 
                    FROM troubleshooting_guides 
                    WHERE is_active = TRUE
                """)
                guide_results = cursor.fetchall()
                print(f"[DEBUG] Found {len(guide_results)} active troubleshooting guides")
            except mysql.connector.Error as e:
                print(f"[DEBUG] Troubleshooting guides table not available: {e}")
                guide_results = []
            
            print("\n[DEBUG] 2. Searching AI responses with feedback...")
            cursor.execute("""
                SELECT ar.id, ar.issue, ar.solution, ar.embedding, 
                       ar.effectiveness_score,
                       COUNT(f.id) as feedback_count,
                       GROUP_CONCAT(CASE 
                           WHEN f.feedback_text IS NOT NULL 
                           THEN CONCAT(
                               'Feedback: ', 
                               f.feedback_text, 
                               ' (', CASE WHEN f.was_helpful THEN 'Helpful' ELSE 'Not Helpful' END, ')'
                           )
                           ELSE NULL 
                       END SEPARATOR '\n') as feedback_texts
                FROM ai_responses ar
                LEFT JOIN feedback f ON ar.id = f.response_id
                WHERE ar.effectiveness_score > 0.3  # Lowered from 0.5
                GROUP BY ar.id
                HAVING feedback_count = 0 OR 
                       (feedback_count > 0 AND 
                        SUM(CASE WHEN f.was_helpful THEN 1 ELSE 0 END) / feedback_count >= 0.4)  # Lowered from 0.6
                ORDER BY ar.effectiveness_score DESC, feedback_count DESC
                LIMIT 10  # Increased from 5
            """)
            ai_results = cursor.fetchall()
            print(f"[DEBUG] Found {len(ai_results)} relevant AI responses")

            print("\n[DEBUG] 3. Combining and ranking all sources...")
            all_sources = []
            
            # Add guides to sources with lower threshold
            for guide in guide_results:
                similarity = calculate_similarity(
                    query_embedding,
                    json.loads(guide[3]),
                    query_data.query,
                    guide[1]
                )
                print(f"[DEBUG] Guide '{guide[1]}' similarity: {similarity:.3f}")
                if similarity > 0.3:  # Lowered from 0.5
                    all_sources.append({
                        'type': 'guide',
                        'title': guide[1],
                        'content': guide[2],
                        'similarity': similarity,
                        'id': guide[0]
                    })
                    print(f"[DEBUG] Found matching guide: '{guide[1]}' with similarity {similarity:.3f}")

            # Add AI responses to sources with lower threshold
            for result in ai_results:
                similarity = calculate_similarity(
                    query_embedding,
                    json.loads(result[3]),
                    query_data.query,
                    result[1]
                )
                print(f"[DEBUG] AI Response '{result[1]}' similarity: {similarity:.3f}")
                if similarity > 0.3:  # Lowered from 0.5
                    all_sources.append({
                        'type': 'ai_response',
                        'issue': result[1],
                        'solution': result[2],
                        'similarity': similarity,
                        'effectiveness': result[4],
                        'feedback': result[6],
                        'id': result[0]
                    })
                    print(f"[DEBUG] Found matching AI response: '{result[1]}' with similarity {similarity:.3f}")

            # Sort by similarity
            all_sources.sort(key=lambda x: x['similarity'], reverse=True)
            print(f"\n[DEBUG] Total matching sources found: {len(all_sources)}")

            if all_sources:
                print(f"[DEBUG] Best match type: {all_sources[0]['type']} with similarity: {all_sources[0]['similarity']:.3f}")
                print(f"[DEBUG] Best match title/issue: {all_sources[0].get('title') or all_sources[0].get('issue')}")
            else:
                print("[DEBUG] No good matches found, falling back to Gemini")

            # Lower the threshold for using existing content
            if all_sources and all_sources[0]['similarity'] > 0.4:  # Lowered from 0.6
                # Get the best matching source
                best_source = all_sources[0]
                
                # Prepare context for Gemini
                context = f"""
                Found relevant {best_source['type']}:
                
                {"Title: " + best_source['title'] if 'title' in best_source else "Issue: " + best_source['issue']}
                {"Content: " + best_source['content'] if 'content' in best_source else "Solution: " + best_source['solution']}
                
                Additional context from other sources:
                """
                
                # Add context from other relevant sources
                for source in all_sources[1:3]:  # Use next 2 best matches
                    context += f"\n- {source['type'].title()}: "
                    context += source['title'] if 'title' in source else source['issue']
                
                # If we have feedback, add it
                if 'feedback' in best_source and best_source['feedback']:
                    context += "\n\nUser feedback from similar issues:\n"
                    context += best_source['feedback']

                # Generate enhanced response using Gemini
                response_prompt = f"""
                You are a tech support assistant. A user has asked: "{query_data.query}"

                We have found relevant information from our database:
                {context}

                Please create a comprehensive response that:
                1. Shows understanding of the issue
                2. Lists key points in a structured way:
                   - For step-by-step solutions, use numbered steps (1., 2., etc.)
                   - For options or alternatives, use bullet points
                3. Provides detailed solution

                Format your response in exactly this structure:

                Understanding: Brief statement showing you understand the user's issue

                Key Points:
                [List points here - use numbers for steps, bullets (-) for options]

                Detailed Solution:
                [Break down the solution into clear sections with numbered steps where appropriate]
                """

                response = gemini_model.generate_content(response_prompt)
                sections = response.text.split('\n\n')

                # Find the sections by their headers
                issue = ""
                options = ""
                solution = ""

                for section in sections:
                    if section.startswith('Understanding:'):
                        issue = section.replace('Understanding:', '').strip()
                    elif section.startswith('Key Points:'):
                        options = section.replace('Key Points:', '').strip()
                    elif section.startswith('Detailed Solution:'):
                        solution = section.replace('Detailed Solution:', '').strip()

                if not all([issue, options, solution]):
                    issue = "Tech Support Response"
                    options = "Here are the steps to help you:"
                    solution = response.text.strip()

                # Store the enhanced response
                new_embedding = get_embedding(query_data.query + " " + issue + " " + solution)
                cursor.execute("""
                    INSERT INTO ai_responses (query, issue, solution, embedding, effectiveness_score)
                    VALUES (%s, %s, %s, %s, %s)
                """, (query_data.query, issue, solution, json.dumps(new_embedding), 0.5))
                
                # Get the last inserted ID
                cursor.execute("SELECT LAST_INSERT_ID()")
                new_id = cursor.fetchone()[0]
                db.commit()

                return {
                    "issue": issue,
                    "options": options,
                    "solution": solution,
                    "source": "combined",
                    "similarity": float(best_source['similarity']) if best_source else 0.0,
                    "type": "initial_response",
                    "response_id": new_id
                }

            # If no good matches, fall back to Gemini
            if not all_sources or all_sources[0]['similarity'] <= 0.4:
                # Get conversation context
                context = get_conversation_context(cursor, query_data.conversation_history)
                
                # Prepare prompt for Gemini
                prompt = f"""
                You are a tech support assistant. A user has asked: "{query_data.query}"
                
                Previous conversation context:
                {context}
                
                Please provide a helpful response in exactly this structure:

                Understanding: Write a brief statement showing you understand the user's issue

                Key Points:
                - For step-by-step instructions, use numbered steps (1., 2., etc.)
                - For troubleshooting options, use bullet points (-)
                - Keep each point clear and concise

                Detailed Solution:
                Provide a detailed explanation that expands on the key points

                Remember:
                - Use numbered steps (1., 2., etc.) for sequential instructions
                - Use bullet points (-) for non-sequential options or alternatives
                - Keep formatting consistent and clean
                """
                
                response = gemini_model.generate_content(prompt)
                sections = response.text.split('\n\n')
                
                # Find the sections by their headers
                issue = ""
                options = ""
                solution = ""

                for section in sections:
                    if section.startswith('Understanding:'):
                        issue = section.replace('Understanding:', '').strip()
                    elif section.startswith('Key Points:'):
                        options = section.replace('Key Points:', '').strip()
                    elif section.startswith('Detailed Solution:'):
                        solution = section.replace('Detailed Solution:', '').strip()

                if not all([issue, options, solution]):
                    issue = "Tech Support Response"
                    options = "Here are the steps to help you:"
                    solution = response.text.strip()
                
                # Store the new response
                new_embedding = get_embedding(query_data.query + " " + issue + " " + solution)
                cursor.execute("""
                    INSERT INTO ai_responses (query, issue, solution, embedding, effectiveness_score)
                    VALUES (%s, %s, %s, %s, %s)
                """, (query_data.query, issue, solution, json.dumps(new_embedding), 0.5))
                
                # Get the last inserted ID
                cursor.execute("SELECT LAST_INSERT_ID()")
                new_id = cursor.fetchone()[0]
                db.commit()
                
                return {
                    "issue": issue,
                    "options": options,
                    "solution": solution,
                    "source": "gemini",
                    "context_used": bool(context),
                    "num_contexts": len(context) if context else 0,
                    "type": "initial_response",
                    "response_id": new_id
                }

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

@app.post("/feedback")
async def record_feedback(feedback: FeedbackData):
    try:
        with get_db_cursor() as (cursor, db):
            # First verify the response exists
            cursor.execute("SELECT id FROM ai_responses WHERE id = %s", (feedback.response_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Response not found")
            
            # Update the effectiveness score based on feedback
            effectiveness_delta = 0.1 if feedback.was_helpful else -0.1
            
            cursor.execute("""
                UPDATE ai_responses 
                SET effectiveness_score = LEAST(GREATEST(effectiveness_score + %s, 0.1), 5.0)
                WHERE id = %s
            """, (effectiveness_delta, feedback.response_id))
            
            # Store the feedback if provided
            cursor.execute("""
                INSERT INTO feedback (response_id, feedback_text, was_helpful)
                VALUES (%s, %s, %s)
            """, (feedback.response_id, feedback.feedback_text, feedback.was_helpful))
            
            db.commit()
            
        return {"status": "success", "message": "Feedback recorded successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error recording feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    print("Starting up FastAPI application...")

@app.on_event("shutdown")
async def shutdown():
    print("Shutting down FastAPI application...")
