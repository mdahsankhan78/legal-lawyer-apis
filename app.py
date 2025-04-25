import os
import re
import jwt
import io
import bcrypt
import secrets
import certifi
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from bson import ObjectId
import fitz  # PyMuPDF
from fastapi import Response
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jinja2 import Template
from pymongo import MongoClient, TEXT
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer
from groq import Groq
from docx import Document
from PIL import Image
import pytesseract

# Set the Tesseract command path (adjust according to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ======================
# Configuration
# ======================
class Config:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://mahsankh77:fkp51svj@cluster1st.o6bsc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1st")
    # MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME = "legal_ai_pro"
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")
    GROQ_API_KEY = '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'
    SESSION_TIMEOUT = 2  # hours
    MAX_FILE_SIZE = 5  # MB
    
    SUPPORTED_MEDIA = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "png": "image/png",
        "jpg": "image/jpeg"
    }
    
    LEGAL_TEMPLATES = {
        "FIR": Template("""FIRST INFORMATION REPORT
Police Station: {{station}}
Date: {{date}}
Section: {{section}}
Complainant: {{complainant}}
Details: {{details}}""")
    }
    
    MODELS = {
        "RAG_MODEL": "llama3-8b-8192",
        "QA_MODEL": "nlpaueb/legal-bert-base-uncased",
        "NER_MODEL": "nlpaueb/legal-bert-base-uncased"
    }
    
    PROMPTS = {
        "LEGAL_ANSWER": """As a Pakistani legal expert, provide:
1. Relevant laws
2. Required procedures
3. Recommended actions
4. Key considerations

Question: {question}
Context: {context}"""
    }

# ======================
# Database Service
# ======================
class DatabaseService:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.client = MongoClient(Config.MONGODB_URI)
            cls._instance.db = cls._instance.client[Config.DB_NAME]
            cls._instance._init_indexes()
        return cls._instance
    
    def _init_indexes(self):
        self.db.users.create_index("email", unique=True)
        self.db.laws.create_index([("text", TEXT)], weights={"text": 2})
        self.db.chat_history.create_index("user_id")

# ======================
# AI Service
# ======================
class LegalAIService:
    def __init__(self):
        self.groq = Groq(api_key=Config.GROQ_API_KEY)
        self.db = DatabaseService().db
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=Config.MODELS["QA_MODEL"],
            tokenizer=AutoTokenizer.from_pretrained(Config.MODELS["QA_MODEL"])
        )
        self.ner_pipeline = pipeline("ner", model=Config.MODELS["NER_MODEL"])

    def rag_query(self, question: str, context: str = "") -> dict:
        legal_context = self._get_legal_context(question)
        full_context = f"{context}\n{legal_context}"
        
        # Truncate the context to a certain number of tokens
        max_tokens = 4000  # Adjust this based on your needs
        truncated_context = self._truncate_text(full_context, max_tokens)
        
        response = self.groq.chat.completions.create(
            messages=[{
                "role": "user",
                "content": Config.PROMPTS["LEGAL_ANSWER"].format(
                    question=question,
                    context=truncated_context
                )
            }],
            model=Config.MODELS["RAG_MODEL"],
            temperature=0.3
        )
        return {
            "response": response.choices[0].message.content,
            "sources": self._extract_sources(response.choices[0].message.content)
        }

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODELS["QA_MODEL"])
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.convert_tokens_to_string(tokens)

    def _get_legal_context(self, question: str) -> str:
        terms = self._extract_legal_terms(question)
        results = self.db.laws.find(
            {"$text": {"$search": " ".join(terms)}},
            {"text": 1, "source": 1}
        ).limit(3)
        return "\n".join(f"[{doc['source']}] {doc['text'][:256]}" for doc in results)

    def _extract_legal_terms(self, text: str) -> List[str]:
        return [e['word'] for e in self.ner_pipeline(text[:384]) 
                if e['entity'] in ['B-LAW', 'I-LAW']]

    def _extract_sources(self, text: str) -> List[str]:
        return list(set(re.findall(r'\[(.*?)\]', text)))

# ======================
# Document Processing
# ======================
class DocumentProcessor:
    @staticmethod
    def process_file(file: UploadFile) -> str:
        content = file.file.read()
        if file.filename.endswith(".pdf"):
            with fitz.open(stream=content, filetype="pdf") as doc:
                return " ".join(page.get_text() for page in doc)
        elif file.filename.endswith(".docx"):
            return "\n".join([p.text for p in Document(io.BytesIO(content)).paragraphs])
        elif file.filename.lower().endswith(("png", "jpg")):
            return pytesseract.image_to_string(Image.open(io.BytesIO(content)))
        raise HTTPException(400, "Unsupported file type")

# ======================
# Authentication
# ======================
class AuthService:
    def __init__(self):
        self.db = DatabaseService().db
        self.secret_key = "alpha_beta_gamma"

    def register(self, email: str, password: str, name: str, role: str = "User"):
        if self.db.users.find_one({"email": email}):
            raise HTTPException(400, "Email already registered")
        
        self.db.users.insert_one({
            "email": email,
            "password": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
            "name": name,
            "role": role, 
            "created_at": datetime.utcnow()
        })

    def authenticate(self, email: str, password: str) -> str:
        user = self.db.users.find_one({"email": email})
        if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
            raise HTTPException(401, "Invalid credentials")
             # JWT payload with user information
        payload = {
            "id": str(user["_id"]),  # Subject of the token (usually the user ID or email)
            "sub": user["email"],  # Subject of the token (usually the user ID or email)
            "name": user["name"],   # You can include other data if you want
            "iat": datetime.utcnow(),  # Issued at
            "exp": datetime.utcnow() + timedelta(hours=24)  # Expiry time (1 hour in this case)
        }

        # Encode the JWT
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token

# ======================
# FastAPI Setup
# ======================
app = FastAPI(title="Legal AI API")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ======================
# Pydantic Models
# ======================
class User(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    role: Optional[str] = Field(default="User", description="User role (User, Lawyer, Admin)")

class UpdateUserRole(BaseModel):
    role: str = Field(..., description="New role for the user (User, Lawyer, Admin)")

class Query(BaseModel):
    question: str
    context: Optional[str] = None

class FIRRequest(BaseModel):
    station: str
    date: str
    section: str 
    complainant: str
    details: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    role: Optional[str] = None
    created_at: datetime

class ChatQuery(BaseModel):
    question: Optional[str] = Field(None, description="The question asked by the user")
    document: Optional[Dict] = Field(None, description="The document uploaded by the user")
    answer: str = Field(..., description="The answer generated by the AI")

class ChatHistory(BaseModel):
    query: List[ChatQuery] = Field(..., description="List of chat queries and answers")
    chatName: Optional[str] = Field(None, description="Name of the chat (first question)")

class LawSearchQuery(BaseModel):
    query: str = Field(..., description="Search query for legal provisions")
    limit: Optional[int] = Field(5, description="Maximum number of results to return", ge=1, le=20)
# ======================
# Endpoints
# ======================
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Vercel"}

@app.post("/auth/register")
async def register(user: User):
    try:
        AuthService().register(user.email, user.password, user.name)
        return {"message": "Registration successful"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        # Authenticate the user and get the token
        token = AuthService().authenticate(form_data.username, form_data.password)
        
        # Fetch the full user object from the database
        user = DatabaseService().db.users.find_one({"email": form_data.username})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert ObjectId to string for the response
        user["_id"] = str(user["_id"])
        
        # Exclude sensitive fields like password
        user.pop("password", None)
        
        # Return the token and user details
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))
    
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, "alpha_beta_gamma", algorithms=["HS256"])
        user_id = payload.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = DatabaseService().db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/users", response_model=List[UserResponse])
async def get_all_users(
    current_user: dict = Depends(get_current_user)
):
    try:
        # Only allow admins to fetch all users
        if current_user.get("role") != "Admin":
            raise HTTPException(status_code=403, detail="Only admins can fetch all users")

        # Fetch all users from the database
        users = list(DatabaseService().db.users.find())

        # Convert ObjectId to string and exclude sensitive fields (e.g., password)
        user_list = []
        for user in users:
            user_data = {
                "id": str(user["_id"]),  # Rename "_id" to "id"
                "email": user["email"],
                "name": user.get("name"),  # Use .get() to handle missing fields safely
                "role": user.get("role", "User"),  # Default to "User" if role is missing
                "created_at": user["created_at"]
            }
            user_list.append(user_data)

        return user_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# to get user by id
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    user: dict = Depends(get_current_user)
):
    try:
        # Convert the user_id string to an ObjectId
        user_object_id = ObjectId(user_id)
        
        # Query the database for the user
        user_data = DatabaseService().db.users.find_one({"_id": user_object_id})
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Return the user data, excluding sensitive information like password
        return {
            "id": str(user_object_id),
            "email": user_data["email"],
            "name": user_data.get("name"),
            "role": user_data.get("role", "User"), 
            "created_at": user_data["created_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/users/update-role/{user_id}")
async def update_user_role(
    user_id: str,
    role_data: UpdateUserRole,
    current_user: dict = Depends(get_current_user)
):
    # Only allow admins to update roles
    if current_user.get("role") != "Admin":
        raise HTTPException(status_code=403, detail="Only admins can update roles")

    # Validate the new role
    if role_data.role not in ["User", "Lawyer", "Admin"]:
        raise HTTPException(status_code=400, detail="Invalid role. Allowed roles: User, Lawyer, Admin")

    # Update the user's role
    result = DatabaseService().db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"role": role_data.role}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": f"User role updated to {role_data.role}"}

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    user: dict = Depends(get_current_user)
):
    try:
        text = DocumentProcessor.process_file(file)
        analysis = LegalAIService().rag_query("Analyze this document", text)
        return analysis
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/query")
async def legal_query(
    query: Query,
    user: dict = Depends(get_current_user)
):
    try:
        return LegalAIService().rag_query(query.question, query.context)
    except Exception as e:
        raise HTTPException(500, str(e))

from fastapi.responses import FileResponse

@app.post("/generate/fir")
async def generate_fir(
    data: FIRRequest,
    user: dict = Depends(get_current_user)
):
    try:
        # Render the FIR template with the provided data
        doc = Config.LEGAL_TEMPLATES["FIR"].render(**data.dict())

        # Create a file-like object in memory
        fir_file = io.BytesIO(doc.encode('utf-8'))

        # We need to write the in-memory BytesIO content to a file on disk to use with FileResponse
        # Set the file path for saving the file temporarily
        file_path = f"static/FIR_{data.station}_{data.date}.txt"

        with open(file_path, "wb") as f:
            f.write(fir_file.getvalue())

        # Return the file response
        return FileResponse(file_path, media_type="application/octet-stream", filename=f"FIR_{data.station}_{data.date}.txt")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/upload/laws")
async def upload_laws(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)):
    try:
        text = DocumentProcessor.process_file(file)
        chunks = [text[i:i+384] for i in range(0, len(text), 384)]
        
        for chunk in chunks:
            DatabaseService().db.laws.insert_one({
                "text": chunk,
                "source": file.filename,
                "timestamp": datetime.utcnow()
            })
        
        return {"message": f"Added {len(chunks)} law chunks"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.put("/laws/update/{law_id}")
async def update_law(
    law_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Fetch the existing law
        existing_law = DatabaseService().db.laws.find_one({"_id": ObjectId(law_id)})
        if not existing_law:
            raise HTTPException(status_code=404, detail="Law not found")

        # Prepare the update data
        update_data = {}
        text = DocumentProcessor.process_file(file)
            
        # If no update data is provided, return an error
        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")

        # Update the law in the database
        result = DatabaseService().db.laws.update_one(
            {"_id": ObjectId(law_id)},
            {"$set": update_data}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Law not found")

        return {"message": "Law updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/laws", response_model=List[Dict])
async def get_all_laws(
    current_user: dict = Depends(get_current_user)
):
    try:
        # Fetch all laws from the database
        laws = list(DatabaseService().db.laws.find())

        # Convert ObjectId to string for each law
        for law in laws:
            law["_id"] = str(law["_id"])

        return laws
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/laws/{law_id}", response_model=Dict)
async def get_law_by_id(
    law_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Fetch the law by ID
        law = DatabaseService().db.laws.find_one({"_id": ObjectId(law_id)})

        if not law:
            raise HTTPException(status_code=404, detail="Law not found")

        # Convert ObjectId to string
        law["_id"] = str(law["_id"])

        return law
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/laws/delete/{law_id}")
async def delete_law(
    law_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Delete the law from the database
        result = DatabaseService().db.laws.delete_one({"_id": ObjectId(law_id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Law not found")

        return {"message": "Law deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/laws/search", response_model=List[Dict])
async def search_laws(
    search: LawSearchQuery,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Perform text search in the laws collection
        results = DatabaseService().db.laws.find(
            {"$text": {"$search": search.query}},
            {"text": 1, "source": 1, "timestamp": 1, "score": {"$meta": "textScore"}}
        ).sort("score", -1).limit(search.limit)

        # Convert results to list of dictionaries and handle ObjectId
        law_results = []
        for doc in results:
            law_data = {
                "id": str(doc["_id"]),
                "text": doc["text"],
                "source": doc.get("source", "Unknown"),
                "timestamp": doc.get("timestamp"),
                "score": doc.get("score", 0)
            }
            law_results.append(law_data)

        return law_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
# Add this endpoint in your backend file
@app.get("/laws/download/{law_id}")
async def download_law(
    law_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        law = DatabaseService().db.laws.find_one({"_id": ObjectId(law_id)})
        if not law:
            raise HTTPException(status_code=404, detail="Law not found")
        
        law_content = law["text"]
        law_file = io.BytesIO(law_content.encode('utf-8'))
        
        filename = f"law_{law.get('source', law_id)}.txt"
        
        return Response(
            content=law_file.getvalue(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
           
# Chat History Endpoints
# ======================
@app.post("/chat-history/add")
async def add_chat_history(
    history: ChatHistory,
    user: dict = Depends(get_current_user)
):
    try:
        history_data = history.dict()
        history_data["user_id"] = user["_id"]  # Add user_id from the authenticated user
        history_data["timestamp"] = datetime.utcnow()  # Add current timestamp
        # Set chatName to the first question or document filename if it exists
        if history_data["query"] and len(history_data["query"]) > 0:
            first_query = history_data["query"][0]
            if first_query.get("question"):
                history_data["chatName"] = first_query["question"]
            elif first_query.get("document"):
                history_data["chatName"] = first_query["document"].get("filename", "Unnamed Document")
        
        # Insert the chat history into the database
        result = DatabaseService().db.chat_history.insert_one(history_data)
        
        # Fetch the inserted document using the inserted_id
        inserted_document = DatabaseService().db.chat_history.find_one({"_id": result.inserted_id})
        
        # Convert all ObjectId fields to strings
        if inserted_document:
            inserted_document["_id"] = str(inserted_document["_id"])
            if "user_id" in inserted_document and isinstance(inserted_document["user_id"], ObjectId):
                inserted_document["user_id"] = str(inserted_document["user_id"])
        
        return {
            "message": "Chat history added successfully",
            "chat_history": inserted_document  # Return the inserted document
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.put("/chat-history/update/{history_id}")
async def update_chat_history(
    history_id: str,
    history: ChatHistory,
    user: dict = Depends(get_current_user)
):
    try:
        history_data = history.dict()

        # Preserve chatName if it already exists
        existing_history = DatabaseService().db.chat_history.find_one(
            {"_id": ObjectId(history_id), "user_id": user["_id"]}
        )
        if existing_history and "chatName" in existing_history:
            history_data["chatName"] = existing_history["chatName"]

        DatabaseService().db.chat_history.update_one(
            {"_id": ObjectId(history_id), "user_id": user["_id"]},
            {"$set": history_data}
        )
        return {"message": "Chat history updated successfully"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/chat-history/get")
async def get_chat_history(
    user: dict = Depends(get_current_user)
):
    try:
        # Fetch chat history for the current user
        histories = list(DatabaseService().db.chat_history.find({"user_id": user["_id"]}).sort("timestamp", -1))

        # Convert ObjectId to string for each history item
        for history in histories:
            history["_id"] = str(history["_id"])  # Convert ObjectId to string
            history["user_id"] = str(history["user_id"])  # Convert ObjectId to string (if user_id is ObjectId)

        return {"histories": histories}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/chat-history/get/{history_id}", response_model=ChatHistory)
async def get_chat_history_by_id(
    history_id: str,
    user: dict = Depends(get_current_user)
):
    try:
        # Fetch the chat history by ID and user_id
        history = DatabaseService().db.chat_history.find_one(
            {"_id": ObjectId(history_id), "user_id": user["_id"]}
        )

        if not history:
            raise HTTPException(status_code=404, detail="Chat history not found")

        # Convert ObjectId to string
        history["_id"] = str(history["_id"])
        history["user_id"] = str(history["user_id"])

        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-history/delete/{history_id}")
async def delete_chat_history(
    history_id: str,
    user: dict = Depends(get_current_user)
):
    try:
        DatabaseService().db.chat_history.delete_one({"_id": ObjectId(history_id), "user_id": user["_id"]})
        return {"message": "Chat history deleted successfully"}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)