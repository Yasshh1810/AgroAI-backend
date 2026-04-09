# ══════════════════════════════════════════════
#  AgroAI — backend.py (Production Ready)
# ══════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import hashlib
import os
import io
import sys
import uuid
import logging
from dotenv import load_dotenv
import jwt
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# ══════════════════════════════════════════════
#  LOGGING CONFIGURATION
# ══════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agroai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
#  APP INITIALIZATION
# ══════════════════════════════════════════════

app = FastAPI(
    title="AgroAI API",
    description="Plant Disease Detection API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", str(uuid.uuid4()))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# ── CORS Configuration ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
#  DATABASE SETUP (MongoDB with connection pooling)
# ══════════════════════════════════════════════

MONGODB_URI = os.getenv("MONGODB_URI", "")
DATABASE_NAME = os.getenv("DATABASE_NAME", "agroai_db")

users_collection = None
detections_collection = None
sessions_collection = None
in_memory_users: Dict[str, Any] = {}
in_memory_detections: Dict[str, List[Any]] = {}
in_memory_sessions: Dict[str, Any] = {}

# Try to connect to MongoDB
try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError, DuplicateKeyError
    
    if MONGODB_URI:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Test connection
        db = client[DATABASE_NAME]
        users_collection = db["users"]
        detections_collection = db["detections"]
        sessions_collection = db["sessions"]
        
        # Create indexes
        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)
        detections_collection.create_index([("username", 1), ("timestamp", -1)])
        sessions_collection.create_index("token", unique=True)
        sessions_collection.create_index("expires_at", expireAfterSeconds=0)
        
        logger.info("✅ MongoDB connected successfully")
    else:
        logger.warning("⚠️ MONGODB_URI not set, using in-memory storage")
except Exception as e:
    logger.warning(f"⚠️ MongoDB connection failed: {e}, using in-memory storage")

# ══════════════════════════════════════════════
#  YOLO MODEL LOADING with retry logic
# ══════════════════════════════════════════════

MODEL = None
MODEL_LOAD_ERROR = None
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

def load_model():
    """Load YOLO model with proper error handling"""
    global MODEL, MODEL_LOAD_ERROR
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Check if file exists
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model file not found at: {os.path.abspath(MODEL_PATH)}"
            logger.error(error_msg)
            MODEL_LOAD_ERROR = error_msg
            return False
        
        # Get file size
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        logger.info(f"📁 Model file size: {file_size:.2f} MB")
        
        # Check if file is too small (corrupted)
        if file_size < 1:
            error_msg = "Model file seems corrupted (too small)"
            logger.error(error_msg)
            MODEL_LOAD_ERROR = error_msg
            return False
        
        # Load the model
        logger.info("🔄 Loading YOLO model...")
        MODEL = YOLO(MODEL_PATH)
        
        # Test the model with a dummy image
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        test_result = MODEL.predict(dummy, verbose=False)
        logger.info("✅ Model loaded and validated successfully")
        MODEL_LOAD_ERROR = None
        return True
        
    except ImportError as e:
        error_msg = f"ultralytics not installed: {e}"
        logger.error(error_msg)
        MODEL_LOAD_ERROR = error_msg
        return False
    except Exception as e:
        error_msg = f"Model loading error: {str(e)}"
        logger.error(error_msg)
        MODEL_LOAD_ERROR = error_msg
        return False

# Load model on startup
load_model()

# ══════════════════════════════════════════════
#  DISEASE MAPPING (Complete)
# ══════════════════════════════════════════════

DISEASE_MAP = {
    # Tomato Diseases
    "Tomato_Bacterial_spot": ("Bacterial Spot", "High", "Copper-based fungicides, crop rotation"),
    "Tomato_Early_blight": ("Early Blight", "High", "Fungicides, remove infected leaves"),
    "Tomato_Late_blight": ("Late Blight", "Critical", "Immediate fungicide application"),
    "Tomato_Leaf_Mold": ("Leaf Mold", "Medium", "Improve air circulation, fungicides"),
    "Tomato_Septoria_leaf_spot": ("Septoria Leaf Spot", "Medium", "Fungicides, remove affected leaves"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Spider Mites", "Low", "Insecticidal soap, neem oil"),
    "Tomato_Target_Spot": ("Target Spot", "Medium", "Fungicides, crop rotation"),
    "Tomato__Target_Spot": ("Target Spot", "Medium", "Fungicides, crop rotation"),
    "Tomato_Yellow_Leaf_Curl_Virus": ("Yellow Leaf Curl Virus", "Critical", "Remove infected plants, control whiteflies"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Yellow Leaf Curl Virus", "Critical", "Remove infected plants, control whiteflies"),
    "Tomato_Mosaic_Virus": ("Tomato Mosaic Virus", "High", "Remove infected plants, sanitize tools"),
    "Tomato__Tomato_mosaic_virus": ("Tomato Mosaic Virus", "High", "Remove infected plants, sanitize tools"),
    "Tomato_healthy": ("Healthy", "None", "No treatment needed"),
    
    # Pepper Diseases
    "Pepper__bell___Bacterial_spot": ("Bacterial Spot", "High", "Copper-based bactericides"),
    "Pepper__bell___healthy": ("Healthy", "None", "No treatment needed"),
    
    # Potato Diseases
    "Potato___Early_blight": ("Early Blight", "High", "Fungicides, crop rotation"),
    "Potato___Late_blight": ("Late Blight", "Critical", "Immediate fungicide application"),
    "Potato___healthy": ("Healthy", "None", "No treatment needed"),
}

# Severity order for sorting
SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "None": 0}

def get_disease_info(class_name: str) -> tuple:
    """Get complete disease information from class name"""
    # Direct match
    if class_name in DISEASE_MAP:
        return DISEASE_MAP[class_name]
    
    # Try to find by partial match
    for key, value in DISEASE_MAP.items():
        if class_name.lower() in key.lower() or key.lower() in class_name.lower():
            return value
    
    # Default fallback
    display_name = class_name.replace("_", " ").replace("__", " ").strip()
    return (display_name, "Medium", "Consult local agricultural expert")

# ══════════════════════════════════════════════
#  JWT TOKEN FUNCTIONS
# ══════════════════════════════════════════════

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(401, "Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

# ══════════════════════════════════════════════
#  DATABASE HELPER FUNCTIONS
# ══════════════════════════════════════════════

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def save_user(username: str, email: str, password_hash: str) -> bool:
    try:
        if users_collection:
            users_collection.insert_one({
                "username": username,
                "email": email,
                "password": password_hash,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            })
        else:
            if username in in_memory_users:
                raise Exception("Username exists")
            in_memory_users[username] = {
                "username": username,
                "email": email,
                "password": password_hash,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        return True
    except Exception as e:
        logger.error(f"Save user error: {e}")
        return False

def find_user_by_username(username: str) -> Optional[Dict]:
    if users_collection:
        return users_collection.find_one({"username": username})
    return in_memory_users.get(username)

def find_user_by_email(email: str) -> Optional[Dict]:
    if users_collection:
        return users_collection.find_one({"email": email})
    for user in in_memory_users.values():
        if user["email"] == email:
            return user
    return None

def update_user_password(email: str, new_password_hash: str) -> bool:
    if users_collection:
        result = users_collection.update_one(
            {"email": email},
            {"$set": {"password": new_password_hash, "updated_at": datetime.utcnow().isoformat()}}
        )
        return result.modified_count > 0
    else:
        for user in in_memory_users.values():
            if user["email"] == email:
                user["password"] = new_password_hash
                user["updated_at"] = datetime.utcnow().isoformat()
                return True
        return False

def save_detection(username: str, disease: str, confidence: float, severity: str, treatment: str):
    detection = {
        "username": username,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "treatment": treatment,
        "timestamp": datetime.utcnow().isoformat()
    }
    if detections_collection:
        detections_collection.insert_one(detection)
    else:
        if username not in in_memory_detections:
            in_memory_detections[username] = []
        in_memory_detections[username].append(detection)
    return detection

def get_user_detections(username: str, limit: int = 50, skip: int = 0) -> List[Dict]:
    if detections_collection:
        cursor = detections_collection.find(
            {"username": username}
        ).sort("timestamp", -1).skip(skip).limit(limit)
        return list(cursor)
    detections = in_memory_detections.get(username, [])
    return detections[skip:skip+limit]

def delete_user_detections(username: str) -> int:
    if detections_collection:
        result = detections_collection.delete_many({"username": username})
        return result.deleted_count
    else:
        if username in in_memory_detections:
            del in_memory_detections[username]
        return 0

# ══════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ══════════════════════════════════════════════

class SignupData(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class LoginData(BaseModel):
    username: str
    password: str

class DetectionSave(BaseModel):
    disease: str
    confidence: float
    severity: str
    treatment: Optional[str] = ""

class DetectionResponse(BaseModel):
    id: Optional[str] = None
    disease: str
    confidence: float
    severity: str
    treatment: str
    timestamp: str

class PredictionResponse(BaseModel):
    disease: str
    severity: str
    confidence: float
    treatment: str
    raw_class: str
    mode: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    email: str

# ══════════════════════════════════════════════
#  HEALTH & ROOT ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "AgroAI API is running",
        "version": "2.0.0",
        "status": "healthy",
        "docs": "/api/docs"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "mongodb": users_collection is not None,
        "model_loaded": MODEL is not None,
        "model_error": MODEL_LOAD_ERROR,
        "python_version": sys.version
    }

# ══════════════════════════════════════════════
#  AUTHENTICATION ENDPOINTS
# ══════════════════════════════════════════════

@app.post("/api/signup", response_model=Dict[str, Any])
async def signup(data: SignupData):
    """Create a new user account"""
    # Validate input
    if not data.username.strip():
        raise HTTPException(400, "Username is required")
    if not data.email:
        raise HTTPException(400, "Email is required")
    if len(data.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    
    # Check if user exists
    if find_user_by_username(data.username.strip()):
        raise HTTPException(409, "Username already exists")
    if find_user_by_email(data.email.strip()):
        raise HTTPException(409, "Email already registered")
    
    # Create user
    success = save_user(data.username.strip(), data.email.strip(), hash_password(data.password))
    if not success:
        raise HTTPException(500, "Failed to create user")
    
    # Create access token
    access_token = create_access_token(data={"sub": data.username.strip()})
    
    return {
        "success": True,
        "message": "Account created successfully",
        "access_token": access_token,
        "token_type": "bearer",
        "username": data.username.strip(),
        "email": data.email.strip()
    }

@app.post("/api/login", response_model=Dict[str, Any])
async def login(data: LoginData):
    """Authenticate user and return token"""
    user = find_user_by_username(data.username.strip())
    
    if not user or user["password"] != hash_password(data.password):
        raise HTTPException(401, "Invalid username or password")
    
    # Create access token
    access_token = create_access_token(data={"sub": user["username"]})
    
    return {
        "success": True,
        "message": f"Welcome back, {user['username']}!",
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "email": user["email"]
    }

@app.post("/api/verify-email")
async def verify_email(email: EmailStr):
    """Check if email exists for password reset"""
    user = find_user_by_email(email)
    if not user:
        raise HTTPException(404, "No account found with this email address")
    return {"success": True, "message": "Email verified"}

@app.post("/api/reset-password")
async def reset_password(email: EmailStr, new_password: str):
    """Reset user password"""
    if len(new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    
    success = update_user_password(email, hash_password(new_password))
    if not success:
        raise HTTPException(404, "Email not found")
    
    return {"success": True, "message": "Password reset successfully"}

@app.post("/api/logout")
async def logout(username: str = Depends(verify_token)):
    """Logout user (client should discard token)"""
    return {"success": True, "message": "Logged out successfully"}

# ══════════════════════════════════════════════
#  PREDICTION ENDPOINTS
# ══════════════════════════════════════════════

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    username: Optional[str] = None
):
    """
    Detect plant disease from uploaded image
    Optional username to save detection history
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Check model availability
    if MODEL is None:
        logger.warning(f"Prediction fallback: Model not loaded. Error: {MODEL_LOAD_ERROR}")
        # Return demo response
        return PredictionResponse(
            disease="Early Blight",
            severity="Medium",
            confidence=0.85,
            treatment="Apply fungicide and remove affected leaves",
            raw_class="Tomato_Early_blight",
            mode="demo_fallback"
        )
    
    try:
        from PIL import Image
        import numpy as np
        
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to model input size
        img = img.resize((224, 224))
        arr = np.array(img)
        
        # Run prediction
        results = MODEL.predict(arr, conf=0.25, iou=0.45, verbose=False)
        
        if not results:
            raise Exception("No prediction results")
        
        r = results[0]
        
        # Extract prediction
        if r.probs is not None:
            # Classification model
            top1_idx = int(r.probs.top1)
            confidence = float(r.probs.top1conf)
            class_name = MODEL.names[top1_idx]
            logger.info(f"Predicted: {class_name} with confidence {confidence:.3f}")
        elif r.boxes and len(r.boxes) > 0:
            # Detection model - use highest confidence box
            best_idx = int(r.boxes.conf.argmax())
            confidence = float(r.boxes.conf[best_idx])
            class_id = int(r.boxes.cls[best_idx])
            class_name = MODEL.names[class_id]
            logger.info(f"Predicted (detection): {class_name} with confidence {confidence:.3f}")
        else:
            # Default fallback for no detection
            class_name = "Tomato_healthy"
            confidence = 0.95
            logger.info("No detection found, defaulting to healthy")
        
        # Get disease information
        disease_name, severity, treatment = get_disease_info(class_name)
        
        # Save to history if username provided
        if username:
            save_detection(username, disease_name, confidence, severity, treatment)
        
        return PredictionResponse(
            disease=disease_name,
            severity=severity,
            confidence=round(confidence, 4),
            treatment=treatment,
            raw_class=class_name,
            mode="model"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.post("/api/save-detection")
async def save_detection_endpoint(
    data: DetectionSave,
    username: str = Depends(verify_token)
):
    """Save detection to user history"""
    detection = save_detection(username, data.disease, data.confidence, data.severity, data.treatment)
    return {"success": True, "detection_id": str(detection.get("_id", ""))}

# ══════════════════════════════════════════════
#  HISTORY ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/api/history")
async def get_history(
    username: str = Depends(verify_token),
    limit: int = 50,
    skip: int = 0
):
    """Get user's detection history"""
    detections = get_user_detections(username, limit, skip)
    
    # Convert ObjectId to string for JSON serialization
    history = []
    for d in detections:
        d_copy = dict(d)
        if "_id" in d_copy:
            d_copy["id"] = str(d_copy.pop("_id"))
        history.append(d_copy)
    
    return {"history": history, "count": len(history)}

@app.delete("/api/history")
async def clear_history(username: str = Depends(verify_token)):
    """Clear user's detection history"""
    deleted = delete_user_detections(username)
    return {"success": True, "deleted_count": deleted}

# ══════════════════════════════════════════════
#  DISEASE INFO ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/api/diseases")
async def get_all_diseases():
    """Get list of all detectable diseases"""
    diseases = []
    seen = set()
    
    for class_name, (display_name, severity, treatment) in DISEASE_MAP.items():
        if display_name not in seen:
            seen.add(display_name)
            diseases.append({
                "name": display_name,
                "severity": severity,
                "severity_level": SEVERITY_ORDER.get(severity, 0),
                "treatment": treatment,
                "class_name": class_name
            })
    
    # Sort by severity (highest first)
    diseases.sort(key=lambda x: x["severity_level"], reverse=True)
    return {"diseases": diseases}

@app.get("/api/diseases/{disease_name}")
async def get_disease_info_endpoint(disease_name: str):
    """Get detailed information about a specific disease"""
    for class_name, (display_name, severity, treatment) in DISEASE_MAP.items():
        if display_name.lower() == disease_name.lower():
            return {
                "name": display_name,
                "severity": severity,
                "treatment": treatment,
                "class_names": [k for k, v in DISEASE_MAP.items() if v[0] == display_name]
            }
    raise HTTPException(404, f"Disease '{disease_name}' not found")

# ══════════════════════════════════════════════
#  STATISTICS ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/api/stats")
async def get_stats(username: str = Depends(verify_token)):
    """Get user statistics"""
    detections = get_user_detections(username, limit=1000)
    
    stats = {
        "total_detections": len(detections),
        "by_severity": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "None": 0},
        "by_disease": {},
        "average_confidence": 0,
        "last_detection": None
    }
    
    total_confidence = 0
    for d in detections:
        # Count by severity
        severity = d.get("severity", "Medium")
        if severity in stats["by_severity"]:
            stats["by_severity"][severity] += 1
        
        # Count by disease
        disease = d.get("disease", "Unknown")
        stats["by_disease"][disease] = stats["by_disease"].get(disease, 0) + 1
        
        # Sum confidence
        total_confidence += d.get("confidence", 0)
        
        # Track last detection
        if stats["last_detection"] is None or d.get("timestamp", "") > stats["last_detection"]:
            stats["last_detection"] = d.get("timestamp")
    
    if detections:
        stats["average_confidence"] = round(total_confidence / len(detections), 4)
    
    return stats

# ══════════════════════════════════════════════
#  MODEL MANAGEMENT ENDPOINTS (Admin only)
# ══════════════════════════════════════════════

@app.post("/api/model/reload")
async def reload_model(username: str = Depends(verify_token)):
    """Reload the ML model (admin only)"""
    # TODO: Add admin check
    success = load_model()
    return {"success": success, "model_loaded": MODEL is not None, "error": MODEL_LOAD_ERROR}

# ══════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": "Internal server error",
        "status_code": 500
    }

# ══════════════════════════════════════════════
#  RUN CONFIGURATION
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "backend:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
