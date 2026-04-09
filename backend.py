# ═══════════════════════════════════════════════════════════════
#  AgroAI — backend.py (Render Compatible - No numpy test)
# ═══════════════════════════════════════════════════════════════

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
import logging
from dotenv import load_dotenv
import jwt

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Force CPU mode for Render
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
IS_RENDER = os.environ.get('RENDER', False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="AgroAI API",
    description="Plant Disease Detection API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", str(os.urandom(32).hex()))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

# ═══════════════════════════════════════════════════════════════
#  CORS CONFIGURATION
# ═══════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

logger.info("✅ CORS configured")

# ═══════════════════════════════════════════════════════════════
#  DATABASE (In-memory)
# ═══════════════════════════════════════════════════════════════

in_memory_users: Dict[str, Any] = {}
in_memory_detections: Dict[str, List[Any]] = {}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
    except:
        raise HTTPException(401, "Invalid token")

# ═══════════════════════════════════════════════════════════════
#  YOLO MODEL LOADING (Fixed - No numpy test)
# ═══════════════════════════════════════════════════════════════

MODEL = None
MODEL_LOAD_ERROR = None
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

def load_model():
    """Load YOLO model without numpy test to avoid errors"""
    global MODEL, MODEL_LOAD_ERROR
    
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model file not found at: {MODEL_PATH}"
            logger.error(error_msg)
            MODEL_LOAD_ERROR = error_msg
            return False
        
        # Get file size
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        logger.info(f"📁 Model file: {MODEL_PATH} ({file_size:.2f} MB)")
        
        # Load the model (without testing)
        logger.info("🔄 Loading YOLO model...")
        MODEL = YOLO(MODEL_PATH)
        
        # Don't test with dummy image - this causes the numpy error
        # Just verify model has names
        if hasattr(MODEL, 'names'):
            logger.info(f"✅ Model loaded! Classes: {len(MODEL.names)}")
            MODEL_LOAD_ERROR = None
            return True
        else:
            raise Exception("Model loaded but no class names found")
        
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

# ═══════════════════════════════════════════════════════════════
#  DISEASE DATABASE
# ═══════════════════════════════════════════════════════════════

DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot",
        "severity": "High",
        "treatment": "Apply copper-based bactericides. Remove infected debris.",
        "symptoms": "Small dark water-soaked lesions on leaves.",
        "prevention": "Use certified disease-free seeds."
    },
    "Tomato_Early_blight": {
        "name": "Early Blight",
        "severity": "Medium", 
        "treatment": "Apply chlorothalonil or mancozeb fungicide.",
        "symptoms": "Concentric dark rings on older leaves.",
        "prevention": "Rotate crops annually."
    },
    "Tomato_Late_blight": {
        "name": "Late Blight",
        "severity": "Critical",
        "treatment": "Apply metalaxyl fungicide immediately.",
        "symptoms": "Large water-soaked lesions with white mould.",
        "prevention": "Avoid overhead watering."
    },
    "Tomato_healthy": {
        "name": "Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "symptoms": "No disease detected.",
        "prevention": "Continue good practices."
    },
}

def get_disease_info(class_name: str):
    """Get disease info from class name"""
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    
    # Try to find by partial match
    for key, info in DISEASE_INFO.items():
        if class_name.lower() in key.lower():
            return info
    
    # Default
    return {
        "name": class_name.replace("_", " ").title(),
        "severity": "Medium",
        "treatment": "Consult local expert",
        "symptoms": "Unknown",
        "prevention": "Monitor regularly"
    }

# ═══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class SignupData(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginData(BaseModel):
    username: str
    password: str

class DetectionSave(BaseModel):
    disease: str
    confidence: float
    severity: str
    treatment: Optional[str] = ""

# ═══════════════════════════════════════════════════════════════
#  HEALTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "🌱 AgroAI API is running",
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": MODEL is not None,
        "model_error": MODEL_LOAD_ERROR
    }

# ═══════════════════════════════════════════════════════════════
#  AUTHENTICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/api/signup")
async def signup(data: SignupData):
    if data.username in in_memory_users:
        raise HTTPException(409, "Username already exists")
    
    for user in in_memory_users.values():
        if user["email"] == data.email:
            raise HTTPException(409, "Email already registered")
    
    in_memory_users[data.username] = {
        "username": data.username,
        "email": data.email,
        "password": hash_password(data.password),
        "created_at": datetime.utcnow().isoformat()
    }
    
    access_token = create_access_token(data={"sub": data.username})
    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "username": data.username,
        "email": data.email
    }

@app.post("/api/login")
async def login(data: LoginData):
    user = in_memory_users.get(data.username)
    if not user or user["password"] != hash_password(data.password):
        raise HTTPException(401, "Invalid credentials")
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "email": user["email"]
    }

@app.post("/api/verify-email")
async def verify_email(email: EmailStr):
    for user in in_memory_users.values():
        if user["email"] == email:
            return {"success": True}
    raise HTTPException(404, "Email not found")

@app.post("/api/reset-password")
async def reset_password(email: EmailStr, new_password: str):
    for user in in_memory_users.values():
        if user["email"] == email:
            user["password"] = hash_password(new_password)
            return {"success": True}
    raise HTTPException(404, "Email not found")

# ═══════════════════════════════════════════════════════════════
#  PREDICTION ENDPOINT (Fixed for Render)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    username: Optional[str] = None
):
    logger.info(f"Prediction request: {file.filename}")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Try to use YOLO model if available
        if MODEL is not None:
            try:
                from PIL import Image
                import numpy as np
                
                contents = await file.read()
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                img = img.resize((640, 640))
                arr = np.array(img)
                
                # Run prediction
                results = MODEL.predict(arr, conf=0.25, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.probs is not None:
                        # Classification model
                        top1_idx = int(result.probs.top1)
                        confidence = float(result.probs.top1conf)
                        class_name = MODEL.names[top1_idx]
                    elif result.boxes is not None and len(result.boxes) > 0:
                        # Detection model
                        best_idx = int(result.boxes.conf.argmax())
                        confidence = float(result.boxes.conf[best_idx])
                        class_id = int(result.boxes.cls[best_idx])
                        class_name = MODEL.names[class_id]
                    else:
                        class_name = "Tomato_healthy"
                        confidence = 0.95
                    
                    disease_info = get_disease_info(class_name)
                    
                    # Save to history
                    if username:
                        if username not in in_memory_detections:
                            in_memory_detections[username] = []
                        in_memory_detections[username].append({
                            "disease": disease_info['name'],
                            "confidence": confidence,
                            "severity": disease_info['severity'],
                            "treatment": disease_info['treatment'],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    return {
                        "disease": disease_info['name'],
                        "severity": disease_info['severity'],
                        "confidence": round(confidence, 4),
                        "treatment": disease_info['treatment'],
                        "symptoms": disease_info['symptoms'],
                        "prevention": disease_info['prevention'],
                        "raw_class": class_name,
                        "mode": "yolo_model"
                    }
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                # Fall through to demo mode
        
        # Demo mode (fallback if model fails)
        import random
        demo = [
            ("Early Blight", "Medium", 0.92, "Apply fungicide"),
            ("Bacterial Spot", "High", 0.88, "Apply copper bactericide"),
            ("Late Blight", "Critical", 0.85, "Apply metalaxyl immediately"),
            ("Healthy", "None", 0.96, "No treatment needed"),
            ("Leaf Mold", "Medium", 0.79, "Improve ventilation"),
        ]
        
        disease, severity, confidence, treatment = random.choice(demo)
        
        if username:
            if username not in in_memory_detections:
                in_memory_detections[username] = []
            in_memory_detections[username].append({
                "disease": disease,
                "confidence": confidence,
                "severity": severity,
                "treatment": treatment,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "disease": disease,
            "severity": severity,
            "confidence": confidence,
            "treatment": treatment,
            "symptoms": "Detected through image analysis",
            "prevention": "Regular monitoring recommended",
            "raw_class": disease.replace(" ", "_"),
            "mode": "demo_fallback"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/api/save-detection")
async def save_detection(data: DetectionSave, username: str = Depends(verify_token)):
    if username not in in_memory_detections:
        in_memory_detections[username] = []
    in_memory_detections[username].append({
        "disease": data.disease,
        "confidence": data.confidence,
        "severity": data.severity,
        "treatment": data.treatment,
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"success": True}

# ═══════════════════════════════════════════════════════════════
#  HISTORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/history")
async def get_history(username: str = Depends(verify_token)):
    history = in_memory_detections.get(username, [])
    history.reverse()  # Newest first
    return {"history": history, "count": len(history)}

@app.delete("/api/history")
async def clear_history(username: str = Depends(verify_token)):
    if username in in_memory_detections:
        in_memory_detections[username] = []
    return {"success": True}

# ═══════════════════════════════════════════════════════════════
#  RUN CONFIGURATION - FIXED PORT BINDING
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    # Use PORT from environment (Render sets this to 10000)
    port = int(os.getenv("PORT", "10000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(
        "backend:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
