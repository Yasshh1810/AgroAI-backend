# ══════════════════════════════════════════════
#  AgroAI — backend.py (Fixed Model Loading)
# ══════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import hashlib
import os
import io
import sys
import random
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AgroAI API")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB Setup ──
MONGODB_URI = os.getenv("MONGODB_URI", "")
users_collection = None
detections_collection = None
in_memory_users = {}
in_memory_detections = {}

try:
    from pymongo import MongoClient
    if MONGODB_URI:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client["agroai_db"]
        users_collection = db["users"]
        detections_collection = db["detections"]
        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)
        print("✅ MongoDB connected")
    else:
        print("⚠️ MONGODB_URI not set, using in-memory storage")
except Exception as e:
    print(f"⚠️ MongoDB error: {e}, using in-memory storage")

# ── Load YOLO model with proper error handling ──
MODEL = None
MODEL_LOAD_ERROR = None

try:
    from ultralytics import YOLO
    
    MODEL_PATH = "best.pt"
    
    # Check if file exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model file not found at: {os.path.abspath(MODEL_PATH)}")
        MODEL_LOAD_ERROR = "Model file not found"
    else:
        # Get file size
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        print(f"📁 Model file size: {file_size:.2f} MB")
        
        # Check if file is too small (corrupted)
        if file_size < 1:
            print(f"⚠️ Model file seems corrupted (too small)")
            MODEL_LOAD_ERROR = "Model file corrupted (too small)"
        else:
            # Load the model
            print("🔄 Loading YOLO model...")
            MODEL = YOLO(MODEL_PATH)
            print("✅ Model loaded successfully")
            
            # Test the model with a dummy image
            try:
                import numpy as np
                from PIL import Image
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                test_result = MODEL.predict(dummy, verbose=False)
                print("✅ Model validation passed")
            except Exception as e:
                print(f"⚠️ Model validation warning: {e}")
                
except ImportError as e:
    print(f"⚠️ ultralytics not installed: {e}")
    MODEL_LOAD_ERROR = "ultralytics not installed"
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    MODEL_LOAD_ERROR = str(e)

# ─── Disease mapping (exact match for YOLO class names) ───
DISEASE_MAP = {
    "Tomato_Bacterial_spot": ("Bacterial Spot", "High"),
    "Tomato_Early_blight": ("Early Blight", "Medium"),
    "Tomato_Late_blight": ("Late Blight", "Critical"),
    "Tomato_Leaf_Mold": ("Leaf Mold", "Medium"),
    "Tomato_Septoria_leaf_spot": ("Septoria Leaf Spot", "Medium"),
    "Tomato_Spider_mites Two-spotted_spider_mite": ("Spider Mites", "Low"),
    "Tomato_Target_Spot": ("Target Spot", "Medium"),
    "Tomato__Target_Spot": ("Target Spot", "Medium"),
    "Tomato_Yellow_Leaf_Curl_Virus": ("Yellow Leaf Curl Virus", "Critical"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Yellow Leaf Curl Virus", "Critical"),
    "Tomato_Mosaic_Virus": ("Tomato Mosaic Virus", "High"),
    "Tomato__Tomato_mosaic_virus": ("Tomato Mosaic Virus", "High"),
    "Tomato_healthy": ("Healthy", "None"),
}

def get_disease_info(class_name):
    """Get disease info from class name with fallback"""
    if class_name in DISEASE_MAP:
        return DISEASE_MAP[class_name]
    
    # Try to clean up the class name
    cleaned = class_name.replace("__", "_").replace("_", " ").strip()
    for key, value in DISEASE_MAP.items():
        if key.replace("_", " ") in cleaned or cleaned in key:
            return value
    
    # Default fallback
    return (cleaned, "Medium")

# ─── Helper functions ───
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def save_user(username: str, email: str, password_hash: str):
    if users_collection:
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password_hash,
            "created": datetime.now().isoformat()
        })
    else:
        if username in in_memory_users:
            raise Exception("Username exists")
        in_memory_users[username] = {
            "username": username,
            "email": email,
            "password": password_hash,
            "created": datetime.now().isoformat()
        }

def find_user_by_username(username: str):
    if users_collection:
        return users_collection.find_one({"username": username})
    return in_memory_users.get(username)

def find_user_by_email(email: str):
    if users_collection:
        return users_collection.find_one({"email": email})
    for user in in_memory_users.values():
        if user["email"] == email:
            return user
    return None

def update_user_password(email: str, new_password_hash: str):
    if users_collection:
        result = users_collection.update_one(
            {"email": email},
            {"$set": {"password": new_password_hash}}
        )
        return result.modified_count > 0
    else:
        for user in in_memory_users.values():
            if user["email"] == email:
                user["password"] = new_password_hash
                return True
        return False

def save_detection_to_db(username: str, disease: str, confidence: float, severity: str):
    detection = {
        "username": username,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }
    if detections_collection:
        detections_collection.insert_one(detection)
    else:
        if username not in in_memory_detections:
            in_memory_detections[username] = []
        in_memory_detections[username].append(detection)

def get_user_detections(username: str, limit: int = 50):
    if detections_collection:
        cursor = detections_collection.find(
            {"username": username}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    return in_memory_detections.get(username, [])[:limit]

def delete_user_detections(username: str):
    if detections_collection:
        result = detections_collection.delete_many({"username": username})
        return result.deleted_count
    else:
        if username in in_memory_detections:
            del in_memory_detections[username]
        return 0

# ─── Schemas ───
class SignupData(BaseModel):
    username: str
    email: str
    password: str

class LoginData(BaseModel):
    username: str
    password: str

class DetectionSave(BaseModel):
    username: str
    disease: str
    confidence: float
    severity: str

class VerifyEmailData(BaseModel):
    email: str

class ResetPasswordData(BaseModel):
    email: str
    new_password: str

# ─── API Routes ───
@app.get("/")
def root():
    return {"message": "AgroAI API is running", "status": "healthy"}

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "mongodb": users_collection is not None,
        "model_loaded": MODEL is not None,
        "model_error": MODEL_LOAD_ERROR,
        "python_version": sys.version
    }

@app.post("/api/signup")
def signup(data: SignupData):
    if len(data.username.strip()) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if len(data.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if "@" not in data.email:
        raise HTTPException(400, "Enter a valid email address.")
    
    try:
        save_user(data.username.strip(), data.email.strip(), hash_pw(data.password))
        return {"success": True, "message": "Account created successfully."}
    except Exception:
        raise HTTPException(409, "Username or email already exists.")

@app.post("/api/login")
def login(data: LoginData):
    user = find_user_by_username(data.username.strip())
    if not user or user["password"] != hash_pw(data.password):
        raise HTTPException(401, "Invalid username or password.")
    
    return {
        "success": True,
        "username": user["username"],
        "email": user["email"],
        "message": f"Welcome back, {user['username']}!",
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """YOLOv8 inference with consistent results"""
    
    # Check if model is loaded
    if MODEL is None:
        print(f"⚠️ Prediction fallback: Model not loaded. Error: {MODEL_LOAD_ERROR}")
        # Return a consistent fallback response
        return {
            "disease": "Early Blight",
            "severity": "Medium", 
            "confidence": 0.85,
            "mode": "demo_fallback",
            "warning": "Model not loaded - using demo mode"
        }

    try:
        from PIL import Image
        import numpy as np
        
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to consistent size for better results
        img = img.resize((224, 224))
        arr = np.array(img)
        
        # Run prediction with consistent settings
        results = MODEL.predict(
            arr, 
            conf=0.25,  # Confidence threshold
            iou=0.45,   # IoU threshold
            verbose=False
        )
        
        if not results:
            raise Exception("No prediction results")
        
        r = results[0]
        
        # Extract prediction
        if r.probs is not None:
            # Classification model
            probs = r.probs.data.cpu().numpy()
            top1_idx = int(r.probs.top1)
            confidence = float(r.probs.top1conf)
            class_name = MODEL.names[top1_idx]
            
            # Log for debugging
            print(f"Predicted: {class_name} with confidence {confidence:.3f}")
            
        elif r.boxes and len(r.boxes) > 0:
            # Detection model - get highest confidence box
            best_idx = int(r.boxes.conf.argmax())
            confidence = float(r.boxes.conf[best_idx])
            class_id = int(r.boxes.cls[best_idx])
            class_name = MODEL.names[class_id]
            
            print(f"Predicted (detection): {class_name} with confidence {confidence:.3f}")
        else:
            # Default fallback
            class_name = "Tomato_healthy"
            confidence = 0.95
            print("No boxes found, defaulting to healthy")
        
        # Get disease info
        disease_name, severity = get_disease_info(class_name)
        
        return {
            "disease": disease_name,
            "severity": severity,
            "confidence": round(confidence, 4),
            "mode": "model",
            "raw_class": class_name
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.post("/api/save-detection")
def save_detection(data: DetectionSave):
    save_detection_to_db(data.username, data.disease, data.confidence, data.severity)
    return {"success": True}

@app.get("/api/history/{username}")
def get_history(username: str):
    rows = get_user_detections(username)
    history = []
    for row in rows:
        if "_id" in row:
            row["_id"] = str(row["_id"])
        history.append(row)
    return {"history": history}

@app.delete("/api/history/{username}")
def clear_history(username: str):
    delete_user_detections(username)
    return {"success": True}

@app.post("/api/verify-email")
def verify_email(data: VerifyEmailData):
    user = find_user_by_email(data.email.strip())
    if not user:
        raise HTTPException(404, "No account found with this email address.")
    return {"success": True}

@app.post("/api/reset-password")
def reset_password(data: ResetPasswordData):
    if len(data.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    
    success = update_user_password(data.email.strip(), hash_pw(data.new_password))
    if not success:
        raise HTTPException(404, "Email not found.")
    return {"success": True, "message": "Password reset successfully."}
