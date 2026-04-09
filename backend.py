# ══════════════════════════════════════════════
#  AgroAI — backend.py (MongoDB Version)
#  FastAPI + MongoDB Atlas | Login / Signup / Detect
#  Model: best.pt (YOLOv8n-cls — 10 tomato classes)
#  Deploy: Render.com
# ══════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import hashlib
import os
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── MongoDB Setup ──
try:
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(MONGODB_URI)
    db = client["agroai_db"]
    users_collection = db["users"]
    detections_collection = db["detections"]
    
    # Create indexes
    users_collection.create_index("username", unique=True)
    users_collection.create_index("email", unique=True)
    detections_collection.create_index([("username", 1), ("timestamp", -1)])
    
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"⚠️ MongoDB connection error: {e}")
    print("Running in fallback mode with in-memory storage")
    # Fallback to in-memory storage
    users_collection = None
    detections_collection = None
    in_memory_users = {}
    in_memory_detections = {}

# ── Load YOLO model ──
try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    
    # Download model if not exists (for Render deployment)
    import requests
    MODEL_PATH = "best.pt"
    
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading best.pt from GitHub release...")
        # Replace with your actual model download URL
        model_url = os.getenv("MODEL_URL", "")
        if model_url:
            response = requests.get(model_url, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded")
    
    MODEL = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if MODEL:
        print(f"✅ Model loaded: {MODEL_PATH}")
    else:
        print("⚠️ best.pt not found — running in DEMO mode")
except ImportError as e:
    MODEL = None
    print(f"⚠️ ultralytics not installed: {e} — running in DEMO mode")
except Exception as e:
    MODEL = None
    print(f"⚠️ Model loading error: {e} — running in DEMO mode")

app = FastAPI(title="AgroAI API", version="1.0.0")

# ── CORS (Allow all origins for frontend) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════
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

class DetectionHistory(BaseModel):
    disease: str
    confidence: float
    severity: str
    timestamp: str

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# Helper functions for database operations
def save_user(username: str, email: str, password_hash: str):
    if users_collection is not None:
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password_hash,
            "created": datetime.now().isoformat()
        })
    else:
        # Fallback to in-memory
        if username in in_memory_users:
            raise Exception("Username exists")
        in_memory_users[username] = {
            "username": username,
            "email": email,
            "password": password_hash,
            "created": datetime.now().isoformat()
        }

def find_user_by_username(username: str):
    if users_collection is not None:
        return users_collection.find_one({"username": username})
    else:
        return in_memory_users.get(username)

def find_user_by_email(email: str):
    if users_collection is not None:
        return users_collection.find_one({"email": email})
    else:
        for user in in_memory_users.values():
            if user["email"] == email:
                return user
        return None

def update_user_password(email: str, new_password_hash: str):
    if users_collection is not None:
        result = users_collection.update_one(
            {"email": email},
            {"$set": {"password": new_password_hash}}
        )
        return result.modified_count > 0
    else:
        for username, user in in_memory_users.items():
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
    if detections_collection is not None:
        detections_collection.insert_one(detection)
    else:
        if username not in in_memory_detections:
            in_memory_detections[username] = []
        in_memory_detections[username].append(detection)

def get_user_detections(username: str, limit: int = 50):
    if detections_collection is not None:
        cursor = detections_collection.find(
            {"username": username}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    else:
        return in_memory_detections.get(username, [])[:limit]

def delete_user_detections(username: str):
    if detections_collection is not None:
        result = detections_collection.delete_many({"username": username})
        return result.deleted_count
    else:
        if username in in_memory_detections:
            del in_memory_detections[username]
        return 0

# ══════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "AgroAI API is running", "status": "healthy"}

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "mongodb": users_collection is not None,
        "model": MODEL is not None
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
        save_user(
            data.username.strip(),
            data.email.strip(),
            hash_pw(data.password)
        )
        return {"success": True, "message": "Account created successfully."}
    except Exception as e:
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

# ══════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════

# Maps YOLOv8 class names → (display label, severity)
SEVERITY_MAP = {
    "Tomato_Bacterial_spot": ("Bacterial Spot", "High"),
    "Tomato_Early_blight": ("Early Blight", "Medium"),
    "Tomato_Late_blight": ("Late Blight", "Critical"),
    "Tomato_Leaf_Mold": ("Leaf Mold", "Medium"),
    "Tomato_Septoria_leaf_spot": ("Septoria Leaf Spot", "Medium"),
    "Tomato_Spider_mites Two-spotted_spider_mite": ("Spider Mites", "Low"),
    "Tomato__Target_Spot": ("Target Spot", "Medium"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Yellow Leaf Curl Virus", "Critical"),
    "Tomato__Tomato_mosaic_virus": ("Tomato Mosaic Virus", "High"),
    "Tomato_healthy": ("Healthy", "None"),
}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """YOLOv8 inference. Falls back to demo if best.pt not found."""
    
    if MODEL is None:
        # Demo fallback with weighted probabilities
        import random
        options = list(SEVERITY_MAP.values())
        # Give higher probability to non-healthy for demo
        weights = [0.3 if opt[0] == "Healthy" else 0.7/9 for opt in options]
        label, sev = random.choices(options, weights=weights)[0]
        conf = round(random.uniform(0.75, 0.98), 4)
        return {"disease": label, "severity": sev, "confidence": conf, "mode": "demo"}

    # Real inference
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        arr = np.array(img)
        results = MODEL.predict(arr, conf=0.1, verbose=False)
        r = results[0]

        if r.probs is not None:
            # Classification model
            idx = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            key = MODEL.names[idx]
        elif r.boxes and len(r.boxes):
            best = int(r.boxes.conf.argmax())
            idx = int(r.boxes.cls[best])
            conf = float(r.boxes.conf[best])
            key = MODEL.names[idx]
        else:
            key, conf = "Tomato_healthy", 1.0

        label, severity = SEVERITY_MAP.get(key, (key, "Medium"))
        return {
            "disease": label,
            "severity": severity,
            "confidence": round(conf, 4),
            "mode": "model"
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.post("/api/save-detection")
def save_detection(data: DetectionSave):
    save_detection_to_db(
        data.username,
        data.disease,
        data.confidence,
        data.severity
    )
    return {"success": True}

@app.get("/api/history/{username}")
def get_history(username: str):
    rows = get_user_detections(username)
    # Convert ObjectId to string for JSON serialization
    history = []
    for row in rows:
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

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
