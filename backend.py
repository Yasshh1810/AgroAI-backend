# ══════════════════════════════════════════════
#  AgroAI — backend.py (Render Compatible)
# ══════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import hashlib
import os
import io
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB Setup with fallback ──
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
        # Create indexes
        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)
        print("✅ MongoDB connected")
    else:
        print("⚠️ MONGODB_URI not set, using in-memory storage")
except Exception as e:
    print(f"⚠️ MongoDB error: {e}, using in-memory storage")

# ── Load YOLO model ──
MODEL = None
try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    
    MODEL_PATH = "best.pt"
    if os.path.exists(MODEL_PATH):
        MODEL = YOLO(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    else:
        print("⚠️ best.pt not found — running in DEMO mode")
except ImportError as e:
    print(f"⚠️ Model error: {e} — running in DEMO mode")

app = FastAPI(title="AgroAI API")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ─── Database helper functions ───
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

# ─── Disease mapping ───
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

# ─── API Routes ───
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
    import random
    
    if MODEL is None:
        # Demo mode
        options = list(SEVERITY_MAP.values())
        weights = [0.3 if opt[0] == "Healthy" else 0.7/9 for opt in options]
        label, sev = random.choices(options, weights=weights)[0]
        conf = round(random.uniform(0.75, 0.98), 4)
        return {"disease": label, "severity": sev, "confidence": conf, "mode": "demo"}

    try:
        from PIL import Image
        import numpy as np
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        arr = np.array(img)
        results = MODEL.predict(arr, conf=0.1, verbose=False)
        r = results[0]

        if r.probs is not None:
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
