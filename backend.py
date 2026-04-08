# ══════════════════════════════════════════════
# AgroAI — backend.py (MongoDB Version)
# ══════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib, os
from datetime import datetime
from pymongo import MongoClient

# ── MongoDB Connection ──
MONGO_URL = os.getenv("MONGO_URL")  # from Render env
client = MongoClient(MONGO_URL)
db = client["agroai_db"]

users_col = db["users"]
detect_col = db["detections"]

# ── YOLO Model ──
try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np, io
    MODEL = YOLO("best.pt") if os.path.exists("best.pt") else None
except:
    MODEL = None

app = FastAPI(title="AgroAI API")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Utils ──
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ══════════════════════════════════════════════
# SCHEMAS
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

# ══════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════

@app.post("/api/signup")
def signup(data: SignupData):
    if len(data.username) < 3:
        raise HTTPException(400, "Username too short")
    if len(data.password) < 6:
        raise HTTPException(400, "Password too short")

    if users_col.find_one({"username": data.username}):
        raise HTTPException(409, "Username already exists")

    if users_col.find_one({"email": data.email}):
        raise HTTPException(409, "Email already exists")

    users_col.insert_one({
        "username": data.username,
        "email": data.email,
        "password": hash_pw(data.password),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    return {"success": True}


@app.post("/api/login")
def login(data: LoginData):
    user = users_col.find_one({
        "username": data.username,
        "password": hash_pw(data.password)
    })

    if not user:
        raise HTTPException(401, "Invalid username or password")

    return {
        "success": True,
        "username": user["username"],
        "email": user["email"]
    }

# ══════════════════════════════════════════════
# DETECTION
# ══════════════════════════════════════════════

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        import random
        data = [
            ("Bacterial Spot","High"),
            ("Early Blight","Medium"),
            ("Late Blight","Critical"),
            ("Healthy","None")
        ]
        name, sev = random.choice(data)
        return {"disease": name, "severity": sev, "confidence": 0.9}

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    arr = np.array(img)

    results = MODEL.predict(arr, conf=0.25, verbose=False)
    r = results[0]

    if r.probs:
        idx = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        label = MODEL.names[idx]
    else:
        label, conf = "Healthy", 1.0

    return {
        "disease": label,
        "severity": "Medium",
        "confidence": round(conf, 4)
    }

# ══════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════

@app.post("/api/save-detection")
def save_detection(data: DetectionSave):
    detect_col.insert_one({
        "username": data.username,
        "disease": data.disease,
        "confidence": data.confidence,
        "severity": data.severity,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    return {"success": True}


@app.get("/api/history/{username}")
def get_history(username: str):
    rows = list(detect_col.find(
        {"username": username},
        {"_id": 0}
    ).sort("timestamp", -1).limit(50))

    return {"history": rows}


@app.delete("/api/history/{username}")
def clear_history(username: str):
    detect_col.delete_many({"username": username})
    return {"success": True}

# ══════════════════════════════════════════════
# FORGOT PASSWORD
# ══════════════════════════════════════════════

@app.post("/api/verify-email")
def verify_email(data: VerifyEmailData):
    if not users_col.find_one({"email": data.email}):
        raise HTTPException(404, "Email not found")
    return {"success": True}


@app.post("/api/reset-password")
def reset_password(data: ResetPasswordData):
    if len(data.new_password) < 6:
        raise HTTPException(400, "Password too short")

    res = users_col.update_one(
        {"email": data.email},
        {"$set": {"password": hash_pw(data.new_password)}}
    )

    if res.modified_count == 0:
        raise HTTPException(404, "Email not found")

    return {"success": True}

# ══════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════

@app.get("/")
def home():
    return {"status": "AgroAI MongoDB backend running"}
