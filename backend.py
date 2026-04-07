"""
AgroAI — BACKEND with YOLOv8 CLASSIFICATION
===========================================
Run: uvicorn backend:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles# ══════════════════════════════════════════════
#  AgroAI — backend.py
#  FastAPI + SQLite  |  Login / Signup / Detect
#  Run: uvicorn backend:app --reload
# ══════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3, hashlib, os
from datetime import datetime

# ── optional: load YOLO if best.pt exists ──
try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np, io
    MODEL = YOLO("best.pt") if os.path.exists("best.pt") else None
except ImportError:
    MODEL = None

app = FastAPI(title="AgroAI API")

# ── CORS (allow frontend to talk to backend) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════
DB = "agroai.db"

def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    UNIQUE NOT NULL,
            email     TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            created   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detections (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL,
            disease    TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity   TEXT NOT NULL,
            timestamp  TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ══════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════
class SignupData(BaseModel):
    username: str
    email:    str
    password: str

class LoginData(BaseModel):
    username: str
    password: str

class DetectionSave(BaseModel):
    username:   str
    disease:    str
    confidence: float
    severity:   str

# ══════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════

@app.post("/api/signup")
def signup(data: SignupData):
    if len(data.username.strip()) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if len(data.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if "@" not in data.email:
        raise HTTPException(400, "Enter a valid email address.")

    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password, created) VALUES (?,?,?,?)",
            (data.username.strip(), data.email.strip(), hash_pw(data.password),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        return {"success": True, "message": "Account created successfully."}
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Username or email already exists.")
    finally:
        conn.close()


@app.post("/api/login")
def login(data: LoginData):
    conn = get_conn()
    row = conn.execute(
        "SELECT username, email FROM users WHERE username=? AND password=?",
        (data.username.strip(), hash_pw(data.password))
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(401, "Invalid username or password.")

    return {
        "success":  True,
        "username": row["username"],
        "email":    row["email"],
        "message":  f"Welcome back, {row['username']}!",
    }

# ══════════════════════════════════════════════
#  DETECTION ROUTES
# ══════════════════════════════════════════════

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Real YOLOv8 inference.
    Falls back to demo result if best.pt is not present.
    """
    if MODEL is None:
        # Demo fallback — replace with real model later
        import random
        NAMES = [
            ("Bacterial Spot",        "High"),
            ("Early Blight",          "Medium"),
            ("Late Blight",           "Critical"),
            ("Leaf Mold",             "Medium"),
            ("Septoria Leaf Spot",    "Medium"),
            ("Spider Mites",          "Low"),
            ("Target Spot",           "Medium"),
            ("Yellow Leaf Curl Virus","Critical"),
            ("Tomato Mosaic Virus",   "High"),
            ("Healthy",               "None"),
        ]
        name, sev = random.choice(NAMES)
        conf = round(random.uniform(0.72, 0.99), 4)
        return {"disease": name, "severity": sev, "confidence": conf}

    # Real inference
    contents = await file.read()
    img      = Image.open(io.BytesIO(contents)).convert("RGB")
    arr      = np.array(img)
    results  = MODEL.predict(arr, conf=0.25, verbose=False)
    r        = results[0]

    SEVERITY_MAP = {
        "Tomato_Bacterial_spot":                        ("Bacterial Spot",         "High"),
        "Tomato_Early_blight":                          ("Early Blight",           "Medium"),
        "Tomato_Late_blight":                           ("Late Blight",            "Critical"),
        "Tomato_Leaf_Mold":                             ("Leaf Mold",              "Medium"),
        "Tomato_Septoria_leaf_spot":                    ("Septoria Leaf Spot",     "Medium"),
        "Tomato_Spider_mites Two-spotted_spider_mite":  ("Spider Mites",           "Low"),
        "Tomato__Target_Spot":                          ("Target Spot",            "Medium"),
        "Tomato__Tomato_YellowLeaf__Curl_Virus":        ("Yellow Leaf Curl Virus", "Critical"),
        "Tomato__Tomato_mosaic_virus":                  ("Tomato Mosaic Virus",    "High"),
        "Tomato_healthy":                               ("Healthy",                "None"),
    }

    if r.probs is not None:
        idx  = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        key  = MODEL.names[idx]
    elif r.boxes and len(r.boxes):
        best = int(r.boxes.conf.argmax())
        idx  = int(r.boxes.cls[best])
        conf = float(r.boxes.conf[best])
        key  = MODEL.names[idx]
    else:
        key, conf = "Tomato_healthy", 1.0

    label, severity = SEVERITY_MAP.get(key, (key, "Medium"))
    return {"disease": label, "severity": severity, "confidence": round(conf, 4)}


@app.post("/api/save-detection")
def save_detection(data: DetectionSave):
    conn = get_conn()
    conn.execute(
        "INSERT INTO detections (username,disease,confidence,severity,timestamp) VALUES (?,?,?,?,?)",
        (data.username, data.disease, data.confidence, data.severity,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.get("/api/history/{username}")
def get_history(username: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT disease,confidence,severity,timestamp FROM detections "
        "WHERE username=? ORDER BY id DESC LIMIT 50",
        (username,)
    ).fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


@app.delete("/api/history/{username}")
def clear_history(username: str):
    conn = get_conn()
    conn.execute("DELETE FROM detections WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return {"success": True}




# ══ FORGOT PASSWORD ══
class VerifyEmailData(BaseModel):
    email: str

class ResetPasswordData(BaseModel):
    email:        str
    new_password: str

@app.post("/api/verify-email")
def verify_email(data: VerifyEmailData):
    conn = get_conn()
    row  = conn.execute("SELECT email FROM users WHERE email=?", (data.email.strip(),)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "No account found with this email address.")
    return {"success": True}

@app.post("/api/reset-password")
def reset_password(data: ResetPasswordData):
    if len(data.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    conn = get_conn()
    cur  = conn.execute("UPDATE users SET password=? WHERE email=?",
                        (hash_pw(data.new_password), data.email.strip()))
    conn.commit(); conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Email not found.")
    return {"success": True, "message": "Password reset successfully."}

# ── Serve frontend files ──
app.mount("/", StaticFiles(directory=".", html=True), name="static")


from pydantic import BaseModel
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import hashlib
import os
import base64
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AgroAI API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== DATABASE ==========
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["agroai"]
users_col = db["users"]
detect_col = db["detections"]

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ========== YOLOv8 MODEL ==========
from ultralytics import YOLO

# Load model with error handling
MODEL_PATH = "best.pt"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
        logger.info(f"Classes: {model.names}")
    else:
        logger.warning(f"⚠️ Model file '{MODEL_PATH}' not found. Running without model.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# Disease information database
DISEASE_INFO = {
    "Bacterial_Spot": {
        "name": "Bacterial Spot",
        "severity": "High",
        "symptoms": "Small, dark, water-soaked spots on leaves that turn brown and necrotic. Spots may have yellow halos.",
        "treatment": "Apply copper-based bactericides. Remove infected leaves. Practice crop rotation.",
        "prevention": "Use resistant varieties. Avoid overhead irrigation. Maintain proper plant spacing."
    },
    "Early_Blight": {
        "name": "Early Blight",
        "severity": "Medium",
        "symptoms": "Dark brown spots with concentric rings (target-like appearance). Leaves turn yellow and drop.",
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb. Remove infected lower leaves.",
        "prevention": "Mulch to prevent soil splash. Water at base of plants. Stake plants for better air flow."
    },
    "Late_Blight": {
        "name": "Late Blight",
        "severity": "Critical",
        "symptoms": "Large, irregular, water-soaked lesions. White fuzzy growth on leaf undersides. Rapid plant death.",
        "treatment": "Apply fungicides immediately (chlorothalonil, mancozeb, or metalaxyl). Remove infected plants.",
        "prevention": "Plant resistant varieties. Avoid overhead watering. Monitor weather conditions."
    },
    "Leaf_Mold": {
        "name": "Leaf Mold",
        "severity": "Medium",
        "symptoms": "Pale green to yellow spots on upper leaf surface. Olive-green to purple mold on underside.",
        "treatment": "Improve air circulation. Apply fungicides. Remove affected leaves.",
        "prevention": "Reduce humidity. Space plants properly. Water early in morning."
    },
    "Septoria_Leaf_Spot": {
        "name": "Septoria Leaf Spot",
        "severity": "Medium",
        "symptoms": "Small, circular spots with gray centers and dark borders. Spots have tiny black specks.",
        "treatment": "Apply fungicides containing copper or chlorothalonil. Remove infected leaves.",
        "prevention": "Avoid overhead watering. Clean up plant debris. Practice crop rotation."
    },
    "Spider_Mites": {
        "name": "Spider Mites",
        "severity": "Low",
        "symptoms": "Tiny yellow/white speckles on leaves. Fine webbing on leaf undersides.",
        "treatment": "Spray with neem oil or insecticidal soap. Introduce predatory mites.",
        "prevention": "Keep plants well-watered. Increase humidity. Regularly inspect plants."
    },
    "Target_Spot": {
        "name": "Target Spot",
        "severity": "Medium",
        "symptoms": "Circular lesions with concentric rings. Lesions may have yellow halos.",
        "treatment": "Apply fungicides (azoxystrobin or pyraclostrobin). Remove infected plant material.",
        "prevention": "Avoid overhead irrigation. Maintain proper spacing. Rotate crops."
    },
    "Yellow_Leaf_Curl_Virus": {
        "name": "Yellow Leaf Curl Virus",
        "severity": "Critical",
        "symptoms": "Leaves curl upward and turn yellow. Stunted growth. Reduced fruit production.",
        "treatment": "Remove infected plants immediately. Control whitefly vectors with insecticides.",
        "prevention": "Use virus-resistant varieties. Install insect screens. Use reflective mulches."
    },
    "Healthy": {
        "name": "Healthy",
        "severity": "None",
        "symptoms": "No visible disease symptoms. Plant appears healthy and vigorous.",
        "treatment": "Continue good cultural practices. No treatment needed.",
        "prevention": "Maintain proper nutrition. Regular monitoring. Good sanitation practices."
    },
    "Tomato_Mosaic_Virus": {
        "name": "Tomato Mosaic Virus",
        "severity": "High",
        "symptoms": "Light and dark green mottling on leaves. Leaves may be distorted.",
        "treatment": "Remove infected plants. Disinfect tools. No cure for infected plants.",
        "prevention": "Use virus-free seeds. Wash hands before handling plants. Use resistant varieties."
    }
}

CONF_THRESHOLD = 0.35

# ========== SCHEMAS ==========
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
    image_data: Optional[str] = None

class VerifyEmailData(BaseModel):
    email: str

class ResetPasswordData(BaseModel):
    email: str
    new_password: str

# ========== AUTH ENDPOINTS ==========
@app.post("/api/signup")
def signup(data: SignupData):
    try:
        if len(data.username.strip()) < 3:
            raise HTTPException(400, "Username must be at least 3 characters")
        if len(data.password) < 6:
            raise HTTPException(400, "Password must be at least 6 characters")
        if "@" not in data.email:
            raise HTTPException(400, "Enter a valid email address")
        
        if users_col.find_one({"$or": [{"username": data.username.strip()}, {"email": data.email.strip()}]}):
            raise HTTPException(409, "Username or email already exists")
        
        users_col.insert_one({
            "username": data.username.strip(),
            "email": data.email.strip(),
            "password": hash_pw(data.password),
            "created": datetime.now(),
        })
        return {"success": True, "message": "Account created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(500, f"Server error: {str(e)}")

@app.post("/api/login")
def login(data: LoginData):
    try:
        user = users_col.find_one({
            "username": data.username.strip(),
            "password": hash_pw(data.password),
        })
        if not user:
            raise HTTPException(401, "Invalid username or password")
        return {"success": True, "username": user["username"], "email": user["email"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(500, f"Server error: {str(e)}")

# ========== PREDICTION ENDPOINT ==========
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded leaf image"""
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # If no model loaded, return fallback response
        if model is None:
            return {
                "disease": "Model Not Loaded",
                "severity": "Medium",
                "confidence": 0.0,
                "symptoms": "The AI model is not yet loaded. Please ensure 'best.pt' exists.",
                "treatment": "Train the model using train_tomato_classifier.py",
                "prevention": "Run training script to generate model file",
                "annotated_url": f"data:image/jpeg;base64,{img_base64}",
                "num_boxes": 0
            }
        
        # Run prediction
        results = model.predict(
            source=img,
            imgsz=224,
            conf=CONF_THRESHOLD,
            verbose=False,
            device="cpu"
        )
        
        result = results[0]
        
        # Get top prediction
        if hasattr(result, 'probs') and result.probs is not None:
            top1_idx = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = model.names[top1_idx]
        else:
            # Fallback for detection model
            if result.boxes is not None and len(result.boxes) > 0:
                top1_idx = int(result.boxes.cls[0])
                confidence = float(result.boxes.conf[0])
                class_name = model.names[top1_idx]
            else:
                return {
                    "disease": "No Disease Detected",
                    "severity": "Low",
                    "confidence": 0.0,
                    "symptoms": "No clear disease pattern detected.",
                    "treatment": "Upload a clearer image of the leaf.",
                    "prevention": "Ensure good lighting and focus.",
                    "num_boxes": 0
                }
        
        # Get disease info
        disease_info = DISEASE_INFO.get(class_name, {
            "name": class_name.replace("_", " "),
            "severity": "Medium",
            "symptoms": "No detailed information available.",
            "treatment": "Consult a local agricultural expert.",
            "prevention": "Maintain good crop management practices."
        })
        
        # Draw prediction on image
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 30], fill=(33, 102, 196))
        draw.text((10, 5), f"{disease_info['name']}: {confidence:.1%}", fill="white")
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "disease": disease_info["name"],
            "severity": disease_info["severity"],
            "confidence": round(confidence, 4),
            "symptoms": disease_info["symptoms"],
            "treatment": disease_info["treatment"],
            "prevention": disease_info["prevention"],
            "annotated_image": img_base64,
            "num_boxes": 1,
            "detections": [{
                "class": class_name,
                "confidence": round(confidence, 4)
            }]
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction error: {str(e)}")

# ========== HISTORY ENDPOINTS ==========
@app.post("/api/save-detection")
def save_detection(data: DetectionSave):
    try:
        detect_col.insert_one({
            "username": data.username,
            "disease": data.disease,
            "confidence": data.confidence,
            "severity": data.severity,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_data": data.image_data
        })
        return {"success": True}
    except Exception as e:
        logger.error(f"Save detection error: {e}")
        raise HTTPException(500, f"Error saving detection: {str(e)}")

@app.get("/api/history/{username}")
def get_history(username: str):
    try:
        rows = list(detect_col.find({"username": username}).sort("_id", -1).limit(50))
        for r in rows:
            r["_id"] = str(r["_id"])
            if "image_data" in r:
                del r["image_data"]
        return {"history": rows}
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(500, f"Error fetching history: {str(e)}")

@app.delete("/api/history/{username}")
def clear_history(username: str):
    try:
        detect_col.delete_many({"username": username})
        return {"success": True}
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(500, f"Error clearing history: {str(e)}")

# ========== FORGOT PASSWORD ==========
@app.post("/api/verify-email")
def verify_email(data: VerifyEmailData):
    try:
        user = users_col.find_one({"email": data.email.strip()})
        if not user:
            raise HTTPException(404, "No account found with this email address")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify email error: {e}")
        raise HTTPException(500, f"Error verifying email: {str(e)}")

@app.post("/api/reset-password")
def reset_password(data: ResetPasswordData):
    try:
        if len(data.new_password) < 6:
            raise HTTPException(400, "Password must be at least 6 characters")
        
        res = users_col.update_one(
            {"email": data.email.strip()},
            {"$set": {"password": hash_pw(data.new_password)}}
        )
        if res.matched_count == 0:
            raise HTTPException(404, "Email not found")
        return {"success": True, "message": "Password reset successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        raise HTTPException(500, f"Error resetting password: {str(e)}")

# ========== HEALTH CHECK ==========
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
