# ══════════════════════════════════════════════
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
    MODEL = MODEL = YOLO(_pt) if os.path.exists(_pt) else None
    if MODEL:
        print(f"✅  Model loaded: {_pt}")
    else:
        print("⚠️   best.pt not found — running in DEMO mode")
except ImportError:
    MODEL = None
    print("⚠️   ultralytics not installed — running in DEMO mode")

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

