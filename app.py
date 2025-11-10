from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import qrcode
import os
import sqlite3
import bcrypt
import datetime
from datetime import datetime, timedelta
from functools import wraps
import random
import secrets
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
import random
import string
import re
from typing import Dict, Optional, Any, List

# Import enhanced AI engine
try:
    from ai_engine import get_nlu, get_model, get_feedback, get_llm
    AI_ENGINE_AVAILABLE = True
except ImportError:
    AI_ENGINE_AVAILABLE = False
    print("Warning: ai_engine.py not found, using fallback regex system")

# Import sklearn-based NLU
try:
    from nlu import predict_intent, load_nlu, initialize_nlu
    NLU_AVAILABLE = True
except ImportError:
    NLU_AVAILABLE = False
    print("Warning: nlu.py not found, using fallback regex system")

# Import program model for scoring
try:
    from program_model import score_programs, featurize as pm_featurize
    PROGRAM_MODEL_AVAILABLE = True
except ImportError:
    PROGRAM_MODEL_AVAILABLE = False
    print("Warning: program_model.py not found, using fallback scoring")

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "supersecretkey"  # required for sessions
app.config['UPLOAD_FOLDER'] = 'static/qr_codes'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)
app.permanent_session_lifetime = timedelta(minutes=60)

# --- Access control decorators (must be defined before use) ---
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

# Allowed gender options used across forms and validations
ALLOWED_GENDERS = { 'Male', 'Female', 'Non-binary', 'Other', 'Prefer not to say' }

# Admin credentials from environment (hashed)
ENV_ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ENV_ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH')
ENV_ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '1234')
if ENV_ADMIN_PASSWORD_HASH:
    ADMIN_PASSWORD_HASH = ENV_ADMIN_PASSWORD_HASH.encode('utf-8')
else:
    ADMIN_PASSWORD_HASH = bcrypt.hashpw(ENV_ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt())
ADMIN_USERNAME = ENV_ADMIN_USERNAME

# Feature flags / settings
BACKDATE_SIGNUPS = os.environ.get('BACKDATE_SIGNUPS', 'false').lower() == 'true'

# CSRF helpers
from flask import abort

def ensure_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)

def require_valid_csrf():
    token = request.form.get('csrf_token')
    if not token or token != session.get('csrf_token'):
        abort(400, description='Invalid CSRF token')

@app.context_processor
def inject_csrf_token():
    return { 'csrf_token': session.get('csrf_token', '') }

@app.before_request
def apply_security_and_idle_timeout():
    # Ensure CSRF token exists
    ensure_csrf_token()
    # Idle timeout
    now = datetime.now()
    last_seen = session.get('last_seen')
    if last_seen:
        try:
            last_dt = datetime.fromisoformat(last_seen)
            if (now - last_dt) > timedelta(minutes=30):
                # Expire session
                session.clear()
        except Exception:
            session.clear()
    session['last_seen'] = now.isoformat(timespec='seconds')

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()

    # --- Core Tables ---
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    username TEXT UNIQUE,
                    password TEXT,
                    age INTEGER,
                    gender TEXT,
                    qr_path TEXT,
                    created_at TEXT,
                    email TEXT,
                    email_verified INTEGER DEFAULT 0
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS checkins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    action TEXT,
                    timestamp TEXT
                )''')

    # --- New Tables for Attendance + Programs ---
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    date TEXT,
                    time TEXT,
                    program_id INTEGER,
                    FOREIGN KEY(username) REFERENCES users(username)
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    category TEXT,
                    description TEXT,
                    day TEXT,
                    recommended_for TEXT,
                    themes TEXT,
                    diet_tags TEXT,
                    scripture_tags TEXT,
                    difficulty INTEGER
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS program_of_day (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE,
                    program_id INTEGER
                )''')

    # --- AI Enhancement Tables (from schemas.sql) ---
    c.execute('''CREATE TABLE IF NOT EXISTS program_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    program_id INTEGER,
                    username TEXT,
                    feedback INTEGER,
                    snapshot_json TEXT
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS model_meta (
                    model_name TEXT PRIMARY KEY,
                    storage_path TEXT,
                    last_trained TEXT,
                    version INTEGER
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )''')

    # --- Lightweight Migrations (safe if columns already exist) ---
    try:
        c.execute('ALTER TABLE users ADD COLUMN created_at TEXT')
    except Exception:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN email TEXT')
    except Exception:
        pass
    try:
        c.execute('ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0')
    except Exception:
        pass
    # programs optional columns
    for alter_sql in (
        "ALTER TABLE programs ADD COLUMN themes TEXT",
        "ALTER TABLE programs ADD COLUMN diet_tags TEXT",
        "ALTER TABLE programs ADD COLUMN scripture_tags TEXT",
        "ALTER TABLE programs ADD COLUMN difficulty INTEGER"
    ):
        try:
            c.execute(alter_sql)
        except Exception:
            pass

    conn.commit()
    conn.close()

def seed_programs_if_empty():
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM programs')
    n = c.fetchone()[0]
    if n and n > 0:
        conn.close()
        return
    
    # Seed actual mandir programs based on real schedule
    programs = [
        # Sunday Morning Program
        (
            'Sunday Morning Program (रविवार सुबह)',
            'Weekly',
            'Puja, Kirtan, Pravachan, Bhog, Aarti, Bhojan | पूजा, कीर्तन, प्रवचन, भोग, आरती, भोजन',
            'Sunday',
            'All',
            'devotional,community,family',
            'vegetarian,traditional',
            'general,spiritual',
            2,
        ),
        # Monday Evening Program - Shiv Puja
        (
            'Monday Program - Bhagwan Shiv Puja (सोमवार शाम)',
            'Weekly',
            'Bhagwan Shiv Puja, Kirtan, Maha Shiv Puran Katha, Aarti, Bhojan | भगवान शिव की पूजा, कीर्तन, महा शिव पुराण कथा, आरती, भोजन',
            'Monday',
            'All',
            'devotional,shiv-focused,spiritual',
            'vegetarian,traditional',
            'shiv-puran,katha',
            2,
        ),
        # Tuesday Evening Program - Hanuman Puja
        (
            'Tuesday Program - Hanuman Swami Puja (मंगलवार शाम)',
            'Weekly',
            'Hanuman Swami Puja, Chalisa, Sundar Kaand, Aarti, Bhojan | हनुमान स्वामी की पूजा, चालीसा, सुंदर कांड, आरती, भोजन',
            'Tuesday',
            'All',
            'devotional,hanuman-focused,spiritual',
            'vegetarian,traditional',
            'hanuman-chalisa,sundar-kaand',
            2,
        ),
        # Saturday Navgraha Puja
        (
            'Navgraha Puja (नवग्रह पूजा)',
            'Weekly',
            'Group Navgraha Puja - Call to join with other families',
            'Saturday',
            'All',
            'devotional,planetary,group-puja',
            'vegetarian,traditional',
            'navgraha',
            2,
        ),
        # Purnima Program (special event)
        (
            'Purnima Program (पूर्णिमा पूजा)',
            'Special',
            'Puja, Kirtan, Pravachan, Bhog, Aarti, Bhojan | पूजा, कीर्तन, प्रवचन, भोग, आरती, भोजन',
            'Purnima',
            'All',
            'devotional,special-event,full-moon',
            'vegetarian,traditional,special',
            'purnima,spiritual',
            2,
        ),
        # Daily Aarti (for other days)
        (
            'Daily Aarti (रोज़ आरती)',
            'Daily',
            'Regular daily aarti service',
            'Daily',
            'All',
            'devotional,daily',
            'vegetarian',
            'general',
            1,
        ),
    ]
    
    for prog in programs:
        c.execute(
            """
            INSERT INTO programs (name, category, description, day, recommended_for, themes, diet_tags, scripture_tags, difficulty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            prog,
        )
    conn.commit()
    conn.close()

# Initialize database
init_db()
seed_programs_if_empty()


# --- Helper functions ---
def get_all_users():
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT name, username, age, gender, qr_path FROM users")
    users = c.fetchall()
    conn.close()
    return users

def get_user_by_username(username):
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT name, username, age, gender, qr_path FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_username_full(username):
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    # Keep qr_path at index 5; append email at index 6
    c.execute("SELECT id, name, username, age, gender, qr_path, email FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_visit_stats(username):
    """Get visit statistics for a user"""
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    # Total visits
    c.execute("SELECT COUNT(*) FROM attendance WHERE username = ?", (username,))
    total_visits = c.fetchone()[0]
    # Last visit
    c.execute("SELECT MAX(date) FROM attendance WHERE username = ?", (username,))
    last_visit = c.fetchone()[0]
    # Visits this month
    month_start = datetime.now().date().replace(day=1).isoformat()
    c.execute("SELECT COUNT(*) FROM attendance WHERE username = ? AND date >= ?", (username, month_start))
    visits_this_month = c.fetchone()[0]
    # Visits last 30 days
    thirty_days_ago = (datetime.now() - timedelta(days=30)).date().isoformat()
    c.execute("SELECT COUNT(*) FROM attendance WHERE username = ? AND date >= ?", (username, thirty_days_ago))
    visits_last_30 = c.fetchone()[0]
    conn.close()
    return {
        'total_visits': total_visits,
        'last_visit': last_visit,
        'visits_this_month': visits_this_month,
        'visits_last_30': visits_last_30
    }

def get_inactive_users(days_threshold=30):
    """Get users who haven't visited in X days"""
    threshold_date = (datetime.now() - timedelta(days=days_threshold)).date().isoformat()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    # Users with no attendance OR last visit before threshold
    c.execute("""
        SELECT u.username, u.name, u.email, 
               COALESCE(MAX(a.date), 'Never') as last_visit,
               COALESCE(COUNT(a.id), 0) as total_visits
        FROM users u
        LEFT JOIN attendance a ON u.username = a.username
        WHERE u.email_verified = 1
        GROUP BY u.username, u.name, u.email
        HAVING last_visit = 'Never' OR last_visit < ?
        ORDER BY last_visit ASC
    """, (threshold_date,))
    users = c.fetchall()
    conn.close()
    return [{'username': u[0], 'name': u[1], 'email': u[2], 'last_visit': u[3], 'total_visits': u[4]} for u in users]

def send_reminder_email(email, name, days_since_visit):
    """Send reminder email (simulated for demo, ready for real SMTP)"""
    subject = f"Namaste {name} - We Miss You at the Mandir!"
    body = f"""Namaste {name},

We noticed it's been {days_since_visit} days since your last visit to the mandir. We'd love to see you again!

The mandir community is always here for you. Come join us for:
- Daily prayers and aarti
- Community gatherings
- Spiritual guidance
- Peaceful reflection

We look forward to welcoming you back soon.

With blessings,
The Mandir Community
"""
    # Simulated email sending (print for demo)
    print(f"\n{'='*60}")
    print(f"[EMAIL DEMO] To: {email}")
    print(f"Subject: {subject}")
    print(f"{'='*60}")
    print(body)
    print(f"{'='*60}\n")
    
    # TODO: Replace with real SMTP when ready
    # import smtplib
    # from email.mime.text import MIMEText
    # msg = MIMEText(body)
    # msg['Subject'] = subject
    # msg['From'] = 'mandir@example.com'
    # msg['To'] = email
    # server = smtplib.SMTP('smtp.example.com', 587)
    # server.send_message(msg)
    # server.quit()
    
    return True

# --- Programs helper ---
def get_program_of_day_id() -> int | None:
    today_name = datetime.now().strftime('%A')  # e.g., 'Monday'
    today_date = datetime.now().date().isoformat()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    # First: explicit program_of_day
    c.execute("SELECT program_id FROM program_of_day WHERE date = ?", (today_date,))
    row = c.fetchone()
    if row and row[0]:
        conn.close()
        return row[0]
    # Fallback to weekday default
    c.execute("SELECT id FROM programs WHERE day = ? ORDER BY id LIMIT 1", (today_name,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# --- Analytics snapshot helpers ---
AGE_BUCKETS = [(0,12),(13,17),(18,29),(30,49),(50,64),(65,200)]

def compute_today_snapshot():
    today_dt = datetime.now()
    today = today_dt.date().isoformat()
    last30 = (today_dt - timedelta(days=30)).date().isoformat()
    last14 = (today_dt - timedelta(days=14)).date().isoformat()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    # attendees today with user profile
    c.execute("""
        SELECT u.username, u.age, u.gender, u.created_at
        FROM attendance a
        JOIN users u ON u.username = a.username
        WHERE a.date = ?
    """, (today,))
    rows = c.fetchall()
    usernames = [r[0] for r in rows]
    # attendance counts last 30 days for frequent visitor metric
    frequent_set = set()
    if usernames:
        qmarks = ','.join(['?']*len(usernames))
        c.execute(f"""
            SELECT username, COUNT(*)
            FROM attendance
            WHERE date >= ? AND username IN ({qmarks})
            GROUP BY username
        """, (last30, *usernames))
        for uname, cnt in c.fetchall():
            if int(cnt) >= 4:
                frequent_set.add(uname)
    conn.close()
    total = len(rows)
    by_gender = {}
    by_bucket = { '0-12':0,'13-17':0,'18-29':0,'30-49':0,'50-64':0,'65+':0 }
    newcomers = 0
    frequent = 0
    for uname, age, gender, created_at in rows:
        by_gender[gender] = by_gender.get(gender,0)+1
        age = int(age)
        if age <= 12: by_bucket['0-12']+=1
        elif age <= 17: by_bucket['13-17']+=1
        elif age <= 29: by_bucket['18-29']+=1
        elif age <= 49: by_bucket['30-49']+=1
        elif age <= 64: by_bucket['50-64']+=1
        else: by_bucket['65+']+=1
        if created_at and created_at >= last14:
            newcomers += 1
        if uname in frequent_set:
            frequent += 1
    return {
        'date': today,
        'total': total,
        'by_gender': by_gender,
        'by_age_bucket': by_bucket,
        'newcomers': newcomers,
        'frequent': frequent
    }

# --- Lightweight on-device learning (Raspberry Pi friendly) ---
# We learn per-program linear weights over snapshot features using a perceptron-style update.

def init_model_storage():
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS model_weights (
                    program_id INTEGER,
                    feature TEXT,
                    weight REAL,
                    PRIMARY KEY(program_id, feature)
                )''')
    conn.commit()
    conn.close()

init_model_storage()

FEATURES = [
    'bias',
    'ratio_children',
    'ratio_seniors',
    'ratio_newcomers',
    'ratio_frequent',
]

LEARNING_RATE = 0.5


def features_from_snapshot(snapshot):
    """Extract features from snapshot (uses new AI engine if available)"""
    if AI_ENGINE_AVAILABLE:
        model = get_model()
        return model.features_from_snapshot(snapshot)
    # Fallback to old implementation
    total = max(snapshot.get('total', 0), 1)
    by_age = snapshot.get('by_age_bucket', {})
    newcomers = snapshot.get('newcomers', 0)
    frequent = snapshot.get('frequent', 0)
    children = by_age.get('0-12', 0) + by_age.get('13-17', 0)
    seniors = by_age.get('65+', 0)
    feats = {
        'bias': 1.0,
        'ratio_children': children / total,
        'ratio_seniors': seniors / total,
        'ratio_newcomers': newcomers / total,
        'ratio_frequent': frequent / total,
    }
    return feats


def load_weights(program_id):
    """Load weights (backward compatibility)"""
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute('SELECT feature, weight FROM model_weights WHERE program_id=?', (program_id,))
    rows = c.fetchall()
    conn.close()
    weights = { f: w for f, w in rows }
    # default missing weights to 0
    for f in FEATURES:
        weights.setdefault(f, 0.0)
    return weights


def score_program(program_id, feats, snapshot=None):
    """Score program using enhanced model if available"""
    if AI_ENGINE_AVAILABLE and snapshot:
        model = get_model()
        return model.score_program(program_id, feats, snapshot)
    # Fallback to old implementation
    w = load_weights(program_id)
    s = 0.0
    for f, v in feats.items():
        s += w.get(f, 0.0) * v
    return s


def update_weights(program_id, feats, label):
    """Update weights using enhanced model if available"""
    if AI_ENGINE_AVAILABLE:
        model = get_model()
        model.update_weights(program_id, feats, label, learning_rate=LEARNING_RATE)
        # Also record structured feedback
        feedback = get_feedback()
        feedback.record_feedback(program_id, 'good' if label > 0 else 'bad', 
                                'simple', snapshot_features=feats)
        return
    # Fallback to old implementation
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    for f, v in feats.items():
        c.execute('SELECT weight FROM model_weights WHERE program_id=? AND feature=?', (program_id, f))
        row = c.fetchone()
        w = row[0] if row else 0.0
        w_new = w + LEARNING_RATE * label * v
        c.execute('REPLACE INTO model_weights(program_id, feature, weight) VALUES(?,?,?)', (program_id, f, w_new))
    conn.commit()
    conn.close()

# Modify suggestion to consider multiple candidates and score them

def fallback_rule_based_program(snapshot):
    """Fallback rule-based program selection when ML model is not available"""
    today_name = datetime.now().strftime('%A')
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM programs WHERE day=? ORDER BY id LIMIT 1", (today_name,))
    result = c.fetchone()
    if not result:
        c.execute("SELECT id, name FROM programs ORDER BY id LIMIT 1")
        result = c.fetchone()
    conn.close()
    if result:
        return result[0], result[1]
    return None, None

def suggest_program_from_snapshot(snapshot):
    # Initialize variables
    best_prob = None
    best_score = None
    
    # Use sklearn-based program model if available
    if PROGRAM_MODEL_AVAILABLE:
        try:
            # Get today's candidate programs
            today_name = datetime.now().strftime('%A')
            conn = sqlite3.connect('temple.db')
            c = conn.cursor()
            c.execute("SELECT id, name FROM programs WHERE day=? ORDER BY id", (today_name,))
            candidates = c.fetchall()
            if not candidates:
                c.execute("SELECT id, name FROM programs ORDER BY id")
                candidates = c.fetchall()
            conn.close()
            
            if not candidates:
                return None
            
            # Get candidate program IDs
            candidate_ids = [pid for pid, _ in candidates]
            
            # Get scores for candidate programs using the new model
            scores = score_programs(snapshot, program_ids=candidate_ids)
            
            if not scores:
                # Fallback to rule-based if no scores
                program_id, name = fallback_rule_based_program(snapshot)
                if not program_id:
                    return None
            else:
                # Get best program (highest score)
                # Sort by score descending
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                best_program_id, best_prob = sorted_scores[0]
                
                # Get program name (should be in candidates, but double-check)
                name = next((nm for pid, nm in candidates if pid == best_program_id), None)
                if name:
                    program_id = best_program_id
                else:
                    # Fallback if program not found
                    program_id, name = fallback_rule_based_program(snapshot)
                    if not program_id:
                        return None
        except Exception as e:
            print(f"Error in program model scoring: {e}")
            # Fallback to old method
            program_id, name = fallback_rule_based_program(snapshot)
            if not program_id:
                return None
    else:
        # Fallback to old perceptron-based scoring
        feats = features_from_snapshot(snapshot)
        today_name = datetime.now().strftime('%A')
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        c.execute("SELECT id, name FROM programs WHERE day=? ORDER BY id", (today_name,))
        candidates = c.fetchall()
        if not candidates:
            c.execute("SELECT id, name FROM programs ORDER BY id")
            candidates = c.fetchall()
        if not candidates:
            conn.close()
            return None
        # pick best by learned score
        best = None
        best_score = -1e9
        for pid, nm in candidates:
            sc = score_program(pid, feats, snapshot)  # Pass snapshot for enhanced model
            if sc > best_score:
                best_score = sc
                best = (pid, nm)
        conn.close()
        program_id, name = best
    # Enhanced AI-like rationale
    total = snapshot['total'] or 1
    by_age = snapshot['by_age_bucket']
    by_gender = snapshot.get('by_gender', {})
    newcomers = snapshot.get('newcomers', 0)
    frequent = snapshot.get('frequent', 0)
    
    children_ratio = (by_age['0-12'] + by_age['13-17'])/total
    seniors_ratio = (by_age['65+'])/total
    newcomer_ratio = newcomers/total
    frequent_ratio = frequent/total
    
    rationale = []
    
    # Use new format if best_prob is available (from sklearn model)
    if best_prob is not None:
        rationale.append(f"Detected {children_ratio*100:.0f}% children and {newcomer_ratio*100:.0f}% newcomers — recommending program ID {program_id} (confidence {best_prob:.0%}).")
    elif best_score is not None:
        # Fallback to old format
        if best_score > 0.5:
            rationale.append(f"AI recommendation: This program historically performs well for similar demographics (confidence: {min(100, int(best_score*100))}%)")
        elif best_score > 0:
            rationale.append(f"AI recommendation: Moderate confidence based on learned patterns")
        else:
            rationale.append("AI recommendation: Using baseline selection while learning continues")
    else:
        rationale.append("AI recommendation: Using baseline selection")
    
    if total == 0:
        rationale.append("No attendance data yet today; using default program")
    elif children_ratio > 0.35:
        rationale.append(f"Noticing {int(children_ratio*100)}% children/teens → prioritizing engaging, story-based content")
    elif seniors_ratio > 0.40:
        rationale.append(f"Detecting {int(seniors_ratio*100)}% seniors → emphasizing comfort, accessible pace, and seated options")
    
    if newcomer_ratio > 0.25:
        rationale.append(f"Observing {int(newcomer_ratio*100)}% newcomers → adding welcoming introduction segments")
    elif frequent_ratio > 0.40:
        rationale.append(f"Strong regular community ({int(frequent_ratio*100)}%) → including deeper discussion opportunities")
    
    if by_gender.get('Prefer not to say',0)/total > 0.15:
        rationale.append("Emphasizing inclusive, privacy-respecting participation given diverse preferences")
    
    # Derive tags with rules
    menu_tags = []
    talk_tags = []
    if children_ratio > 0.35:
        menu_tags += ['mild-spice','kid-friendly']
        talk_tags += ['youth-focus','interactive']
    if seniors_ratio > 0.40:
        menu_tags += ['low-spice','soft-texture']
        talk_tags += ['accessible-pace']
    if newcomer_ratio > 0.25:
        talk_tags += ['introductory','community-welcome']
    if frequent_ratio > 0.40:
        talk_tags += ['deeper-discussion']
    if by_gender.get('Prefer not to say',0)/total > 0.15:
        talk_tags += ['inclusive']
    menu_tags = sorted(set(menu_tags))
    talk_tags = sorted(set(talk_tags))
    conn.close()
    return {
        'program_id': program_id,
        'program_name': name,
        'rationale': rationale,
        'menu_tags': menu_tags,
        'talk_tags': talk_tags,
        'snapshot': snapshot
    }

@app.route('/train-outcome', methods=['POST'])
@admin_required
def train_outcome():
    require_valid_csrf()
    outcome = request.form.get('outcome')  # 'good' or 'bad'
    program_id = request.form.get('program_id')
    if not program_id or outcome not in ('good','bad'):
        return 'program_id and valid outcome required', 400
    program_id = int(program_id)
    snap = compute_today_snapshot()
    feats = features_from_snapshot(snap)
    label = 1 if outcome == 'good' else -1
    update_weights(program_id, feats, label)
    flash('Thanks! Model updated for today\'s outcome.', 'success')
    return redirect(url_for('program_plan'))

@app.route('/suggest-program')
@admin_required
def suggest_program():
    snap = compute_today_snapshot()
    suggestion = suggest_program_from_snapshot(snap)
    if not suggestion:
        return { 'message': 'No program candidates found for today' }, 200
    return suggestion, 200

@app.route('/apply-program-of-day', methods=['POST'])
@admin_required
def apply_program_of_day():
    require_valid_csrf()
    program_id = request.form.get('program_id')
    if not program_id:
        return 'program_id required', 400
    today = datetime.now().date().isoformat()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("INSERT INTO program_of_day(date, program_id) VALUES(?, ?) ON CONFLICT(date) DO UPDATE SET program_id=excluded.program_id", (today, program_id))
    conn.commit()
    conn.close()
    flash('Program of the day applied.', 'success')
    return redirect(url_for('admin_page'))

# --- Decorators ---
# Ensure admin_required decorator is defined before any routes use it
# try:
#     admin_required
# except NameError:
#     def admin_required(f):
#         @wraps(f)
#         def decorated(*args, **kwargs):
#             if not session.get('admin'):
#                 return redirect(url_for('admin_login'))
#             return f(*args, **kwargs)
#         return decorated

# --- 2FA remember helpers ---
TWOFA_COOKIE_NAME = 'twofa_ok'
TWOFA_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days

def _twofa_signer() -> TimestampSigner:
    return TimestampSigner(app.secret_key)

def create_twofa_cookie_value(username: str) -> str:
    signer = _twofa_signer()
    return signer.sign(username.encode('utf-8')).decode('utf-8')

def validate_twofa_cookie(username: str, token: str) -> bool:
    if not token:
        return False
    signer = _twofa_signer()
    try:
        value = signer.unsign(token, max_age=TWOFA_MAX_AGE_SECONDS)
        return value.decode('utf-8') == username
    except (BadSignature, SignatureExpired):
        return False

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    require_valid_csrf()
    name = request.form['name']
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']
    age = request.form['age']
    gender = request.form['gender']
    if gender not in ALLOWED_GENDERS:
        return "Invalid gender selection.", 400

    # Check if username is unique
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return "Username already exists. Please choose another.", 400

    # Hash the password with bcrypt
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # For real registrations we keep the actual timestamp unless simulation mode is enabled
    days_ago = 0
    if BACKDATE_SIGNUPS:
        if random.random() < 0.8:
            days_ago = random.randint(15, 180)
        else:
            days_ago = random.randint(0, 14)
    created_dt = datetime.now() - timedelta(days=days_ago)
    created_at = created_dt.isoformat(timespec='seconds')

    # Generate QR code
    qr_data = f"{username}-{age}-{gender}"
    qr_img = qrcode.make(qr_data)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{username}.png")
    qr_img.save(file_path)

    # Save user info
    c.execute("INSERT INTO users (name, username, password, age, gender, qr_path, created_at, email, email_verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
              (name, username, hashed_pw, age, gender, file_path, created_at, email.lower()))

    # Optionally create some past attendance within last 30 days to simulate regulars
    # Heavier chance for users with older created_at
    probable_visits = 0
    if days_ago >= 30:
        probable_visits = random.randint(2, 8)
    elif days_ago >= 14:
        probable_visits = random.randint(1, 5)
    else:
        probable_visits = random.randint(0, 3)

    if BACKDATE_SIGNUPS and probable_visits > 0:
        today = datetime.now().date()
        chosen_days = set()
        for _ in range(probable_visits):
            d = today - timedelta(days=random.randint(1, 30))
            if d in chosen_days:
                continue
            chosen_days.add(d)
            date_str = d.isoformat()
            time_str = f"{random.randint(8,20):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
            # use weekday default (no program_of_day backfill for historical)
            program_id = None
            # Insert attendance and checkin
            c.execute("INSERT INTO attendance (username, date, time, program_id) VALUES (?, ?, ?, ?)",
                      (username, date_str, time_str, program_id))
            c.execute("INSERT INTO checkins (username, action, timestamp) VALUES (?, 'qr_scan', ?)",
                      (username, f"{date_str}T{time_str}"))

    conn.commit()
    conn.close()

    # Email verification step
    code = f'{random.randint(100000, 999999)}'
    session.clear()
    session.permanent = True
    session['email_verification'] = {
        'username': username,
        'code': code,
        'email': email.lower()
    }
    print(f"[EMAIL DEMO] Verification code for {email}: {code}")
    return redirect(url_for('verify_email'))

@app.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    info = session.get('email_verification')
    if not info:
        return redirect(url_for('login'))
    username = info.get('username')
    expected_code = info.get('code')
    if request.method == 'POST':
        require_valid_csrf()
        code = request.form['code']
        if code == expected_code:
            conn = sqlite3.connect('temple.db')
            c = conn.cursor()
            c.execute("UPDATE users SET email_verified=1 WHERE username=?", (username,))
            conn.commit()
            # Email verified, can log in now
            c.execute("SELECT name, username, age, gender, qr_path, email FROM users WHERE username=?", (username,))
            user = c.fetchone()
            conn.close()
            # Log in the user
            session.clear()
            session.permanent = True
            session['user'] = {
                'name': user[0],
                'username': user[1],
                'age': user[2],
                'gender': user[3],
                'qr_path': user[4],
                'email': user[5],
            }
            flash('Email verified and account activated!','success')
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect verification code.','danger')
    return render_template('verify_email.html', email=info.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        require_valid_csrf()
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        c.execute("SELECT name, username, age, gender, qr_path, password, email_verified, email FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        # Must check email_verified!
        if not user or not user[6]:
            return "Account not verified (check your email)", 400

        # If a valid 2FA cookie exists, skip 2FA
        token = request.cookies.get(TWOFA_COOKIE_NAME)
        if token and bcrypt.checkpw(password.encode('utf-8'), user[5]) and validate_twofa_cookie(username, token):
            session.clear()
            session.permanent = True
            session['user'] = {
                'name': user[0],
                'username': user[1],
                'age': user[2],
                'gender': user[3],
                'qr_path': user[4]
            }
            now = datetime.now().isoformat(timespec='seconds')
            conn = sqlite3.connect('temple.db')
            c = conn.cursor()
            c.execute("INSERT INTO checkins (username, action, timestamp) VALUES (?, ?, ?)", (username, 'login', now))
            conn.commit()
            conn.close()
            resp = redirect(url_for('dashboard'))
            # Refresh cookie validity window
            refreshed = create_twofa_cookie_value(username)
            resp.set_cookie(TWOFA_COOKIE_NAME, refreshed, max_age=TWOFA_MAX_AGE_SECONDS, secure=True, httponly=True, samesite='Lax')
            return resp

        if bcrypt.checkpw(password.encode('utf-8'), user[5]):
            # Generate and "send" 2FA code
            code = f'{random.randint(100000, 999999)}'
            session.clear()
            session.permanent = True
            session['pending_2fa'] = {
                'username': username,
                'code': code
            }
            print(f"[2FA DEMO] 2FA code for {username}: {code}")
            return redirect(url_for('twofa'))
        else:
            return "Invalid username or password", 400
    return render_template('login.html')

@app.route('/2fa', methods=['GET', 'POST'])
def twofa():
    info = session.get('pending_2fa')
    if not info:
        return redirect(url_for('login'))
    username = info.get('username')
    expected_code = info.get('code')
    if request.method == 'POST':
        require_valid_csrf()
        code = request.form['code']
        if code == expected_code:
            # Actually log user in (lookup DB info for session)
            conn = sqlite3.connect('temple.db')
            c = conn.cursor()
            c.execute("SELECT name, username, age, gender, qr_path, email FROM users WHERE username=?", (username,))
            u = c.fetchone()
            conn.close()
            session.clear()
            session.permanent = True
            session['user'] = {
                'name': u[0],
                'username': u[1],
                'age': u[2],
                'gender': u[3],
                'qr_path': u[4],
                'email': u[5],
            }
            # Log check-in event
            now = datetime.now().isoformat(timespec='seconds')
            conn = sqlite3.connect('temple.db')
            c = conn.cursor()
            c.execute("INSERT INTO checkins (username, action, timestamp) VALUES (?, ?, ?)", (username, 'login', now))
            conn.commit()
            conn.close()
            # Issue remember-2FA cookie for 7 days
            token = create_twofa_cookie_value(username)
            resp = redirect(url_for('dashboard'))
            resp.set_cookie(TWOFA_COOKIE_NAME, token, max_age=TWOFA_MAX_AGE_SECONDS, secure=True, httponly=True, samesite='Lax')
            return resp
        else:
            flash('Incorrect 2FA code.','danger')
    return render_template('twofa.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user = session['user']
    visit_stats = get_user_visit_stats(user['username'])
    return render_template('dashboard.html', user=user, visit_stats=visit_stats)

@app.route('/logout', methods=['POST'])
def logout():
    require_valid_csrf()
    user = session.get('user')
    if user:
        now = datetime.now().isoformat(timespec='seconds')
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        c.execute("INSERT INTO checkins (username, action, timestamp) VALUES (?, ?, ?)", (user['username'], 'logout', now))
        conn.commit()
        conn.close()
    session.pop('user', None)
    return redirect(url_for('login'))  # goes to login page

@app.route('/scan/<username>')
def scan_qr(username):
    # Simulate scanning a QR for a user and log it
    now_dt = datetime.now()
    now_iso = now_dt.isoformat(timespec='seconds')
    date_str = now_dt.date().isoformat()
    time_str = now_dt.strftime('%H:%M:%S')
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    # Only log if user exists
    c.execute("SELECT 1 FROM users WHERE username=?", (username,))
    if c.fetchone():
        # Check active program of the day
        program_id = get_program_of_day_id()
        # Log checkin event
        c.execute("INSERT INTO checkins (username, action, timestamp) VALUES (?, ?, ?)", (username, 'qr_scan', now_iso))
        # Log attendance with optional program
        if program_id is not None:
            c.execute(
                "INSERT INTO attendance (username, date, time, program_id) VALUES (?, ?, ?, ?)",
                (username, date_str, time_str, program_id),
            )
        else:
            c.execute(
                "INSERT INTO attendance (username, date, time, program_id) VALUES (?, ?, ?, NULL)",
                (username, date_str, time_str),
            )
        conn.commit()
        conn.close()
        return f"QR scan recorded for {username} at {now_iso}. Program: {program_id if program_id is not None else 'None'}. <a href='{url_for('admin_page')}'>Back to admin</a>", 200
    conn.close()
    return "User not found.", 404

# --- Enhanced AI Chat with conversational understanding ---
# Helper functions for intent handlers
def get_today_stats():
    """Get today's statistics"""
    snap = compute_today_snapshot()
    total = snap.get('total', 0)
    by_age = snap.get('by_age_bucket', {})
    by_gender = snap.get('by_gender', {})
    
    stats_text = f"Today's Statistics:\n\n"
    stats_text += f"Total Attendees: {total}\n"
    if by_age:
        stats_text += f"\nAge Distribution:\n"
        for bucket, count in by_age.items():
            if count > 0:
                stats_text += f"  {bucket}: {count}\n"
    if by_gender:
        stats_text += f"\nGender Distribution:\n"
        for gender, count in by_gender.items():
            if count > 0:
                stats_text += f"  {gender}: {count}\n"
    
    return {'answer': stats_text, 'type': 'statistics'}

def adjust_menu(current_program, adjustment_type, session_obj=None):
    """Adjust menu based on adjustment type"""
    # Get current program context from session
    if session_obj is None:
        session_obj = session
    context = session_obj.get('current_program_context')
    if not context:
        return {'answer': 'No program in context. Please suggest a program first.', 'type': 'error'}
    
    plan = context.get('plan', {})
    menu = plan.get('menu', [])
    
    # Apply adjustments
    if adjustment_type == 'kid_friendly':
        menu = ['Vegetable pulao (mild)', 'Fruit cups', 'Banana sheera']
    elif adjustment_type == 'senior_friendly':
        menu = ['Khichdi', 'Cucumber raita', 'Soft roti']
    elif adjustment_type == 'traditional':
        menu = ['Puri-bhaji', 'Kadhi-chawal']
    else:
        menu = menu  # Keep current
    
    # Update context
    plan['menu'] = menu
    context['plan'] = plan
    session_obj['current_program_context'] = context
    
    return {'answer': f'Menu updated to: {", ".join(menu)}', 'type': 'menu_updated', 'menu': menu}

def change_program(new_program_name):
    """Change to a different program"""
    if not new_program_name:
        return {'answer': 'Please specify which program you want to switch to.', 'type': 'error'}
    
    # Look up program by name
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM programs WHERE name LIKE ?", (f'%{new_program_name.strip()}%',))
    program_row = c.fetchone()
    conn.close()
    
    if not program_row:
        return {'answer': f'Program "{new_program_name}" not found.', 'type': 'error'}
    
    program_id, program_name = program_row
    snap = compute_today_snapshot()
    suggestion = suggest_program_from_snapshot(snap)
    if suggestion:
        suggestion['program_id'] = program_id
        suggestion['program_name'] = program_name
        suggestion['rationale'] = [f'Program manually changed to {program_name} by admin request.']
    else:
        suggestion = {
            'program_id': program_id,
            'program_name': program_name,
            'rationale': [f'Program manually changed to {program_name} by admin request.'],
            'menu_tags': [],
            'talk_tags': [],
            'snapshot': snap
        }
    
    plan = generate_program_plan(snap, suggestion)
    
    # Persist program of the day selection
    today = datetime.now().date().isoformat()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO program_of_day(date, program_id) VALUES(?, ?) "
        "ON CONFLICT(date) DO UPDATE SET program_id=excluded.program_id",
        (today, program_id)
    )
    conn.commit()
    conn.close()
    
    # Update session context for further modifications
    session['current_program_context'] = {
        'program_id': program_id,
        'program_name': program_name,
        'plan': plan
    }
    
    schedule_lines = '\n'.join([f"{s['time']}: {s['item']}" for s in plan.get('schedule', [])])
    response = (
        f'Understood. I have switched today\'s program to **{program_name}** '
        f'and applied it for all upcoming check-ins.\n\nSchedule:\n{schedule_lines}'
    )
    
    return {
        'answer': response,
        'type': 'program_changed',
        'program_data': suggestion,
        'plan': plan
    }

# Entity extraction helpers
def extract_entities(query: str) -> Dict[str, Any]:
    """Extract entities from query (program names, dates, counts)"""
    entities = {}
    query_lower = query.lower()
    
    # Extract program names (check against programs table)
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT name FROM programs")
    all_programs = [row[0].lower() for row in c.fetchall()]
    conn.close()
    
    for prog in all_programs:
        # Check for program name mentions
        prog_keywords = prog.split()
        if any(kw in query_lower for kw in prog_keywords if len(kw) > 3):
            entities['program_name'] = prog
            break
    
    # Extract dates
    date_patterns = [
        r'today', r'this week', r'this month', r'last week', r'last month',
        r'monday', r'tuesday', r'wednesday', r'thursday', r'friday', r'saturday', r'sunday'
    ]
    for pattern in date_patterns:
        if re.search(pattern, query_lower):
            entities['date'] = pattern
            break
    
    # Extract numbers/counts
    numbers = re.findall(r'\d+', query)
    if numbers:
        entities['numbers'] = [int(n) for n in numbers]
    
    return entities

def _map_nlu_intent_to_query_type(intent: str, entities: Dict) -> str:
    """Map NLU intent to query type"""
    # Direct mappings
    if intent in ['greeting', 'thanks', 'goodbye', 'capabilities', 'suggestion_request', 
                  'statistics_query', 'newcomer_query', 'frequent_visitor_query']:
        intent_map = {
            'greeting': 'greeting',
            'thanks': 'thanks',
            'goodbye': 'goodbye',
            'capabilities': 'capabilities',
            'suggestion_request': 'suggest_program',
            'statistics_query': 'trends',
            'newcomer_query': 'newcomers',
            'frequent_visitor_query': 'frequent_visitors',
        }
        return intent_map.get(intent, 'clarify')
    
    # Attendance query - check time period
    if intent == 'attendance_query':
        time_period = entities.get('time_period', 'today')
        if 'week' in time_period:
            return 'attendance_week'
        elif 'month' in time_period:
            return 'attendance_month'
        return 'attendance_today'
    
    # Demographic query - check specific type
    if intent == 'demographic_query':
        query_lower = entities.get('raw_query', '').lower()
        if 'average' in query_lower or 'mean' in query_lower:
            return 'avg_age'
        elif any(w in query_lower for w in ['child', 'kid', 'teen', 'youth']):
            return 'children_count'
        elif any(w in query_lower for w in ['senior', 'elder', 'older']):
            return 'seniors_count'
        return 'age_distribution'
    
    # Program query
    if intent == 'program_query':
        query_lower = entities.get('raw_query', '').lower()
        if 'popular' in query_lower or 'most' in query_lower:
            return 'popular_program'
        elif any(w in query_lower for w in ['show', 'display', 'view', 'full', 'complete']):
            return 'show_full_plan'
        return 'program_today'
    
    # Modification request
    if intent == 'modification_request':
        mod_type = entities.get('modification_type', '')
        if mod_type == 'menu':
            return 'modify_menu'
        elif mod_type in ['sermon', 'talk']:
            return 'modify_sermon'
        elif mod_type == 'schedule':
            return 'modify_schedule'
        return 'modify_program'
    
    return 'clarify'

def _generate_sql_for_intent(query_type: str, entities: Dict) -> Optional[str]:
    """Generate SQL query based on intent type and entities"""
    sql_map = {
        'attendance_today': "SELECT COUNT(*) FROM attendance WHERE date = date('now')",
        'attendance_week': "SELECT COUNT(*) FROM attendance WHERE date >= date('now', '-7 days')",
        'attendance_month': "SELECT COUNT(*) FROM attendance WHERE date >= date('now', '-30 days')",
        'avg_age': "SELECT AVG(age) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now')",
        'children_count': "SELECT COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') AND (u.age <= 17)",
        'seniors_count': "SELECT COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') AND u.age >= 65",
        'age_distribution': "SELECT CASE WHEN age <= 12 THEN '0-12' WHEN age <= 17 THEN '13-17' WHEN age <= 29 THEN '18-29' WHEN age <= 49 THEN '30-49' WHEN age <= 64 THEN '50-64' ELSE '65+' END as bucket, COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') GROUP BY bucket",
        'gender_dist': "SELECT gender, COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') GROUP BY gender",
        'program_today': "SELECT p.name FROM program_of_day pod JOIN programs p ON p.id = pod.program_id WHERE pod.date = date('now')",
        'popular_program': "SELECT p.name, COUNT(*) as cnt FROM attendance a JOIN programs p ON a.program_id = p.id GROUP BY p.id ORDER BY cnt DESC LIMIT 1",
        'newcomers': "SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-14 days')",
        'frequent_visitors': "SELECT username, COUNT(*) as visits FROM attendance WHERE date >= date('now', '-30 days') GROUP BY username HAVING visits >= 4",
    }
    return sql_map.get(query_type)

def parse_chat_query(query, conversation_history=None):
    """Parse chat query using sklearn NLU, fallback to regex"""
    # Use sklearn NLU if available
    if NLU_AVAILABLE:
        try:
            # Predict intent using sklearn NLU
            result = predict_intent(query)
            intent = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            # Extract entities (program names, dates, counts)
            entities = extract_entities(query)
            
            # Map NLU intents to handlers
            intent_map = {
                'attendance_query': 'stats_total',
                'demographic_query': 'stats_total',
                'statistics_query': 'stats_total',
                'suggestion_request': 'suggest_program',
                'modification_request': 'modify_menu',  # Default to menu, can be refined
                'program_query': 'program_today',
                'newcomer_query': 'newcomers',
                'frequent_visitor_query': 'frequent_visitors',
            }
            
            # Map intent to handler type
            handler_type = intent_map.get(intent, intent)
            
            # Special handling for modification requests
            if intent == 'modification_request':
                query_lower = query.lower()
                has_program_name = bool(entities.get('program_name'))
                wants_change = any(word in query_lower for word in ['change', 'switch', 'different', 'instead', 'another', 'swap'])
                if 'menu' in query_lower or 'food' in query_lower:
                    handler_type = 'modify_menu'
                elif 'program' in query_lower or (has_program_name and wants_change):
                    handler_type = 'change_program'
                elif 'sermon' in query_lower or 'talk' in query_lower:
                    handler_type = 'modify_sermon'
            
            return {
                'type': handler_type,
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'query': query,
                'raw_query': query,
                'sql': None  # Will be handled by execute_chat_query
            }
        except Exception as e:
            print(f"Error in NLU prediction: {e}")
            # Fallback to regex
            return _parse_chat_query_regex(query, conversation_history)
    
    # Fallback to regex-based parsing
    return _parse_chat_query_regex(query, conversation_history)

def _parse_chat_query_regex(query, conversation_history=None):
    """Original regex-based parsing (fallback)"""
    q_lower = query.lower().strip()
    
    # Greetings and conversational
    if re.search(r'^(hi|hello|hey|greetings|namaste|good morning|good afternoon|good evening)', q_lower):
        return {'type': 'greeting', 'sql': None}
    if re.search(r'(thank|thanks|appreciate|helpful)', q_lower):
        return {'type': 'thanks', 'sql': None}
    if re.search(r'(bye|goodbye|see you|farewell)', q_lower):
        return {'type': 'goodbye', 'sql': None}
    if re.search(r'what.*can.*you.*do|what.*help|capabilities|abilities', q_lower):
        return {'type': 'capabilities', 'sql': None}
    
    # Attendance queries (expanded patterns)
    attendance_patterns = [
        (r'(how many|count|number|total).*(people|attendees|visitors|guests|folks).*(today|now|present)', 'attendance_today'),
        (r'(attendance|people came|show up).*(today|now)', 'attendance_today'),
        (r'(how many|count|number).*(this week|weekly|past 7 days|last week)', 'attendance_week'),
        (r'(attendance|people).*(this week|weekly)', 'attendance_week'),
        (r'(how many|count|number).*(this month|monthly|past 30 days|last month)', 'attendance_month'),
        (r'(attendance|people).*(this month|monthly)', 'attendance_month'),
    ]
    for pattern, qtype in attendance_patterns:
        if re.search(pattern, q_lower):
            return {'type': qtype, 'sql': {
                'attendance_today': "SELECT COUNT(*) FROM attendance WHERE date = date('now')",
                'attendance_week': "SELECT COUNT(*) FROM attendance WHERE date >= date('now', '-7 days')",
                'attendance_month': "SELECT COUNT(*) FROM attendance WHERE date >= date('now', '-30 days')"
            }[qtype]}
    
    # Demographic queries (expanded)
    if re.search(r'(age|demographic|demographics|age group|age range)', q_lower):
        if re.search(r'(average|mean|typical|median)', q_lower):
            return {'type': 'avg_age', 'sql': "SELECT AVG(age) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now')"}
        if re.search(r'(how many|count|number).*(child|kid|teen|youth|young|minor)', q_lower):
            return {'type': 'children_count', 'sql': "SELECT COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') AND (u.age <= 17)"}
        if re.search(r'(how many|count|number).*(senior|elder|older|retired)', q_lower):
            return {'type': 'seniors_count', 'sql': "SELECT COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') AND u.age >= 65"}
        # Default age question
        return {'type': 'age_distribution', 'sql': "SELECT CASE WHEN age <= 12 THEN '0-12' WHEN age <= 17 THEN '13-17' WHEN age <= 29 THEN '18-29' WHEN age <= 49 THEN '30-49' WHEN age <= 64 THEN '50-64' ELSE '65+' END as bucket, COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') GROUP BY bucket"}
    
    # Gender distribution (expanded)
    if re.search(r'(gender|male|female|men|women|boys|girls|distribution|breakdown)', q_lower):
        return {'type': 'gender_dist', 'sql': "SELECT gender, COUNT(*) FROM users u JOIN attendance a ON u.username = a.username WHERE a.date = date('now') GROUP BY gender"}
    
    # Program queries (expanded)
    program_patterns = [
        (r'(what|which|today).*(program|schedule|plan|event|service|session)', 'program_today'),
        (r'(program|schedule).*(today|now|current)', 'program_today'),
        (r'(popular|most|best|favorite|favourite).*(program|service|event)', 'popular_program'),
        (r'(which program|what program).*(most|popular|attended)', 'popular_program'),
        (r'(show|display|see|view).*(full|complete|entire|whole).*(plan|program|schedule)', 'show_full_plan'),
        (r'(show|display|see|view).*(plan|program|schedule)', 'show_full_plan'),
    ]
    for pattern, qtype in program_patterns:
        if re.search(pattern, q_lower):
            if qtype == 'show_full_plan':
                return {'type': qtype, 'sql': None}
            return {'type': qtype, 'sql': {
                'program_today': "SELECT p.name FROM program_of_day pod JOIN programs p ON p.id = pod.program_id WHERE pod.date = date('now')",
                'popular_program': "SELECT p.name, COUNT(*) as cnt FROM attendance a JOIN programs p ON a.program_id = p.id GROUP BY p.id ORDER BY cnt DESC LIMIT 1"
            }[qtype]}
    
    # Newcomer queries (expanded)
    if re.search(r'(new|newcomer|recent|just signed|new signup|recently joined|new member)', q_lower):
        return {'type': 'newcomers', 'sql': "SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-14 days')"}
    
    # Frequent visitor queries (expanded)
    if re.search(r'(regular|frequent|often|repeat|loyal|consistent).*(visitor|attendee|member|people)', q_lower):
        return {'type': 'frequent_visitors', 'sql': "SELECT username, COUNT(*) as visits FROM attendance WHERE date >= date('now', '-30 days') GROUP BY username HAVING visits >= 4"}
    
    # Statistics and trends
    if re.search(r'(trend|pattern|change|growth|increase|decrease|compare|comparison)', q_lower):
        return {'type': 'trends', 'sql': None}
    
    # Program suggestions and modifications
    if re.search(r'(suggest|recommend|advice|what should|what would|recommendation|program.*for.*today|today.*program)', q_lower):
        return {'type': 'suggest_program', 'sql': None}
    
    # Program modification requests
    if re.search(r'(don\'t like|don\'t want|change|modify|edit|different|instead|instead of|prefer|rather)', q_lower):
        # Check if in context of program discussion
        if conversation_history:
            recent = ' '.join([h.get('content', '').lower() for h in conversation_history[-3:]])
            if any(word in recent for word in ['program', 'menu', 'sermon', 'schedule']):
                return {'type': 'modify_program', 'sql': None, 'query': query}
    
    # Menu modifications
    if re.search(r'(menu|food|prasad|dish|cook|chef).*(change|modify|different|instead|add|remove|include)', q_lower):
        return {'type': 'modify_menu', 'sql': None, 'query': query}
    
    # Sermon modifications
    if re.search(r'(sermon|talk|discourse|pandit|speech).*(change|modify|different|instead|focus|emphasize)', q_lower):
        return {'type': 'modify_sermon', 'sql': None, 'query': query}
    
    # Schedule modifications
    if re.search(r'(schedule|timing|time|when).*(change|modify|different|adjust)', q_lower):
        return {'type': 'modify_schedule', 'sql': None, 'query': query}
    
    # Apply program commands
    if re.search(r'(apply|use|set|go with|accept|confirm).*(program|this|that|it)', q_lower):
        return {'type': 'apply_program', 'sql': None}
    
    # Default: try to understand intent or ask for clarification
    return {'type': 'clarify', 'sql': None, 'query': query}

def execute_chat_query(query_obj, session_obj=None):
    # Conversational responses
    if query_obj['type'] == 'greeting':
        return {
            'answer': 'Namaste! 🙏 I\'m your AI assistant for the mandir. I can help you with:\n\n• **Statistics** - Ask about attendance, demographics, trends\n• **Program Planning** - I can suggest and customize today\'s program based on who\'s attending\n• **Modifications** - Just tell me what you want to change and I\'ll adjust it!\n\nTry asking: "Suggest a program for today" or "How many people came today?"',
            'type': 'conversational'
        }
    if query_obj['type'] == 'thanks':
        return {
            'answer': 'You\'re welcome! Feel free to ask if you need anything else.',
            'type': 'conversational'
        }
    if query_obj['type'] == 'goodbye':
        return {
            'answer': 'Namaste! Have a blessed day.',
            'type': 'conversational'
        }
    if query_obj['type'] == 'capabilities':
        return {
            'answer': 'I can help you with:\n• Attendance statistics (today, this week, this month)\n• Demographics (age breakdowns, gender distribution)\n• Program information (today\'s program, popular programs)\n• Newcomer and frequent visitor insights\n• Trends and patterns\n\nJust ask naturally - I understand various ways of phrasing questions!',
            'type': 'conversational'
        }
    
    # Route to new NLU-based handlers
    if query_obj['type'] == 'stats_total':
        return get_today_stats()
    
    if query_obj['type'] == 'suggest_program':
        snap = compute_today_snapshot()
        suggestion = suggest_program_from_snapshot(snap)
        if suggestion:
            # Store program context in session
            if session_obj:
                plan = generate_program_plan(snap, suggestion)
                session_obj['current_program_context'] = {
                    'program_id': suggestion['program_id'],
                    'program_name': suggestion['program_name'],
                    'plan': plan
                }
            return {
                'answer': f'Based on today\'s demographics ({snap.get("total", 0)} attendees), I recommend:\n\n**Program:** {suggestion["program_name"]}\n\n**Why?**\n' + '\n'.join([f'• {r}' for r in suggestion["rationale"][:3]]) + f'\n\n**Menu:** {", ".join(suggestion.get("menu_tags", ["standard"])[:3])}\n**Focus:** {", ".join(suggestion.get("talk_tags", ["general"])[:3])}\n\nWould you like me to show the full schedule and outline?',
                'type': 'program_suggestion',
                'program_data': suggestion,
                'plan': generate_program_plan(snap, suggestion) if suggestion else None
            }
        else:
            return {
                'answer': 'I don\'t have enough data yet to make a recommendation. Please check back after some attendance is recorded.',
                'type': 'suggestion'
            }
    
    if query_obj['type'] == 'modify_menu':
        # Extract adjustment type from query
        query_lower = query_obj.get('raw_query', '').lower()
        adjustment_type = 'kid_friendly'  # default
        if 'kid' in query_lower or 'child' in query_lower:
            adjustment_type = 'kid_friendly'
        elif 'senior' in query_lower or 'elder' in query_lower:
            adjustment_type = 'senior_friendly'
        elif 'traditional' in query_lower:
            adjustment_type = 'traditional'
        
        return adjust_menu(None, adjustment_type, session_obj)
    
    if query_obj['type'] == 'change_program':
        # Extract program name from entities or query
        entities = query_obj.get('entities', {})
        program_name = entities.get('program_name', '')
        if not program_name:
            # Try to extract from raw query
            query_lower = query_obj.get('raw_query', '').lower()
            # Look for program keywords
            if 'sunday' in query_lower:
                program_name = 'Sunday'
            elif 'monday' in query_lower:
                program_name = 'Monday'
            elif 'tuesday' in query_lower:
                program_name = 'Tuesday'
            elif 'navgraha' in query_lower:
                program_name = 'Navgraha'
            elif 'purnima' in query_lower:
                program_name = 'Purnima'
        
        if program_name:
            return change_program(program_name)
        else:
            return {'answer': 'Please specify which program you want to change to (e.g., "Sunday", "Monday", "Navgraha").', 'type': 'error'}
    
    if query_obj['type'] == 'trends':
        # Calculate some trend data
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        today = datetime.now().date().isoformat()
        week_ago = (datetime.now() - timedelta(days=7)).date().isoformat()
        c.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
        today_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (week_ago,))
        week_ago_count = c.fetchone()[0]
        change = today_count - week_ago_count
        trend_text = f"increased by {change}" if change > 0 else f"decreased by {abs(change)}" if change < 0 else "remained the same"
        conn.close()
        return {
            'answer': f'Attendance trends:\n• Today: {today_count} people\n• Same day last week: {week_ago_count} people\n• Change: {trend_text}\n\nWould you like more detailed analysis?',
            'type': 'trends'
        }
    if query_obj['type'] == 'suggest_program' or query_obj['type'] == 'suggestion':
        snap = compute_today_snapshot()
        suggestion = suggest_program_from_snapshot(snap)
        if suggestion:
            # Store program context in session
            if session_obj:
                plan = generate_program_plan(snap, suggestion)
                session_obj['current_program_context'] = {
                    'program_id': suggestion['program_id'],
                    'program_name': suggestion['program_name'],
                    'plan': plan
                }
            return {
                'answer': f'Based on today\'s demographics ({snap.get("total", 0)} attendees), I recommend:\n\n**Program:** {suggestion["program_name"]}\n\n**Why?**\n' + '\n'.join([f'• {r}' for r in suggestion["rationale"][:3]]) + f'\n\n**Menu:** {", ".join(suggestion.get("menu_tags", ["standard"])[:3])}\n**Focus:** {", ".join(suggestion.get("talk_tags", ["general"])[:3])}\n\nWould you like me to show the full schedule and outline? Or would you like to modify anything?',
                'type': 'program_suggestion',
                'program_data': suggestion,
                'plan': generate_program_plan(snap, suggestion) if suggestion else None
            }
        else:
            return {
                'answer': 'I don\'t have enough data yet to make a recommendation. Please check back after some attendance is recorded.',
                'type': 'suggestion'
            }
    
    # Program modification handlers
    if query_obj['type'] == 'modify_program' and session_obj:
        context = session_obj.get('current_program_context')
        if not context:
            return {
                'answer': 'I don\'t have a program in context right now. Try asking me to suggest a program first, then I can help modify it!',
                'type': 'error'
            }
        query = query_obj.get('query', '').lower()
        plan = context.get('plan', {})
        
        # Try to parse direct modifications
        # Check if they're specifying menu changes
        if re.search(r'(menu|food|prasad|dish|cook)', query):
            # Delegate to menu modification
            query_obj['type'] = 'modify_menu'
            return execute_chat_query(query_obj, session_obj)
        
        # Check if they're specifying sermon changes
        if re.search(r'(sermon|talk|discourse|pandit|speech)', query):
            # Delegate to sermon modification
            query_obj['type'] = 'modify_sermon'
            return execute_chat_query(query_obj, session_obj)
        
        # Check if they want a different program entirely
        if re.search(r'(different program|other program|change program|switch)', query):
            # Suggest a new program
            snap = compute_today_snapshot()
            suggestion = suggest_program_from_snapshot(snap)
            if suggestion:
                plan = generate_program_plan(snap, suggestion)
                session_obj['current_program_context'] = {
                    'program_id': suggestion['program_id'],
                    'program_name': suggestion['program_name'],
                    'plan': plan
                }
                return {
                    'answer': f'Got it! Let me suggest a different program:\n\n**Program:** {suggestion["program_name"]}\n\n**Why this one?**\n' + '\n'.join([f'• {r}' for r in suggestion["rationale"][:2]]) + '\n\nWould you like to see the full plan or modify anything?',
                    'type': 'program_suggestion',
                    'program_data': suggestion,
                    'plan': plan
                }
        
        # General modification request
        changes = []
        if re.search(r'(menu|food|prasad)', query):
            changes.append('menu')
        if re.search(r'(sermon|talk|discourse)', query):
            changes.append('sermon')
        if re.search(r'(schedule|timing)', query):
            changes.append('schedule')
        if re.search(r'(kids|children|youth)', query):
            changes.append('kids_focus')
        if re.search(r'(senior|elder|older)', query):
            changes.append('senior_focus')
        
        return {
            'answer': f'Understood! I can help modify the {", ".join(changes) if changes else "program"}. What specifically would you like to change? For example:\n• "Make the menu more kid-friendly"\n• "Focus the sermon on newcomers"\n• "Add more interactive elements"\n• Or tell me exactly what you want!',
            'type': 'modify_program',
            'context': context
        }
    
    if query_obj['type'] == 'modify_menu' and session_obj:
        context = session_obj.get('current_program_context')
        query = query_obj.get('query', '').lower()
        # Extract menu preferences with better parsing
        new_menu = []
        menu_keywords = {
            'mild': ['mild', 'soft', 'gentle', 'bland'],
            'kid': ['kid', 'children', 'young', 'child-friendly', 'kids'],
            'spicy': ['spicy', 'hot', 'strong', 'pungent'],
            'vegetarian': ['vegetarian', 'veg'],
            'light': ['light', 'simple', 'easy'],
            'sweet': ['sweet', 'dessert', 'sheera', 'halwa'],
            'traditional': ['traditional', 'classic', 'regular']
        }
        
        # Check for specific dishes mentioned
        specific_dishes = {
            'khichdi': ['khichdi', 'kichdi'],
            'pulao': ['pulao', 'pilaf'],
            'biryani': ['biryani'],
            'puri': ['puri'],
            'bhaji': ['bhaji'],
            'upma': ['upma'],
            'sheera': ['sheera'],
            'raita': ['raita']
        }
        
        found_specific = []
        for dish, words in specific_dishes.items():
            if any(w in query for w in words):
                found_specific.append(dish.title())
        
        if found_specific:
            new_menu = found_specific
        else:
            for key, words in menu_keywords.items():
                if any(w in query for w in words):
                    if key == 'mild':
                        new_menu.extend(['Khichdi', 'Cucumber raita'])
                    elif key == 'kid':
                        new_menu.extend(['Vegetable pulao (mild)', 'Fruit cups'])
                    elif key == 'spicy':
                        new_menu.extend(['Biryani', 'Spicy dal'])
                    elif key == 'light':
                        new_menu.extend(['Upma', 'Banana sheera'])
                    elif key == 'sweet':
                        new_menu.extend(['Banana sheera', 'Halwa'])
                    elif key == 'traditional':
                        new_menu.extend(['Puri-bhaji', 'Kadhi-chawal'])
        
        if not new_menu:
            new_menu = ['Puri-bhaji', 'Kadhi-chawal']  # default
        
        if context:
            context['plan']['menu'] = new_menu
            session_obj['current_program_context'] = context
        
        return {
            'answer': f'Perfect! I\'ve updated the menu to: {", ".join(new_menu)}. The chefs can prepare these dishes. Want to adjust anything else?',
            'type': 'menu_updated',
            'menu': new_menu
        }
    
    if query_obj['type'] == 'modify_sermon' and session_obj:
        context = session_obj.get('current_program_context')
        query = query_obj.get('query', '').lower()
        new_outline = []
        
        if 'newcomer' in query or 'new' in query:
            new_outline.append('Warm welcome and introduction to traditions')
            new_outline.append('Basic concepts explained simply')
        elif 'regular' in query or 'deep' in query:
            new_outline.append('Advanced discussion for experienced attendees')
            new_outline.append('Philosophical exploration')
        elif 'kids' in query or 'children' in query:
            new_outline.append('Story-based teaching for children')
            new_outline.append('Interactive Q&A')
        else:
            new_outline = ['Adapted message based on your request', 'Community-focused reflection']
        
        if context:
            context['plan']['sermon_outline'] = new_outline
            session_obj['current_program_context'] = context
        
        return {
            'answer': f'Perfect! I\'ve updated the sermon outline:\n' + '\n'.join([f'• {o}' for o in new_outline]) + '\n\nAnything else you\'d like to adjust?',
            'type': 'sermon_updated',
            'outline': new_outline
        }
    
    if query_obj['type'] == 'apply_program' and session_obj:
        context = session_obj.get('current_program_context')
        if not context:
            return {
                'answer': 'I don\'t have a program ready to apply. Let me suggest one first!',
                'type': 'error'
            }
        program_id = context.get('program_id')
        today = datetime.now().date().isoformat()
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        c.execute("INSERT INTO program_of_day(date, program_id) VALUES(?, ?) ON CONFLICT(date) DO UPDATE SET program_id=excluded.program_id", (today, program_id))
        conn.commit()
        conn.close()
        return {
            'answer': f'Done! I\'ve applied "{context.get("program_name")}" as today\'s program. All QR scans will now be associated with this program. The plan is ready for the chefs and pandits! 🙏',
            'type': 'program_applied',
            'program_id': program_id
        }
    if query_obj['type'] == 'clarify':
        # Try to give helpful suggestions based on keywords found
        query = query_obj.get('query', '').lower()
        suggestions = []
        if any(w in query for w in ['people', 'attend', 'come', 'visit']):
            suggestions.append('Try: "How many people came today?"')
        if any(w in query for w in ['age', 'old', 'young']):
            suggestions.append('Try: "What\'s the average age?" or "How many children are here?"')
        if any(w in query for w in ['program', 'event', 'schedule']):
            suggestions.append('Try: "What\'s today\'s program?"')
        if not suggestions:
            suggestions = ['Try: "How many people came today?"', 'Try: "What\'s the average age?"', 'Try: "What\'s today\'s program?"']
        return {
            'answer': f'I\'m not sure I understood that. Here are some things I can help with:\n\n' + '\n'.join(suggestions) + '\n\nOr ask me "What can you do?" to see all my capabilities.',
            'type': 'clarify'
        }
    if query_obj['type'] == 'help':
        return {
            'answer': 'I can answer questions about:\n- Attendance (today, this week, this month)\n- Demographics (age, gender, newcomers, frequent visitors)\n- Programs (today\'s program, popular programs)\n- Statistics and trends\n\nTry asking: "How many people came today?" or "What\'s the average age?"',
            'type': 'help'
        }
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    try:
        c.execute(query_obj['sql'])
        result = c.fetchone()
        conn.close()
        if query_obj['type'] == 'attendance_today':
            count = result[0] or 0
            if count == 0:
                return {'answer': 'No one has checked in today yet. The mandir is ready for visitors!', 'type': 'attendance'}
            return {'answer': f'Great question! Today we have {count} {"person" if count == 1 else "people"} who have checked in so far. The community is gathering! 🙏', 'type': 'attendance'}
        elif query_obj['type'] == 'attendance_week':
            count = result[0] or 0
            return {'answer': f'This week has been wonderful - {count} {"person" if count == 1 else "people"} have attended. The mandir community is thriving!', 'type': 'attendance'}
        elif query_obj['type'] == 'attendance_month':
            count = result[0] or 0
            return {'answer': f'Looking at this month, we\'ve had {count} {"person" if count == 1 else "people"} attend. A strong and growing community!', 'type': 'attendance'}
        elif query_obj['type'] == 'avg_age':
            avg = round(result[0] or 0, 1)
            if avg == 0:
                return {'answer': 'I don\'t have age data for today\'s attendees yet. Check back after some people check in!', 'type': 'demographic'}
            return {'answer': f'The average age of today\'s attendees is {avg} years. This helps us understand our community better!', 'type': 'demographic'}
        elif query_obj['type'] == 'children_count':
            count = result[0] or 0
            if count == 0:
                return {'answer': 'No children or teens have checked in today yet.', 'type': 'demographic'}
            return {'answer': f'We have {count} {"child" if count == 1 else "children/teens"} attending today. It\'s wonderful to see young people connecting with the mandir!', 'type': 'demographic'}
        elif query_obj['type'] == 'seniors_count':
            count = result[0] or 0
            if count == 0:
                return {'answer': 'No seniors have checked in today yet.', 'type': 'demographic'}
            return {'answer': f'We have {count} {"senior" if count == 1 else "seniors"} attending today. Their wisdom and presence enrich our community!', 'type': 'demographic'}
        elif query_obj['type'] == 'gender_dist':
            c = conn.cursor()
            c.execute(query_obj['sql'])
            rows = c.fetchall()
            conn.close()
            if not rows:
                return {'answer': 'No attendance data for today yet.', 'type': 'demographic'}
            dist = ', '.join([f'{g}: {c}' for g, c in rows])
            return {'answer': f'Here\'s the gender distribution for today: {dist}. Diversity makes our community stronger!', 'type': 'demographic'}
        elif query_obj['type'] == 'age_distribution':
            c = conn.cursor()
            c.execute(query_obj['sql'])
            rows = c.fetchall()
            conn.close()
            if rows:
                dist = ', '.join([f'{b}: {c} people' for b, c in rows])
                return {'answer': f'Age groups attending today: {dist}. We serve all generations!', 'type': 'demographic'}
            else:
                return {'answer': 'No attendance data for today yet.', 'type': 'demographic'}
        elif query_obj['type'] == 'program_today':
            if result:
                return {'answer': f'Today\'s program is: {result[0]}. It\'s been selected to best serve our community!', 'type': 'program'}
            else:
                return {'answer': 'No program has been explicitly set for today yet. The suggestion engine can recommend one based on who\'s attending!', 'type': 'program'}
        elif query_obj['type'] == 'popular_program':
            if result:
                return {'answer': f'The most popular program is {result[0]} with {result[1]} {"visit" if result[1] == 1 else "visits"}! It clearly resonates with our community.', 'type': 'program'}
            else:
                return {'answer': 'We don\'t have enough attendance data yet to identify the most popular program. Check back as more people attend!', 'type': 'program'}
        elif query_obj['type'] == 'newcomers':
            count = result[0] or 0
            if count == 0:
                return {'answer': 'No new signups in the last 14 days. We always welcome new members to our community!', 'type': 'demographic'}
            return {'answer': f'We\'ve had {count} {"new member" if count == 1 else "new members"} join us in the last 14 days. How wonderful to see our community growing! 🌱', 'type': 'demographic'}
        elif query_obj['type'] == 'frequent_visitors':
            c = conn.cursor()
            c.execute(query_obj['sql'])
            rows = c.fetchall()
            conn.close()
            count = len(rows)
            if count == 0:
                return {'answer': 'We don\'t have any frequent visitors (4+ visits in last 30 days) yet. Regular attendance builds strong community bonds!', 'type': 'demographic'}
            return {'answer': f'We have {count} {"frequent visitor" if count == 1 else "frequent visitors"} - people who\'ve attended 4 or more times in the last month. Their dedication is inspiring! 🙏', 'type': 'demographic'}
        else:
            return {'answer': f'Result: {result[0] if result else "No data"}', 'type': 'general'}
    except Exception as e:
        conn.close()
        return {'answer': 'I apologize, but I encountered an issue processing that question. Could you try rephrasing it? For example, ask "How many people came today?" or "What\'s the average age?"', 'type': 'error'}

@app.route('/chat', methods=['POST'])
@admin_required
def chat():
    require_valid_csrf()
    query = request.form.get('query', '').strip()
    if not query:
        return {'answer': 'Please ask a question.', 'type': 'error'}, 400
    
    # Maintain conversation context
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'current_program_context' not in session:
        session['current_program_context'] = None
    
    # Store query in history (keep last 5)
    session['chat_history'].append({'role': 'user', 'content': query})
    if len(session['chat_history']) > 10:
        session['chat_history'] = session['chat_history'][-10:]
    
    query_obj = parse_chat_query(query, session.get('chat_history'))
    response = execute_chat_query(query_obj, session)
    
    # Store response in history
    session['chat_history'].append({'role': 'assistant', 'content': response.get('answer', '')})
    
    return response, 200

@app.route('/admin')
@admin_required
def admin_page():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    users = get_all_users()
    total_users = len(users)
    if total_users > 0:
        avg_age = round(sum([int(u[2]) for u in users]) / total_users, 1)
        gender_counts = {}
        for u in users:
            gender_counts[u[3]] = gender_counts.get(u[3], 0) + 1
    else:
        avg_age = 0
        gender_counts = {}

    # --- Analytics ---
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    today = datetime.now().date()
    days = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]  # oldest to newest
    signup_counts = []
    login_counts = []
    qr_counts = []
    for day in days:
        next_day = (datetime.fromisoformat(day) + timedelta(days=1)).isoformat()
        c.execute("SELECT COUNT(*) FROM users WHERE created_at >= ? AND created_at < ?", (day, next_day))
        signup_counts.append(c.fetchone()[0])
        c.execute("SELECT COUNT(*) FROM checkins WHERE action = 'login' AND timestamp >= ? AND timestamp < ?", (day, next_day))
        login_counts.append(c.fetchone()[0])
        c.execute("SELECT COUNT(*) FROM checkins WHERE action = 'qr_scan' AND timestamp >= ? AND timestamp < ?", (day, next_day))
        qr_counts.append(c.fetchone()[0])
    conn.close()

    return render_template('admin.html', users=users, total=total_users, avg_age=avg_age, gender_counts=gender_counts,
        days=days, signup_counts=signup_counts, login_counts=login_counts, qr_counts=qr_counts)

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        require_valid_csrf()
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and bcrypt.checkpw(password.encode('utf-8'), ADMIN_PASSWORD_HASH):
            session.clear()
            session.permanent = True
            session['admin'] = True
            return redirect(url_for('admin_page'))
        else:
            return "Invalid admin credentials", 403
    return render_template('admin_login.html')

@app.route('/admin-logout', methods=['POST'])
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/edit/<username>', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(username):
    user = get_user_by_username_full(username)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_page'))
    if request.method == 'POST':
        require_valid_csrf()
        new_name = request.form['name']
        new_age = request.form['age']
        new_gender = request.form['gender']
        if new_gender not in ALLOWED_GENDERS:
            return "Invalid gender selection.", 400
        new_email = request.form['email']
        new_password = request.form.get('password', '')
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        if new_password.strip():
            hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            c.execute("UPDATE users SET name=?, age=?, gender=?, email=?, password=? WHERE username=?",
                      (new_name, new_age, new_gender, new_email, hashed_pw, username))
        else:
            c.execute("UPDATE users SET name=?, age=?, gender=?, email=? WHERE username=?",
                      (new_name, new_age, new_gender, new_email, username))
        conn.commit()
        conn.close()
        flash('User updated successfully.', 'success')
        return redirect(url_for('admin_page'))
    return render_template('edit_user.html', user=user)

@app.route('/admin/delete/<username>', methods=['POST'])
@admin_required
def admin_delete_user(username):
    require_valid_csrf()
    user = get_user_by_username_full(username)
    if user:
        qr_path = user[5]
        if qr_path and os.path.exists(qr_path):
            os.remove(qr_path)
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
        conn.close()
        flash('User deleted successfully.', 'success')
    else:
        flash('User not found.', 'danger')
    return redirect(url_for('admin_page'))

@app.route('/admin/purge-simulated', methods=['POST'])
@admin_required
def admin_purge_simulated():
    require_valid_csrf()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    # Identify simulated users by username prefix or demo email domain
    c.execute("SELECT qr_path, username FROM users WHERE username LIKE 'sim_%' OR email LIKE '%@example.com'")
    rows = c.fetchall()
    paths = [row[0] for row in rows if row and row[0]]
    usernames = [row[1] for row in rows if row and row[1]]
    if usernames:
        qmarks = ','.join(['?']*len(usernames))
        # Delete dependent rows first
        c.execute(f"DELETE FROM attendance WHERE username IN ({qmarks})", (*usernames,))
        c.execute(f"DELETE FROM checkins WHERE username IN ({qmarks})", (*usernames,))
        # Then users
        c.execute(f"DELETE FROM users WHERE username IN ({qmarks})", (*usernames,))
    conn.commit()
    conn.close()
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    flash('Purged simulated users and related data.', 'success')
    return redirect(url_for('admin_page'))

@app.route('/admin/reminders')
@admin_required
def admin_reminders():
    days = int(request.args.get('days', 30))
    inactive = get_inactive_users(days)
    return render_template('reminders.html', inactive_users=inactive, days_threshold=days)

@app.route('/admin/send-reminder/<username>', methods=['POST'])
@admin_required
def admin_send_reminder(username):
    require_valid_csrf()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT name, email FROM users WHERE username = ? AND email_verified = 1", (username,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        flash('User not found or email not verified.', 'danger')
        return redirect(url_for('admin_reminders'))
    
    name, email = user
    # Calculate days since last visit
    c.execute("SELECT MAX(date) FROM attendance WHERE username = ?", (username,))
    last_visit_str = c.fetchone()[0]
    conn.close()
    
    if last_visit_str and last_visit_str != 'Never':
        last_visit = datetime.fromisoformat(last_visit_str).date()
        days_since = (datetime.now().date() - last_visit).days
    else:
        days_since = 999  # Never visited
    
    send_reminder_email(email, name, days_since)
    flash(f'Reminder sent to {name} ({email})', 'success')
    return redirect(url_for('admin_reminders'))

@app.route('/admin/send-reminders-bulk', methods=['POST'])
@admin_required
def admin_send_reminders_bulk():
    require_valid_csrf()
    days = int(request.form.get('days', 30))
    inactive = get_inactive_users(days)
    sent = 0
    for user in inactive:
        if user['email']:
            last_visit_str = user['last_visit']
            if last_visit_str and last_visit_str != 'Never':
                last_visit = datetime.fromisoformat(last_visit_str).date()
                days_since = (datetime.now().date() - last_visit).days
            else:
                days_since = 999
            send_reminder_email(user['email'], user['name'], days_since)
            sent += 1
    flash(f'Sent {sent} reminder emails to inactive users.', 'success')
    return redirect(url_for('admin_reminders'))

@app.route('/admin/feedback', methods=['POST'])
@admin_required
def admin_feedback():
    """Admin feedback endpoint for program suggestions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400
        
        program_id = data.get('program_id')
        label = data.get('label')
        
        if program_id is None or label is None:
            return jsonify({'status': 'error', 'message': 'program_id and label required'}), 400
        
        program_id = int(program_id)
        label = int(label)
        
        # Validate label
        if label not in [1, -1]:
            return jsonify({'status': 'error', 'message': 'label must be 1 or -1'}), 400
        
        # Get snapshot
        snapshot = compute_today_snapshot()
        
        # Get admin username (use 'admin' as default since admin doesn't have username in session)
        admin_username = session.get('admin_username', 'admin')
        
        # Insert feedback into database
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        today = datetime.now().isoformat()
        snapshot_json = json.dumps(snapshot)
        
        c.execute(
            "INSERT INTO program_feedback(date, program_id, username, feedback, snapshot_json) VALUES (?,?,?,?,?)",
            (today, program_id, admin_username, label, snapshot_json)
        )
        conn.commit()
        conn.close()
        
        # Optional: trigger online update if program_model is available
        # Only update on positive feedback (label=1)
        if PROGRAM_MODEL_AVAILABLE and label == 1:
            try:
                from program_model import online_update
                online_update(snapshot, program_id, learning_rate=0.01)
            except Exception as e:
                print(f"Warning: Could not perform online update: {e}")
        
        return jsonify({'status': 'ok', 'message': 'Feedback recorded successfully'})
    
    except Exception as e:
        print(f"Error in admin_feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def get_setting(key: str, default: str = '') -> str:
    """Get a setting from the database"""
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default

def set_setting(key: str, value: str):
    """Set a setting in the database"""
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    c.execute("INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?, ?, ?)", 
             (key, value, now))
    conn.commit()
    conn.close()

@app.route('/admin/ai-settings', methods=['GET', 'POST'])
@admin_required
def admin_ai_settings():
    """Admin page for AI/LLM configuration"""
    if request.method == 'POST':
        require_valid_csrf()
        # Update LLM settings
        llm_enabled = request.form.get('llm_enabled') == 'on'
        llm_api_key = request.form.get('llm_api_key', '').strip()
        llm_provider = request.form.get('llm_provider', 'openai')
        
        # Store in database
        set_setting('llm_enabled', 'true' if llm_enabled else 'false')
        if llm_api_key:
            set_setting('llm_api_key', llm_api_key)
        set_setting('llm_provider', llm_provider)
        
        flash('LLM settings updated successfully.', 'success')
        return redirect(url_for('admin_ai_settings'))
    
    # Get settings from database (default to false/off)
    llm_enabled = get_setting('llm_enabled', 'false').lower() == 'true'
    llm_api_key = get_setting('llm_api_key', '')
    llm_provider = get_setting('llm_provider', 'openai')
    
    # Fallback to environment variables if not in database (for backward compatibility)
    if not llm_api_key:
        llm_api_key = os.environ.get('LLM_API_KEY', '')
    if not llm_enabled and os.environ.get('LLM_ENABLED', 'false').lower() == 'true':
        llm_enabled = True
    
    ai_engine_status = 'Available' if AI_ENGINE_AVAILABLE else 'Not Available (using fallback)'
    
    # Get feedback stats if available
    feedback_stats = {}
    if AI_ENGINE_AVAILABLE:
        feedback = get_feedback()
        feedback_stats = feedback.get_feedback_stats()
    
    return render_template('ai_settings.html', 
                         llm_enabled=llm_enabled,
                         llm_api_key=llm_api_key,
                         llm_provider=llm_provider,
                         llm_has_key=bool(llm_api_key),
                         ai_engine_status=ai_engine_status,
                         feedback_stats=feedback_stats)

@app.route('/admin/feedback-stats')
@admin_required
def admin_feedback_stats():
    """View feedback statistics"""
    if not AI_ENGINE_AVAILABLE:
        flash('AI engine not available.', 'warning')
        return redirect(url_for('admin_page'))
    
    feedback = get_feedback()
    program_id = request.args.get('program_id', type=int)
    stats = feedback.get_feedback_stats(program_id)
    
    # Get program names
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM programs")
    programs = c.fetchall()
    conn.close()
    
    return render_template('feedback_stats.html', 
                         feedback_stats=stats,
                         programs=programs,
                         selected_program_id=program_id)

@app.route('/admin/metrics')
@admin_required
def admin_metrics():
    """Admin evaluation dashboard for feedback metrics"""
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()

    # Total feedback
    c.execute("SELECT COUNT(*) FROM program_feedback")
    total_feedback = c.fetchone()[0] or 0

    # Positive / negative counts
    c.execute("SELECT COUNT(*) FROM program_feedback WHERE feedback = 1")
    positive_feedback = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM program_feedback WHERE feedback = -1")
    negative_feedback = c.fetchone()[0] or 0

    accuracy = (positive_feedback / total_feedback * 100) if total_feedback else None

    # Program frequency histogram
    c.execute("""
        SELECT COALESCE(p.name, 'Program ' || f.program_id) as program_name,
               COUNT(*) as cnt
        FROM program_feedback f
        LEFT JOIN programs p ON f.program_id = p.id
        GROUP BY f.program_id
        ORDER BY cnt DESC
    """)
    program_rows = c.fetchall()
    program_labels = [row[0] for row in program_rows]
    program_counts = [row[1] for row in program_rows]

    # Feedback trend by date
    c.execute("""
        SELECT strftime('%Y-%m-%d', date) as day,
               SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) as positive,
               SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) as negative
        FROM program_feedback
        GROUP BY day
        ORDER BY day ASC
    """)
    trend_rows = c.fetchall()
    conn.close()

    trend_dates = [row[0] for row in trend_rows]
    trend_positive = [row[1] for row in trend_rows]
    trend_negative = [row[2] for row in trend_rows]

    return render_template(
        'admin_metrics.html',
        total_feedback=total_feedback,
        positive_feedback=positive_feedback,
        negative_feedback=negative_feedback,
        accuracy=accuracy,
        program_labels=program_labels,
        program_counts=program_counts,
        trend_dates=trend_dates,
        trend_positive=trend_positive,
        trend_negative=trend_negative
    )

@app.route('/admin/retrain-model', methods=['POST'])
@admin_required
def admin_retrain_model():
    """Retrain the program recommendation model using feedback data"""
    try:
        from retrain import retrain_model, check_model_status
        
        # Check status first
        status = check_model_status()
        
        # Retrain the model
        result = retrain_model(min_samples=3)  # Lower threshold for admin-initiated retraining
        
        if result['success']:
            flash(f"Model retrained successfully! Used {result['samples']} feedback samples across {result['programs']} programs.", 'success')
        else:
            flash(f"Retraining failed: {result['message']}", 'danger')
        
        return jsonify(result)
    
    except ImportError:
        return jsonify({
            'success': False,
            'message': 'retrain.py module not available'
        }), 500
    except Exception as e:
        print(f"Error in admin_retrain_model: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/admin/model-status')
@admin_required
def admin_model_status():
    """Get model status information"""
    try:
        from retrain import check_model_status
        status = check_model_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

def _random_name():
    first = random.choice(['Khushi','Dhruv','Aaryan','Eshaan','Ayush','Rishi','Jaineel','Riddhi','Diya','Deetya','Mahek','Shreeya','Aarna','Sana'])
    last = random.choice(['Sharma','Patel','Rao','Iyer','Gupta','Khan','Singh','Joshi','Mehta','Desai'])
    return f"{first} {last}"

def _random_username():
    return 'sim_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def _pick_gender():
    # Weighted random roughly balanced
    options = ['Male','Female','Non-binary','Other','Prefer not to say']
    weights = [0.45,0.45,0.04,0.03,0.03]
    return random.choices(options, weights=weights, k=1)[0]

@app.route('/admin/simulate', methods=['GET','POST'])
@admin_required
def admin_simulate():
    if request.method == 'POST':
        require_valid_csrf()
        try:
            count = int(request.form.get('count','0'))
            age_min = int(request.form.get('age_min','5'))
            age_max = int(request.form.get('age_max','80'))
            gender_sel = request.form.get('gender','Random')
            newcomer_percent = int(request.form.get('newcomer_percent','20'))  # 0-100
            frequent_percent = int(request.form.get('frequent_percent','50'))  # 0-100
            newcomer_percent = max(0, min(100, newcomer_percent))
            frequent_percent = max(0, min(100, frequent_percent))
            if count <= 0 or age_min < 0 or age_max < age_min:
                flash('Invalid input values.', 'danger')
                return redirect(url_for('admin_simulate'))
        except Exception:
            flash('Invalid input values.', 'danger')
            return redirect(url_for('admin_simulate'))

        today_dt = datetime.now()
        date_str = today_dt.date().isoformat()
        conn = sqlite3.connect('temple.db')
        c = conn.cursor()
        created = 0
        for i in range(count):
            age = random.randint(age_min, age_max)
            gender = _pick_gender() if gender_sel == 'Random' else gender_sel
            username = _random_username()
            name = _random_name()

            # newcomer/frequent flags
            is_newcomer = (random.randint(0,99) < newcomer_percent)
            is_frequent = (random.randint(0,99) < frequent_percent)

            # Backdate account creation: newcomers ~ last 14 days; others 15-180 days
            if is_newcomer:
                days_ago = random.randint(0, 14)
            else:
                days_ago = random.randint(15, 180)
            created_at = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec='seconds')

            # Create user
            hashed_pw = bcrypt.hashpw('TempPass123!'.encode('utf-8'), bcrypt.gensalt())
            qr_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{username}.png")
            qr_img = qrcode.make(f"{username}-{age}-{gender}")
            qr_img.save(qr_path)
            try:
                c.execute("INSERT INTO users (name, username, password, age, gender, qr_path, created_at, email, email_verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                          (name, username, hashed_pw, age, gender, qr_path, created_at, f"{username}@example.com"))
                created += 1
            except Exception:
                pass

            # Seed past attendance in last 30 days
            visits = 0
            if is_frequent:
                visits = random.randint(4, 8)
            elif not is_newcomer:
                visits = random.randint(1, 4)
            else:
                visits = random.randint(0, 2)

            used_days = set()
            for _ in range(visits):
                d = today_dt.date() - timedelta(days=random.randint(1, 30))
                if d in used_days:
                    continue
                used_days.add(d)
                date_past = d.isoformat()
                time_past = f"{random.randint(8,20):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
                c.execute("INSERT INTO attendance (username, date, time, program_id) VALUES (?, ?, ?, NULL)", (username, date_past, time_past))
                c.execute("INSERT INTO checkins (username, action, timestamp) VALUES (?, 'qr_scan', ?)", (username, f"{date_past}T{time_past}"))

            # Today attendance
            time_str = datetime.now().strftime('%H:%M:%S')
            program_id = get_program_of_day_id()
            if program_id is not None:
                c.execute("INSERT INTO attendance (username, date, time, program_id) VALUES (?, ?, ?, ?)", (username, date_str, time_str, program_id))
            else:
                c.execute("INSERT INTO attendance (username, date, time, program_id) VALUES (?, ?, ?, NULL)", (username, date_str, time_str))
            c.execute("INSERT INTO checkins (username, action, timestamp) VALUES (?, 'qr_scan', ?)", (username, datetime.now().isoformat(timespec='seconds')))
        conn.commit()
        conn.close()
        flash(f'Simulated {count} attendees (created {created} new users).', 'success')
        return redirect(url_for('admin_page'))
    return render_template('simulate.html')

# --- Program plan generator ---
def generate_program_plan(snapshot, suggestion):
    total = snapshot.get('total', 0)
    by_age = snapshot.get('by_age_bucket', {})
    by_gender = snapshot.get('by_gender', {})
    menu_tags = suggestion.get('menu_tags', [])
    talk_tags = suggestion.get('talk_tags', [])
    program_name = suggestion.get('program_name', '')
    today_name = datetime.now().strftime('%A')

    # Build schedule based on actual mandir programs
    schedule = []
    
    if 'Sunday' in program_name or today_name == 'Sunday':
        # Sunday Morning Program: 9:30am - 11:30am
        schedule = [
            { 'time': '09:30', 'item': 'Puja (पूजा)' },
            { 'time': '10:00', 'item': 'Kirtan (कीर्तन)' },
            { 'time': '10:30', 'item': 'Pravachan (प्रवचन)' },
            { 'time': '11:00', 'item': 'Bhog (भोग)' },
            { 'time': '11:15', 'item': 'Aarti (आरती)' },
            { 'time': '11:30', 'item': 'Bhojan (भोजन)' },
        ]
    elif 'Monday' in program_name or today_name == 'Monday':
        # Monday Evening Program: 6:30pm - 7:45pm
        schedule = [
            { 'time': '18:30', 'item': 'Bhagwan Shiv Puja (भगवान शिव की पूजा)' },
            { 'time': '19:00', 'item': 'Kirtan (कीर्तन)' },
            { 'time': '19:15', 'item': 'Maha Shiv Puran Katha (महा शिव पुराण कथा)' },
            { 'time': '19:30', 'item': 'Aarti (आरती)' },
            { 'time': '19:45', 'item': 'Bhojan (भोजन)' },
        ]
    elif 'Tuesday' in program_name or today_name == 'Tuesday':
        # Tuesday Evening Program: 6:30pm - 7:45pm
        schedule = [
            { 'time': '18:30', 'item': 'Hanuman Swami Puja (हनुमान स्वामी की पूजा)' },
            { 'time': '19:00', 'item': 'Hanuman Chalisa (चालीसा)' },
            { 'time': '19:15', 'item': 'Sundar Kaand (सुंदर कांड)' },
            { 'time': '19:30', 'item': 'Aarti (आरती)' },
            { 'time': '19:45', 'item': 'Bhojan (भोजन)' },
        ]
    elif 'Navgraha' in program_name or today_name == 'Saturday':
        # Saturday Navgraha Puja: 10:00am - 10:30am
        schedule = [
            { 'time': '10:00', 'item': 'Navgraha Puja (नवग्रह पूजा)' },
            { 'time': '10:30', 'item': 'Aarti and Prasad' },
        ]
    elif 'Purnima' in program_name:
        # Purnima Program: 6:30pm - 7:45pm
        schedule = [
            { 'time': '18:30', 'item': 'Purnima Puja (पूर्णिमा पूजा)' },
            { 'time': '19:00', 'item': 'Kirtan (कीर्तन)' },
            { 'time': '19:15', 'item': 'Pravachan (प्रवचन)' },
            { 'time': '19:30', 'item': 'Bhog and Aarti (भोग, आरती)' },
            { 'time': '19:45', 'item': 'Bhojan (भोजन)' },
        ]
    else:
        # Daily Aarti times: 8:00am, 12:00pm, 7:00pm
        schedule = [
            { 'time': '08:00', 'item': 'Morning Aarti (सुबह आरती)' },
            { 'time': '12:00', 'item': 'Afternoon Aarti (दोपहर आरती)' },
            { 'time': '19:00', 'item': 'Evening Aarti (शाम आरती)' },
        ]
    
    # Add daily aarti reminder for all programs
    if today_name not in ['Sunday', 'Monday', 'Tuesday', 'Saturday']:
        schedule.insert(0, { 'time': '08:00', 'item': 'Daily Morning Aarti (रोज़ सुबह आरती)' })
        schedule.append({ 'time': '12:00', 'item': 'Daily Afternoon Aarti (रोज़ दोपहर आरती)' })
        schedule.append({ 'time': '19:00', 'item': 'Daily Evening Aarti (रोज़ शाम आरती)' })

    # Sermon outline (Pravachan/Katha) based on program
    outline = []
    
    if 'Monday' in program_name or 'Shiv' in program_name:
        outline = [
            'Introduction to Maha Shiv Puran Katha',
            'Stories of Lord Shiva\'s compassion and wisdom',
            'Lessons for daily life and spiritual growth',
            'Closing prayers and reflection'
        ]
        if 'introductory' in talk_tags:
            outline.insert(1, 'Welcome newcomers with brief overview of Shiv Puja significance')
        if 'youth-focus' in talk_tags:
            outline.insert(2, 'Engaging stories suitable for children and families')
    elif 'Tuesday' in program_name or 'Hanuman' in program_name:
        outline = [
            'Hanuman Chalisa recitation and meaning',
            'Sundar Kaand - Stories of Hanuman\'s devotion',
            'Lessons on devotion, strength, and service',
            'Closing prayers and reflection'
        ]
        if 'introductory' in talk_tags:
            outline.insert(1, 'Introduction to Hanuman Swami and his significance')
        if 'youth-focus' in talk_tags:
            outline.insert(2, 'Interactive storytelling for children about Hanuman\'s adventures')
    elif 'Sunday' in program_name:
        outline = [
            'Welcome and introduction to today\'s spiritual theme',
            'Key teachings from scriptures',
            'Practical applications for family and community',
            'Closing reflection and gratitude'
        ]
        if 'introductory' in talk_tags:
            outline[0] = 'Warm welcome; introduction to mandir traditions and today\'s program'
        if 'youth-focus' in talk_tags:
            outline.insert(2, 'Story-oriented segment for children/teens with practical values')
    elif 'Purnima' in program_name:
        outline = [
            'Significance of Purnima (Full Moon) in Hindu tradition',
            'Special prayers and blessings for the community',
            'Reflection on spiritual growth and renewal',
            'Closing prayers and community blessings'
        ]
    else:
        # Default outline
        if 'introductory' in talk_tags:
            outline.append('Warm welcome; brief introduction to the day\'s scripture')
        if 'youth-focus' in talk_tags:
            outline.append('Story-oriented segment for children/teens; practical values')
        if 'deeper-discussion' in talk_tags:
            outline.append('Short deeper-dive for regulars; reflective exercise')
        if 'inclusive' in talk_tags:
            outline.append('Emphasize inclusivity and privacy-respecting participation')
        if not outline:
            outline = [
                'Central theme and key verses',
                'Practical takeaways for family and community',
                'Closing reflection and gratitude'
            ]

    # Menu suggestions from tags
    menu = []
    if 'low-spice' in menu_tags:
        menu += ['Khichdi', 'Cucumber raita']
    if 'soft-texture' in menu_tags:
        menu += ['Upma', 'Banana sheera']
    if 'kid-friendly' in menu_tags:
        menu += ['Vegetable pulao (mild)', 'Fruit cups']
    if 'mild-spice' in menu_tags and not menu:
        menu += ['Jeera rice', 'Dal tadka (mild)']
    if not menu:
        menu = ['Puri-bhaji', 'Kadhi-chawal']

    # TV announcements (for display around mandir)
    tv = [
        f"Welcome! Today's attendees: {total} people",
        'Daily Aarti times: 8:00am, 12:00pm, 7:00pm | रोज़ आरती का समय',
    ]
    
    # Add program-specific announcements
    if 'Sunday' in program_name:
        tv.append('Sunday Morning Program: 9:30am - 11:30am | रविवार सुबह')
    elif 'Monday' in program_name:
        tv.append('Monday Evening Program: 6:30pm - 7:45pm | सोमवार शाम - Bhagwan Shiv Puja')
    elif 'Tuesday' in program_name:
        tv.append('Tuesday Evening Program: 6:30pm - 7:45pm | मंगलवार शाम - Hanuman Swami Puja')
    elif 'Navgraha' in program_name:
        tv.append('Saturday Navgraha Puja: 10:00am - 10:30am | नवग्रह पूजा - Call to join with families')
    elif 'Purnima' in program_name:
        tv.append('Special Purnima Program: 6:30pm - 7:45pm | पूर्णिमा पूजा')
    
    # Gender-aware reminder
    if by_gender.get('Prefer not to say',0)/max(total or 1,1) > 0.15:
        tv.append('Your privacy matters—join, participate and connect at your comfort level')

    # Optional LLM enhancement (if enabled by admin - opt-in only, off by default)
    llm_enabled = get_setting('llm_enabled', 'false').lower() == 'true'
    llm_api_key = get_setting('llm_api_key', '')
    llm_provider = get_setting('llm_provider', 'openai')
    
    # Fallback to environment variables for backward compatibility
    if not llm_api_key:
        llm_api_key = os.environ.get('LLM_API_KEY', '')
    if not llm_enabled:
        llm_enabled = os.environ.get('LLM_ENABLED', 'false').lower() == 'true'
    
    if llm_enabled and llm_api_key:
        try:
            from llm_integration import generate_sermon_outline as llm_sermon, generate_menu_suggestions as llm_menu
            
            # Determine theme from program
            theme = program_name
            if 'Shiv' in program_name or 'Monday' in program_name:
                theme = "Lord Shiva's compassion and wisdom"
            elif 'Hanuman' in program_name or 'Tuesday' in program_name:
                theme = "Hanuman's devotion and strength"
            elif 'Sunday' in program_name:
                theme = "Spiritual teachings and community"
            elif 'Purnima' in program_name:
                theme = "Full moon significance and renewal"
            else:
                theme = "Spiritual guidance and community"
            
            # Generate LLM-enhanced sermon outline (only aggregate demographics, no personal data)
            llm_outline = llm_sermon(theme, snapshot, program_name, llm_api_key, llm_provider)
            if llm_outline:
                # Combine base outline with LLM suggestions
                outline = llm_outline[:3] + outline[:2]  # Take first 3 from LLM, keep 2 from base
                outline = outline[:5]  # Limit to 5 points
            
            # Generate LLM-enhanced menu suggestions
            llm_menu_items = llm_menu(snapshot, program_name, llm_api_key, llm_provider)
            if llm_menu_items:
                # Use LLM suggestions, fallback to base if empty
                menu = llm_menu_items[:4] if llm_menu_items else menu
        except ImportError:
            print("Warning: llm_integration.py not available")
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}")
            # Continue with base outline/menu on error

    return {
        'program_name': suggestion.get('program_name'),
        'program_id': suggestion.get('program_id'),
        'rationale': suggestion.get('rationale', []),
        'schedule': schedule,
        'sermon_outline': outline,
        'menu': menu,
        'tv_announcements': tv,
        'snapshot': snapshot,
        'tags': { 'menu': menu_tags, 'talk': talk_tags },
        'llm_enhanced': llm_enabled if AI_ENGINE_AVAILABLE else False
    }

@app.route('/program-plan')
@admin_required
def program_plan():
    snap = compute_today_snapshot()
    suggestion = suggest_program_from_snapshot(snap)
    if not suggestion:
        flash('No program suggestions available. Seed programs or attendance first.', 'danger')
        return redirect(url_for('admin_page'))

    plan = generate_program_plan(snap, suggestion)

    # Optional LLM enhancement if OPENAI_API_KEY and openai package available
    try:
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            import json
            import urllib.request
            # Minimal, offline-friendly placeholder: skip network call; keep deterministic plan.
            # Hook point: send plan+snapshot+tags to your LLM service and replace outline/menu text.
            pass
    except Exception:
        pass

    return render_template('program_plan.html', plan=plan)

@app.route('/today')
def today_program():
    # If program_of_day is set, keep program_name from DB; otherwise use suggestion
    snap = compute_today_snapshot()
    # Try to get applied program
    today_date = datetime.now().date().isoformat()
    conn = sqlite3.connect('temple.db')
    c = conn.cursor()
    c.execute("SELECT p.id, p.name FROM program_of_day pod JOIN programs p ON p.id = pod.program_id WHERE pod.date = ?", (today_date,))
    row = c.fetchone()
    conn.close()
    if row:
        # Build a minimal suggestion object using selected program
        suggestion = { 'program_id': row[0], 'program_name': row[1], 'rationale': ['Applied by admin'], 'menu_tags': [], 'talk_tags': [] }
    else:
        suggestion = suggest_program_from_snapshot(snap)
        if not suggestion:
            return render_template('today.html', plan=None)
    plan = generate_program_plan(snap, suggestion)
    return render_template('today.html', plan=plan)

if __name__ == '__main__':
    # Initialize NLU model if available
    if NLU_AVAILABLE:
        try:
            initialize_nlu()
        except Exception as e:
            print(f"Warning: Could not initialize NLU: {e}")
    
    app.run(debug=True, port=5001)