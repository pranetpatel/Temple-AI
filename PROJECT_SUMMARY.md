# Temple AI System - Project Summary for AI Ideation

## Project Overview
A Flask-based web application for mandirs (Hindu temples) that uses AI to personalize daily programs based on attendee demographics. The system tracks user attendance via QR codes, analyzes demographic data, and suggests appropriate programs, menus, and sermon topics.

## Core Purpose
- **User Registration**: Attendees create profiles with name, age, gender, email
- **QR Code System**: Each user gets a unique QR code for check-in
- **Demographic Analysis**: System tracks who's attending each day (age groups, gender, newcomers vs regulars)
- **AI Program Suggestions**: Based on demographics, suggests appropriate programs, menu items, and sermon topics
- **Admin Dashboard**: Manage users, view analytics, interact with AI chatbot, send reminders

## Technical Stack
- **Backend**: Flask (Python)
- **Database**: SQLite
- **Authentication**: bcrypt password hashing, 2FA with remember-me cookies (7 days)
- **Security**: CSRF protection, session management, secure cookies
- **QR Codes**: qrcode library
- **Frontend**: Jinja2 templates, HTML/CSS/JavaScript

## Database Schema

### Core Tables:
1. **users**: id, name, username, password (hashed), age, gender, qr_path, created_at, email, email_verified
2. **attendance**: id, username, date, time, program_id (links to programs table)
3. **checkins**: id, username, action (login/qr_scan/logout), timestamp
4. **programs**: id, name, category, description, day, recommended_for, themes, diet_tags, scripture_tags, difficulty
5. **program_of_day**: id, date (unique), program_id (today's active program)
6. **model_weights**: program_id, feature, weight (for machine learning)

## Actual Mandir Programs (Real Schedule)
1. **Sunday Morning** (9:30am-11:30am): Puja, Kirtan, Pravachan, Bhog, Aarti, Bhojan
2. **Monday Evening** (6:30pm-7:45pm): Bhagwan Shiv Puja, Kirtan, Maha Shiv Puran Katha, Aarti, Bhojan
3. **Tuesday Evening** (6:30pm-7:45pm): Hanuman Swami Puja, Chalisa, Sundar Kaand, Aarti, Bhojan
4. **Saturday** (10:00am-10:30am): Navgraha Puja (group puja)
5. **Purnima** (6:30pm-7:45pm): Special full moon program
6. **Daily Aarti**: 8:00am, 12:00pm, 7:00pm

## Current AI/ML Components

### 1. Demographic Snapshot System
- **Function**: `compute_today_snapshot()`
- **Tracks**: Total attendees, age buckets (0-12, 13-17, 18-29, 30-49, 50-64, 65+), gender distribution, newcomers (last 14 days), frequent visitors (4+ visits in 30 days)

### 2. Feature Engineering
- **Function**: `features_from_snapshot()`
- **Features**: 
  - `bias` (constant 1.0)
  - `ratio_children` (ages 0-17 / total)
  - `ratio_seniors` (ages 65+ / total)
  - `ratio_newcomers` (newcomers / total)
  - `ratio_frequent` (frequent visitors / total)

### 3. On-Device Learning (Perceptron-style)
- **Purpose**: Lightweight ML suitable for Raspberry Pi
- **Functions**: 
  - `load_weights(program_id)`: Loads learned weights per program
  - `score_program(program_id, feats)`: Scores programs using linear model (dot product of weights × features)
  - `update_weights(program_id, feats, label)`: Updates weights based on admin feedback (+1 for "Good", -1 for "Needs Improvement")
- **Learning Rate**: 0.5
- **Storage**: SQLite table `model_weights`

### 4. Program Suggestion Engine
- **Function**: `suggest_program_from_snapshot()`
- **Process**:
  1. Computes demographic snapshot
  2. Extracts features from snapshot
  3. Scores all candidate programs for today's weekday
  4. Selects highest-scoring program
  5. Generates AI-like rationale based on demographics
  6. Derives menu tags and talk tags using rule-based logic
- **Rationale Examples**:
  - "AI recommendation: This program historically performs well for similar demographics (confidence: X%)"
  - "Noticing 35% children/teens → prioritizing engaging, story-based content"
  - "Detecting 40% seniors → emphasizing comfort, accessible pace"
  - "Observing 25% newcomers → adding welcoming introduction segments"

### 5. Program Plan Generator
- **Function**: `generate_program_plan()`
- **Outputs**:
  - **Schedule**: Time-based activities matching actual mandir schedule
  - **Sermon Outline**: Program-specific (e.g., Maha Shiv Puran Katha for Monday, Hanuman Chalisa for Tuesday)
  - **Menu Suggestions**: Based on demographic tags (kid-friendly, low-spice, soft-texture, etc.)
  - **TV Announcements**: Bilingual (English/Hindi) for display around mandir

### 6. AI Chatbot System
- **Functions**: 
  - `parse_chat_query()`: Natural language understanding using regex patterns
  - `execute_chat_query()`: Executes queries and returns conversational responses
- **Capabilities**:
  - **Statistics Queries**: "How many people came today?", "What's the average age?", "Show me gender distribution"
  - **Program Queries**: "What's today's program?", "Suggest a program for today", "Show full plan"
  - **Demographic Queries**: "How many newcomers?", "How many frequent visitors?", "Age breakdown"
  - **Trends**: Week-over-week comparisons
  - **Program Modifications**: "Make menu more kid-friendly", "Focus sermon on newcomers", "Change program"
  - **Conversational**: Greetings, thanks, goodbyes, capabilities
- **Context Management**: Maintains conversation history and program context for modifications
- **Pattern Matching**: Uses regex to identify intent, then generates SQL queries or conversational responses

## Key Features

### User Features:
- Registration with email verification
- Login with 2FA (remembered for 7 days)
- Personal dashboard showing visit history (total visits, last visit, monthly stats)
- QR code for check-in

### Admin Features:
- User management (view, edit, delete)
- Analytics dashboard (signups, logins, QR scans over time - Chart.js visualization)
- Program suggestion system with "Apply for Today" button
- AI chatbot for natural language queries
- Attendance simulation tool (generates test data)
- Reminder system (identifies inactive users, sends email reminders)
- Program plan viewer (detailed schedule, sermon outline, menu, announcements)
- Training system (admin provides feedback: "Good" or "Needs Improvement" to improve ML model)

### Public Features:
- `/today` route: Public-facing page showing today's program plan

## Data Flow

1. **Registration** → User creates profile → QR code generated → Email verification
2. **Check-in** → QR scan → Logged to `attendance` table → Associated with `program_of_day` if active
3. **Daily Snapshot** → System analyzes today's attendees → Computes demographics
4. **AI Suggestion** → Features extracted → Programs scored → Best program selected → Rationale generated
5. **Admin Review** → Admin views suggestion → Can apply program → Can provide feedback
6. **Learning** → Feedback updates model weights → Future suggestions improve

## Current AI Limitations & Opportunities

### Current Approach:
- **Pattern Matching**: Chatbot uses regex patterns (limited to specific phrasings)
- **Rule-Based Logic**: Menu/talk tags derived from hardcoded rules
- **Simple Linear Model**: Perceptron-style learning (good for Raspberry Pi, but limited complexity)
- **No LLM Integration**: Currently no external AI API calls

### Potential Enhancements (Ideation Areas):
1. **Natural Language Understanding**: 
   - Replace regex with more sophisticated NLP
   - Intent classification
   - Entity extraction
   - Sentiment analysis

2. **Advanced ML Models**:
   - Decision trees for program selection
   - Clustering for user segmentation
   - Time series forecasting for attendance prediction
   - Collaborative filtering for program recommendations

3. **LLM Integration**:
   - Use GPT/Claude for generating sermon outlines
   - Dynamic menu suggestions based on demographics
   - Personalized program descriptions
   - More natural chatbot responses

4. **Predictive Analytics**:
   - Predict attendance for upcoming days
   - Suggest optimal program times
   - Identify at-risk users (likely to stop attending)

5. **Content Generation**:
   - Auto-generate sermon topics based on demographics
   - Generate menu suggestions with recipes
   - Create personalized welcome messages

6. **Multi-Modal AI**:
   - Image recognition for QR codes (if needed)
   - Voice commands for admin
   - Text-to-speech for announcements

7. **Contextual Awareness**:
   - Seasonal adjustments (festivals, holidays)
   - Weather-based suggestions
   - Historical pattern recognition

8. **Personalization**:
   - Individual user preferences
   - Family group recommendations
   - Personalized spiritual content

## Constraints
- **Raspberry Pi Deployment**: Must be lightweight, low-resource
- **Offline Capability**: Should work without internet (or with minimal API calls)
- **Privacy**: User data must be handled securely
- **Cultural Sensitivity**: Must respect Hindu traditions and practices

## Key Files
- `app.py`: Main Flask application (1973 lines)
- `templates/`: HTML templates (admin.html, dashboard.html, reminders.html, etc.)
- `static/qr_codes/`: Generated QR code images
- `temple.db`: SQLite database

## Current State
- Fully functional system with user registration, QR check-in, admin dashboard
- Basic AI program suggestion with on-device learning
- Pattern-matching chatbot
- Real mandir schedule integration
- Email reminder system (simulated, ready for SMTP)

---

**Goal**: Enhance the AI capabilities to be more intelligent, natural, and helpful while maintaining the lightweight, on-device nature suitable for Raspberry Pi deployment.