"""
Lightweight AI Engine for Temple System
- Natural Language Understanding (NLU)
- Enhanced ML Model for Program Suggestions
- Structured Feedback Learning
- Optional LLM Integration
"""

import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json

# Lightweight NLU: Intent classification using keyword matching + simple patterns
class LightweightNLU:
    """Lightweight Natural Language Understanding without external dependencies"""
    
    def __init__(self):
        # Intent patterns with keywords and weights
        self.intent_patterns = {
            'attendance_query': {
                'keywords': ['attendance', 'people', 'came', 'visitors', 'guests', 'attendees', 'how many', 'count', 'number'],
                'time_modifiers': ['today', 'this week', 'this month', 'weekly', 'monthly', 'last week', 'last month'],
                'weight': 1.0
            },
            'demographic_query': {
                'keywords': ['age', 'demographic', 'gender', 'male', 'female', 'children', 'kids', 'seniors', 'elderly', 'average age'],
                'weight': 1.0
            },
            'program_query': {
                'keywords': ['program', 'schedule', 'plan', 'event', 'service', 'today', 'what', 'which', 'show'],
                'weight': 1.0
            },
            'suggestion_request': {
                'keywords': ['suggest', 'recommend', 'what should', 'what would', 'advice', 'recommendation'],
                'weight': 1.0
            },
            'modification_request': {
                'keywords': ['change', 'modify', 'edit', 'different', 'instead', 'prefer', 'rather', "don't like", "don't want"],
                'weight': 1.0
            },
            'statistics_query': {
                'keywords': ['statistics', 'stats', 'trends', 'pattern', 'compare', 'growth', 'increase', 'decrease'],
                'weight': 1.0
            },
            'newcomer_query': {
                'keywords': ['new', 'newcomer', 'recent', 'just signed', 'new signup', 'new member'],
                'weight': 1.0
            },
            'frequent_visitor_query': {
                'keywords': ['regular', 'frequent', 'often', 'repeat', 'loyal', 'consistent'],
                'weight': 1.0
            },
            'greeting': {
                'keywords': ['hi', 'hello', 'hey', 'greetings', 'namaste', 'good morning', 'good afternoon', 'good evening'],
                'weight': 2.0  # Higher weight for greetings
            },
            'thanks': {
                'keywords': ['thank', 'thanks', 'appreciate', 'helpful'],
                'weight': 2.0
            },
            'goodbye': {
                'keywords': ['bye', 'goodbye', 'see you', 'farewell'],
                'weight': 2.0
            },
            'capabilities': {
                'keywords': ['what can you do', 'what help', 'capabilities', 'abilities', 'help'],
                'weight': 1.0
            }
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'time_period': r'(today|this week|this month|weekly|monthly|last week|last month|past \d+ days)',
            'number': r'(\d+)',
            'program_name': r'(sunday|monday|tuesday|wednesday|thursday|friday|saturday|purnima|navgraha)',
            'modification_type': r'(menu|food|sermon|talk|schedule|program)',
        }
    
    def classify_intent(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Classify user intent using keyword matching and context"""
        query_lower = query.lower().strip()
        
        # Score each intent
        intent_scores = {}
        for intent, pattern in self.intent_patterns.items():
            score = 0.0
            # Check keyword matches
            for keyword in pattern['keywords']:
                if keyword in query_lower:
                    score += pattern['weight']
            intent_scores[intent] = score
        
        # Get top intent
        if not intent_scores or max(intent_scores.values()) == 0:
            return {'intent': 'unknown', 'confidence': 0.0, 'entities': {}}
        
        top_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(1.0, intent_scores[top_intent] / 5.0)  # Normalize confidence
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Context-aware adjustments
        if conversation_history:
            entities.update(self._extract_context_entities(conversation_history))
        
        return {
            'intent': top_intent,
            'confidence': confidence,
            'entities': entities,
            'raw_query': query
        }
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {}
        
        # Time period
        time_match = re.search(self.entity_patterns['time_period'], query)
        if time_match:
            entities['time_period'] = time_match.group(1)
        
        # Numbers
        numbers = re.findall(self.entity_patterns['number'], query)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]
        
        # Program name
        program_match = re.search(self.entity_patterns['program_name'], query)
        if program_match:
            entities['program_name'] = program_match.group(1)
        
        # Modification type
        mod_match = re.search(self.entity_patterns['modification_type'], query)
        if mod_match:
            entities['modification_type'] = mod_match.group(1)
        
        return entities
    
    def _extract_context_entities(self, history: List[Dict]) -> Dict[str, Any]:
        """Extract entities from conversation history"""
        entities = {}
        recent = ' '.join([h.get('content', '').lower() for h in history[-3:]])
        
        # Check for program context
        if any(word in recent for word in ['program', 'menu', 'sermon', 'schedule']):
            entities['has_program_context'] = True
        
        return entities


# Enhanced ML Model: Decision Tree (lightweight, interpretable)
class ProgramSuggestionModel:
    """Enhanced ML model for program suggestions using decision tree logic"""
    
    def __init__(self, db_path: str = 'temple.db'):
        self.db_path = db_path
        self.features = ['ratio_children', 'ratio_seniors', 'ratio_newcomers', 'ratio_frequent', 'total_attendees']
    
    def features_from_snapshot(self, snapshot: Dict) -> Dict[str, float]:
        """Extract features from demographic snapshot"""
        total = max(snapshot.get('total', 0), 1)
        by_age = snapshot.get('by_age_bucket', {})
        newcomers = snapshot.get('newcomers', 0)
        frequent = snapshot.get('frequent', 0)
        
        children = by_age.get('0-12', 0) + by_age.get('13-17', 0)
        seniors = by_age.get('65+', 0)
        
        return {
            'ratio_children': children / total,
            'ratio_seniors': seniors / total,
            'ratio_newcomers': newcomers / total,
            'ratio_frequent': frequent / total,
            'total_attendees': total
        }
    
    def score_program(self, program_id: int, features: Dict[str, float], snapshot: Dict) -> float:
        """Score a program using decision tree + learned weights"""
        # Load learned weights
        weights = self._load_weights(program_id)
        
        # Base score from learned weights (linear component)
        linear_score = sum(weights.get(f, 0.0) * features.get(f, 0.0) for f in self.features)
        
        # Decision tree rules (interpretable, lightweight)
        tree_score = self._decision_tree_score(program_id, features, snapshot)
        
        # Combine: 70% learned weights, 30% decision tree
        final_score = 0.7 * linear_score + 0.3 * tree_score
        
        return final_score
    
    def _decision_tree_score(self, program_id: int, features: Dict, snapshot: Dict) -> float:
        """Lightweight decision tree scoring"""
        score = 0.0
        
        # Rule 1: High children ratio → prefer Sunday morning (family-friendly)
        if features['ratio_children'] > 0.35:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT name FROM programs WHERE id = ?", (program_id,))
            prog_name = c.fetchone()
            conn.close()
            if prog_name and 'Sunday' in prog_name[0]:
                score += 0.5
        
        # Rule 2: High seniors ratio → prefer accessible programs
        if features['ratio_seniors'] > 0.40:
            score += 0.3
        
        # Rule 3: High newcomers → prefer welcoming programs
        if features['ratio_newcomers'] > 0.25:
            score += 0.2
        
        # Rule 4: High frequent visitors → prefer deeper content
        if features['ratio_frequent'] > 0.40:
            score += 0.2
        
        return min(1.0, score)
    
    def _load_weights(self, program_id: int) -> Dict[str, float]:
        """Load learned weights from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT feature, weight FROM model_weights WHERE program_id=?', (program_id,))
        rows = c.fetchall()
        conn.close()
        weights = {f: w for f, w in rows}
        # Default missing weights to 0
        for f in self.features:
            weights.setdefault(f, 0.0)
        return weights
    
    def update_weights(self, program_id: int, features: Dict[str, float], label: int, learning_rate: float = 0.5):
        """Update weights based on feedback (perceptron-style update)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for feature, value in features.items():
            if feature not in self.features:
                continue
            c.execute('SELECT weight FROM model_weights WHERE program_id=? AND feature=?', (program_id, feature))
            row = c.fetchone()
            w = row[0] if row else 0.0
            w_new = w + learning_rate * label * value
            c.execute('REPLACE INTO model_weights(program_id, feature, weight) VALUES(?,?,?)', 
                     (program_id, feature, w_new))
        
        conn.commit()
        conn.close()


# Structured Feedback System
class FeedbackCollector:
    """Structured feedback collection and learning"""
    
    def __init__(self, db_path: str = 'temple.db'):
        self.db_path = db_path
        self._init_feedback_table()
    
    def _init_feedback_table(self):
        """Initialize feedback table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS program_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        program_id INTEGER,
                        username TEXT,
                        feedback INTEGER,
                        snapshot_json TEXT
                    )''')
        # Ensure new columns exist if the table was created with the old schema
        try:
            c.execute("ALTER TABLE program_feedback ADD COLUMN username TEXT")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE program_feedback ADD COLUMN feedback INTEGER")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE program_feedback ADD COLUMN snapshot_json TEXT")
        except Exception:
            pass
        conn.commit()
        conn.close()
    
    def record_feedback(self, program_id: int, outcome: str, feedback_type: str = 'simple',
                       details: Optional[str] = None, snapshot_features: Optional[Dict] = None):
        """Record structured feedback"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        today = datetime.now().date().isoformat()
        now = datetime.now().isoformat(timespec='seconds')
        features_json = json.dumps(snapshot_features) if snapshot_features else None
        # Map outcome string to numeric feedback
        feedback_value = None
        if isinstance(outcome, (int, float)):
            feedback_value = 1 if outcome >= 1 else -1
        elif isinstance(outcome, str):
            feedback_value = 1 if outcome.lower() in ('good', 'positive', '1', 'true', 'yes') else -1
        else:
            feedback_value = -1

        c.execute('''INSERT INTO program_feedback 
                    (date, program_id, username, feedback, snapshot_json)
                    VALUES (?, ?, ?, ?, ?)''',
                 (today, program_id, 'admin', feedback_value, features_json))
        
        conn.commit()
        conn.close()
    
    def get_feedback_stats(self, program_id: Optional[int] = None) -> Dict:
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if program_id:
            c.execute('''SELECT feedback, COUNT(*) FROM program_feedback 
                         WHERE program_id = ? GROUP BY feedback''', (program_id,))
        else:
            c.execute('''SELECT feedback, COUNT(*) FROM program_feedback 
                         GROUP BY feedback''')
        
        stats = dict(c.fetchall())
        conn.close()
        return stats


# Optional LLM Integration (offline-first, admin toggle)
class LLMIntegration:
    """Optional LLM integration for content generation"""
    
    def __init__(self, enabled: bool = False, api_key: Optional[str] = None, provider: str = 'openai'):
        self.enabled = enabled
        self.api_key = api_key
        self.provider = provider
    
    def generate_sermon_outline(self, program_name: str, demographics: Dict, 
                               base_outline: List[str]) -> List[str]:
        """Generate enhanced sermon outline using LLM (if enabled)"""
        if not self.enabled or not self.api_key:
            return base_outline  # Return base outline if LLM disabled
        
        try:
            # Placeholder for LLM call
            # In production, this would call OpenAI/Anthropic API
            # For now, return enhanced base outline
            enhanced = base_outline.copy()
            if demographics.get('total', 0) > 0:
                enhanced.insert(0, f"Personalized for {demographics.get('total')} attendees")
            return enhanced
        except Exception:
            return base_outline  # Fallback to base
    
    def generate_menu_suggestions(self, program_name: str, demographics: Dict,
                                 base_menu: List[str]) -> List[str]:
        """Generate enhanced menu suggestions using LLM (if enabled)"""
        if not self.enabled or not self.api_key:
            return base_menu  # Return base menu if LLM disabled
        
        try:
            # Placeholder for LLM call
            # In production, this would call LLM API
            return base_menu
        except Exception:
            return base_menu  # Fallback to base


# Initialize global instances
_nlu = None
_model = None
_feedback = None
_llm = None

def get_nlu() -> LightweightNLU:
    """Get or create NLU instance"""
    global _nlu
    if _nlu is None:
        _nlu = LightweightNLU()
    return _nlu

def get_model() -> ProgramSuggestionModel:
    """Get or create ML model instance"""
    global _model
    if _model is None:
        _model = ProgramSuggestionModel()
    return _model

def get_feedback() -> FeedbackCollector:
    """Get or create feedback collector"""
    global _feedback
    if _feedback is None:
        _feedback = FeedbackCollector()
    return _feedback

def get_llm(enabled: bool = False, api_key: Optional[str] = None) -> LLMIntegration:
    """Get or create LLM integration"""
    global _llm
    if _llm is None or _llm.enabled != enabled:
        _llm = LLMIntegration(enabled=enabled, api_key=api_key)
    return _llm