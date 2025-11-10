"""
Lightweight Natural Language Understanding using sklearn
TF-IDF + LogisticRegression for intent classification
"""

import os
import pickle
import re
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Default training examples (intent -> list of example utterances)
DEFAULT_TRAINING_DATA = {
    'greeting': [
        'hi', 'hello', 'hey', 'greetings', 'namaste',
        'good morning', 'good afternoon', 'good evening',
        'hi there', 'hello there'
    ],
    'attendance_query': [
        'how many people came today',
        'how many attendees today',
        'count of visitors',
        'number of people today',
        'attendance today',
        'how many people this week',
        'attendance this month',
        'how many visitors',
        'total attendees',
        'people count'
    ],
    'demographic_query': [
        'what is the average age',
        'age distribution',
        'demographics',
        'how many children',
        'how many seniors',
        'age breakdown',
        'gender distribution',
        'demographic breakdown',
        'age groups',
        'what ages are here'
    ],
    'program_query': [
        'what is today program',
        'what is the schedule',
        'show me the program',
        'today schedule',
        'what program today',
        'show program plan',
        'display schedule',
        'what event today',
        'today event',
        'current program'
    ],
    'suggestion_request': [
        'suggest a program',
        'recommend program',
        'what should we do',
        'what program should we use',
        'recommendation for today',
        'suggest program for today',
        'what would you suggest',
        'give me a suggestion',
        'program suggestion',
        'recommend today program'
    ],
    'modification_request': [
        'change the menu',
        'modify program',
        'edit schedule',
        'different menu',
        'change sermon',
        'modify sermon',
        'update menu',
        'change food',
        'different program',
        'edit program'
    ],
    'statistics_query': [
        'show statistics',
        'trends',
        'attendance trends',
        'compare attendance',
        'growth statistics',
        'statistics report',
        'show trends',
        'attendance patterns',
        'statistical analysis',
        'data trends'
    ],
    'newcomer_query': [
        'how many newcomers',
        'new members',
        'recent signups',
        'new signups',
        'newcomers today',
        'recent members',
        'new people',
        'just signed up',
        'new registrations',
        'recent registrations'
    ],
    'frequent_visitor_query': [
        'frequent visitors',
        'regular attendees',
        'loyal members',
        'repeat visitors',
        'consistent attendees',
        'regulars',
        'frequent attendees',
        'regular visitors',
        'loyal visitors',
        'consistent visitors'
    ],
    'thanks': [
        'thank you',
        'thanks',
        'appreciate it',
        'helpful',
        'thank you very much',
        'thanks a lot',
        'much appreciated',
        'grateful',
        'appreciate',
        'thank'
    ],
    'goodbye': [
        'bye',
        'goodbye',
        'see you',
        'farewell',
        'see you later',
        'bye bye',
        'goodbye for now',
        'see you soon',
        'take care',
        'until next time'
    ],
    'capabilities': [
        'what can you do',
        'what help can you provide',
        'your capabilities',
        'what are your abilities',
        'help me',
        'what do you do',
        'your features',
        'what can you help with',
        'show capabilities',
        'what functions'
    ],
    'unknown': [
        'xyzabc123',  # Placeholder for unknown
        'random text',
        'nonsense query'
    ]
}

MODELS_DIR = 'models'
NLU_MODEL_PATH = os.path.join(MODELS_DIR, 'nlu_model.pkl')
NLU_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'nlu_vectorizer.pkl')


def ensure_models_dir():
    """Ensure models directory exists"""
    os.makedirs(MODELS_DIR, exist_ok=True)


def train_nlu(examples: Optional[Dict[str, List[str]]] = None) -> Tuple[Pipeline, List[str]]:
    """
    Train NLU model on example utterances
    
    Args:
        examples: Dict mapping intent labels to lists of example utterances.
                  If None, uses DEFAULT_TRAINING_DATA
    
    Returns:
        Tuple of (trained pipeline, list of intent labels)
    """
    if examples is None:
        examples = DEFAULT_TRAINING_DATA
    
    # Prepare training data
    X = []  # utterances
    y = []  # intent labels
    
    for intent, utterances in examples.items():
        for utterance in utterances:
            X.append(utterance.lower().strip())
            y.append(intent)
    
    # Create pipeline: TF-IDF vectorizer + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # unigrams and bigrams
            max_features=1000,
            min_df=1,
            max_df=0.95
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='ovr',  # one-vs-rest for multi-class
            C=1.0
        ))
    ])
    
    # Train
    pipeline.fit(X, y)
    
    # Get unique labels (intents)
    intents = sorted(list(set(y)))
    
    # Save model
    ensure_models_dir()
    with open(NLU_MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"NLU model trained on {len(X)} examples, {len(intents)} intents")
    print(f"Model saved to {NLU_MODEL_PATH}")
    
    return pipeline, intents


def load_nlu() -> Optional[Tuple[Pipeline, List[str]]]:
    """
    Load trained NLU model and return pipeline + intents
    
    Returns:
        Tuple of (pipeline, intents) if model exists, None otherwise
    """
    if not os.path.exists(NLU_MODEL_PATH):
        return None
    
    try:
        with open(NLU_MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Get intents from the classifier classes
        intents = pipeline.named_steps['clf'].classes_.tolist()
        
        return pipeline, intents
    except Exception as e:
        print(f"Error loading NLU model: {e}")
        return None


def predict_intent(text: str, clf: Optional[Pipeline] = None, 
                  vect: Optional[Any] = None) -> Dict[str, Any]:
    """
    Predict intent for given text
    
    Args:
        text: Input text to classify
        clf: Optional pre-loaded pipeline (if None, loads from disk)
        vect: Deprecated (kept for compatibility, not used)
    
    Returns:
        Dict with 'intent', 'confidence', and 'all_scores'
    """
    # Load model if not provided
    if clf is None:
        result = load_nlu()
        if result is None:
            # Fallback: return unknown with low confidence
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'all_scores': {}
            }
        clf, intents = result
    else:
        intents = clf.named_steps['clf'].classes_.tolist()
    
    # Preprocess text
    text_clean = text.lower().strip()
    
    # Predict
    try:
        # Get prediction probabilities
        probs = clf.predict_proba([text_clean])[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probs)
        predicted_intent = intents[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Get all scores
        all_scores = {intents[i]: float(probs[i]) for i in range(len(intents))}
        
        # Fallback for low confidence
        if confidence < 0.3:
            predicted_intent = 'unknown'
            confidence = 0.0
        
        return {
            'intent': predicted_intent,
            'confidence': confidence,
            'all_scores': all_scores
        }
    except Exception as e:
        print(f"Error predicting intent: {e}")
        return {
            'intent': 'unknown',
            'confidence': 0.0,
            'all_scores': {}
        }


def initialize_nlu():
    """Initialize NLU model - train if not exists"""
    if load_nlu() is None:
        print("NLU model not found. Training new model...")
        train_nlu()
    else:
        print("NLU model loaded successfully")


if __name__ == '__main__':
    # Train model if run directly
    print("Training NLU model...")
    train_nlu()
    print("Training complete!")
    
    # Test
    test_queries = [
        "how many people came today",
        "what is today's program",
        "suggest a program",
        "hi there",
        "thank you",
        "random nonsense text"
    ]
    
    pipeline, intents = load_nlu()
    print(f"\nTesting with {len(intents)} intents:")
    for query in test_queries:
        result = predict_intent(query, clf=pipeline)
        print(f"'{query}' -> {result['intent']} (confidence: {result['confidence']:.2f})")

