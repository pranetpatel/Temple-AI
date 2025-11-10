"""
Improved ML Model for Program Recommendations
Uses DictVectorizer + LogisticRegression/SGDClassifier
Supports incremental updates via partial_fit
"""

import os
import pickle
from typing import Dict, List, Optional, Any, Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

MODELS_DIR = 'models'
PROGRAM_MODEL_PATH = os.path.join(MODELS_DIR, 'program_model.pkl')
PROGRAM_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'program_vectorizer.pkl')
PROGRAM_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'program_label_encoder.pkl')


def ensure_models_dir():
    """Ensure models directory exists"""
    os.makedirs(MODELS_DIR, exist_ok=True)


def featurize(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert demographic snapshot to feature dictionary
    
    Args:
        snapshot: Dict with keys like 'total', 'by_age_bucket', 'newcomers', 'frequent', etc.
    
    Returns:
        Dict of feature names to values
    """
    total = max(snapshot.get('total', 0), 1)
    by_age = snapshot.get('by_age_bucket', {})
    newcomers = snapshot.get('newcomers', 0)
    frequent = snapshot.get('frequent', 0)
    by_gender = snapshot.get('by_gender', {})
    
    # Age ratios
    children = by_age.get('0-12', 0) + by_age.get('13-17', 0)
    young_adults = by_age.get('18-29', 0)
    middle_aged = by_age.get('30-49', 0)
    older_adults = by_age.get('50-64', 0)
    seniors = by_age.get('65+', 0)
    
    features = {
        'ratio_children': children / total,
        'ratio_young_adults': young_adults / total,
        'ratio_middle_aged': middle_aged / total,
        'ratio_older_adults': older_adults / total,
        'ratio_seniors': seniors / total,
        'ratio_newcomers': newcomers / total,
        'ratio_frequent': frequent / total,
        'total_attendees': total,
        'has_children': 1.0 if children > 0 else 0.0,
        'has_seniors': 1.0 if seniors > 0 else 0.0,
        'has_newcomers': 1.0 if newcomers > 0 else 0.0,
        'has_frequent': 1.0 if frequent > 0 else 0.0,
    }
    
    # Gender ratios
    for gender in ['Male', 'Female', 'Non-binary', 'Other', 'Prefer not to say']:
        count = by_gender.get(gender, 0)
        features[f'ratio_{gender.lower().replace(" ", "_")}'] = count / total
    
    return features


def train_program_model(X_dicts: List[Dict[str, float]], y_labels: List[int]) -> Tuple[DictVectorizer, Any, LabelEncoder]:
    """
    Train program recommendation model
    
    Args:
        X_dicts: List of feature dictionaries (from featurize())
        y_labels: List of program IDs (labels)
    
    Returns:
        Tuple of (vectorizer, classifier, label_encoder)
    """
    # Create vectorizer
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(X_dicts)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    
    # Train classifier
    # Use LogisticRegression for initial training
    # For incremental updates, use SGDClassifier
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='ovr',  # one-vs-rest for multi-class
        C=1.0
    )
    
    classifier.fit(X, y_encoded)
    
    # Save model components
    ensure_models_dir()
    with open(PROGRAM_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(PROGRAM_MODEL_PATH, 'wb') as f:
        pickle.dump(classifier, f)
    with open(PROGRAM_LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Program model trained on {len(X_dicts)} examples, {len(label_encoder.classes_)} programs")
    print(f"Model saved to {MODELS_DIR}/")
    
    return vectorizer, classifier, label_encoder


def load_program_model() -> Optional[Tuple[DictVectorizer, Any, LabelEncoder]]:
    """
    Load trained program model
    
    Returns:
        Tuple of (vectorizer, classifier, label_encoder) if exists, None otherwise
    """
    if not all(os.path.exists(p) for p in [PROGRAM_VECTORIZER_PATH, PROGRAM_MODEL_PATH, PROGRAM_LABEL_ENCODER_PATH]):
        return None
    
    try:
        with open(PROGRAM_VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(PROGRAM_MODEL_PATH, 'rb') as f:
            classifier = pickle.load(f)
        with open(PROGRAM_LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return vectorizer, classifier, label_encoder
    except Exception as e:
        print(f"Error loading program model: {e}")
        return None


def score_programs(snapshot: Dict[str, Any], program_ids: Optional[List[int]] = None) -> Dict[int, float]:
    """
    Score programs for given snapshot
    
    Args:
        snapshot: Demographic snapshot
        program_ids: Optional list of program IDs to score. If None, scores all known programs.
    
    Returns:
        Dict mapping program_id -> score
    """
    result = load_program_model()
    if result is None:
        # Return default scores if model not trained
        return {pid: 0.0 for pid in (program_ids or [])}
    
    vectorizer, classifier, label_encoder = result
    
    # Featurize snapshot
    features = featurize(snapshot)
    X = vectorizer.transform([features])
    
    # Get predictions (probabilities)
    try:
        probs = classifier.predict_proba(X)[0]
        
        # Map encoded labels back to program IDs
        program_id_to_score = {}
        for encoded_label, prob in enumerate(probs):
            program_id = label_encoder.inverse_transform([encoded_label])[0]
            program_id_to_score[int(program_id)] = float(prob)
        
        # Filter to requested program_ids if provided
        if program_ids:
            program_id_to_score = {pid: program_id_to_score.get(pid, 0.0) for pid in program_ids}
        
        return program_id_to_score
    except Exception as e:
        print(f"Error scoring programs: {e}")
        return {pid: 0.0 for pid in (program_ids or [])}


def online_update(snapshot: Dict[str, Any], label: int, learning_rate: float = 0.01):
    """
    Perform online/incremental update to model using SGD
    
    Args:
        snapshot: Demographic snapshot
        label: Program ID that was selected (positive example)
        learning_rate: Learning rate for SGD update
    """
    result = load_program_model()
    if result is None:
        # Can't update if model doesn't exist - need to train first
        print("Warning: Cannot perform online update - model not trained yet")
        return
    
    vectorizer, classifier, label_encoder = result
    
    # Check if label is in known classes
    known_program_ids = label_encoder.classes_
    if label not in known_program_ids:
        print(f"Warning: Program ID {label} not in trained model. Skipping update.")
        return
    
    # Featurize snapshot
    features = featurize(snapshot)
    X = vectorizer.transform([features])
    
    # Encode label
    y_encoded = label_encoder.transform([label])[0]
    
    # Use SGDClassifier for incremental learning
    # If current classifier is LogisticRegression, convert to SGD
    if isinstance(classifier, LogisticRegression):
        # Create new SGDClassifier initialized with LogisticRegression weights
        sgd_classifier = SGDClassifier(
            loss='log',  # logistic loss for probability estimates
            learning_rate='constant',
            eta0=learning_rate,
            random_state=42,
            warm_start=True,
            max_iter=1  # Single iteration for incremental update
        )
        
        # Initialize with current model's structure
        sgd_classifier.classes_ = classifier.classes_
        sgd_classifier.coef_ = classifier.coef_.copy()
        sgd_classifier.intercept_ = classifier.intercept_.copy()
        
        classifier = sgd_classifier
    
    # Perform partial_fit (incremental update)
    if hasattr(classifier, 'partial_fit'):
        try:
            # Partial fit expects classes parameter on first call
            if not hasattr(classifier, '_partial_fit_called'):
                classifier.partial_fit(X, [y_encoded], classes=label_encoder.classes_)
                classifier._partial_fit_called = True
            else:
                classifier.partial_fit(X, [y_encoded])
        except Exception as e:
            print(f"Error in partial_fit: {e}")
            return
    else:
        # Fallback: retrain with new data point
        # In production, you'd maintain a data buffer
        print("Warning: Classifier doesn't support partial_fit. Skipping incremental update.")
        return
    
    # Save updated model
    ensure_models_dir()
    with open(PROGRAM_MODEL_PATH, 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"Model updated with program_id={label}, learning_rate={learning_rate}")


def initialize_program_model():
    """Initialize program model - create empty model if needed"""
    if load_program_model() is None:
        print("Program model not found. Train with train_program_model() first.")
    else:
        print("Program model loaded successfully")


if __name__ == '__main__':
    # Example usage
    print("Program Model Module")
    print("Use train_program_model() to train, score_programs() to score, online_update() to update")

