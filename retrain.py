"""
Periodic Retraining Script for Program Recommendation Model
Reads feedback from program_feedback table and retrains the model
"""

import sqlite3
import json
from typing import List, Dict, Any, Tuple
from program_model import train_program_model, featurize, load_program_model

def load_feedback_data(db_path: str = 'temple.db') -> Tuple[List[Dict[str, float]], List[int]]:
    """
    Load training data from program_feedback table
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        Tuple of (feature_dicts, program_ids)
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Get all feedback entries with valid snapshot data
    c.execute("""
        SELECT program_id, feedback, snapshot_json
        FROM program_feedback
        WHERE snapshot_json IS NOT NULL AND snapshot_json != ''
        ORDER BY date DESC
    """)
    
    rows = c.fetchall()
    conn.close()
    
    X_dicts = []  # Feature dictionaries
    y_labels = []  # Program IDs (labels)
    
    for program_id, feedback, snapshot_json in rows:
        try:
            # Parse snapshot JSON
            snapshot = json.loads(snapshot_json)
            
            # Only use positive feedback (label=1) for training
            # Negative feedback (label=-1) indicates the program wasn't good for that snapshot
            # We can use it as negative examples or skip it
            if feedback == 1:
                # Positive example: this program worked well for this snapshot
                features = featurize(snapshot)
                X_dicts.append(features)
                y_labels.append(program_id)
            # Optionally, we could use negative feedback to train against other programs
            # For now, we'll only use positive feedback
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Skipping invalid feedback entry: {e}")
            continue
    
    return X_dicts, y_labels

def retrain_model(db_path: str = 'temple.db', min_samples: int = 5) -> Dict[str, Any]:
    """
    Retrain the program recommendation model using feedback data
    
    Args:
        db_path: Path to SQLite database
        min_samples: Minimum number of samples required to train
    
    Returns:
        Dict with training results
    """
    print("Loading feedback data...")
    X_dicts, y_labels = load_feedback_data(db_path)
    
    if len(X_dicts) < min_samples:
        return {
            'success': False,
            'message': f'Not enough training data. Need at least {min_samples} samples, got {len(X_dicts)}.',
            'samples': len(X_dicts)
        }
    
    print(f"Found {len(X_dicts)} training samples")
    print(f"Training model on {len(set(y_labels))} unique programs...")
    
    try:
        # Train the model
        vectorizer, classifier, label_encoder = train_program_model(X_dicts, y_labels)
        
        # Get some statistics
        from collections import Counter
        program_counts = Counter(y_labels)
        
        return {
            'success': True,
            'message': f'Model retrained successfully on {len(X_dicts)} samples',
            'samples': len(X_dicts),
            'programs': len(set(y_labels)),
            'program_distribution': dict(program_counts)
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error during training: {str(e)}',
            'samples': len(X_dicts)
        }

def check_model_status(db_path: str = 'temple.db') -> Dict[str, Any]:
    """
    Check the status of the current model and available training data
    
    Returns:
        Dict with status information
    """
    # Check if model exists
    model_loaded = load_program_model() is not None
    
    # Count feedback entries
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM program_feedback WHERE feedback = 1")
    positive_feedback_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM program_feedback WHERE feedback = -1")
    negative_feedback_count = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT program_id) FROM program_feedback WHERE feedback = 1")
    unique_programs = c.fetchone()[0]
    conn.close()
    
    return {
        'model_exists': model_loaded,
        'positive_feedback': positive_feedback_count,
        'negative_feedback': negative_feedback_count,
        'unique_programs': unique_programs,
        'total_feedback': positive_feedback_count + negative_feedback_count
    }

if __name__ == '__main__':
    # Command-line usage
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        # Show status
        status = check_model_status()
        print("Model Status:")
        print(f"  Model exists: {status['model_exists']}")
        print(f"  Positive feedback: {status['positive_feedback']}")
        print(f"  Negative feedback: {status['negative_feedback']}")
        print(f"  Unique programs: {status['unique_programs']}")
        print(f"  Total feedback: {status['total_feedback']}")
    else:
        # Retrain model
        print("Starting model retraining...")
        result = retrain_model()
        
        if result['success']:
            print(f"✓ {result['message']}")
            print(f"  Samples: {result['samples']}")
            print(f"  Programs: {result['programs']}")
            if 'program_distribution' in result:
                print(f"  Program distribution: {result['program_distribution']}")
        else:
            print(f"✗ {result['message']}")
            sys.exit(1)

