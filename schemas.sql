-- Additional database schemas for AI enhancements
-- Run only if tables don't exist

-- Structured feedback table for program recommendations
CREATE TABLE IF NOT EXISTS program_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT,
  program_id INTEGER,
  username TEXT,
  feedback INTEGER,
  snapshot_json TEXT
);

-- Model metadata table for tracking ML models
CREATE TABLE IF NOT EXISTS model_meta (
  model_name TEXT PRIMARY KEY,
  storage_path TEXT,
  last_trained TEXT,
  version INTEGER
);

