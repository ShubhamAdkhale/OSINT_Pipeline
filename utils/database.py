import sqlite3
import os

def save_to_db(records, db_path="data/osint.db"):
    """Save records to SQLite database"""
    if not records:
        print("⚠️  No records to save to database")
        return
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create table with additional fields for flexibility
    cur.execute("""CREATE TABLE IF NOT EXISTS osint_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        platform TEXT, 
        user TEXT, 
        timestamp TEXT, 
        text TEXT, 
        url TEXT, 
        sentiment REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    # Insert records
    for r in records:
        try:
            cur.execute("INSERT INTO osint_data (platform, user, timestamp, text, url, sentiment) VALUES (?, ?, ?, ?, ?, ?)",
                        (r.get("platform", "unknown"), 
                         r.get("user", "unknown"), 
                         r.get("timestamp", ""), 
                         r.get("text", ""), 
                         r.get("url", ""), 
                         r.get("sentiment", 0)))
        except Exception as e:
            print(f"Error inserting record: {e}")
            continue
    
    conn.commit()
    conn.close()
    print(f"💾 Saved {len(records)} records to {db_path}")