from collectors.facebook_collector import fetch_facebook
from collectors.twitter_collector import fetch_twitter
from collectors.reddit_collector import fetch_reddit
from collectors.github_collector import fetch_github

from utils.cleaner import clean_text, filter_english
from utils.database import save_to_db
from utils.sentiment import add_sentiment

def run_pipeline():
    data = []
    data.extend(fetch_twitter("AI", 10))
    data.extend(fetch_reddit("technology", 10))
    data.extend(fetch_facebook("open source intelligence", 5))
    data.extend(fetch_github("leak", 5))

    processed_data = []
    for i, d in enumerate(data):
        try:
            if not isinstance(d, dict):
                print(f"⚠️  Skipping non-dict item {i}")
                continue
                
            # Ensure text field exists - check multiple possible field names
            text_fields = ['text', 'content', 'description', 'title', 'body']
            for field in text_fields:
                if field in d and d[field]:
                    d['text'] = str(d[field])
                    break
            else:
                d['text'] = ''  # No text field found
                
            # Ensure other required fields exist
            d.setdefault('platform', 'unknown')
            d.setdefault('user', 'unknown')
            d.setdefault('timestamp', '2024-01-01 00:00:00')
            d.setdefault('url', '')
            
            # Clean the text
            d["text"] = clean_text(d["text"])
            
            processed_data.append(d)
            
        except Exception as e:
            print(f"⚠️  Error processing item {i}: {e}")
            continue
    
    print(f"✅ Successfully processed: {len(processed_data)} items")
    
    # Filter and analyze
    if processed_data:
        try:
            english_data = filter_english(processed_data)
            print(f"🌍 English content: {len(english_data)} items")
            
            sentiment_data = add_sentiment(english_data)
            save_to_db(sentiment_data)
            print(f"💾 Saved {len(sentiment_data)} OSINT records to database")
        except Exception as e:
            print(f"❌ Error in analysis/saving: {e}")
    else:
        print("❌ No valid data to process")

def view_database():
    """View all records in the database"""
    import sqlite3
    import pandas as pd
    import os
    
    try:
        db_path = "data/osint.db"
        
        # Check if database file exists
        if not os.path.exists(db_path):
            print("❌ Database file does not exist yet. Run the pipeline first.")
            return
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='osint_data'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("❌ Database table 'osint_data' does not exist yet")
            conn.close()
            return
        
        # Read data
        df = pd.read_sql_query("SELECT * FROM osint_data", conn)
        
        print("📊 DATABASE CONTENTS:")
        print(f"Total records: {len(df)}")
        print("\n" + "="*80)
        
        if len(df) > 0:
            # Display in a readable format
            for idx, row in df.iterrows():
                print(f"\nRecord {idx + 1}:")
                print(f"  Platform: {row['platform']}")
                print(f"  User: {row['user']}")
                if pd.notna(row['text']):
                    text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                    print(f"  Text: {text_preview}")
                print(f"  Timestamp: {row['timestamp']}")
                print(f"  Sentiment: {row['sentiment']:.2f}")
                print(f"  URL: {row['url']}")
                print("-" * 40)
        else:
            print("No records found in database")
            
        conn.close()
        
    except Exception as e:
        print(f"Error accessing database: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OSINT Data Collection Pipeline')
    parser.add_argument('--view-db', action='store_true', help='View database contents')
    parser.add_argument('--run', action='store_true', help='Run the data collection pipeline')
    
    args = parser.parse_args()
    
    if args.view_db:
        view_database()
    elif args.run:
        run_pipeline()
    else:
        # Default: run pipeline then show database
        run_pipeline()
        print("\n" + "="*50)
        view_database()
