import re
from langdetect import detect, DetectorFactory
import logging

# Make langdetect more deterministic
DetectorFactory.seed = 0

# Set up logging to avoid langdetect warnings
logging.getLogger('langdetect').setLevel(logging.ERROR)

def clean_text(text):
    """Clean text by removing URLs and special characters"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^A-Za-z0-9\s.,!?]", "", text)  # Keep basic punctuation
        text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
        return text.strip()
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

def filter_english(records):
    """Filter records to only include English text"""
    english_records = []
    
    for r in records:
        try:
            text = r.get("text", "")
            if text and detect(text) == "en":
                english_records.append(r)
        except:
            # If language detection fails, skip the record
            continue
            
    return english_records