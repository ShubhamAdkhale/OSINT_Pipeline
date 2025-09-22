from textblob import TextBlob

def add_sentiment(records):
    for r in records:
        r["sentiment"] = TextBlob(r["text"]).sentiment.polarity
    return records