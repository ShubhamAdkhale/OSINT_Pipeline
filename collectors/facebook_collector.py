import os
from facebook import GraphAPI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def fetch_facebook(query="OSINT", limit=20):
    """
    Due to Facebook API restrictions, this function now returns simulated data
    for demonstration and testing purposes.
    
    Args:
        query (str): Search query (used to generate relevant sample data)
        limit (int): Maximum number of items to return
    Returns:
        list: List of dictionaries containing simulated Facebook data
    """
    print("⚠️ Facebook Graph API search feature is currently unavailable")
    print("Using simulated data for demonstration purposes")
    
    # Generate sample data based on the query
    samples = []
    topics = ["AI", "cybersecurity", "data privacy", "social media", "technology"]
    
    for i in range(min(limit, 5)):
        samples.append({
            "platform": "facebook",
            "user": f"Demo User {i+1}",
            "user_id": f"demo_{i+1}",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "text": f"Sample post about {query} and {topics[i]} #OSINT #Research",
            "post_id": f"sample_{i+1}",
            "url": f"https://facebook.com/demo/post_{i+1}",
            "likes": 100 + i * 10,
            "shares": 20 + i * 5
        })
    
    return samples
        
if __name__ == "__main__":
    # Test the function
    results = fetch_facebook(query="cybersecurity", limit=5)
    for post in results:
        print(f"\nPost by {post['user']}:")
        print(f"Text: {post['text'][:100]}...")  # Print first 100 chars
        print(f"URL: {post['url']}")
        print("-" * 50)
