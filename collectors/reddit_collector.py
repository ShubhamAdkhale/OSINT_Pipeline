import praw
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def fetch_reddit(query="OSINT", limit=20):
    """
    Fetch posts from Reddit using PRAW
    Args:
        query (str): Search query
        limit (int): Maximum number of posts to fetch
    Returns:
        list: List of dictionaries containing post data
    """
    try:
        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_ID"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent="OSINT_Scraper/1.0"
        )

        # Search for posts
        results = []
        for submission in reddit.subreddit("all").search(query, limit=limit):
            # Convert timestamp to datetime
            created_time = datetime.fromtimestamp(submission.created_utc)
            
            # Extract post data
            results.append({
                "platform": "reddit",
                "user": str(submission.author),
                "subreddit": str(submission.subreddit),
                "timestamp": str(created_time),
                "title": submission.title,
                "text": submission.selftext,
                "score": submission.score,
                "url": f"https://reddit.com{submission.permalink}",
                "num_comments": submission.num_comments,
                "is_original_content": submission.is_original_content
            })

        return results

    except Exception as e:
        print(f"Error fetching Reddit data: {str(e)}")
        return []

def fetch_reddit_comments(post_id, limit=20):
    """
    Fetch comments from a specific Reddit post
    Args:
        post_id (str): Reddit post ID
        limit (int): Maximum number of comments to fetch
    Returns:
        list: List of dictionaries containing comment data
    """
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_ID"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent="OSINT_Scraper/1.0"
        )

        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Flatten comment tree
        
        comments = []
        for comment in submission.comments.list()[:limit]:
            created_time = datetime.fromtimestamp(comment.created_utc)
            
            comments.append({
                "platform": "reddit",
                "user": str(comment.author),
                "timestamp": str(created_time),
                "text": comment.body,
                "score": comment.score,
                "url": f"https://reddit.com{comment.permalink}",
                "is_submitter": comment.is_submitter
            })

        return comments

    except Exception as e:
        print(f"Error fetching Reddit comments: {str(e)}")
        return []

if __name__ == "__main__":
    # Test post collection
    print("Testing post collection...")
    posts = fetch_reddit(query="cybersecurity", limit=5)
    for post in posts:
        print(f"\nPost in r/{post['subreddit']} by {post['user']}:")
        print(f"Title: {post['title']}")
        print(f"Score: {post['score']}")
        print(f"URL: {post['url']}")
        print("-" * 50)

    # Test comment collection for the first post if available
    if posts:
        print("\nTesting comment collection...")
        post_id = posts[0]['url'].split('comments/')[1].split('/')[0]
        comments = fetch_reddit_comments(post_id, limit=3)
        for comment in comments:
            print(f"\nComment by {comment['user']}:")
            print(f"Text: {comment['text'][:100]}...")  # Print first 100 chars
            print(f"Score: {comment['score']}")
            print("-" * 50)
