import tweepy
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()

def create_api():
    """
    Create and authenticate Twitter API instance
    Returns:
        tweepy.API: Authenticated API instance
    """
    # Check if credentials exist
    if not os.getenv("TWITTER_KEY") or not os.getenv("TWITTER_SECRET"):
        print("⚠️ Twitter API credentials not found in environment variables")
        print("Please create a .env file with your Twitter API credentials:")
        print("TWITTER_KEY=your_api_key")
        print("TWITTER_SECRET=your_api_secret")
        return None
        
    try:
        auth = tweepy.AppAuthHandler(
            os.getenv("TWITTER_KEY"),
            os.getenv("TWITTER_SECRET")
        )
        api = tweepy.API(auth, wait_on_rate_limit=True)
        return api
    except Exception as e:
        print(f"⚠️ Error creating Twitter API instance: {str(e)}")
        print("This might be due to API access limitations.")
        print("Consider using an alternative data source or upgrading your Twitter API access level.")
        return None

def fetch_twitter(query="OSINT", limit=20, include_replies=False, include_retweets=False):
    """
    Fetch tweets using Twitter API
    Args:
        query (str): Search query
        limit (int): Maximum number of tweets to fetch
        include_replies (bool): Whether to include replies
        include_retweets (bool): Whether to include retweets
    Returns:
        list: List of dictionaries containing tweet data
    """
    try:
        api = create_api()
        if not api:
            return []

        # Modify query based on parameters
        if not include_retweets:
            query += " -filter:retweets"
        if not include_replies:
            query += " -filter:replies"

        # Add language filter
        query += " lang:en"

        results = []
        for tweet in tweepy.Cursor(api.search_tweets, 
                                 q=query,
                                 tweet_mode="extended",
                                 include_entities=True).items(limit):
            
            # Extract hashtags
            hashtags = [tag['text'] for tag in tweet.entities.get('hashtags', [])]
            
            # Extract user mentions
            mentions = [mention['screen_name'] 
                       for mention in tweet.entities.get('user_mentions', [])]
            
            # Extract media URLs
            media_urls = []
            if hasattr(tweet, 'extended_entities') and 'media' in tweet.extended_entities:
                for media in tweet.extended_entities['media']:
                    if media['type'] == 'photo':
                        media_urls.append(media['media_url_https'])
                    elif media['type'] == 'video':
                        variants = media['video_info']['variants']
                        # Get the highest bitrate video URL
                        video_urls = [v['url'] for v in variants if v['content_type'] == 'video/mp4']
                        if video_urls:
                            media_urls.append(video_urls[0])

            # Build tweet data dictionary
            tweet_data = {
                "platform": "twitter",
                "tweet_id": tweet.id_str,
                "user": tweet.user.screen_name,
                "user_display_name": tweet.user.name,
                "user_followers": tweet.user.followers_count,
                "user_verified": tweet.user.verified,
                "timestamp": str(tweet.created_at),
                "text": tweet.full_text,
                "url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
                "retweet_count": tweet.retweet_count,
                "favorite_count": tweet.favorite_count,
                "hashtags": hashtags,
                "mentions": mentions,
                "media_urls": media_urls,
                "is_retweet": hasattr(tweet, 'retweeted_status'),
                "source": tweet.source,
                "coordinates": tweet.coordinates.coordinates if tweet.coordinates else None,
                "place": tweet.place.full_name if tweet.place else None,
            }
            
            results.append(tweet_data)

        return results

    except Exception as e:
        print(f"Error fetching Twitter data: {str(e)}")
        return []

def fetch_user_tweets(username, limit=20):
    """
    Fetch tweets from a specific user
    Args:
        username (str): Twitter username (without @)
        limit (int): Maximum number of tweets to fetch
    Returns:
        list: List of dictionaries containing tweet data
    """
    try:
        api = create_api()
        if not api:
            return []

        results = []
        for tweet in tweepy.Cursor(api.user_timeline, 
                                 screen_name=username,
                                 tweet_mode="extended",
                                 include_entities=True).items(limit):
            
            # Extract hashtags and mentions
            hashtags = [tag['text'] for tag in tweet.entities.get('hashtags', [])]
            mentions = [mention['screen_name'] 
                       for mention in tweet.entities.get('user_mentions', [])]

            tweet_data = {
                "platform": "twitter",
                "tweet_id": tweet.id_str,
                "timestamp": str(tweet.created_at),
                "text": tweet.full_text,
                "url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
                "retweet_count": tweet.retweet_count,
                "favorite_count": tweet.favorite_count,
                "hashtags": hashtags,
                "mentions": mentions,
                "is_retweet": hasattr(tweet, 'retweeted_status')
            }
            
            results.append(tweet_data)

        return results

    except Exception as e:
        print(f"Error fetching user tweets: {str(e)}")
        return []

def save_to_json(tweets, filename="twitter_data.json"):
    """
    Save tweets to a JSON file
    Args:
        tweets (list): List of tweet dictionaries
        filename (str): Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tweets, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")

if __name__ == "__main__":
    # Test the main search function
    print("Testing Twitter search...")
    tweets = fetch_twitter(query="cybersecurity", limit=5)
    
    for tweet in tweets:
        print("\nTweet Details:")
        print(f"User: @{tweet['user']} ({tweet['user_display_name']})")
        print(f"Text: {tweet['text'][:100]}...")  # First 100 chars
        print(f"Time: {tweet['timestamp']}")
        print(f"Engagement: {tweet['retweet_count']} RTs, {tweet['favorite_count']} Likes")
        if tweet['hashtags']:
            print(f"Hashtags: {tweet['hashtags']}")
        if tweet['media_urls']:
            print(f"Media: {tweet['media_urls']}")
        print(f"URL: {tweet['url']}")
        print("-" * 50)
    
    # Test user timeline function
    print("\nTesting user timeline...")
    user_tweets = fetch_user_tweets("cybersecurity", limit=3)
    
    for tweet in user_tweets:
        print("\nTweet Details:")
        print(f"Time: {tweet['timestamp']}")
        print(f"Text: {tweet['text'][:100]}...")
        print(f"URL: {tweet['url']}")
        print("-" * 50)
    
    # Save results to JSON if any tweets were fetched
    if tweets:
        save_to_json(tweets, "recent_cybersecurity_tweets.json")