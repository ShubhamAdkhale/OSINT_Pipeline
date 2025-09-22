import os
from github import Github
from github.GithubException import GithubException, RateLimitExceededException
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def handle_rate_limit(g):
    """Handle GitHub API rate limiting"""
    try:
        rate_limit = g.get_rate_limit()
        core_rate_limit = rate_limit.core
        if core_rate_limit.remaining == 0:
            reset_timestamp = core_rate_limit.reset.timestamp()
            sleep_time = reset_timestamp - time.time()
            if sleep_time > 0:
                print(f"Rate limit exceeded. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time + 1)
    except Exception as e:
        print(f"Warning: Could not check rate limit - {str(e)}")

def fetch_github(query="OSINT", days_back=7, limit=20):
    """
    Fetch GitHub repositories and issues based on search query
    Args:
        query (str): Search query
        days_back (int): Number of days to look back
        limit (int): Maximum number of items to fetch per category
    Returns:
        list: List of dictionaries containing GitHub data
    """
    if not isinstance(query, str) or not query.strip():
        print("Error: Invalid query provided")
        return []
    
    if not isinstance(days_back, int) or days_back < 1:
        print("Error: Invalid days_back parameter")
        return []
    
    if not isinstance(limit, int) or limit < 1:
        print("Error: Invalid limit parameter")
        return []
    try:
        # Initialize GitHub API with token
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            print("⚠️ GitHub token not found in environment variables")
            print("Please create a .env file with your GitHub token:")
            print("GITHUB_TOKEN=your_github_token")
            print("⚠️ Using sample data instead")
            
            # Return sample data
            return [{
                "platform": "github",
                "type": "repository",
                "text": f"Sample GitHub repository about {query}",
                "description": "This is a sample repository description",
                "user": "sample_user",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "url": "https://github.com/sample/repo",
                "stars": 100,
                "language": "Python"
            }]
        
        g = Github(github_token)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_query = f" created:>{start_date.strftime('%Y-%m-%d')}"
        
        results = []
        
        # Search repositories
        print("Searching GitHub repositories...")
        try:
            handle_rate_limit(g)
            repos = g.search_repositories(query + date_query, sort='stars', order='desc')
            
            repo_count = 0
            for repo in repos:
                if repo_count >= limit:
                    break
                    
                try:
                    results.append({
                        "platform": "github",
                        "type": "repository",
                        "name": repo.name,
                        "user": repo.owner.login if repo.owner else "Unknown",
                        "timestamp": str(repo.created_at) if repo.created_at else "",
                        "text": repo.description or "",
                        "url": repo.html_url or "",
                        "stars": repo.stargazers_count or 0,
                        "forks": repo.forks_count or 0,
                        "language": repo.language or "Unknown",
                        "topics": list(repo.get_topics()) if hasattr(repo, 'get_topics') else [],
                        "last_updated": str(repo.updated_at) if repo.updated_at else "",
                        "is_fork": bool(repo.fork),
                        "open_issues": repo.open_issues_count or 0
                    })
                    repo_count += 1
                except (AttributeError, GithubException) as e:
                    print(f"Error processing repository: {str(e)}")
                    continue
                    
        except RateLimitExceededException:
            print("Rate limit exceeded while searching repositories")
            handle_rate_limit(g)
        except GithubException as e:
            print(f"GitHub API error while searching repositories: {str(e)}")

        # Search issues and pull requests
        print("Searching GitHub issues and PRs...")
        try:
            handle_rate_limit(g)
            issues = g.search_issues(query + date_query, sort='comments', order='desc')
            
            issue_count = 0
            for issue in issues:
                if issue_count >= limit:
                    break
                    
                try:
                    results.append({
                        "platform": "github",
                        "type": "issue" if not getattr(issue, 'pull_request', None) else "pull_request",
                        "title": issue.title if hasattr(issue, 'title') else "No title",
                        "user": issue.user.login if issue.user else "Unknown",
                        "timestamp": str(issue.created_at) if issue.created_at else "",
                        "text": issue.body or "",
                        "url": issue.html_url if hasattr(issue, 'html_url') else "",
                        "repository": issue.repository.full_name if hasattr(issue, 'repository') else "Unknown",
                        "comments": issue.comments if hasattr(issue, 'comments') else 0,
                        "labels": [label.name for label in issue.labels] if hasattr(issue, 'labels') else [],
                        "state": issue.state if hasattr(issue, 'state') else "unknown",
                        "closed": issue.closed_at is not None if hasattr(issue, 'closed_at') else False,
                        "closed_at": str(issue.closed_at) if issue.closed_at else None
                    })
                    issue_count += 1
                except (AttributeError, GithubException) as e:
                    print(f"Error processing issue: {str(e)}")
                    continue
                    
        except RateLimitExceededException:
            print("Rate limit exceeded while searching issues")
            handle_rate_limit(g)
        except GithubException as e:
            print(f"GitHub API error while searching issues: {str(e)}")

        # Search code
        print("Searching GitHub code...")
        try:
            handle_rate_limit(g)
            code_query = f"{query} language:python language:javascript language:java"
            code_results = g.search_code(code_query, order='desc')
            
            code_count = 0
            for code in code_results:
                if code_count >= limit:
                    break
                    
                try:
                    # Get file extension safely
                    file_ext = code.name.split('.')[-1] if '.' in code.name else "unknown"
                    
                    # Try to get content safely
                    try:
                        content = code.decoded_content.decode('utf-8') if code.decoded_content else ""
                        if len(content) > 5000:
                            content = content[:5000] + "... (content truncated)"
                    except (AttributeError, UnicodeDecodeError):
                        content = "Unable to decode content"
                    
                    results.append({
                        "platform": "github",
                        "type": "code",
                        "file_name": code.name if hasattr(code, 'name') else "unknown",
                        "user": code.repository.owner.login if hasattr(code.repository, 'owner') else "Unknown",
                        "timestamp": str(getattr(code, 'last_modified', datetime.now())),
                        "text": content,
                        "url": code.html_url if hasattr(code, 'html_url') else "",
                        "repository": code.repository.full_name if hasattr(code.repository, 'full_name') else "Unknown",
                        "path": code.path if hasattr(code, 'path') else "",
                        "language": file_ext
                    })
                    code_count += 1
                except (AttributeError, GithubException) as e:
                    print(f"Error processing code file: {str(e)}")
                    continue
                    
        except RateLimitExceededException:
            print("Rate limit exceeded while searching code")
            handle_rate_limit(g)
        except GithubException as e:
            print(f"GitHub API error while searching code: {str(e)}")

        return results

    except Exception as e:
        print(f"Error fetching GitHub data: {str(e)}")
        return []

def get_repository_details(repo_name, include_commits=True, commit_limit=10):
    """
    Get detailed information about a specific repository
    Args:
        repo_name (str): Repository name in format "owner/repo"
        include_commits (bool): Whether to include recent commits
        commit_limit (int): Maximum number of commits to fetch
    Returns:
        dict: Repository details
    """
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GitHub token not found in environment variables")
        
        g = Github(github_token)
        repo = g.get_repo(repo_name)
        
        # Get basic repository information
        details = {
            "platform": "github",
            "type": "repository_details",
            "name": repo.name,
            "full_name": repo.full_name,
            "owner": repo.owner.login,
            "description": repo.description,
            "url": repo.html_url,
            "created_at": str(repo.created_at),
            "updated_at": str(repo.updated_at),
            "pushed_at": str(repo.pushed_at),
            "language": repo.language,
            "topics": repo.get_topics(),
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "watchers": repo.watchers_count,
            "open_issues": repo.open_issues_count,
            "default_branch": repo.default_branch,
            "license": repo.license.name if repo.license else None,
            "size": repo.size,
            "is_fork": repo.fork,
            "has_wiki": repo.has_wiki,
            "has_pages": repo.has_pages,
            "archived": repo.archived
        }
        
        if include_commits:
            # Get recent commits
            commits = []
            for commit in repo.get_commits()[:commit_limit]:
                commits.append({
                    "sha": commit.sha,
                    "author": commit.author.login if commit.author else "Unknown",
                    "date": str(commit.commit.author.date),
                    "message": commit.commit.message,
                    "url": commit.html_url
                })
            details["recent_commits"] = commits
        
        return details

    except Exception as e:
        print(f"Error fetching repository details: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the GitHub collector
    print("Testing GitHub collector...")
    
    # Test basic search
    results = fetch_github(query="cybersecurity", days_back=7, limit=5)
    
    print("\nSearch Results:")
    for item in results:
        print(f"\nType: {item['type']}")
        print(f"User: {item['user']}")
        if item['type'] == 'repository':
            print(f"Name: {item['name']}")
            print(f"Stars: {item['stars']}")
        elif item['type'] in ['issue', 'pull_request']:
            print(f"Title: {item['title']}")
            print(f"Comments: {item['comments']}")
        print(f"URL: {item['url']}")
        print("-" * 50)
    
    # Test repository details
    print("\nTesting repository details...")
    repo_details = get_repository_details("octocat/Hello-World")
    if repo_details:
        print(f"Repository: {repo_details['full_name']}")
        print(f"Stars: {repo_details['stars']}")
        print(f"Description: {repo_details['description']}")
        if 'recent_commits' in repo_details:
            print("\nRecent commits:")
            for commit in repo_details['recent_commits'][:3]:
                print(f"- {commit['message'][:50]}...")