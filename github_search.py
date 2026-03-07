#!/usr/bin/env python3
"""GitHub repository search and filtering functionality."""
import os
import json
import requests
import time
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

class GitHubSearcher:
    def __init__(self, token: str = None):
        self.token = token or os.environ.get('GITHUB_TOKEN')
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "QLoRA-Training-Pipeline"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def search_repos(self, query: str, min_stars: int = 100, min_forks: int = 50, 
                    max_results: int = 50, languages: List[str] = None) -> List[Dict]:
        """Search GitHub repositories with filtering."""
        if languages is None:
            languages = ['python', 'javascript']
        
        repos = []
        page = 1
        per_page = min(100, max_results)
        
        # Build search query
        language_query = ' '.join([f'language:{lang}' for lang in languages])
        full_query = f"{query} {language_query} stars:>={min_stars} forks:>={min_forks}"
        
        logging.info(f"Searching with query: {full_query}")
        
        while len(repos) < max_results:
            url = f"{self.base_url}/search/repositories"
            params = {
                "q": full_query,
                "sort": "stars",
                "order": "desc",
                "page": page,
                "per_page": per_page
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'items' not in data or not data['items']:
                    break
                
                for repo in data['items']:
                    if len(repos) >= max_results:
                        break
                    
                    repo_info = {
                        'name': repo['full_name'],
                        'url': repo['clone_url'],
                        'stars': repo['stargazers_count'],
                        'forks': repo['forks_count'],
                        'language': repo['language'],
                        'description': repo.get('description', ''),
                        'created_at': repo['created_at'],
                        'updated_at': repo['updated_at']
                    }
                    repos.append(repo_info)
                
                logging.info(f"Found {len(repos)} repositories so far...")
                
                # Check rate limit
                remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                if remaining < 10:
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    wait_time = max(reset_time - time.time(), 60)
                    logging.info(f"Rate limit low, waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time)
                
                page += 1
                time.sleep(1)  # Be nice to GitHub
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error searching repositories: {e}")
                break
        
        # Sort by stars then forks
        repos.sort(key=lambda x: (x['stars'], x['forks']), reverse=True)
        return repos[:max_results]
    
    def save_repo_list(self, repos: List[Dict], output_file: str):
        """Save repository list to file."""
        # Save as simple URL list for compatibility
        with open(output_file, 'w') as f:
            for repo in repos:
                f.write(f"{repo['url']}\n")
        
        # Also save detailed JSON
        json_file = output_file.replace('.txt', '_details.json')
        with open(json_file, 'w') as f:
            json.dump(repos, f, indent=2)
        
        logging.info(f"Saved {len(repos)} repositories to {output_file}")

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search GitHub repositories')
    parser.add_argument('--query', default='', help='Search query')
    parser.add_argument('--min-stars', type=int, default=100, help='Minimum stars')
    parser.add_argument('--min-forks', type=int, default=50, help='Minimum forks')
    parser.add_argument('--max-results', type=int, default=50, help='Maximum results')
    parser.add_argument('--languages', default='python,javascript', help='Comma-separated languages')
    parser.add_argument('--output', default='repos_searched.txt', help='Output file')
    
    args = parser.parse_args()
    
    # Load config if available
    config_file = 'config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        # Override with config values
        args.min_stars = config.get('min_stars', args.min_stars)
        args.min_forks = config.get('min_forks', args.min_forks)
        args.max_results = config.get('max_repos', args.max_results)
        languages = config.get('search_languages', args.languages)
    else:
        languages = args.languages
    
    searcher = GitHubSearcher()
    repos = searcher.search_repos(
        query=args.query,
        min_stars=args.min_stars,
        min_forks=args.min_forks,
        max_results=args.max_results,
        languages=[l.strip() for l in languages.split(',')]
    )
    
    searcher.save_repo_list(repos, args.output)

if __name__ == '__main__':
    main()
