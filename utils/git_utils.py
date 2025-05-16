# ======== GitHub Repository Search ========

@lru_cache(maxsize=32)
def search_github_repos(query: str, n: int = 5) -> List[Dict[str, Any]]:
    """Search GitHub for repositories matching the query, with caching for efficiency."""
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": n
    }
    
    try:
        logger.info(f"Searching GitHub for: {query}")
        response = requests.get(f"{GITHUB_API_URL}/search/repositories", headers=headers, params=params)
        response.raise_for_status()
        repos = response.json().get("items", [])
        
        result = []
        for repo in repos:
            # Get additional details about each repo
            languages_url = repo.get("languages_url")
            languages = []
            if languages_url:
                lang_response = requests.get(languages_url, headers=headers)
                if lang_response.status_code == 200:
                    languages = list(lang_response.json().keys())
            
            # Get README content asynchronously to improve performance
            readme_content = get_readme(repo["full_name"], headers)
            
            # Create embedding for the repository
            embedding_text = f"{repo['name']} {repo['description'] or ''} {' '.join(repo.get('topics', []))}"
            
            repo_info = {
                "name": repo["name"],
                "full_name": repo["full_name"],
                "description": repo["description"] or "",
                "url": repo["html_url"],
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "languages": languages,
                "license": repo.get("license", {}).get("spdx_id") if repo.get("license") else None,
                "topics": repo.get("topics", []),
                "readme_summary": summarize_readme(readme_content),
                #"embedding": get_embeddings(embedding_text)
            }
            result.append(repo_info)
        
        return result
    except Exception as e:
        logger.error(f"Error searching GitHub: {e}")
        return []

@lru_cache(maxsize=64)
def get_readme(repo_full_name: str, headers: Dict[str, str]) -> str:
    """Get README content for a repository with caching."""
    try:
        # Try to get the default README
        response = requests.get(
            f"{GITHUB_API_URL}/repos/{repo_full_name}/readme",
            headers=headers
        )
        
        if response.status_code == 200:
            content = response.json().get("content", "")
            encoding = response.json().get("encoding", "")
            
            if encoding == "base64" and content:
                return base64.b64decode(content).decode('utf-8', errors='replace')
        
        return ""
    except Exception as e:
        logger.error(f"Error getting README: {e}")
        return ""

# ======== Semantic Matching Functions ========

def rank_repos_by_similarity(subproject: Dict[str, Any], repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank repositories by semantic similarity to subproject with improved algorithm."""
    if not repos or not subproject.get("embedding") or len(subproject["embedding"]) == 0:
        return repos
        
    # Calculate multiple similarity factors
    results = []
    for repo in repos:
        if not repo.get("embedding") or len(repo["embedding"]) == 0:
            continue
        
        # Calculate semantic similarity
        semantic_similarity = compute_similarity(subproject["embedding"], repo["embedding"])
        
        # Calculate tech requirement match score 
        tech_match_score = 0
        if repo.get("topics") and subproject.get("tech_requirements"):
            # Case-insensitive matching of tech requirements with repo topics
            repo_topics_lower = [topic.lower() for topic in repo["topics"]]
            matched_techs = sum(1 for req in subproject["tech_requirements"] 
                               if any(req.lower() in topic for topic in repo_topics_lower))
            tech_match_score = matched_techs / len(subproject["tech_requirements"]) if subproject["tech_requirements"] else 0
        
        # Factor in repository popularity (stars) as a quality signal
        popularity_score = min(1.0, repo.get("stars", 0) / 10000)  # Cap at 10k stars for normalization
        
        # Combined score with weights 
        combined_score = (
            0.5 * semantic_similarity +  # Semantic similarity is most important
            0.3 * tech_match_score +     # Technical match is important
            0.2 * popularity_score       # Popularity provides some quality assurance
        )
        
        results.append((combined_score, repo))
    
    # Sort by combined score
    results.sort(reverse=True, key=lambda x: x[0])
    
    # Return sorted repositories
    return [repo for _, repo in results]
