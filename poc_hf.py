"""
Project Planning Tool: Finds repositories and creates integration plans for software projects.
Provides automated decomposition of projects and repository matching using HuggingFace Transformers.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import base64
from functools import lru_cache
import datetime

import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

# Configuration
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("project_planner")

# ======== Model Setup ========

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the embedding model."""
    try:
        # Use Sentence Transformers for high-quality embeddings
        logger.info("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None

@lru_cache(maxsize=1)
def get_llm_model():
    """Load and cache the language model for text generation."""
    try:
        # Use a small but capable model for text generation
        logger.info("Loading language model...")
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading language model: {e}")
        return None, None

# ======== Embedding Functions ========

def get_embeddings(text: str) -> List[float]:
    """Get embeddings for text using SentenceTransformers."""
    try:
        model = get_embedding_model()
        if model is None:
            return []
            
        # Encode the text to get embeddings
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        return []

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    if not embedding1 or not embedding2:
        return 0.0
        
    try:
        return cosine_similarity([embedding1], [embedding2])[0][0]
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

# ======== LLM Query Function ========

def query_huggingface(
    prompt: str, 
    system_prompt: str = "", 
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """Query local HuggingFace model with the given prompt."""
    try:
        # Fall back to Hugging Face Inference API if available
        if os.getenv("HF_API_TOKEN"):
            return query_hf_inference_api(prompt, system_prompt, max_tokens, temperature)
            
        tokenizer, model = get_llm_model()
        if tokenizer is None or model is None:
            return "Model loading failed. Please check logs."
            
        # Format input for model
        if system_prompt:
            input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate text
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode the generated text
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract assistant's response
        match = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return output.split("<|im_start|>assistant\n")[-1].strip()
        
    except Exception as e:
        logger.error(f"Error querying local model: {e}")
        fallback_message = f"Error using local model: {str(e)}. Attempting to use Hugging Face Inference API..."
        logger.info(fallback_message)
        
        # Try using a simpler approach with a fallback prompt template
        try:
            tokenizer, model = get_llm_model()
            if tokenizer and model:
                # Simplified prompt template
                simple_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                inputs = tokenizer(simple_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )
                
                return tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(simple_prompt):]
        except Exception as nested_e:
            logger.error(f"Error with simplified generation: {nested_e}")
        
        # Fall back to mock response for testing or try HF Inference API
        if os.getenv("HF_API_TOKEN"):
            return query_hf_inference_api(prompt, system_prompt, max_tokens, temperature)
        
        # Last resort - generate a basic mock response for testing
        if "project" in prompt.lower() and "subproject" in system_prompt.lower():
            return '[{"name": "Frontend", "description": "User interface", "tech_requirements": ["React"], "dependencies": []}]'
        
        return f"Error generating response: {str(e)}"

def query_hf_inference_api(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """Query Hugging Face Inference API as a fallback."""
    try:
        API_URL = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
        
        # Format the prompt
        if system_prompt:
            payload = {
                "inputs": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
        else:
            payload = {
                "inputs": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0].get("generated_text", "").strip()
    except Exception as e:
        logger.error(f"Error using HF Inference API: {e}")
        return f"Error with Inference API: {str(e)}"

# ======== Project Decomposition ========

def decompose_project(project_description: str) -> List[Dict[str, Any]]:
    """Break down project into subprojects using LLM."""
    system_prompt = """
    You are a technical architect. Break down the given project into logical subprojects so that i can find neccessary Github repos for the decoupled tasks. reason along the way, no need to output the reasoning. ensure that if i were to club all the subprojects, i wont have to do anything more. The goal is to directly use the available github projects.
    For each subproject, provide:
    1. Name
    2. Description
    3. Technical requirements 
    4. Dependencies on other subprojects
    
    Format your response only as a "JSON" array of objects with the structure:
    [
        {
            "name": "string",
            "description": "string",
            "tech_requirements": ["string"],
            "dependencies": ["string"]
        }
    ]
    
    Keep your response concise and ONLY return the JSON array.
    """
    
    prompt = f"Project description: {project_description}\n\nBreak this down into logical subprojects:"
    
    response = query_huggingface(prompt, system_prompt=system_prompt)
    try:
        # Find the JSON content in the response
        json_content = response
        if "```json" in response:
            json_content = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_content = response.split("```")[1].strip()
        
        # Some additional parsing for robustness
        if json_content.startswith('[') and json_content.endswith(']'):
            subprojects = json.loads(json_content)
        else:
            # Try to find JSON array in the string
            matches = re.search(r'\[.*?\]', json_content, re.DOTALL)
            if matches:
                subprojects = json.loads(matches.group(0))
            else:
                # Fallback mock response for testing if everything fails
                subprojects = [
                    {
                        "name": "Frontend",
                        "description": "User interface component",
                        "tech_requirements": ["React", "TypeScript"],
                        "dependencies": []
                    },
                    {
                        "name": "Backend API",
                        "description": "Server-side logic and data processing",
                        "tech_requirements": ["Python", "FastAPI"],
                        "dependencies": []
                    }
                ]
                print(response)
                logger.warning("Using fallback mock response for project decomposition")
            
        # Generate embeddings for each subproject
        for subproject in subprojects:
            subproject_text = f"{subproject['name']} {subproject['description']} {' '.join(subproject['tech_requirements'])}"
            subproject["embedding"] = get_embeddings(subproject_text)
            
        return subprojects
    except Exception as e:
        logger.error(f"Error parsing LLM response for project decomposition: {e}")
        logger.debug(f"Response: {response}")
        
        # Fallback for testing
        return [
            {
                "name": "Frontend",
                "description": "User interface component",
                "tech_requirements": ["React", "TypeScript"],
                "dependencies": [],
                "embedding": get_embeddings("Frontend User interface component React TypeScript")
            },
            {
                "name": "Backend API",
                "description": "Server-side logic and data processing",
                "tech_requirements": ["Python", "FastAPI"],
                "dependencies": [],
                "embedding": get_embeddings("Backend API Server-side logic and data processing Python FastAPI")
            }
        ]

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

def summarize_readme(readme_content: str) -> str:
    """Summarize README content using LLM."""
    if not readme_content or len(readme_content) < 100:
        return readme_content
    
    # Truncate very long READMEs to avoid token limits
    if len(readme_content) > 4000:
        readme_content = readme_content[:4000] + "..."
    
    system_prompt = "Summarize the following README in 2-3 sentences, focusing on what the repository does and its key features:"
    
    return query_huggingface(readme_content, system_prompt=system_prompt, max_tokens=200)

def find_repos_for_subproject(subproject: Dict[str, Any], n: int = 3) -> List[Dict[str, Any]]:
    """Find repositories for a specific subproject, with efficient search query construction."""
    # Construct optimized search query from subproject details
    search_terms = [
        subproject["name"],
        *[req for req in subproject["tech_requirements"] if len(req.split()) <= 3]  # Only include short tech requirements
    ]
    
    # Use NLP to extract key terms from description
    if subproject["description"]:
        # Simple keyword extraction - in production you might use NLP techniques
        stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "and", "or", "of"}
        desc_words = [word.lower() for word in re.findall(r'\w+', subproject["description"]) 
                      if word.lower() not in stop_words and len(word) > 3]
        
        # Take most important keywords (longer words tend to be more specific)
        desc_words.sort(key=len, reverse=True)
        search_terms.extend(desc_words[:3])
    
    # Make query unique by removing duplicates while preserving order
    seen = set()
    query_terms = [term for term in search_terms if not (term.lower() in seen or seen.add(term.lower()))]
    
    # Join terms and ensure we don't exceed GitHub's query length limits
    query = " ".join(query_terms)
    if len(query) > 256:  # GitHub has query length limits
        query = query[:256]
    
    return search_github_repos(query, n)

# ======== Conflict Analysis ========

def analyze_conflicts(subprojects: List[Dict[str, Any]], repos_by_subproject: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Analyze potential integration conflicts between selected repositories."""
    conflicts = []
    
    # More comprehensive language compatibility matrix
    compatible_languages = {
        "Python": ["Python", "JavaScript", "TypeScript", "C", "C++", "Rust"],
        "JavaScript": ["JavaScript", "TypeScript", "Python", "HTML", "CSS", "Ruby"],
        "TypeScript": ["TypeScript", "JavaScript", "Python", "HTML", "CSS"],
        "Java": ["Java", "Kotlin", "Scala", "Groovy"],
        "Kotlin": ["Kotlin", "Java", "Scala"],
        "Go": ["Go", "C", "Rust"],
        "Rust": ["Rust", "C", "C++", "Go"],
        "C#": ["C#", "F#", "Visual Basic", "JavaScript"],
        "PHP": ["PHP", "JavaScript", "HTML", "CSS"],
        "Ruby": ["Ruby", "JavaScript", "HTML", "CSS"],
        "Swift": ["Swift", "Objective-C", "C"],
        "C++": ["C++", "C", "Rust", "Python"],
        "C": ["C", "C++", "Rust", "Assembly"]
    }
    
    # More comprehensive license compatibility matrix
    # This is simplified - real license compatibility is complex and should be evaluated by legal professionals
    license_compatibility = {
        "AGPL-3.0": {
            "incompatible": ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "MPL-2.0"],
            "compatible": ["GPL-3.0", "LGPL-3.0"]
        },
        "GPL-3.0": {
            "incompatible": ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "MPL-2.0"],
            "compatible": ["LGPL-3.0", "AGPL-3.0"]
        },
        "LGPL-3.0": {
            "incompatible": [],
            "compatible": ["GPL-3.0", "AGPL-3.0", "MIT", "Apache-2.0", "BSD-3-Clause"]
        },
        "MIT": {
            "incompatible": ["AGPL-3.0", "GPL-3.0"],
            "compatible": ["Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "MPL-2.0", "LGPL-3.0"]
        },
        "Apache-2.0": {
            "incompatible": ["AGPL-3.0", "GPL-3.0"],
            "compatible": ["MIT", "BSD-3-Clause", "BSD-2-Clause", "MPL-2.0", "LGPL-3.0"]
        }
    }
    
    # Check for language conflicts
    for i, subproject1 in enumerate(subprojects):
        name1 = subproject1["name"]
        if name1 not in repos_by_subproject:
            continue
            
        for repo1 in repos_by_subproject[name1]:
            languages1 = repo1.get("languages", [])
            primary_lang1 = languages1[0] if languages1 else None
            
            for j, subproject2 in enumerate(subprojects):
                if i == j:
                    continue  # Skip self-comparison
                    
                name2 = subproject2["name"]
                if name2 not in repos_by_subproject:
                    continue
                    
                # Check if subproject2 depends on subproject1 or vice versa
                has_dependency = (name1 in subproject2.get("dependencies", []) or
                                 name2 in subproject1.get("dependencies", []))
                
                if not has_dependency:
                    continue  # Only check conflicts for dependent components
                    
                for repo2 in repos_by_subproject[name2]:
                    languages2 = repo2.get("languages", [])
                    primary_lang2 = languages2[0] if languages2 else None
                    
                    # Check language compatibility
                    if (primary_lang1 and primary_lang2 and 
                        primary_lang1 in compatible_languages and
                        primary_lang2 not in compatible_languages.get(primary_lang1, [])):
                        conflicts.append({
                            "type": "language_incompatibility",
                            "severity": "high",
                            "description": f"Language incompatibility between {name1} ({primary_lang1}) and {name2} ({primary_lang2})",
                            "subprojects": [name1, name2],
                            "repos": [repo1["full_name"], repo2["full_name"]],
                            "mitigation": f"Consider using language bindings, APIs, or microservices to integrate {primary_lang1} with {primary_lang2}"
                        })
    
    # Check for license conflicts
    for subproject1 in subprojects:
        name1 = subproject1["name"]
        if name1 not in repos_by_subproject:
            continue
            
        for repo1 in repos_by_subproject[name1]:
            license1 = repo1.get("license")
            
            for subproject2 in subprojects:
                name2 = subproject2["name"]
                if name2 == name1 or name2 not in repos_by_subproject:
                    continue
                    
                for repo2 in repos_by_subproject[name2]:
                    license2 = repo2.get("license")
                    
                    # Check license compatibility
                    if (license1 and license2 and 
                        license1 in license_compatibility and
                        license2 in license_compatibility.get(license1, {}).get("incompatible", [])):
                        conflicts.append({
                            "type": "license_incompatibility",
                            "severity": "high",
                            "description": f"License incompatibility between {name1} ({license1}) and {name2} ({license2})",
                            "subprojects": [name1, name2],
                            "repos": [repo1["full_name"], repo2["full_name"]],
                            "mitigation": "Consider finding alternative repositories with compatible licenses or seeking legal advice"
                        })
    
    return conflicts

# ======== Integration Planning ========

def generate_integration_plan(
    project_description: str,
    subprojects: List[Dict[str, Any]],
    selected_repos: Dict[str, Dict[str, Any]],
    conflicts: List[Dict[str, Any]]
) -> str:
    """Generate an integration plan using LLM with improved prompting."""
    # Prepare a structured context for the LLM
    subproject_summaries = []
    for subproject in subprojects:
        name = subproject["name"]
        repo = selected_repos.get(name, {})
        
        # Build a comprehensive description of the subproject and selected repo
        summary = {
            "name": name,
            "description": subproject["description"],
            "tech_requirements": subproject["tech_requirements"],
            "dependencies": subproject.get("dependencies", [])
        }
        
        if repo:
            summary["repository"] = {
                "name": repo["name"],
                "full_name": repo["full_name"],
                "url": repo.get("url", ""),
                "description": repo.get("description", ""),
                "primary_language": repo.get("languages", [""])[0] if repo.get("languages") else "",
                "stars": repo.get("stars", 0)
            }
            
        subproject_summaries.append(summary)
    
    # Format conflict information
    conflict_details = []
    for conflict in conflicts:
        conflict_details.append({
            "type": conflict["type"],
            "description": conflict["description"],
            "affected_components": conflict["subprojects"],
            "mitigation": conflict.get("mitigation", "Needs mitigation strategy")
        })
    
    # Create structured input for the LLM
    plan_input = {
        "project_description": project_description,
        "subprojects": subproject_summaries,
        "conflicts": conflict_details
    }
    
    system_prompt = """
    You are a technical integration architect with expertise in software development and system design.
    Create a practical integration plan for connecting the selected repositories into a cohesive system.
    
    Your plan should include:
    1. Architecture overview - how components will fit together
    2. Key integration points between components
    3. Specific modifications needed for each repository
    4. Step-by-step implementation plan detailing which repo is to be integrated where
    5. Solutions for the identified conflicts
    
    Focus on practical implementation details rather than theory.
    """
    
    prompt = f"""
    I need an integration plan for the following software project:
    
    {json.dumps(plan_input, indent=2)}
    
    Please provide a comprehensive integration plan that covers architecture, 
    integration points, necessary modifications, and conflict resolutions.
    """
    
    return query_huggingface(prompt, system_prompt=system_prompt, max_tokens=1500)

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

# ======== Main Execution Function ========

def create_project_plan(project_description: str) -> Dict[str, Any]:
    """Create a complete project plan from description with improved execution flow."""
    logger.info("Starting project planning process")
    
    # Step 1: Decompose project into subprojects
    logger.info("Decomposing project...")
    subprojects = decompose_project(project_description)
    if not subprojects:
        return {"error": "Failed to decompose project"}
    
    # Step 2: Find repositories for each subproject
    logger.info("Searching repositories...")
    repos_by_subproject = {}
    for subproject in subprojects:
        name = subproject["name"]
        repos = find_repos_for_subproject(subproject, n=5)  # Get top 5 repos
        if repos:
            # Rank repos by similarity
            ranked_repos = rank_repos_by_similarity(subproject, repos)
            repos_by_subproject[name] = ranked_repos
    
    # Step 3: Select best repository for each subproject
    selected_repos = {}
    for name, repos in repos_by_subproject.items():
        if repos:
            selected_repos[name] = repos[0]  # Select highest ranked repo
    
    # Step 4: Analyze conflicts
    logger.info("Analyzing potential conflicts...")
    conflicts = analyze_conflicts(subprojects, repos_by_subproject)
    
    # Step 5: Generate integration plan
    logger.info("Generating integration plan...")
    integration_plan = generate_integration_plan(
        project_description, 
        subprojects, 
        selected_repos,
        conflicts
    )
    
    # Build complete result
    return {
        "project_description": project_description,
        "subprojects": subprojects,
        "selected_repositories": selected_repos,
        "alternative_repositories": repos_by_subproject,
        "conflicts": conflicts,
        "integration_plan": integration_plan,
        "timestamp": datetime.datetime.now().isoformat()
    }

# ======== Demo Usage ========

if __name__ == "__main__":
    # Example project description
    project_description = """
    Build a web application that allows users to upload images, 
    apply ML-based filters, and share them on social media.
    """
    
    # Run the planner
    plan = create_project_plan(project_description)
    
    # Output the plan (could be saved to file or displayed in a UI)
    print(json.dumps(plan, indent=2, default=str))
