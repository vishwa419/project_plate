
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
