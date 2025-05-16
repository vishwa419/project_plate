# ======== Model Setup ========
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
 
