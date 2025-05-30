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
