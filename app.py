import os
import sys
import json
from flask import Flask, render_template, request, jsonify
from poc_hf import *

# Add the directory containing project_planner.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(".")))


app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/plan', methods=['POST'])
def generate_project_plan():
    """Generate project plan endpoint."""
    try:
        # Get project description from form
        project_description = request.form.get('project_description', '').strip()
        
        # Validate input
        if not project_description:
            return jsonify({
                'error': 'Please provide a project description.',
                'status': 'error'
            }), 400
        
        # Generate project plan
        plan = create_project_plan(project_description)
        
        # Render plan template for HTMX partial response
        return render_template('project_plan.html', plan=plan)
    
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Error generating project plan: {e}")
        return jsonify({
            'error': 'An error occurred while generating the project plan.',
            'details': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Ensure environment variables are set for GitHub/HuggingFace tokens if needed
    # os.environ['GITHUB_TOKEN'] = 'your_github_token'
    # os.environ['HF_API_TOKEN'] = 'your_huggingface_token'
    
    app.run(debug=True)
