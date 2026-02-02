"""
REST API Server V3 for YAVA Intent Classifier
Implements OpenAPI spec from openapi_skill_v3.yaml

NEW IN V3: 3-Key KNN Embeddings Search
- intent_embedding, description_embedding, example_embedding
- Cosine similarity semantic matching (10-20ms)
- Configurable boosting per embedding key
- Automatic fallback to BM25 keyword search

Deployment: Render Web Service
Backend: Elasticsearch + 3-Key KNN + RAG + LLM (OpenAI GPT-4o)
Port: 8000
Endpoints: /v3/classify, /v3/intents, /v3/health
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from typing import Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import V3 functions (Elasticsearch 3-Key KNN + RAG)
from src.skill_v3 import classify_intent, get_intents, health_check

app = Flask(__name__)
CORS(app)  # Enable CORS for Watson Orchestrate

# API Version prefix
API_PREFIX = "/v3"

#===============================================================================
# TOOL 1: CLASSIFY (Primary Tool)
#===============================================================================
@app.route(f"{API_PREFIX}/classify", methods=['POST'])
def classify():
    """
    POST /v3/classify
    
    Primary tool: Full NLU classification with 3-key KNN embeddings + all features.
    
    NEW IN V3:
    - 3-key KNN embeddings search (intent, description, example)
    - Cosine similarity for semantic matching
    - Configurable boosting per embedding key
    - Automatic fallback to BM25
    """
    try:
        data = request.get_json()
        
        if not data or 'user_input' not in data:
            return jsonify({
                "error": "Missing required field: user_input",
                "details": "Request body must include 'user_input' field"
            }), 400
        
        # Extract parameters
        user_input = data['user_input']
        conversation_id = data.get('conversation_id')
        member_id = data.get('member_id')
        context_aware = data.get('context_aware', True)
        force_llm = data.get('force_llm', False)
        use_knn = data.get('use_knn', True)  # NEW IN V3: Enable KNN search
        
        # Call skill_v3.py classify_intent with KNN support
        result = classify_intent(
            user_input=user_input,
            conversation_id=conversation_id,
            member_id=member_id,
            context_aware=context_aware,
            force_llm=force_llm,
            use_knn=use_knn
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": "Classification failed",
            "details": str(e)
        }), 500


#===============================================================================
# TOOL 2: LIST INTENTS (Secondary Tool)
#===============================================================================
@app.route(f"{API_PREFIX}/intents", methods=['GET'])
def list_intents():
    """
    GET /v3/intents
    
    Secondary tool: List all available intents from Elasticsearch.
    """
    try:
        result = get_intents()
        
        # Check for errors
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": "Failed to fetch intents",
            "details": str(e),
            "source": "elasticsearch"
        }), 500


#===============================================================================
# HEALTH CHECK
#===============================================================================
@app.route(f"{API_PREFIX}/health", methods=['GET'])
def health_check_endpoint():
    """
    GET /v3/health
    
    Health check for Render deployment. Includes KNN status.
    """
    try:
        result = health_check()
        
        if result.get("status") == "healthy":
            return jsonify(result), 200
        else:
            return jsonify(result), 503
    
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "service": "yava-intent-classifier-v3"
        }), 503


#===============================================================================
# ROOT / INFO
#===============================================================================
@app.route("/", methods=['GET'])
@app.route(f"{API_PREFIX}/", methods=['GET'])
def info():
    """API information endpoint."""
    return jsonify({
        "service": "YAVA Intent Classifier API V3",
        "version": "3.0.0",
        "status": "running",
        "architecture": "Elasticsearch + 3-Key KNN Embeddings + RAG + LLM (OpenAI GPT-4o)",
        "features": {
            "knn_embeddings": "3-key semantic search (intent, description, example)",
            "cosine_similarity": "Fast vector matching (10-20ms)",
            "fallback": "BM25 keyword search",
            "llm": "GPT-4o enhancement for low confidence"
        },
        "endpoints": {
            "classify": f"{API_PREFIX}/classify [POST]",
            "list_intents": f"{API_PREFIX}/intents [GET]",
            "health": f"{API_PREFIX}/health [GET]",
            "docs": "/openapi.yaml"
        },
        "documentation": "See openapi_skill_v3.yaml for full API spec"
    }), 200


#===============================================================================
# OPENAPI SPEC ENDPOINT
#===============================================================================
@app.route("/openapi.yaml", methods=['GET'])
def serve_openapi_spec():
    """Serve OpenAPI spec file."""
    try:
        import yaml
        spec_path = os.path.join(os.path.dirname(__file__), 'openapi_skill_v3.yaml')
        
        with open(spec_path, 'r') as f:
            spec = yaml.safe_load(f)
        
        return jsonify(spec), 200
    except Exception as e:
        return jsonify({
            "error": "Failed to load OpenAPI spec",
            "details": str(e)
        }), 500


#===============================================================================
# ERROR HANDLERS
#===============================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            f"{API_PREFIX}/classify [POST]",
            f"{API_PREFIX}/intents [GET]",
            f"{API_PREFIX}/health [GET]",
            "/openapi.yaml [GET]"
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500


#===============================================================================
# MAIN
#===============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  YAVA Intent Classifier API V3                          ║
║  3-Key KNN + Elasticsearch + RAG + LLM (GPT-4o)         ║
╚══════════════════════════════════════════════════════════╝

🆕 NEW IN V3:
   • 3-Key KNN Embeddings (intent, description, example)
   • Cosine Similarity Semantic Search (10-20ms)
   • Configurable Boosting per Embedding Key
   • Automatic Fallback to BM25 Keyword Search

📍 Endpoints:
   POST {API_PREFIX}/classify     - Classify user intent (primary tool)
   GET  {API_PREFIX}/intents      - List all intents (secondary tool)
   GET  {API_PREFIX}/health       - Health check
   GET  /openapi.yaml             - API specification

🔧 Environment:
   ELASTICSEARCH_HOST: {os.getenv('ELASTICSEARCH_HOST', 'NOT SET')}
   ELASTICSEARCH_USERNAME: {os.getenv('ELASTICSEARCH_USERNAME', 'NOT SET')}
   OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}
   SENTENCE_TRANSFORMER_MODEL: {os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')}
   ENABLE_KNN: {os.getenv('ENABLE_KNN', 'true')}
   PORT: {port}

🚀 Starting server on http://0.0.0.0:{port}
    """)
    
    app.run(host="0.0.0.0", port=port, debug=debug)
