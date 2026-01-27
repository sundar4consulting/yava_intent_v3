"""
YAVA Intent Classifier Package

V2: Elasticsearch + RAG + LLM Hybrid Architecture
"""

# V2 Classifier (Elasticsearch-backed)
from .classifier_v2 import (
    get_hybrid_classifier_v2,
    ElasticsearchVectorStore,
    ElasticsearchRAGClassifier,
    RAGLLMIntentClassifierV2
)

# V2 Skill Interface (Watson Orchestrate)
from .skill_v2 import (
    classify_intent,
    extract_slots,
    detect_multi_intent,
    get_disambiguation,
    get_intents,
    health_check
)

__version__ = "2.0.0"
__all__ = [
    # V2 Primary
    "get_hybrid_classifier_v2",
    "classify_intent",
    "get_intents",
    "health_check",
    # V2 Additional Tools
    "extract_slots",
    "detect_multi_intent",
    "get_disambiguation",
    # V2 Classes
    "ElasticsearchVectorStore",
    "ElasticsearchRAGClassifier",
    "RAGLLMIntentClassifierV2"
]


