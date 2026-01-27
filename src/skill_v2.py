"""
Watson Orchestrate Skill Interface V2 - Elasticsearch-backed RAG+LLM Classifier

NEW IN V2:
==========
- Elasticsearch vector store backend (scalable to millions of vectors)
- OpenAI GPT-4o LLM integration for low-confidence cases
- Basic Authentication support for Elasticsearch
- Cost and performance tracking
- All V1 features (slots, multi-intent, disambiguation, context)

ARCHITECTURE:
=============
V1: InMemoryVectorStore (< 10K vectors)
V2: Elasticsearch cluster (millions of vectors, production-ready)

DEPLOYMENT:
===========
Requires environment variables:
- ELASTICSEARCH_HOST: ES cluster URL
- ELASTICSEARCH_USERNAME: Username (default: elastic)
- ELASTICSEARCH_PASSWORD: Password
- OPENAI_API_KEY: OpenAI API key for LLM features

USAGE:
======
Same API as skill.py but uses classifier_v2 backend.
"""

import json
import os
from typing import Dict, List, Optional
from .classifier_v2 import get_hybrid_classifier_v2


#===============================================================================
# TOOL 1: FULL CLASSIFICATION (Primary Tool - RAG+LLM Hybrid)
#===============================================================================
def classify_intent(user_input: str, 
                   conversation_id: Optional[str] = None,
                   member_id: Optional[str] = None,
                   context_aware: bool = True,
                   force_llm: bool = False) -> Dict:
    """
    Tool: Full NLU Classification with Elasticsearch RAG + OpenAI LLM Enhancement.
    
    V2 ENHANCEMENTS:
        - Elasticsearch kNN vector search (fast, scalable)
        - Automatic LLM invocation when confidence < 0.75
        - LLM reasoning for ambiguous cases
        - Cost tracking ($0 for RAG, ~$0.01-0.03 for LLM calls)
        - Performance metrics (RAG: 15-25ms, LLM: 500-1500ms)
    
    Features (from V1):
        - Intent classification with confidence
        - Slot extraction (dates, IDs, amounts, names)
        - Multi-intent detection (compound sentences)
        - Context-aware boosting from conversation history
        - Disambiguation when intents are ambiguous
    
    Args:
        user_input: User's message
        conversation_id: Session identifier
        member_id: Member ID (for session key)
        context_aware: Enable context boosting (default: True)
        force_llm: Force LLM invocation regardless of confidence (default: False)
    
    Returns:
        - intent: Primary detected intent
        - agent: Target agent for routing
        - confidence: Classification confidence (0-1)
        - classification_method: "RAG-only" or "RAG+LLM"
        - llm_invoked: Whether LLM was used
        - llm_reasoning: LLM explanation (if invoked)
        - vector_store: "elasticsearch" (V2 indicator)
        - slots: Extracted entities
        - multi_intents: Array if multiple detected
        - needs_disambiguation: True if clarification needed
        - performance: {rag_time_ms, llm_time_ms, llm_cost_usd}
    """
    session_id = conversation_id or f"watson-{member_id}" if member_id else "default"
    
    # Initialize V2 classifier (Elasticsearch + LLM)
    classifier = get_hybrid_classifier_v2(
        elasticsearch_host=os.getenv("ELASTICSEARCH_HOST"),
        elasticsearch_username=os.getenv("ELASTICSEARCH_USERNAME"),
        elasticsearch_password=os.getenv("ELASTICSEARCH_PASSWORD"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_threshold=float(os.getenv("LLM_THRESHOLD", "0.75")),
        enable_llm=os.getenv("ENABLE_LLM", "true").lower() == "true",
        verify_certs=os.getenv("ELASTICSEARCH_VERIFY_CERTS")

    )
    
    # V2 hybrid classification
    result = classifier.classify(
        utterance=user_input,
        session_id=session_id,
        force_llm=force_llm,
        context_metadata={
            "member_id": member_id,
            "channel": "watson_orchestrate"
        }
    )
    
    # Get session history for context
    session_history = classifier.rag_classifier.session_manager.get(session_id, n=5)
    
    # Build response
    response = {
        # Primary Classification
        "intent": result["intent"],
        "agent": result.get("agent_routing", "FallbackAgent"),
        "category": result.get("category", "unknown"),
        "confidence": result["confidence"],
        
        # V2-Specific Fields
        "classification_method": result.get("classification_method", "RAG-only"),
        "llm_invoked": result.get("llm_invoked", False),
        "vector_store": result.get("vector_store", "elasticsearch"),
        "query_result_action": result.get("queryResultAction", "RAG"),
        
        # LLM Enhancement (if invoked)
        "llm_reasoning": result.get("llm_reasoning", ""),
        "llm_confidence": result.get("llm_confidence"),
        "llm_entities": result.get("llm_entities", {}),
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": result.get("clarification_question", ""),
        "policy_reference": result.get("policy_reference", ""),
        
        # RAG Comparison (if LLM was invoked)
        "rag_result": result.get("rag_result"),
        "rag_vs_llm_agreement": result.get("rag_vs_llm_agreement"),
        "llm_override": result.get("llm_override", False),
        
        # Slot Filling
        "slots": result.get("slots", {}),
        "entities": result.get("entities", {}),
        "merged_slots": result.get("merged_slots", {}),
        
        # Multi-Intent Detection
        "has_multi_intents": result.get("has_multi_intents", False),
        "multi_intents": result.get("multi_intents"),
        
        # Disambiguation
        "needs_disambiguation": result.get("needs_disambiguation", False),
        "disambiguation": result.get("disambiguation", {}),
        "candidates": result.get("candidates", []),
        
        # Session
        "session": {
            "conversation_id": session_id,
            "recent_intents": [h["intent"] for h in session_history],
            "turn_count": len(session_history)
        },
        
        # Performance Metrics
        "performance": {
            "processing_time_ms": result.get("processing_time_ms", 0),
            "rag_time_ms": result.get("rag_time_ms", 0),
            "llm_time_ms": result.get("llm_time_ms", 0),
            "llm_cost_usd": result.get("llm_cost_usd", 0.0),
            "top_match_score": result.get("top_match_score", 0.0)
        }
    }
    
    return response


#===============================================================================
# TOOL 2: SLOT EXTRACTION
#===============================================================================
def extract_slots(user_input: str, intent: Optional[str] = None) -> Dict:
    """
    Tool: Extract entities/parameters from user utterance.
    
    Same as V1 - uses SlotFiller from base classifier.
    
    Extracts:
        - Dates, Claim IDs, Member IDs, Currency, Names
        - Pharmacy names, Medication names, Procedure codes
    
    Returns:
        - extracted_slots: {slot_type: value}
        - slot_count: Number of slots found
        - slot_types: List of slot types found
    """
    classifier = get_hybrid_classifier_v2()
    
    # Extract slots using base RAG classifier's slot filler
    slots = classifier.rag_classifier.slot_filler.extract_slots(user_input, intent or "general")
    
    # Get required slots for intent
    required = classifier.rag_classifier.slot_filler.get_missing_required_slots(intent or "general", {})
    missing = classifier.rag_classifier.slot_filler.get_missing_required_slots(intent or "general", slots)
    
    return {
        "extracted_slots": slots,
        "slot_count": len(slots),
        "slot_types": list(slots.keys()),
        "required_slots": required,
        "missing_required": missing,
        "slot_prompts": _generate_slot_prompts(missing),
        "original_input": user_input
    }


def _generate_slot_prompts(missing_slots: List[str]) -> Dict[str, str]:
    """Generate user-friendly prompts for missing slots."""
    prompts = {
        "member_id": "Could you please provide your member ID?",
        "date_of_service": "What was the date of service?",
        "claim_id": "Do you have a claim ID or reference number?",
        "pharmacy_name": "Which pharmacy would you like to use?",
        "medication_name": "What medication do you need?",
        "provider_name": "What is the provider or doctor's name?",
        "amount": "What is the amount in question?",
        "procedure_code": "Do you have the procedure or service code?"
    }
    return {slot: prompts.get(slot, f"Please provide the {slot.replace('_', ' ')}") 
            for slot in missing_slots}


#===============================================================================
# TOOL 3: MULTI-INTENT DETECTION
#===============================================================================
def detect_multi_intent(user_input: str, 
                        conversation_id: Optional[str] = None) -> Dict:
    """
    Tool: Detect if user utterance contains multiple intents.
    
    Same as V1 - uses MultiIntentDetector from base classifier.
    
    Detects compound sentences like:
        - "I need to refill my prescription AND check my claims"
        - "What's my deductible and also my copay for specialists?"
    
    Returns:
        - has_multiple_intents: Boolean
        - intent_count: Number of intents detected
        - intents: Array of {segment, intent, confidence, agent}
    """
    session_id = conversation_id or "default"
    classifier = get_hybrid_classifier_v2()
    
    # Check for multiple intents
    has_multi = classifier.rag_classifier.multi_intent_detector.has_multiple_intents(user_input)
    
    if not has_multi:
        # Single intent - classify normally
        result = classifier.classify(user_input, session_id)
        return {
            "has_multiple_intents": False,
            "intent_count": 1,
            "intents": [{
                "segment": user_input,
                "intent": result["intent"],
                "confidence": result["confidence"],
                "agent": result.get("agent_routing", "FallbackAgent")
            }],
            "suggested_order": [result["intent"]],
            "combined_response_possible": True
        }
    
    # Split and classify each segment
    segments = classifier.rag_classifier.multi_intent_detector.split_utterance(user_input)
    intents = []
    
    for segment in segments:
        if segment.strip():
            seg_result = classifier.classify(segment, session_id)
            intents.append({
                "segment": segment,
                "intent": seg_result["intent"],
                "confidence": seg_result["confidence"],
                "agent": seg_result.get("agent_routing", "FallbackAgent"),
                "priority": seg_result.get("priority", 3)
            })
    
    # Sort by priority
    sorted_intents = sorted(intents, key=lambda x: x.get("priority", 3))
    
    # Check if same agent handles all
    unique_agents = set(i["agent"] for i in intents)
    combined_possible = len(unique_agents) == 1
    
    return {
        "has_multiple_intents": len(intents) > 1,
        "intent_count": len(intents),
        "intents": intents,
        "suggested_order": [i["intent"] for i in sorted_intents],
        "combined_response_possible": combined_possible,
        "unique_agents": list(unique_agents),
        "original_input": user_input
    }


#===============================================================================
# TOOL 4: DISAMBIGUATION
#===============================================================================
def get_disambiguation(user_input: str, top_k: int = 3) -> Dict:
    """
    Tool: Get disambiguation options when intent is unclear.
    
    Same as V1 - uses DisambiguationEngine from base classifier.
    
    Returns:
        - needs_disambiguation: Boolean
        - prompt: Human-friendly question to ask user
        - options: Array of {option_number, intent, description, agent}
    """
    classifier = get_hybrid_classifier_v2()
    candidates = classifier.rag_classifier.get_candidates(user_input, top_k=top_k)
    disambiguation = classifier.rag_classifier.disambiguation_engine.generate_disambiguation(candidates, user_input)
    
    return {
        "needs_disambiguation": disambiguation["needed"],
        "reason": disambiguation.get("reason", ""),
        "prompt": disambiguation.get("prompt", ""),
        "options": disambiguation.get("options", []),
        "candidates": candidates,
        "confidence_gap": round(candidates[0]["score"] - candidates[1]["score"], 3) if len(candidates) >= 2 else 1.0,
        "recommendation": "Ask user for clarification" if disambiguation["needed"] else "Proceed with top intent",
        "original_input": user_input
    }


def resolve_disambiguation(conversation_id: str, 
                          selected_option: int,
                          original_utterance: str) -> Dict:
    """
    Tool: Process user's disambiguation selection.
    
    Args:
        conversation_id: Session identifier
        selected_option: User's choice (1, 2, or 3)
        original_utterance: The original ambiguous utterance
    
    Returns:
        - resolved_intent: The selected intent
        - agent: Target agent for routing
    """
    classifier = get_hybrid_classifier_v2()
    candidates = classifier.rag_classifier.get_candidates(original_utterance, top_k=3)
    
    if selected_option < 1 or selected_option > len(candidates):
        return {
            "status": "error",
            "error": f"Invalid option. Please select 1-{len(candidates)}"
        }
    
    selected = candidates[selected_option - 1]
    
    # Track in session
    classifier.rag_classifier.session_manager.add(
        conversation_id, original_utterance, 
        selected["intent"], selected["score"],
        slots={}
    )
    
    return {
        "status": "resolved",
        "resolved_intent": selected["intent"],
        "intent_id": selected["intent_id"],
        "agent": selected["agent"],
        "category": selected["category"],
        "confidence": selected["score"],
        "conversation_id": conversation_id
    }


#===============================================================================
# TOOL 5: CONTEXT/SESSION MANAGEMENT
#===============================================================================
def get_session_context(conversation_id: str, turns: int = 5) -> Dict:
    """
    Tool: Get full conversation context including slot memory.
    
    Returns:
        - history: Recent conversation turns
        - slot_memory: Accumulated slots across conversation
        - recent_intents: List of recent intents
    """
    classifier = get_hybrid_classifier_v2()
    history = classifier.rag_classifier.session_manager.get(conversation_id, n=turns)
    slot_memory = classifier.rag_classifier.session_manager.get_slot_memory(conversation_id)
    pending = classifier.rag_classifier.session_manager.get_pending_intents(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "history": history,
        "turn_count": len(history),
        "recent_intents": [h["intent"] for h in history],
        "slot_memory": slot_memory,
        "pending_intents": pending,
        "has_pending": len(pending) > 0,
        "context_summary": _summarize_context(history)
    }


def clear_session(conversation_id: str) -> Dict:
    """Tool: Clear session history and slot memory."""
    classifier = get_hybrid_classifier_v2()
    
    # Clear session
    if conversation_id in classifier.rag_classifier.session_manager.sessions:
        del classifier.rag_classifier.session_manager.sessions[conversation_id]
    if conversation_id in classifier.rag_classifier.session_manager.slot_memory:
        del classifier.rag_classifier.session_manager.slot_memory[conversation_id]
    
    return {
        "status": "cleared",
        "conversation_id": conversation_id
    }


#===============================================================================
# TOOL 6: INTENT CATALOG (FROM ELASTICSEARCH)
#===============================================================================
def get_intents() -> Dict:
    """
    Tool: List all available intents from Elasticsearch index.
    
    V2 CHANGE: Queries Elasticsearch for unique intents instead of static knowledge base.
    """
    classifier = get_hybrid_classifier_v2()
    
    try:
        # Aggregation query to get unique intents with example counts
        agg_query = {
            "size": 0,
            "aggs": {
                "unique_intents": {
                    "terms": {
                        "field": "intent_name",
                        "size": 100
                    },
                    "aggs": {
                        "categories": {
                            "terms": {
                                "field": "intent_category",
                                "size": 1
                            }
                        },
                        "sample_utterance": {
                            "top_hits": {
                                "size": 1,
                                "_source": ["example_utterance", "metadata"]
                            }
                        }
                    }
                }
            }
        }
        
        response = classifier.rag_classifier.vector_store.es.search(
            index=classifier.rag_classifier.vector_store.index_name,
            body=agg_query
        )
        
        # Parse aggregations
        intents = []
        by_category = {}
        
        for bucket in response["aggregations"]["unique_intents"]["buckets"]:
            intent_name = bucket["key"]
            example_count = bucket["doc_count"]
            
            # Get category
            category_buckets = bucket["categories"]["buckets"]
            category = category_buckets[0]["key"] if category_buckets else "unknown"
            
            # Get sample utterance
            sample_hits = bucket["sample_utterance"]["hits"]["hits"]
            sample_utterance = sample_hits[0]["_source"]["example_utterance"] if sample_hits else ""
            
            intent_data = {
                "intent_name": intent_name,
                "category": category,
                "example_count": example_count,
                "sample_utterance": sample_utterance
            }
            
            intents.append(intent_data)
            
            # Group by category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(intent_data)
        
        return {
            "source": "elasticsearch",
            "total_count": len(intents),
            "categories": list(by_category.keys()),
            "by_category": by_category,
            "intents": intents,
            "index_name": classifier.rag_classifier.vector_store.index_name
        }
    
    except Exception as e:
        return {
            "error": f"Failed to fetch intents from Elasticsearch: {str(e)}",
            "source": "elasticsearch"
        }


def get_intent_details(intent_name: str) -> Dict:
    """
    Tool: Get details for a specific intent from Elasticsearch.
    
    V2 CHANGE: Queries Elasticsearch for intent examples instead of static knowledge base.
    """
    classifier = get_hybrid_classifier_v2()
    
    try:
        # Query for all examples of this intent
        query = {
            "query": {
                "term": {
                    "intent_name": intent_name
                }
            },
            "size": 10,
            "_source": ["intent_id", "intent_name", "intent_category", "example_utterance", "metadata"]
        }
        
        response = classifier.rag_classifier.vector_store.es.search(
            index=classifier.rag_classifier.vector_store.index_name,
            body=query
        )
        
        if response["hits"]["total"]["value"] == 0:
            return {
                "found": False,
                "error": f"Intent '{intent_name}' not found in Elasticsearch",
                "source": "elasticsearch"
            }
        
        # Parse results
        examples = []
        intent_data = None
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            
            if not intent_data:
                intent_data = {
                    "intent_id": source.get("intent_id", ""),
                    "intent_name": source["intent_name"],
                    "category": source.get("intent_category", "unknown"),
                    "example_count": response["hits"]["total"]["value"]
                }
            
            examples.append({
                "utterance": source["example_utterance"],
                "metadata": source.get("metadata", {})
            })
        
        intent_data["examples"] = examples
        
        return {
            "found": True,
            "intent": intent_data,
            "source": "elasticsearch",
            "index_name": classifier.rag_classifier.vector_store.index_name
        }
    
    except Exception as e:
        return {
            "found": False,
            "error": f"Failed to fetch intent details: {str(e)}",
            "source": "elasticsearch"
        }


#===============================================================================
# TOOL 7: METRICS & HEALTH CHECK
#===============================================================================
def get_classifier_metrics() -> Dict:
    """
    Tool: Get classification performance and cost metrics.
    
    V2-SPECIFIC: Tracks RAG vs LLM usage, costs, latency.
    
    Returns:
        - total_classifications: Total queries processed
        - rag_only_count: Queries handled by RAG alone
        - llm_invoked_count: Queries that triggered LLM
        - llm_agreement_count: RAG and LLM agreed
        - llm_override_count: LLM changed RAG result
        - total_llm_cost_usd: Total OpenAI API costs
        - avg_cost_per_classification: Average cost per query
        - avg_rag_time_ms: Average RAG latency
        - avg_llm_time_ms: Average LLM latency
    """
    classifier = get_hybrid_classifier_v2()
    metrics = classifier.get_metrics()
    
    return {
        "service": "yava-intent-classifier-v2",
        "vector_store": "elasticsearch",
        "metrics": metrics,
        "cost_breakdown": {
            "rag_percentage": metrics.get("rag_percentage", 0),
            "llm_percentage": metrics.get("llm_percentage", 0),
            "avg_cost_per_query": f"${metrics.get('avg_cost_per_classification', 0):.4f}",
            "total_llm_cost": f"${metrics.get('total_llm_cost_usd', 0):.2f}"
        },
        "performance": {
            "avg_rag_latency_ms": round(metrics.get("avg_rag_time_ms", 0), 1),
            "avg_llm_latency_ms": round(metrics.get("avg_llm_time_ms", 0), 1)
        },
        "accuracy": {
            "llm_agreement_rate": f"{metrics.get('llm_agreement_rate', 0):.1%}",
            "llm_override_rate": f"{metrics.get('llm_override_rate', 0):.1%}"
        }
    }


def health_check() -> Dict:
    """Tool: Health check endpoint with Elasticsearch status."""
    try:
        classifier = get_hybrid_classifier_v2()
        
        # Check Elasticsearch index stats
        index_stats = classifier.rag_classifier.vector_store.get_index_stats()
        
        return {
            "status": "healthy",
            "service": "yava-intent-classifier-v2",
            "version": "2.0.0",
            "features": [
                "elasticsearch_vector_store",
                "openai_llm_enhancement",
                "intent_classification",
                "slot_extraction",
                "multi_intent_detection",
                "disambiguation",
                "context_awareness",
                "session_management"
            ],
            "intent_count": 47,
            "elasticsearch": {
                "connected": True,
                "index_name": classifier.rag_classifier.vector_store.index_name,
                "document_count": index_stats.get("document_count", 0),
                "size_mb": index_stats.get("size_mb", 0)
            },
            "llm": {
                "enabled": classifier.enable_llm,
                "model": classifier.model if classifier.enable_llm else None,
                "threshold": classifier.llm_threshold
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "yava-intent-classifier-v2"
        }


#===============================================================================
# INTERNAL HELPERS
#===============================================================================
def _summarize_context(history: List[Dict]) -> str:
    """Summarize conversation context."""
    if not history:
        return "New conversation - no prior context"
    
    intents = [h["intent"] for h in history]
    unique_intents = list(dict.fromkeys(intents))
    
    if len(unique_intents) == 1:
        return f"User has been asking about {unique_intents[0]} ({len(history)} turns)"
    else:
        return f"User has discussed: {', '.join(unique_intents)} ({len(history)} turns)"


#===============================================================================
# MAIN ROUTER - Watson Orchestrate Entry Point
#===============================================================================
def main(params: Dict) -> Dict:
    """
    Main entry point for Watson Orchestrate skill/tool calls (V2).
    
    Supported actions:
        - classify: Full NLU classification with Elasticsearch + LLM (default)
        - extract_slots: Extract entities from utterance
        - detect_multi: Detect multiple intents
        - disambiguate: Get disambiguation options
        - resolve_disambiguate: Process disambiguation selection
        - context: Get session context
        - clear_context: Clear session
        - intents: List all intents
        - intent_details: Get specific intent details
        - metrics: Get classification metrics (V2-specific)
        - health: Health check with ES status
    """
    action = params.get("action", "classify")
    
    # CLASSIFY (default - Elasticsearch RAG + LLM)
    if action == "classify":
        return classify_intent(
            params.get("user_input", ""),
            params.get("conversation_id"),
            params.get("member_id"),
            params.get("context_aware", True),
            params.get("force_llm", False)
        )
    
    # SLOT EXTRACTION
    elif action == "extract_slots":
        return extract_slots(
            params.get("user_input", ""),
            params.get("intent")
        )
    
    # MULTI-INTENT DETECTION
    elif action == "detect_multi":
        return detect_multi_intent(
            params.get("user_input", ""),
            params.get("conversation_id")
        )
    
    # DISAMBIGUATION
    elif action == "disambiguate":
        return get_disambiguation(
            params.get("user_input", ""),
            params.get("top_k", 3)
        )
    
    elif action == "resolve_disambiguate":
        return resolve_disambiguation(
            params.get("conversation_id", "default"),
            params.get("selected_option", 1),
            params.get("original_utterance", "")
        )
    
    # CONTEXT MANAGEMENT
    elif action == "context":
        return get_session_context(
            params.get("conversation_id", "default"),
            params.get("turns", 5)
        )
    
    elif action == "clear_context":
        return clear_session(params.get("conversation_id", "default"))
    
    # INTENT CATALOG
    elif action == "intents":
        return get_intents()
    
    elif action == "intent_details":
        return get_intent_details(params.get("intent_name", ""))
    
    # METRICS (V2-specific)
    elif action == "metrics":
        return get_classifier_metrics()
    
    # HEALTH
    elif action == "health":
        return health_check()
    
    else:
        return {
            "error": f"Unknown action: {action}",
            "available_actions": [
                "classify", "extract_slots", "detect_multi",
                "disambiguate", "resolve_disambiguate",
                "context", "clear_context",
                "intents", "intent_details", "metrics", "health"
            ]
        }


if __name__ == "__main__":
    # Test V2 classification with Elasticsearch + LLM
    print("=== TEST: V2 Classification (Elasticsearch + LLM) ===")
    result = classify_intent(
        "I need to refill my prescription for Lipitor and also check my claim from January 15th",
        "test-session-v2"
    )
    print(json.dumps(result, indent=2))
    
    print("\n=== TEST: Metrics ===")
    metrics = get_classifier_metrics()
    print(json.dumps(metrics, indent=2))
    
    print("\n=== TEST: Health Check ===")
    health = health_check()
    print(json.dumps(health, indent=2))


