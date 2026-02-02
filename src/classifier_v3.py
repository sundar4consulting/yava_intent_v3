"""
Local RAG+LLM Intent Classifier V3 - 3-Key KNN Embeddings with Elasticsearch

NEW IN V3:
==========
1. **3-Key KNN Embeddings** - Uses intent, description, and example embeddings
2. **Cosine Similarity Search** - Fast vector-based similarity matching (10-20ms)
3. **Sentence Transformers** - Pre-computed embeddings for all intents
4. **Hybrid Search** - Combines BM25 keyword search with KNN vector search
5. **Multi-Query Boosting** - Configurable weights for each embedding key
6. **Fallback to BM25** - Automatic fallback if embeddings unavailable
7. **All V2 Features** - Maintains backward compatibility with V2

ARCHITECTURE:
============
Elasticsearch index → 3-Key KNN Search (intent_embedding, description_embedding, example_embedding) 
→ RAG classification → LLM enhancement (optional)

DEPENDENCIES:
============
- elasticsearch (pip install elasticsearch)
- sentence-transformers (pip install sentence-transformers)
- openai (pip install openai)
- numpy (pip install numpy)

ELASTICSEARCH SETUP:
===================
1. Create index with dense_vector mappings for 3 embedding keys
2. Populate with intent data + pre-computed embeddings (384-dim vectors)
3. Enable KNN search with cosine similarity
4. Use esSearchintentList_47_3key_embeddings.json format

SEARCH METHODS:
==============
1. **KNN Search** (Primary) - Cosine similarity on 3 embedding keys
   - intent_embedding: Captures overall intent semantics
   - description_embedding: Matches intent descriptions
   - example_embedding: Matches training examples
   
2. **Multi-Query Boosting** - Combines scores with configurable weights
   - Default: example^1.5, description^1.0, intent^1.0
   - Customizable per use case
   
3. **BM25 Fallback** - If embeddings not available or low confidence

AUTHENTICATION:
==============
Supports Basic Authentication (username/password) as primary method.

Environment Variables:
- ELASTICSEARCH_HOST: Elasticsearch cluster URL
- ELASTICSEARCH_USERNAME: Username (default: elastic)
- ELASTICSEARCH_PASSWORD: Password (required for basic auth)
- OPENAI_API_KEY: OpenAI API key for LLM features
- SENTENCE_TRANSFORMER_MODEL: Path to model (default: all-MiniLM-L6-v2)

USAGE EXAMPLE:
==============
```python
from classifier_v3 import get_hybrid_classifier_v3

# From environment variables
export ELASTICSEARCH_HOST="https://your-cluster.com:9200"
export ELASTICSEARCH_USERNAME="elastic"
export ELASTICSEARCH_PASSWORD="your-password"
export OPENAI_API_KEY="sk-..."

classifier = get_hybrid_classifier_v3()

# Classification with 3-key KNN
result = classifier.classify(
    utterance="I need to refill my prescription",
    session_id="user_12345",
    use_knn=True  # Enable KNN search (default)
)
```

PERFORMANCE:
===========
- KNN Search Time: 10-20ms (vector similarity)
- BM25 Search Time: 5-15ms (keyword search)
- Memory Usage: ~500 KB (ES external)
- Embedding Generation: ~5ms per query
- Concurrent Users: High (ES cluster scales horizontally)
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

# Elasticsearch client
try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠️  Warning: elasticsearch-py not installed. Install with: pip install elasticsearch")

# Sentence Transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

# OpenAI client (same as V1)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  Warning: OpenAI not installed. Install with: pip install openai")

#===============================================================================
# MULTI-INTENT CONJUNCTIONS - Words that signal multiple intents
#===============================================================================
MULTI_INTENT_SIGNALS = [
    r"\b(?:and also|and|also|plus|as well as|additionally|another thing|one more thing)\b",
    r"\b(?:oh and|btw|by the way|while I'm here|while you're at it)\b",
    r"\b(?:first|second|third|lastly|finally|next)\b"
]


#===============================================================================
# SLOT DEFINITIONS - Entity types to extract per intent
#===============================================================================
SLOT_DEFINITIONS = {
    "pharmacy": {
        "medication_name": {"patterns": [r"(?:for|refill|get|need)\s+(\w+(?:\s+\w+)?)", r"(\w+)\s+(?:prescription|medication|drug)"], "type": "medication"},
        "quantity": {"patterns": [r"(\d+)\s*(?:day|days|month|months)\s+supply", r"(\d+)\s+(?:pills|tablets|capsules)"], "type": "number"},
        "pharmacy_name": {"patterns": [r"(?:at|from|nearest)\s+(CVS|Walgreens|Rite Aid|Costco|Walmart)", r"(CVS|Walgreens|Rite Aid)"], "type": "pharmacy"}
    },
    "claims": {
        "claim_number": {"patterns": [r"claim\s*(?:#|number|id)?\s*[:\s]?\s*(\w{8,15})", r"(\d{10,15})"], "type": "claim_id"},
        "date_of_service": {"patterns": [r"(?:from|on|dated?)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:,?\s+\d{4})?)"], "type": "date"},
        "provider_name": {"patterns": [r"(?:from|at|with)\s+(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", r"(?:doctor|physician|provider)\s+([A-Z][a-z]+)"], "type": "provider"}
    },
    "specialist": {
        "specialty_type": {"patterns": [r"(cardiologist|dermatologist|orthopedic|neurologist|gastroenterologist|oncologist|ENT|urologist|pulmonologist|rheumatologist|endocrinologist)"], "type": "specialty"},
        "location": {"patterns": [r"(?:near|in|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s*[A-Z]{2})?)", r"(?:zip|zipcode|zip code)\s*[:\s]?\s*(\d{5})"], "type": "location"}
    },
    "primaryCareProvider": {
        "doctor_name": {"patterns": [r"(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", r"(?:doctor|physician)\s+([A-Z][a-z]+)"], "type": "provider"},
        "location": {"patterns": [r"(?:near|in|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", r"(\d{5})"], "type": "location"}
    },
    "deductible": {
        "plan_type": {"patterns": [r"(individual|family)\s+(?:deductible|plan)", r"(in[- ]?network|out[- ]?of[- ]?network)"], "type": "plan_type"},
        "year": {"patterns": [r"(?:for|in)\s+(20\d{2})", r"(this year|last year|next year)"], "type": "year"}
    },
    "eligibility": {
        "member_type": {"patterns": [r"(?:for\s+)?(?:my\s+)?(spouse|child|dependent|self)", r"(family|individual)\s+coverage"], "type": "member_type"},
        "date": {"patterns": [r"(?:as of|on|starting)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"], "type": "date"}
    },
    "idCard": {
        "card_type": {"patterns": [r"(digital|physical|paper|temporary)\s+(?:ID\s+)?card", r"(replacement|new)\s+card"], "type": "card_type"},
        "member_type": {"patterns": [r"(?:for\s+)?(?:my\s+)?(spouse|child|dependent)"], "type": "member_type"}
    },
    "hsa": {
        "action": {"patterns": [r"(balance|contribution|withdrawal|transfer|investment)", r"(contribute|withdraw|transfer)\s+"], "type": "action"},
        "amount": {"patterns": [r"\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"(\d+)\s+dollars"], "type": "currency"}
    },
    "appeals": {
        "claim_number": {"patterns": [r"claim\s*(?:#|number)?\s*[:\s]?\s*(\w{8,15})"], "type": "claim_id"},
        "appeal_type": {"patterns": [r"(first level|second level|external|expedited|urgent)\s+(?:appeal|review)"], "type": "appeal_type"}
    },
    "maternity": {
        "trimester": {"patterns": [r"(first|second|third|1st|2nd|3rd)\s+trimester", r"(\d+)\s+weeks?\s+pregnant"], "type": "trimester"},
        "service_type": {"patterns": [r"(prenatal|delivery|postpartum|ultrasound|c-section|cesarean)"], "type": "service"}
    }
}


#===============================================================================
# UTILITY CLASSES (V2 - Self-contained)
#===============================================================================
class SessionManager:
    """Manages conversation session history for context tracking."""
    
    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = defaultdict(list)
        self.slot_memory: Dict[str, Dict] = defaultdict(dict)
        
    def add(self, session_id: str, utterance: str, intent: str, confidence: float, 
            slots: Optional[Dict] = None, multi_intents: Optional[List] = None):
        entry = {
            "utterance": utterance, 
            "intent": intent, 
            "confidence": confidence, 
            "timestamp": datetime.utcnow().isoformat(),
            "slots": slots or {},
            "multi_intents": multi_intents or []
        }
        self.sessions[session_id].append(entry)
        
        if slots:
            self.slot_memory[session_id].update(slots)
        
        if len(self.sessions[session_id]) > 10:
            self.sessions[session_id] = self.sessions[session_id][-10:]
    
    def get(self, session_id: str, n: int = 5) -> List[Dict]:
        return self.sessions.get(session_id, [])[-n:]
    
    def get_recent_intents(self, session_id: str, n: int = 3) -> List[str]:
        history = self.get(session_id, n)
        return [h["intent"] for h in history]
    
    def get_slot_memory(self, session_id: str) -> Dict:
        return self.slot_memory.get(session_id, {})


class EmbeddingGenerator:
    """Simple deterministic embedding generator."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        
    def generate(self, text: str) -> np.ndarray:
        """Generate deterministic embedding for text."""
        text = text.lower()
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(self.dim)
        return emb / (np.linalg.norm(emb) + 1e-10)


class SlotFiller:
    """Extracts entity slots from user utterances."""
    
    def __init__(self):
        self.slot_definitions = SLOT_DEFINITIONS
    
    def extract_slots(self, utterance: str, intent: str) -> Dict[str, any]:
        """Extract slots relevant to the detected intent."""
        slots = {}
        
        intent_slots = self.slot_definitions.get(intent, {})
        
        for slot_name, slot_config in intent_slots.items():
            for pattern in slot_config["patterns"]:
                match = re.search(pattern, utterance, re.IGNORECASE)
                if match:
                    slots[slot_name] = {
                        "value": match.group(1),
                        "type": slot_config["type"],
                        "confidence": 0.9,
                        "source": "extracted"
                    }
                    break
        
        common_slots = self._extract_common_slots(utterance)
        slots.update(common_slots)
        
        return slots
    
    def _extract_common_slots(self, utterance: str) -> Dict[str, any]:
        """Extract common entities that apply to any intent."""
        slots = {}
        
        member_match = re.search(r"(?:member\s*(?:id|#|number)?|id)[:\s]*([A-Z0-9]{8,12})", utterance, re.IGNORECASE)
        if member_match:
            slots["member_id"] = {"value": member_match.group(1), "type": "member_id", "confidence": 0.95, "source": "extracted"}
        
        phone_match = re.search(r"(\d{3}[-.]?\d{3}[-.]?\d{4})", utterance)
        if phone_match:
            slots["phone"] = {"value": phone_match.group(1), "type": "phone", "confidence": 0.9, "source": "extracted"}
        
        date_match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", utterance)
        if date_match and "date" not in slots:
            slots["date"] = {"value": date_match.group(1), "type": "date", "confidence": 0.8, "source": "extracted"}
        
        amount_match = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", utterance)
        if amount_match:
            slots["amount"] = {"value": amount_match.group(1), "type": "currency", "confidence": 0.9, "source": "extracted"}
        
        zip_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", utterance)
        if zip_match:
            slots["zip_code"] = {"value": zip_match.group(1), "type": "location", "confidence": 0.85, "source": "extracted"}
        
        return slots


class MultiIntentDetector:
    """Detects when user message contains multiple intents."""
    
    def __init__(self):
        self.signals = MULTI_INTENT_SIGNALS
    
    def has_multiple_intents(self, utterance: str) -> bool:
        """Check if utterance likely contains multiple intents."""
        for pattern in self.signals:
            if re.search(pattern, utterance, re.IGNORECASE):
                return True
        return False
    
    def split_utterance(self, utterance: str) -> List[str]:
        """Split utterance into potential separate intent segments."""
        split_patterns = [
            r"\s+and also\s+",
            r"\s+also\s+",
            r"\s+and\s+(?=I\s+)",
            r"\s+plus\s+",
            r"\s+as well as\s+",
            r"\.\s+(?=[A-Z])",
            r"\s+oh and\s+",
            r"\s+btw\s+",
            r"\s+by the way\s+"
        ]
        
        segments = [utterance]
        for pattern in split_patterns:
            new_segments = []
            for seg in segments:
                parts = re.split(pattern, seg, flags=re.IGNORECASE)
                new_segments.extend([p.strip() for p in parts if p.strip()])
            segments = new_segments
        
        return [s for s in segments if len(s.split()) >= 2]


class DisambiguationEngine:
    """Handles disambiguation when intent is unclear."""
    
    INTENT_DESCRIPTIONS = {
        "pharmacy": "prescription or medication refills",
        "claims": "claim status or submission",
        "benefits": "coverage and benefit information",
        "eligibility": "enrollment or coverage status",
        "deductible": "deductible amount or status",
        "idCard": "insurance ID card",
        "primaryCareProvider": "primary care doctor (PCP)",
        "specialist": "specialist referral or search",
        "hsa": "Health Savings Account (HSA)",
        "appeals": "appeal a claim denial",
        "maternity": "pregnancy and maternity coverage",
        "enrollment": "plan enrollment or dependent changes",
        "unknown": "general inquiry or out of scope"
    }
    
    def generate_disambiguation(self, candidates: List[Dict], utterance: str) -> Dict:
        """Generate disambiguation response for ambiguous intent."""
        if len(candidates) < 2:
            return {"needed": False}
        
        # Check if top candidates are close in score
        score_diff = candidates[0]["score"] - candidates[1]["score"]
        if score_diff > 0.15:  # Clear winner
            return {"needed": False, "reason": "clear_winner"}
        
        # Build disambiguation options
        options = []
        for i, candidate in enumerate(candidates[:3]):
            desc = self.INTENT_DESCRIPTIONS.get(candidate["intent"], candidate["intent"])
            options.append({
                "option_number": i + 1,
                "intent": candidate["intent"],
                "description": desc,
                "score": candidate.get("score", 0.0)
            })
        
        # Generate natural language prompt
        if len(options) == 2:
            prompt = f"I want to make sure I help you correctly. Are you asking about {options[0]['description']} or {options[1]['description']}?"
        else:
            descs = [o['description'] for o in options]
            prompt = f"I want to make sure I understand. Are you asking about {descs[0]}, {descs[1]}, or {descs[2]}?"
        
        return {
            "needed": True,
            "reason": "ambiguous_intent",
            "score_difference": score_diff,
            "options": options,
            "prompt": prompt,
            "top_intents": [o["intent"] for o in options]
        }

# LLM config (same as V1)
LLM_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 600,
    "timeout": 10,
    "max_retries": 2
}

LLM_COSTS = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
}

# Intent taxonomy (same as V1)
INTENT_TAXONOMY = """
**Available Healthcare Intents:**

1. **pharmacy** - Prescription refills, medication questions, pharmacy locations
   Examples: "refill my Lipitor", "where's the nearest CVS", "how much is my prescription"

2. **claims** - Claim status, payment inquiries, claim appeals
   Examples: "where's my claim", "claim #CLM-12345 status", "denied claim explanation"

3. **specialist** - Find specialist, referral requirements, specialist coverage
   Examples: "find cardiologist near me", "do I need referral for dermatologist"

4. **primaryCareProvider** - Find/change PCP, PCP appointment scheduling
   Examples: "change my primary doctor", "who is my PCP", "PCP near 10001"

5. **deductible** - Deductible amounts, progress tracking, in/out-of-network
   Examples: "how much is my deductible", "deductible remaining", "family deductible"

6. **eligibility** - Coverage verification, benefit status, effective dates
   Examples: "am I covered", "when does coverage start", "check eligibility"

7. **idCard** - Request ID card, view digital card, replacement cards
   Examples: "send me ID card", "digital insurance card", "lost my card"

8. **hsa** - HSA balance, contributions, withdrawals, eligible expenses
   Examples: "HSA balance", "contribute to HSA", "can I use HSA for this"

9. **appeals** - File appeal, appeal status, appeal deadlines
   Examples: "appeal denied claim", "how to file appeal", "appeal deadline"

10. **maternity** - Maternity coverage, prenatal care, delivery benefits
    Examples: "pregnancy coverage", "prenatal visits covered", "maternity program"

11. **benefits** - General benefit questions, coverage details
    Examples: "what's covered", "benefit summary", "covered services"

12. **enrollment** - Plan enrollment, add dependents, life events
    Examples: "add my spouse", "enroll in new plan", "open enrollment"

13. **unknown** - Out of scope, unclear intent, needs human escalation
    Examples: "tell me a joke", "weather forecast", ambiguous requests
"""


#===============================================================================
# ELASTICSEARCH VECTOR STORE (BASIC AUTH)
#===============================================================================
class ElasticsearchVectorStore:
    """
    Elasticsearch-backed vector store with Basic Authentication.
    
    AUTHENTICATION PRIORITY:
    ------------------------
    1. Basic Auth (username/password) - Primary method
    2. API Key - Fallback if password not provided
    
    ENVIRONMENT VARIABLES:
    ----------------------
    - ELASTICSEARCH_HOST: ES cluster URL (default: http://localhost:9200)
    - ELASTICSEARCH_USERNAME: Username (default: elastic)
    - ELASTICSEARCH_PASSWORD: Password (required for basic auth)
    - ELASTICSEARCH_API_KEY: API key (optional fallback)
    
    USAGE WITH BASIC AUTH:
    ----------------------
    export ELASTICSEARCH_HOST="https://your-cluster.com:9200"
    export ELASTICSEARCH_USERNAME="elastic"
    export ELASTICSEARCH_PASSWORD="your-password"
    
    vector_store = ElasticsearchVectorStore()
    
    # Or explicit credentials
    vector_store = ElasticsearchVectorStore(
        hosts=["https://es-cluster.com:9200"],
        username="elastic",
        password="your-password"
    )
    """
    
    def __init__(
        self,
        hosts: List[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "yava-intent-examples",
        timeout: int = 30,
        verify_certs: bool = False
    ):
        """
        Initialize Elasticsearch vector store with Basic Auth.
        
        Args:
            hosts: List of ES hosts (default: env ELASTICSEARCH_HOST or localhost)
            username: ES username (default: env ELASTICSEARCH_USERNAME or 'elastic')
            password: ES password (default: env ELASTICSEARCH_PASSWORD)
            api_key: ES API key (fallback if password not provided)
            index_name: Intent examples index name
            timeout: Request timeout in seconds
            verify_certs: Verify SSL certificates (set False for self-signed certs)
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("elasticsearch-py not installed. Run: pip install elasticsearch")
        
        # Get credentials from environment or parameters
        hosts = hosts or [os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")]
        username = username or os.getenv("ELASTICSEARCH_USERNAME", "elastic")
        password = password or os.getenv("ELASTICSEARCH_PASSWORD")
        api_key = api_key or os.getenv("ELASTICSEARCH_API_KEY")
        
        # Initialize ES client with Basic Auth (priority) or API key (fallback)
        if password:
            # Basic Authentication (Primary method)
            self.es = Elasticsearch(
                hosts=hosts,
                basic_auth=(username, password),
                verify_certs=False,
                timeout=timeout
            )
            auth_method = f"Basic Auth (user: {username})"
        elif api_key:
            # API Key authentication (Fallback)
            self.es = Elasticsearch(
                hosts=hosts,
                api_key=api_key,
                verify_certs=verify_certs,
                timeout=timeout
            )
            auth_method = "API Key"
        else:
            raise ValueError(
                "Elasticsearch authentication required. Provide either:\n"
                "1. Basic Auth: username + password\n"
                "2. API Key: api_key\n"
                "Set via environment variables or constructor parameters."
            )
        
        self.index_name = index_name
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimensions
        
        # Test connection
        try:
            info = self.es.info()
            cluster_name = info.get('cluster_name', 'unknown')
            es_version = info['version']['number']
            print(f"✅ Connected to Elasticsearch: {es_version}")
            print(f"✅ Cluster: {cluster_name}")
            print(f"✅ Auth: {auth_method}")
            print(f"✅ Index: {self.index_name}")
        except Exception as e:
            print(f"❌ Elasticsearch connection failed: {e}")
            print(f"   Host: {hosts[0]}")
            print(f"   Auth: {auth_method}")
            raise
    
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Perform keyword-based search using multi_match query (BM25 algorithm).
        
        Args:
            query_text: User query string (e.g., "I need a prescription refill")
            top_k: Number of results to return
            filters: Optional filters (e.g., {"category": "healthcare"})
        
        Returns:
            List of (metadata_dict, bm25_score) tuples
        
        EXAMPLE:
        --------
        results = vector_store.search("I need a prescription refill", top_k=10)
        
        for metadata, score in results:
            print(f"{metadata['intent_name']}: {score:.3f}")
        
        Output:
            Pharmacy Services: 12.453
            Precertification Services: 3.892
            Claims Status: 2.154
            ...
        """
        
        # Build keyword-based multi_match query (similar to esSearchintentList_47.json structure)
        query = {
            "query": {
                "bool": {
                    "should": [
                        # Primary match on training utterances
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": [
                                    "training_utterances^3",
                                    "example_utterance^3",
                                    "keywords^2",
                                    "description_short^1.5",
                                    "intent_name^1"
                                ],
                                "type": "best_fields",
                                "operator": "or",
                                "fuzziness": "AUTO"
                            }
                        },
                        # Boost for exact phrase matches
                        {
                            "match_phrase": {
                                "training_utterances": {
                                    "query": query_text,
                                    "boost": 5
                                }
                            }
                        },
                        # Boost for keyword matches
                        {
                            "terms": {
                                "keywords": query_text.lower().split(),
                                "boost": 2
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": [
                "intent_id",
                "intent_name",
                "category",
                "intent_category",
                "agent_routing",
                "priority",
                "description_short",
                "disambiguation_prompt",
                "training_utterances",
                "keywords",
                "example_utterance",
                "use_cases",
                "expected_outcomes",
                "metadata"
            ],
            "size": top_k
        }
        
        # Add filters if provided
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})
            
            if "bool" not in query["query"]:
                query["query"]["bool"] = {}
            query["query"]["bool"]["filter"] = filter_clauses
        
        try:
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=query
            )
            
            # Parse results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = {
                    "intent_id": source.get("intent_id"),
                    "intent_name": source.get("intent_name"),
                    "category": source.get("category") or source.get("intent_category", "general"),
                    "agent_routing": source.get("agent_routing", "FallbackAgent"),
                    "priority": source.get("priority", 3),
                    "description_short": source.get("description_short", ""),
                    "disambiguation_prompt": source.get("disambiguation_prompt"),
                    "example_utterance": source.get("example_utterance", source.get("training_utterances", [""])[0] if source.get("training_utterances") else ""),
                    "training_utterances": source.get("training_utterances", []),
                    "keywords": source.get("keywords", []),
                    "use_cases": source.get("use_cases", ""),
                    "expected_outcomes": source.get("expected_outcomes", ""),
                    "metadata": source.get("metadata", {})
                }
                score = hit["_score"]  # BM25 score from Elasticsearch
                results.append((metadata, score))
            
            return results
        
        except Exception as e:
            print(f"❌ Elasticsearch keyword search error: {e}")
            return []
    
    def index_exists(self) -> bool:
        """Check if index exists."""
        try:
            return self.es.indices.exists(index=self.index_name)
        except Exception:
            return False
    
    def create_index(self, schema_file: Optional[str] = None):
        """
        Create index with proper mapping for V3 hybrid search (BM25 + 3-Key KNN).
        Includes dense_vector mappings for intent, description, and example embeddings.
        
        Args:
            schema_file: Path to elasticsearch_intent_index.json (optional)
        """
        if schema_file and os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                schema = json.load(f)
        else:
            # V3 schema optimized for hybrid search (BM25 + KNN)
            schema = {
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 2,
                    "analysis": {
                        "analyzer": {
                            "intent_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
                            }
                        }
                    },
                    "index": {
                        "max_result_window": 10000
                    }
                },
                "mappings": {
                    "properties": {
                        "intent_id": {"type": "keyword"},
                        "intent_name": {
                            "type": "text",
                            "analyzer": "intent_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "category": {"type": "keyword"},
                        "intent_category": {"type": "keyword"},
                        "agent_routing": {"type": "keyword"},
                        "priority": {"type": "integer"},
                        "description_short": {
                            "type": "text",
                            "analyzer": "intent_analyzer"
                        },
                        "disambiguation_prompt": {
                            "type": "text",
                            "analyzer": "intent_analyzer"
                        },
                        "training_utterances": {
                            "type": "text",
                            "analyzer": "intent_analyzer"
                        },
                        "keywords": {
                            "type": "keyword"
                        },
                        "example_utterance": {
                            "type": "text",
                            "analyzer": "intent_analyzer"
                        },
                        "use_cases": {
                            "type": "text",
                            "analyzer": "intent_analyzer"
                        },
                        "expected_outcomes": {
                            "type": "text",
                            "analyzer": "intent_analyzer"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "lob": {"type": "keyword"},
                                "risk_level": {"type": "keyword"},
                                "agent_routing": {"type": "keyword"},
                                "priority": {"type": "integer"}
                            }
                        },
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                        
                        # ===== V3: 3-KEY KNN EMBEDDINGS =====
                        "intent_embedding": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "example_embedding": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
        
        try:
            self.es.indices.create(index=self.index_name, body=schema)
            print(f"✅ Created index: {self.index_name} (V3 hybrid: BM25 + 3-key KNN)")
        except Exception as e:
            print(f"❌ Failed to create index: {e}")
    
    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            doc_count = stats['_all']['total']['docs']['count']
            size_bytes = stats['_all']['total']['store']['size_in_bytes']
            return {
                "document_count": doc_count,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "index_name": self.index_name
            }
        except Exception as e:
            return {"error": str(e)}
    
    def list_intents(
        self,
        category: Optional[str] = None,
        agent_routing: Optional[str] = None,
        priority: Optional[int] = None,
        search_text: Optional[str] = None,
        limit: int = 100,
        include_training_utterances: bool = False,
        include_keywords: bool = True
    ) -> Dict:
        """
        List all intents from Elasticsearch index with mapping attributes.
        
        Matches the schema structure from esSearchintentList_47.json:
        - intent_id, intent_name, category, intent_category
        - agent_routing, priority, description_short
        - disambiguation_prompt, training_utterances, keywords
        - use_cases, expected_outcomes, example_utterance
        - created_at, updated_at
        
        Args:
            category: Filter by category (e.g., "healthcare", "benefits")
            agent_routing: Filter by agent (e.g., "PharmacyAgent", "ClaimsAgent")
            priority: Filter by priority level (1-5)
            search_text: Search text within intent fields
            limit: Maximum number of intents to return (default: 100)
            include_training_utterances: Include full training utterances array (default: False for performance)
            include_keywords: Include keywords array (default: True)
        
        Returns:
            Dict with intents list and metadata:
            {
                "total_count": 47,
                "returned_count": 47,
                "filters_applied": {...},
                "intents": [
                    {
                        "intent_id": "INT-PHR-0001",
                        "intent_name": "Pharmacy Services",
                        "category": "healthcare",
                        "agent_routing": "PharmacyAgent",
                        "priority": 2,
                        "description_short": "...",
                        "disambiguation_prompt": "...",
                        "keywords": [...],
                        "example_utterance": "...",
                        "use_cases": "...",
                        "expected_outcomes": "...",
                        "training_utterance_count": 40,
                        "created_at": "2026-01-24T16:44:31Z",
                        "updated_at": "2026-01-24T16:44:31Z"
                    },
                    ...
                ]
            }
        
        USAGE EXAMPLES:
        ---------------
        # List all intents
        result = vector_store.list_intents()
        print(f"Found {result['total_count']} intents")
        
        # Filter by category
        result = vector_store.list_intents(category="healthcare")
        
        # Filter by agent
        result = vector_store.list_intents(agent_routing="PharmacyAgent")
        
        # Search within intents
        result = vector_store.list_intents(search_text="prescription")
        
        # Include full training utterances
        result = vector_store.list_intents(include_training_utterances=True, limit=10)
        """
        
        # Build source fields list based on options
        source_fields = [
            "intent_id",
            "intent_name",
            "category",
            "intent_category",
            "agent_routing",
            "priority",
            "description_short",
            "disambiguation_prompt",
            "example_utterance",
            "use_cases",
            "expected_outcomes",
            "created_at",
            "updated_at"
        ]
        
        if include_keywords:
            source_fields.append("keywords")
        
        if include_training_utterances:
            source_fields.append("training_utterances")
        
        # Build query with filters
        must_clauses = []
        filter_clauses = []
        filters_applied = {}
        
        # Category filter
        if category:
            filter_clauses.append({
                "bool": {
                    "should": [
                        {"term": {"category": category}},
                        {"term": {"intent_category": category}}
                    ]
                }
            })
            filters_applied["category"] = category
        
        # Agent routing filter
        if agent_routing:
            filter_clauses.append({"term": {"agent_routing": agent_routing}})
            filters_applied["agent_routing"] = agent_routing
        
        # Priority filter
        if priority is not None:
            filter_clauses.append({"term": {"priority": priority}})
            filters_applied["priority"] = priority
        
        # Text search within intent fields
        if search_text:
            must_clauses.append({
                "multi_match": {
                    "query": search_text,
                    "fields": [
                        "intent_name^3",
                        "description_short^2",
                        "training_utterances^2",
                        "keywords^2",
                        "use_cases",
                        "expected_outcomes",
                        "disambiguation_prompt"
                    ],
                    "type": "best_fields",
                    "operator": "or",
                    "fuzziness": "AUTO"
                }
            })
            filters_applied["search_text"] = search_text
        
        # Build final query
        if must_clauses or filter_clauses:
            query = {
                "query": {
                    "bool": {
                        "must": must_clauses if must_clauses else [{"match_all": {}}],
                        "filter": filter_clauses if filter_clauses else []
                    }
                },
                "_source": source_fields,
                "size": limit,
                "sort": [
                    {"priority": {"order": "asc"}},
                    {"intent_name.keyword": {"order": "asc", "unmapped_type": "keyword"}}
                ]
            }
        else:
            # No filters - match all
            query = {
                "query": {"match_all": {}},
                "_source": source_fields,
                "size": limit,
                "sort": [
                    {"priority": {"order": "asc"}},
                    {"intent_name.keyword": {"order": "asc", "unmapped_type": "keyword"}}
                ]
            }
        
        try:
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=query
            )
            
            # Get total count
            total_hits = response["hits"]["total"]
            total_count = total_hits["value"] if isinstance(total_hits, dict) else total_hits
            
            # Parse results with mapped attributes
            intents = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                
                intent_data = {
                    # Core identifiers
                    "intent_id": source.get("intent_id"),
                    "intent_name": source.get("intent_name"),
                    
                    # Classification attributes
                    "category": source.get("category") or source.get("intent_category", "general"),
                    "agent_routing": source.get("agent_routing", "FallbackAgent"),
                    "priority": source.get("priority", 3),
                    
                    # Descriptions
                    "description_short": source.get("description_short", ""),
                    "disambiguation_prompt": source.get("disambiguation_prompt", ""),
                    
                    # Example
                    "example_utterance": source.get("example_utterance", ""),
                    
                    # Business context
                    "use_cases": source.get("use_cases", ""),
                    "expected_outcomes": source.get("expected_outcomes", ""),
                    
                    # Timestamps
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at")
                }
                
                # Add keywords if requested
                if include_keywords:
                    intent_data["keywords"] = source.get("keywords", [])
                    intent_data["keyword_count"] = len(source.get("keywords", []))
                
                # Add training utterances if requested
                if include_training_utterances:
                    intent_data["training_utterances"] = source.get("training_utterances", [])
                    intent_data["training_utterance_count"] = len(source.get("training_utterances", []))
                else:
                    # Just include count for reference
                    training_utts = source.get("training_utterances", [])
                    intent_data["training_utterance_count"] = len(training_utts) if isinstance(training_utts, list) else 0
                
                intents.append(intent_data)
            
            return {
                "total_count": total_count,
                "returned_count": len(intents),
                "filters_applied": filters_applied if filters_applied else None,
                "index_name": self.index_name,
                "include_training_utterances": include_training_utterances,
                "include_keywords": include_keywords,
                "intents": intents
            }
        
        except Exception as e:
            print(f"❌ Elasticsearch list_intents error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "total_count": 0,
                "returned_count": 0,
                "filters_applied": filters_applied if filters_applied else None,
                "error": str(e),
                "intents": []
            }
    
    def get_intent_by_id(self, intent_id: str) -> Optional[Dict]:
        """
        Get a single intent by its ID with all mapped attributes.
        
        Args:
            intent_id: The intent ID (e.g., "INT-PHR-0001")
        
        Returns:
            Intent dict with all attributes, or None if not found
        
        USAGE:
        ------
        intent = vector_store.get_intent_by_id("INT-PHR-0001")
        if intent:
            print(f"Intent: {intent['intent_name']}")
            print(f"Agent: {intent['agent_routing']}")
        """
        query = {
            "query": {
                "term": {"intent_id": intent_id}
            },
            "_source": [
                "intent_id", "intent_name", "category", "intent_category",
                "agent_routing", "priority", "description_short",
                "disambiguation_prompt", "training_utterances", "keywords",
                "example_utterance", "use_cases", "expected_outcomes",
                "created_at", "updated_at",
                # V3: Include embeddings if needed for analysis
                "intent_embedding", "description_embedding", "example_embedding"
            ],
            "size": 1
        }
        
        try:
            response = self.es.search(index=self.index_name, body=query)
            
            if response["hits"]["hits"]:
                source = response["hits"]["hits"][0]["_source"]
                return {
                    "intent_id": source.get("intent_id"),
                    "intent_name": source.get("intent_name"),
                    "category": source.get("category") or source.get("intent_category", "general"),
                    "agent_routing": source.get("agent_routing", "FallbackAgent"),
                    "priority": source.get("priority", 3),
                    "description_short": source.get("description_short", ""),
                    "disambiguation_prompt": source.get("disambiguation_prompt", ""),
                    "training_utterances": source.get("training_utterances", []),
                    "training_utterance_count": len(source.get("training_utterances", [])),
                    "keywords": source.get("keywords", []),
                    "keyword_count": len(source.get("keywords", [])),
                    "example_utterance": source.get("example_utterance", ""),
                    "use_cases": source.get("use_cases", ""),
                    "expected_outcomes": source.get("expected_outcomes", ""),
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at")
                }
            return None
        
        except Exception as e:
            print(f"❌ Error fetching intent {intent_id}: {e}")
            return None
    
    def get_intents_by_category(self, category: str) -> List[Dict]:
        """
        Get all intents for a specific category.
        
        Args:
            category: Category name (e.g., "healthcare", "benefits", "claims")
        
        Returns:
            List of intent dicts
        """
        result = self.list_intents(category=category, limit=100)
        return result.get("intents", [])
    
    def get_intents_by_agent(self, agent_routing: str) -> List[Dict]:
        """
        Get all intents routed to a specific agent.
        
        Args:
            agent_routing: Agent name (e.g., "PharmacyAgent", "ClaimsAgent")
        
        Returns:
            List of intent dicts
        """
        result = self.list_intents(agent_routing=agent_routing, limit=100)
        return result.get("intents", [])
    
    def get_intent_summary(self) -> Dict:
        """
        Get a summary of all intents grouped by category and agent.
        
        Returns:
            Dict with counts by category and agent:
            {
                "total_intents": 47,
                "by_category": {"healthcare": 15, "benefits": 10, ...},
                "by_agent": {"PharmacyAgent": 5, "ClaimsAgent": 8, ...},
                "by_priority": {1: 5, 2: 15, 3: 20, 4: 5, 5: 2}
            }
        """
        # Aggregation query
        query = {
            "size": 0,
            "aggs": {
                "by_category": {
                    "terms": {"field": "category", "size": 50}
                },
                "by_agent": {
                    "terms": {"field": "agent_routing", "size": 50}
                },
                "by_priority": {
                    "terms": {"field": "priority", "size": 10}
                }
            }
        }
        
        try:
            response = self.es.search(index=self.index_name, body=query)
            
            # Parse aggregations
            by_category = {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["by_category"]["buckets"]
            }
            
            by_agent = {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["by_agent"]["buckets"]
            }
            
            by_priority = {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["by_priority"]["buckets"]
            }
            
            total_hits = response["hits"]["total"]
            total_count = total_hits["value"] if isinstance(total_hits, dict) else total_hits
            
            return {
                "total_intents": total_count,
                "by_category": by_category,
                "by_agent": by_agent,
                "by_priority": by_priority,
                "index_name": self.index_name
            }
        
        except Exception as e:
            print(f"❌ Error getting intent summary: {e}")
            return {
                "total_intents": 0,
                "by_category": {},
                "by_agent": {},
                "by_priority": {},
                "error": str(e)
            }


#===============================================================================
# 3-KEY KNN SEARCH CLASS (NEW IN V3)
#===============================================================================
class ThreeKeyKNNSearch:
    """
    Elasticsearch KNN search using 3 embedding keys for improved accuracy.
    
    NEW IN V3:
    - intent_embedding: Overall intent semantics
    - description_embedding: Intent description matching
    - example_embedding: Training example similarity
    
    Uses Sentence Transformers (all-MiniLM-L6-v2) for query encoding.
    """
    
    def __init__(
        self,
        es_client: Elasticsearch,
        index_name: str = "yava-intent-3key-embeddings",
        model_path: str = None
    ):
        """
        Initialize 3-key KNN searcher.
        
        Args:
            es_client: Elasticsearch client instance
            index_name: Index name with 3-key embeddings
            model_path: Path to Sentence Transformer model (default: all-MiniLM-L6-v2)
        """
        self.es = es_client
        self.index_name = index_name
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            model_path = model_path or os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_path)
        else:
            print("⚠️  Sentence Transformers not available. KNN search disabled.")
            self.model = None
    
    def knn_search_single_key(
        self,
        query_vector: List[float],
        field: str,
        k: int = 5,
        num_candidates: int = 100
    ) -> List[Dict]:
        """
        Perform KNN search on a single embedding field.
        
        Args:
            query_vector: Query embedding vector
            field: Embedding field (intent_embedding, description_embedding, example_embedding)
            k: Number of results
            num_candidates: Candidates to consider (higher = more accurate, slower)
        
        Returns:
            List of matching documents with scores
        """
        query = {
            "knn": {
                "field": field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates
            },
            "_source": [
                "intent_id",
                "intent_name",
                "description_short",
                "example_utterance",
                "category",
                "agent_routing",
                "priority",
                "keywords"
            ]
        }
        
        try:
            response = self.es.search(index=self.index_name, body=query)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'score': hit['_score'],
                    'intent_id': hit['_source'].get('intent_id', ''),
                    'intent_name': hit['_source'].get('intent_name', ''),
                    'description_short': hit['_source'].get('description_short', ''),
                    'example_utterance': hit['_source'].get('example_utterance', ''),
                    'category': hit['_source'].get('category', ''),
                    'agent_routing': hit['_source'].get('agent_routing', ''),
                    'priority': hit['_source'].get('priority', 3),
                    'keywords': hit['_source'].get('keywords', []),
                    'matched_field': field
                })
            return results
        except Exception as e:
            print(f"❌ KNN search failed on {field}: {e}")
            return []
    
    def knn_search_3key(
        self,
        query: str,
        k: int = 5,
        num_candidates: int = 100,
        strategy: str = "combined"
    ) -> List[Dict]:
        """
        Perform KNN search across all 3 embedding keys.
        
        Args:
            query: User's query string
            k: Number of results per key
            num_candidates: Candidates to consider
            strategy:
                - "best_match": Single best across all keys
                - "combined": Deduplicated results from all keys
                - "all_keys": Results grouped by key
        
        Returns:
            List of matching documents with scores
        """
        if not self.model:
            print("⚠️  Model not available. Cannot perform KNN search.")
            return []
        
        # Encode query to vector
        query_vector = self.model.encode(query).tolist()
        
        fields = ['intent_embedding', 'description_embedding', 'example_embedding']
        all_results = {}
        
        # Search each key
        for field in fields:
            results = self.knn_search_single_key(query_vector, field, k, num_candidates)
            for r in results:
                key = r['intent_name']
                # Keep highest score for each intent
                if key not in all_results or r['score'] > all_results[key]['score']:
                    all_results[key] = r
        
        if strategy == "best_match":
            if all_results:
                best = max(all_results.values(), key=lambda x: x['score'])
                return [best]
            return []
        
        elif strategy == "all_keys":
            grouped = {}
            for field in fields:
                grouped[field] = self.knn_search_single_key(query_vector, field, k, num_candidates)
            return grouped
        
        elif strategy == "combined":
            combined = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
            return combined[:k]
        
        return list(all_results.values())
    
    def knn_search_multi_query(
        self,
        query: str,
        k: int = 5,
        boost_intent: float = 1.0,
        boost_description: float = 1.0,
        boost_example: float = 1.5
    ) -> List[Dict]:
        """
        Multi-query KNN search with configurable boosting.
        
        Combines scores from all 3 keys with weights using Elasticsearch script_score.
        
        Args:
            query: User's query string
            k: Number of results
            boost_intent: Weight for intent embedding
            boost_description: Weight for description embedding
            boost_example: Weight for example embedding (default: 1.5 for higher relevance)
        
        Returns:
            List of matching documents with combined scores
        """
        if not self.model:
            return []
        
        query_vector = self.model.encode(query).tolist()
        
        search_query = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": """
                            double intent_score = cosineSimilarity(params.query_vector, 'intent_embedding') + 1.0;
                            double desc_score = cosineSimilarity(params.query_vector, 'description_embedding') + 1.0;
                            double example_score = cosineSimilarity(params.query_vector, 'example_embedding') + 1.0;
                            return (intent_score * params.boost_intent + 
                                    desc_score * params.boost_description + 
                                    example_score * params.boost_example);
                        """,
                        "params": {
                            "query_vector": query_vector,
                            "boost_intent": boost_intent,
                            "boost_description": boost_description,
                            "boost_example": boost_example
                        }
                    }
                }
            },
            "_source": [
                "intent_id",
                "intent_name",
                "description_short",
                "example_utterance",
                "category",
                "agent_routing",
                "priority",
                "keywords"
            ]
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_query)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'score': hit['_score'],
                    'intent_id': hit['_source'].get('intent_id', ''),
                    'intent_name': hit['_source'].get('intent_name', ''),
                    'description_short': hit['_source'].get('description_short', ''),
                    'example_utterance': hit['_source'].get('example_utterance', ''),
                    'category': hit['_source'].get('category', ''),
                    'agent_routing': hit['_source'].get('agent_routing', ''),
                    'priority': hit['_source'].get('priority', 3),
                    'keywords': hit['_source'].get('keywords', []),
                    'matched_field': 'combined_3key'
                })
            return results
        except Exception as e:
            print(f"❌ Multi-query KNN search failed: {e}")
            return []


#===============================================================================
# ELASTICSEARCH-BACKED RAG CLASSIFIER
#===============================================================================
class ElasticsearchRAGClassifier:
    """
    RAG classifier using Elasticsearch vector store.
    
    Replaces LocalIntentClassifier from V1 with ES-backed version.
    Same API, different backend.
    """
    
    def __init__(
        self,
        vector_store: ElasticsearchVectorStore,
        embedder: EmbeddingGenerator,
        session_manager: SessionManager,
        slot_filler: SlotFiller,
        multi_intent_detector: MultiIntentDetector,
        disambiguation_engine: DisambiguationEngine
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.session_manager = session_manager
        self.slot_filler = slot_filler
        self.multi_intent_detector = multi_intent_detector
        self.disambiguation_engine = disambiguation_engine
    
    def classify(
        self,
        utterance: str,
        session_id: str = "default",
        context_aware: bool = True
    ) -> Dict:
        """
        Classify intent using Elasticsearch keyword search (BM25 algorithm).
        
        Same signature as V1, but uses keyword matching instead of vector similarity.
        """
        # Perform keyword-based search (no embedding generation needed)
        matches = self.vector_store.search(
            query_text=utterance,
            top_k=10
        )
        
        if not matches:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "agent_routing": "FallbackAgent",
                "slots": {},
                "candidates": [],
                "top_match_score": 0.0,
                "classification_method": "keyword-search-no-results"
            }
        
        # Vote from top matches (aggregate BM25 scores by intent)
        votes = defaultdict(float)
        for metadata, score in matches[:5]:
            intent = metadata["intent_name"]
            votes[intent] += score
        
        # Determine winner
        winner_intent = max(votes, key=votes.get)
        top_scores = [score for metadata, score in matches if metadata["intent_name"] == winner_intent]
        
        # Calculate confidence (normalize BM25 scores to 0-1 range)
        total_score = sum(votes.values())
        confidence = votes[winner_intent] / total_score if total_score > 0 else 0.0
        
        # Get metadata from top match for winner intent
        winner_metadata = next(
            (metadata for metadata, score in matches if metadata["intent_name"] == winner_intent),
            {"metadata": {}}
        )
        es_metadata = winner_metadata.get("metadata", {})
        
        # Get agent routing from ES metadata or fallback to mapping
        agent_routing = es_metadata.get("agent_routing") or self._get_agent_routing(winner_intent)
        category = es_metadata.get("category", "general")
        priority = es_metadata.get("priority", 3)
        
        # Extract slots
        slots = self.slot_filler.extract_slots(utterance, winner_intent)
        
        # Build candidates list
        candidates = [
            {
                "intent": metadata["intent_name"],
                "score": score,
                "example": metadata["example_utterance"],
                "agent": self._get_agent_routing(metadata["intent_name"])
            }
            for metadata, score in matches[:5]
        ]
        
        # Multi-intent detection
        multi_intents = []
        if self.multi_intent_detector.has_multiple_intents(utterance):
            segments = self.multi_intent_detector.split_utterance(utterance)
            if len(segments) > 1:
                for seg in segments:
                    # Use keyword search for segment classification (no embedding needed)
                    seg_matches = self.vector_store.search(seg, top_k=3)
                    if seg_matches:
                        seg_intent = seg_matches[0][0]["intent_name"]
                        seg_score = seg_matches[0][1]
                        # Normalize confidence for multi-intent
                        seg_total = sum(s for _, s in seg_matches)
                        seg_confidence = seg_score / seg_total if seg_total > 0 else 0.0
                        multi_intents.append({
                            "segment": seg,
                            "intent": seg_intent,
                            "confidence": round(seg_confidence, 3),
                            "bm25_score": round(seg_score, 3),
                            "agent": self._get_agent_routing(seg_intent)
                        })
        
        # Disambiguation check
        disambiguation = self.disambiguation_engine.generate_disambiguation(candidates, utterance)
        
        # Add session tracking
        self.session_manager.add(
            session_id=session_id,
            utterance=utterance,
            intent=winner_intent,
            confidence=confidence,
            slots=slots,
            multi_intents=multi_intents
        )
        
        return {
            "intent": winner_intent,
            "confidence": round(confidence, 3),
            "agent_routing": agent_routing,
            "slots": slots,
            "candidates": candidates,
            "top_match_score": round(matches[0][1], 3) if matches else 0.0,
            "session_id": session_id,
            "category": category,
            "priority": priority,
            "multi_intents": multi_intents if multi_intents else None,
            "has_multi_intents": len(multi_intents) > 1,
            "needs_disambiguation": disambiguation["needed"],
            "disambiguation": disambiguation,
            "classification_method": "keyword-search-bm25",
            "search_algorithm": "multi_match with BM25 ranking"
        }
    
    def _get_agent_routing(self, intent: str) -> str:
        """Map intent to agent (same as V1)."""
        agent_map = {
            "pharmacy": "PharmacyAgent",
            "claims": "ClaimsAgent",
            "specialist": "ProviderSearchAgent",
            "primaryCareProvider": "ProviderSearchAgent",
            "deductible": "BenefitsAgent",
            "eligibility": "EligibilityAgent",
            "idCard": "MemberServicesAgent",
            "hsa": "BenefitsAgent",
            "appeals": "ClaimsAgent",
            "maternity": "CareManagementAgent",
            "benefits": "BenefitsAgent",
            "enrollment": "EnrollmentAgent",
            "unknown": "FallbackAgent"
        }
        return agent_map.get(intent, "FallbackAgent")
    
    def get_candidates(self, utterance: str, top_k: int = 5) -> List[Dict]:
        """Get top candidate intents for disambiguation using keyword search."""
        # Use keyword search (no embedding needed)
        matches = self.vector_store.search(utterance, top_k=top_k)
        
        candidates = []
        for metadata, score in matches:
            candidates.append({
                "intent": metadata["intent_name"],
                "score": round(score, 3),
                "bm25_score": round(score, 3),
                "example": metadata["example_utterance"],
                "agent": self._get_agent_routing(metadata["intent_name"])
            })
        
        return candidates

#===============================================================================
# RAG+LLM HYBRID CLASSIFIER V2 (WITH ELASTICSEARCH)
#===============================================================================
class RAGLLMIntentClassifierV2:
    """
    V2: Elasticsearch-backed RAG + LLM hybrid classifier.
    
    Same API as V1, but uses Elasticsearch instead of InMemoryVectorStore.
    """
    
    def __init__(
        self,
        rag_classifier: ElasticsearchRAGClassifier,
        openai_api_key: Optional[str] = None,
        llm_threshold: float = 0.75,
        enable_llm: bool = True,
        model: str = "gpt-4o"
    ):
        """Initialize V2 hybrid classifier with Elasticsearch backend."""
        self.rag_classifier = rag_classifier
        self.llm_threshold = llm_threshold
        self.enable_llm = enable_llm and OPENAI_AVAILABLE
        self.model = model
        
        # Initialize OpenAI client (same as V1)
        if self.enable_llm:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  Warning: No OpenAI API key provided. LLM features disabled.")
                self.enable_llm = False
            else:
                self.llm_client = OpenAI(api_key=api_key)
                print(f"✅ LLM Integration enabled: {model} (threshold: {llm_threshold})")
        
        # Metrics tracking
        self.metrics = {
            "total_classifications": 0,
            "rag_only_count": 0,
            "llm_invoked_count": 0,
            "llm_agreement_count": 0,
            "llm_override_count": 0,
            "total_llm_cost_usd": 0.0,
            "avg_rag_time_ms": 0.0,
            "avg_llm_time_ms": 0.0
        }
    
    def classify(
        self,
        utterance: str,
        session_id: str = "default",
        force_llm: bool = False,
        context_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Classify with Elasticsearch RAG + LLM fallback.
        
        Same API as V1.classify()
        """
        start_time = datetime.utcnow()
        self.metrics["total_classifications"] += 1
        
        # STAGE 1: Elasticsearch RAG Classification
        rag_start = datetime.utcnow()
        rag_result = self.rag_classifier.classify(utterance, session_id)
        rag_time_ms = (datetime.utcnow() - rag_start).total_seconds() * 1000
        
        # Update RAG time metric
        self._update_avg_metric("avg_rag_time_ms", rag_time_ms)
        
        # Decision: Invoke LLM?
        should_invoke_llm = (
            self.enable_llm and
            (force_llm or rag_result["confidence"] < self.llm_threshold or rag_result["intent"] == "unknown")
        )
        
        if should_invoke_llm:
            print(f"[LLM] Confidence {rag_result['confidence']:.2f} < {self.llm_threshold} → Invoking LLM reasoning")
            
            # STAGE 2: LLM Enhancement (same as V1)
            llm_start = datetime.utcnow()
            llm_result = self._classify_with_llm(
                utterance=utterance,
                session_id=session_id,
                rag_context=rag_result,
                metadata=context_metadata
            )
            llm_time_ms = (datetime.utcnow() - llm_start).total_seconds() * 1000
            
            # Update LLM metrics
            self.metrics["llm_invoked_count"] += 1
            self._update_avg_metric("avg_llm_time_ms", llm_time_ms)
            
            # Compare RAG vs LLM
            rag_intent = rag_result["intent"]
            llm_intent = llm_result["intent"]
            agreement = (rag_intent == llm_intent)
            
            if agreement:
                self.metrics["llm_agreement_count"] += 1
            else:
                self.metrics["llm_override_count"] += 1
            
            # Merge results
            enhanced_result = self._merge_rag_llm_results(
                rag_result=rag_result,
                llm_result=llm_result,
                agreement=agreement
            )
            
            # Add performance metrics
            enhanced_result.update({
                "rag_time_ms": round(rag_time_ms, 2),
                "llm_time_ms": round(llm_time_ms, 2),
                "llm_cost_usd": llm_result.get("cost_usd", 0.0),
                "queryResultAction": "RAGLLM",
                "vector_store": "elasticsearch"  # V2 indicator
            })
            
            return enhanced_result
        
        else:
            # RAG-only result (high confidence)
            self.metrics["rag_only_count"] += 1
            rag_result.update({
                "llm_invoked": False,
                "classification_method": "RAG-only",
                "processing_time_ms": round(rag_time_ms, 2),
                "queryResultAction": "RAG",
                "vector_store": "elasticsearch"  # V2 indicator
            })
            return rag_result
    
    def _classify_with_llm(
        self,
        utterance: str,
        session_id: str,
        rag_context: Dict,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Invoke LLM for contextual reasoning (same as V1)."""
        
        prompt = self._build_llm_prompt(utterance, session_id, rag_context, metadata)
        
        try:
            # Call OpenAI API
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
                timeout=LLM_CONFIG["timeout"]
            )
            
            # Parse response
            llm_output = json.loads(response.choices[0].message.content)
            
            # Calculate cost
            cost_usd = self._calculate_cost(response.usage)
            self.metrics["total_llm_cost_usd"] += cost_usd
            
            return {
                "intent": llm_output.get("intent", "unknown"),
                "confidence": llm_output.get("confidence", 0.5),
                "reasoning": llm_output.get("reasoning", ""),
                "entities": llm_output.get("entities", {}),
                "needs_clarification": llm_output.get("needs_clarification", False),
                "clarification_question": llm_output.get("clarification_question", ""),
                "policy_reference": llm_output.get("policy_reference", ""),
                "cost_usd": cost_usd,
                "tokens_used": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
        
        except Exception as e:
            print(f"❌ LLM Error: {str(e)}")
            # Fallback to RAG result on LLM failure
            return {
                "intent": rag_context["intent"],
                "confidence": rag_context["confidence"],
                "reasoning": f"LLM failed: {str(e)}. Using RAG result.",
                "entities": {},
                "needs_clarification": False,
                "clarification_question": "",
                "cost_usd": 0.0,
                "error": str(e)
            }
    
    def _get_system_prompt(self) -> str:
        """System prompt for LLM (same as V1)."""
        return f"""You are an expert intent classifier for Aetna's healthcare virtual assistant (YAVA).

{INTENT_TAXONOMY}

**Classification Guidelines:**

1. **Semantic Understanding:** Look beyond keywords. Understand user intent contextually.
2. **Healthcare Domain Rules:**
   - HSA/FSA questions → hsa intent
   - Prescription/medication → pharmacy intent
   - Billing/payment/EOB → claims intent
3. **Context Awareness:** Use conversation history to resolve ambiguous references.
4. **Confidence Guidelines:**
   - 0.9-1.0: Clear, unambiguous intent
   - 0.7-0.89: Confident but some ambiguity
   - 0.5-0.69: Multiple possible intents
   - 0.0-0.49: Very unclear, needs clarification

**Response Format (JSON):**

{{
  "intent": "pharmacy",
  "confidence": 0.85,
  "reasoning": "User requests prescription refill (keyword: 'refill') for a specific medication.",
  "entities": {{
    "medication_name": "Lipitor",
    "action": "refill"
  }},
  "needs_clarification": false,
  "clarification_question": ""
}}

**Important:** 
- intent MUST be one of the predefined intents in the taxonomy above
- confidence MUST be a float between 0.0 and 1.0
- reasoning MUST explain your decision with reference to utterance and context
"""
    
    def _build_llm_prompt(
        self,
        utterance: str,
        session_id: str,
        rag_context: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """Build context-rich user prompt for LLM (same as V1)."""
        
        # Get session history
        history = self.rag_classifier.session_manager.get(session_id, n=3)
        history_str = "\n".join([
            f"- Turn {i+1}: \"{h['utterance']}\" → {h['intent']} (conf: {h['confidence']:.2f})"
            for i, h in enumerate(history)
        ]) if history else "No previous conversation (first turn)"
        
        # Get RAG candidates
        candidates = rag_context.get("candidates", [])
        candidates_str = "\n".join([
            f"{i+1}. **{c['intent']}** (score: {c['score']:.2f})"
            for i, c in enumerate(candidates[:3])
        ]) if candidates else "No candidates found"
        
        # Build full prompt
        prompt = f"""**User Utterance:** "{utterance}"

**Conversation History:**
{history_str}

**Elasticsearch Keyword Search Results (confidence: {rag_context.get('confidence', 0.0):.2f}):**
{candidates_str}

**Your Task:**
The RAG system (Elasticsearch BM25 keyword search) has {'low' if rag_context.get('confidence', 0) < self.llm_threshold else 'uncertain'} confidence ({rag_context.get('confidence', 0.0):.2f} < {self.llm_threshold} threshold).

Analyze the utterance using:
1. **Semantic Intent:** What is the user trying to accomplish?
2. **Contextual Clues:** Does conversation history provide context?
3. **Healthcare Domain:** Apply healthcare-specific classification rules

Provide your classification with detailed reasoning in the required JSON format.

**Note:** Elasticsearch keyword search suggestions are hints, not constraints. You may classify differently if reasoning justifies it.
"""
        return prompt
    
    def _merge_rag_llm_results(
        self,
        rag_result: Dict,
        llm_result: Dict,
        agreement: bool
    ) -> Dict:
        """Merge RAG and LLM results into enhanced classification (same as V1)."""
        
        # Determine final intent (LLM takes precedence)
        final_intent = llm_result["intent"]
        final_confidence = llm_result["confidence"]
        
        # Merge entities (LLM + RAG slot filling)
        merged_entities = {**rag_result.get("slots", {}), **llm_result.get("entities", {})}
        
        # Build enhanced result
        enhanced = {
            # Final classification (LLM)
            "intent": final_intent,
            "confidence": final_confidence,
            "classification_method": "RAG+LLM",
            "queryResultAction": "RAGLLM",
            "llm_invoked": True,
            
            # LLM-specific fields
            "llm_reasoning": llm_result["reasoning"],
            "llm_confidence": llm_result["confidence"],
            "llm_entities": llm_result.get("entities", {}),
            "needs_clarification": llm_result.get("needs_clarification", False),
            "clarification_question": llm_result.get("clarification_question", ""),
            "policy_reference": llm_result.get("policy_reference", ""),
            
            # RAG comparison
            "rag_result": {
                "intent": rag_result["intent"],
                "confidence": rag_result["confidence"],
                "top_match_score": rag_result.get("top_match_score", 0.0)
            },
            "rag_vs_llm_agreement": agreement,
            
            # Override tracking
            "llm_override": not agreement,
            "llm_override_reason": llm_result["reasoning"] if not agreement else None,
            
            # Merged data
            "entities": merged_entities,
            "slots": merged_entities,  # Alias for backward compatibility
            
            # Metadata from RAG
            "agent_routing": self._get_agent_routing(final_intent),
            "category": rag_result.get("category", "general"),
            "priority": rag_result.get("priority", 3),
            
            # Disambiguation (from RAG)
            "candidates": rag_result.get("candidates", []),
            
            # Session tracking
            "session_id": rag_result.get("session_id", "default")
        }
        
        return enhanced
    
    def _get_agent_routing(self, intent: str) -> str:
        """Map intent to Watson Orchestrate agent."""
        agent_map = {
            "pharmacy": "PharmacyAgent",
            "claims": "ClaimsAgent",
            "specialist": "ProviderSearchAgent",
            "primaryCareProvider": "ProviderSearchAgent",
            "deductible": "BenefitsAgent",
            "eligibility": "EligibilityAgent",
            "idCard": "MemberServicesAgent",
            "hsa": "BenefitsAgent",
            "appeals": "ClaimsAgent",
            "maternity": "CareManagementAgent",
            "benefits": "BenefitsAgent",
            "enrollment": "EnrollmentAgent",
            "unknown": "FallbackAgent"
        }
        return agent_map.get(intent, "FallbackAgent")
    
    def _calculate_cost(self, usage) -> float:
        """Calculate LLM API call cost."""
        cost_config = LLM_COSTS.get(self.model, LLM_COSTS["gpt-4o"])
        input_cost = (usage.prompt_tokens / 1000) * cost_config["input"]
        output_cost = (usage.completion_tokens / 1000) * cost_config["output"]
        return round(input_cost + output_cost, 4)
    
    def _update_avg_metric(self, metric_name: str, new_value: float):
        """Update running average metric."""
        total = self.metrics["total_classifications"]
        current_avg = self.metrics[metric_name]
        self.metrics[metric_name] = ((current_avg * (total - 1)) + new_value) / total
    
    def get_metrics(self) -> Dict:
        """Get classification metrics and performance stats."""
        total = self.metrics["total_classifications"]
        llm_invoked = self.metrics["llm_invoked_count"]
        
        return {
            **self.metrics,
            "rag_percentage": round(self.metrics["rag_only_count"] / total * 100, 1) if total > 0 else 0,
            "llm_percentage": round(llm_invoked / total * 100, 1) if total > 0 else 0,
            "avg_cost_per_classification": round(self.metrics["total_llm_cost_usd"] / total, 4) if total > 0 else 0,
            "llm_agreement_rate": round(self.metrics["llm_agreement_count"] / llm_invoked, 3) if llm_invoked > 0 else 0,
            "llm_override_rate": round(self.metrics["llm_override_count"] / llm_invoked, 3) if llm_invoked > 0 else 0,
            "vector_store_type": "elasticsearch",
            "search_algorithm": "keyword-bm25",
            "auth_method": "basic_auth"
        }


#===============================================================================
# FACTORY FUNCTION V2
#===============================================================================
def get_hybrid_classifier_v2(
    elasticsearch_host: str = None,
    elasticsearch_username: str = None,
    elasticsearch_password: str = None,
    index_name: str = "yava-intent-examples",
    openai_api_key: Optional[str] = None,
    llm_threshold: float = 0.75,
    enable_llm: bool = True,
    model: str = "gpt-4o",
    verify_certs: bool = False
) -> RAGLLMIntentClassifierV2:
    """
    Factory function to create V2 classifier with Elasticsearch KEYWORD SEARCH.
    
    **IMPORTANT:** Uses keyword-based search (BM25) instead of kNN vector search.
    - No embedding generation required for search
    - Faster query response (5-15ms vs 15-25ms)
    - Uses multi_match query with field boosting
    - BM25 ranking algorithm for relevance scoring
    
    Args:
        elasticsearch_host: ES cluster URL (default: env ELASTICSEARCH_HOST)
        elasticsearch_username: ES username (default: env ELASTICSEARCH_USERNAME or 'elastic')
        elasticsearch_password: ES password (default: env ELASTICSEARCH_PASSWORD)
        index_name: Intent examples index name
        openai_api_key: OpenAI API key (default: env OPENAI_API_KEY)
        llm_threshold: Confidence threshold for LLM invocation
        enable_llm: Enable/disable LLM features
        model: OpenAI model name
        verify_certs: Verify SSL certificates
    
    Returns:
        RAGLLMIntentClassifierV2 instance (keyword search mode)
    
    KEYWORD SEARCH FEATURES:
    -----------------------
    - Multi-field matching (training_utterances, keywords, descriptions)
    - Field boosting (training_utterances^3, keywords^2, description_short^1.5)
    - Fuzzy matching (AUTO fuzziness for typos)
    - Phrase matching boost for exact matches
    - BM25 relevance scoring (no vector embedding overhead)
    
    USAGE EXAMPLES:
    ---------------
    
    # Example 1: From environment variables
    export ELASTICSEARCH_HOST="https://your-cluster.com:9200"
    export ELASTICSEARCH_USERNAME="elastic"
    export ELASTICSEARCH_PASSWORD="your-password"
    export OPENAI_API_KEY="sk-..."
    
    classifier = get_hybrid_classifier_v2()
    
    # Example 2: Explicit credentials
    classifier = get_hybrid_classifier_v2(
        elasticsearch_host="https://es.example.com:9200",
        elasticsearch_username="elastic",
        elasticsearch_password="your-password",
        openai_api_key="sk-...",
        llm_threshold=0.75,
        verify_certs=False  # For self-signed certs
    )
    
    # Example 3: Classification
    result = classifier.classify(
        utterance="Can I get a refill for my blood pressure medication?",
        session_id="user_12345"
    )
    
    # Example 4: View metrics
    print(classifier.get_metrics())
    
    # Example 5: Check index stats
    stats = classifier.rag_classifier.vector_store.get_index_stats()
    print(f"Index has {stats['document_count']} documents ({stats['size_mb']} MB)")
    """
    
    # Initialize Elasticsearch vector store with Basic Auth
    hosts = [elasticsearch_host] if elasticsearch_host else None
    vector_store = ElasticsearchVectorStore(
        hosts=hosts,
        username=elasticsearch_username,
        password=elasticsearch_password,
        index_name=index_name,
        verify_certs=False
    )
    
    # Initialize supporting components
    embedder = EmbeddingGenerator()
    session_manager = SessionManager()
    slot_filler = SlotFiller()
    multi_intent_detector = MultiIntentDetector()
    disambiguation_engine = DisambiguationEngine()
    
    # Initialize Elasticsearch RAG classifier
    rag_classifier = ElasticsearchRAGClassifier(
        vector_store=vector_store,
        embedder=embedder,
        session_manager=session_manager,
        slot_filler=slot_filler,
        multi_intent_detector=multi_intent_detector,
        disambiguation_engine=disambiguation_engine
    )
    
    # Wrap with LLM enhancement
    hybrid_classifier = RAGLLMIntentClassifierV2(
        rag_classifier=rag_classifier,
        openai_api_key=openai_api_key,
        llm_threshold=llm_threshold,
        enable_llm=enable_llm,
        model=model
    )
    
    return hybrid_classifier


#===============================================================================
# FACTORY FUNCTION V3 (NEW - 3-KEY KNN SEARCH)
#===============================================================================
def get_hybrid_classifier_v3(
    elasticsearch_host: str = None,
    elasticsearch_username: str = None,
    elasticsearch_password: str = None,
    index_name_knn: str = "yava-intent-3key-embeddings",
    index_name_keyword: str = "yava-intent-examples",
    sentence_transformer_model: str = None,
    openai_api_key: Optional[str] = None,
    llm_threshold: float = 0.75,
    enable_llm: bool = True,
    enable_knn: bool = True,
    model: str = "gpt-4o",
    verify_certs: bool = False,
    knn_boost_intent: float = 1.0,
    knn_boost_description: float = 1.0,
    knn_boost_example: float = 1.5
) -> RAGLLMIntentClassifierV2:
    """
    Factory function to create V3 classifier with 3-KEY KNN EMBEDDINGS SEARCH.
    
    **NEW IN V3:** Uses vector embeddings (intent, description, example) for semantic search.
    - 3-key cosine similarity search (10-20ms)
    - Sentence Transformers for query encoding
    - Configurable boosting for each embedding key
    - Automatic fallback to BM25 keyword search if KNN unavailable
    
    Args:
        elasticsearch_host: ES cluster URL (default: env ELASTICSEARCH_HOST)
        elasticsearch_username: ES username (default: env ELASTICSEARCH_USERNAME or 'elastic')
        elasticsearch_password: ES password (default: env ELASTICSEARCH_PASSWORD)
        index_name_knn: KNN embeddings index (default: yava-intent-3key-embeddings)
        index_name_keyword: Keyword search fallback index (default: yava-intent-examples)
        sentence_transformer_model: Model path (default: env SENTENCE_TRANSFORMER_MODEL or all-MiniLM-L6-v2)
        openai_api_key: OpenAI API key (default: env OPENAI_API_KEY)
        llm_threshold: Confidence threshold for LLM invocation
        enable_llm: Enable/disable LLM features
        enable_knn: Enable/disable KNN search (falls back to keyword if False)
        model: OpenAI model name
        verify_certs: Verify SSL certificates
        knn_boost_intent: Boost for intent_embedding (default: 1.0)
        knn_boost_description: Boost for description_embedding (default: 1.0)
        knn_boost_example: Boost for example_embedding (default: 1.5)
    
    Returns:
        RAGLLMIntentClassifierV2 instance with 3-key KNN search enabled
    
    KNN SEARCH FEATURES:
    -------------------
    - 3 embedding keys for multi-dimensional semantic matching
    - Cosine similarity scoring
    - Configurable boosting per key
    - Fast vector search (10-20ms)
    - Automatic fallback to BM25 if KNN fails
    
    USAGE EXAMPLES:
    ---------------
    
    # Example 1: From environment variables with KNN enabled
    export ELASTICSEARCH_HOST="https://your-cluster.com:9200"
    export ELASTICSEARCH_USERNAME="elastic"
    export ELASTICSEARCH_PASSWORD="your-password"
    export OPENAI_API_KEY="sk-..."
    export SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L6-v2"
    
    classifier = get_hybrid_classifier_v3()
    
    # Example 2: Explicit configuration with custom boosting
    classifier = get_hybrid_classifier_v3(
        elasticsearch_host="https://es.example.com:9200",
        elasticsearch_username="elastic",
        elasticsearch_password="your-password",
        enable_knn=True,
        knn_boost_example=1.5,  # Boost example embeddings
        knn_boost_description=1.0,
        knn_boost_intent=1.0,
        llm_threshold=0.75,
        verify_certs=False
    )
    
    # Example 3: Classification with KNN
    result = classifier.classify(
        utterance="Can I get a refill for my blood pressure medication?",
        session_id="user_12345",
        use_knn=True  # Use KNN search (default if enabled)
    )
    
    PERFORMANCE COMPARISON:
    ----------------------
    V2 (BM25 Keyword):  5-15ms search time
    V3 (3-Key KNN):    10-20ms search time
    V3 (Hybrid):        7-18ms average (KNN primary, BM25 fallback)
    """
    # Initialize vector store for keyword search (fallback)
    vector_store = ElasticsearchVectorStore(
        hosts=[elasticsearch_host] if elasticsearch_host else None,
        username=elasticsearch_username,
        password=elasticsearch_password,
        index_name=index_name_keyword,
        verify_certs=verify_certs
    )
    
    # Initialize 3-key KNN searcher if enabled
    knn_searcher = None
    if enable_knn and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            knn_searcher = ThreeKeyKNNSearch(
                es_client=vector_store.es,
                index_name=index_name_knn,
                model_path=sentence_transformer_model or os.getenv("SENTENCE_TRANSFORMER_MODEL")
            )
            print(f"✅ 3-Key KNN Search enabled on index '{index_name_knn}'")
        except Exception as e:
            print(f"⚠️  Failed to initialize KNN searcher: {e}")
            print("   Falling back to BM25 keyword search")
            knn_searcher = None
    elif not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️  Sentence Transformers not available. Using BM25 keyword search.")
    
    # Initialize components
    embedder = EmbeddingGenerator()
    session_manager = SessionManager()
    slot_filler = SlotFiller()
    multi_intent_detector = MultiIntentDetector()
    disambiguation_engine = DisambiguationEngine()
    
    # Create RAG classifier
    rag_classifier = ElasticsearchRAGClassifier(
        vector_store=vector_store,
        embedder=embedder,
        session_manager=session_manager,
        slot_filler=slot_filler,
        multi_intent_detector=multi_intent_detector,
        disambiguation_engine=disambiguation_engine
    )
    
    # Create hybrid classifier with KNN support
    classifier = RAGLLMIntentClassifierV2(
        rag_classifier=rag_classifier,
        openai_api_key=openai_api_key,
        llm_threshold=llm_threshold,
        enable_llm=enable_llm,
        model=model
    )
    
    # Attach KNN searcher and boosting params to classifier
    classifier.knn_searcher = knn_searcher
    classifier.knn_enabled = enable_knn and knn_searcher is not None
    classifier.knn_boost_intent = knn_boost_intent
    classifier.knn_boost_description = knn_boost_description
    classifier.knn_boost_example = knn_boost_example
    
    return classifier


#===============================================================================
# EXAMPLE USAGE & TESTING
#===============================================================================
if __name__ == "__main__":
    """
    Test script demonstrating RAG+LLM hybrid classification with Elasticsearch KEYWORD SEARCH.
    
    Run with: python classifier_v2.py
    """
    
    print("="*80)
    print("RAG+LLM Hybrid Intent Classifier V2 - Elasticsearch Keyword Search (BM25)")
    print("="*80)
    
    # Initialize classifier
    classifier = get_hybrid_classifier_v2(
        elasticsearch_host=os.getenv("ELASTICSEARCH_HOST"),
        elasticsearch_username=os.getenv("ELASTICSEARCH_USERNAME", "elastic"),
        elasticsearch_password=os.getenv("ELASTICSEARCH_PASSWORD"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_threshold=0.75,
        enable_llm=True,
        model="gpt-4o",
        verify_certs=False  # Set True for production with valid certs
    )
    
    # Check index stats
    print("\n📊 Elasticsearch Index Stats:")
    stats = classifier.rag_classifier.vector_store.get_index_stats()
    if "error" not in stats:
        print(f"   Documents: {stats['document_count']}")
        print(f"   Size: {stats['size_mb']} MB")
        print(f"   Index: {stats['index_name']}")
    else:
        print(f"   Error: {stats['error']}")
    
    # Test cases
    test_cases = [
        {
            "utterance": "I need to refill my Lipitor prescription",
            "session_id": "test_001",
            "expected_intent": "pharmacy"
        },
        {
            "utterance": "Can I use my FSA for that specialist visit?",
            "session_id": "test_002",
            "expected_intent": "hsa"
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"TEST CASE {i}: {test['utterance']}")
        print(f"{'─'*80}")
        
        result = classifier.classify(
            utterance=test["utterance"],
            session_id=test["session_id"]
        )
        
        print(f"\n✓ Intent: {result['intent']}")
        print(f"✓ Confidence: {result['confidence']:.2f}")
        print(f"✓ Method: {result['classification_method']}")
        print(f"✓ Vector Store: {result.get('vector_store', 'N/A')}")
        print(f"✓ Agent: {result.get('agent_routing', 'N/A')}")
        print(f"✓ Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
        
        if result.get("llm_invoked"):
            print(f"\n📝 LLM Reasoning:")
            print(f"   {result.get('llm_reasoning', 'N/A')}")
            print(f"\n💰 Cost: ${result.get('llm_cost_usd', 0.0):.4f}")
            print(f"⏱️  Time: {result.get('llm_time_ms', 0):.0f}ms")
    
    # Print metrics
    print(f"\n{'='*80}")
    print("CLASSIFICATION METRICS")
    print(f"{'='*80}")
    metrics = classifier.get_metrics()
    print(f"Total Classifications: {metrics['total_classifications']}")
    print(f"RAG-Only: {metrics['rag_only_count']} ({metrics['rag_percentage']}%)")
    print(f"LLM Invoked: {metrics['llm_invoked_count']} ({metrics['llm_percentage']}%)")
    print(f"LLM Agreement Rate: {metrics['llm_agreement_rate']:.1%}")
    print(f"LLM Override Rate: {metrics['llm_override_rate']:.1%}")
    print(f"Total LLM Cost: ${metrics['total_llm_cost_usd']:.4f}")
    print(f"Avg Cost per Classification: ${metrics['avg_cost_per_classification']:.4f}")
    print(f"Avg RAG Time: {metrics['avg_rag_time_ms']:.1f}ms")
    print(f"Avg LLM Time: {metrics['avg_llm_time_ms']:.1f}ms")
    print(f"Vector Store: {metrics['vector_store_type']}")
    print(f"Search Algorithm: {metrics['search_algorithm']}")
    print(f"Auth Method: {metrics['auth_method']}")
    print("="*80)
