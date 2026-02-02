#!/usr/bin/env python3
"""
3-Key KNN Search against Elasticsearch
Uses Elasticsearch's native KNN search for intent, description, and example embeddings.
"""

import json
import os
import sys
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from util.elasticsearch_client import ElasticsearchClient

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '../Transformer')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'esSearchintentList_47_3key_embeddings.json')

# Elasticsearch index name for 3-key embeddings
INDEX_NAME = "yava-intent-3key-embeddings"

# Embedding dimension (all-MiniLM-L6-v2 produces 384-dim vectors)
EMBEDDING_DIM = 384


class ThreeKeyKNNSearch:
    """
    Elasticsearch KNN search using 3 embedding keys:
    - intent_embedding
    - description_embedding  
    - example_embedding
    """
    
    def __init__(self, index_name: str = INDEX_NAME):
        self.index_name = index_name
        self.es_client = ElasticsearchClient()
        self.model = SentenceTransformer(MODEL_PATH)
        
    def create_index_with_knn_mappings(self) -> Dict:
        """
        Create Elasticsearch index with dense_vector mappings for KNN search.
        """
        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "index": {
                    "knn": True  # Enable KNN for this index
                }
            },
            "mappings": {
                "properties": {
                    "intent_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "description_short": {"type": "text"},
                    "example_utterance": {"type": "text"},
                    "category": {"type": "keyword"},
                    "intent_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "description_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "example_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        try:
            # Delete if exists
            if self.es_client.client.indices.exists(index=self.index_name):
                print(f"⚠️  Deleting existing index '{self.index_name}'...")
                self.es_client.client.indices.delete(index=self.index_name)
            
            # Create new index
            response = self.es_client.client.indices.create(
                index=self.index_name,
                body=index_body
            )
            print(f"✅ Index '{self.index_name}' created with KNN mappings")
            return {"success": True, "response": response}
        except Exception as e:
            print(f"❌ Failed to create index: {e}")
            return {"success": False, "error": str(e)}
    
    def index_embeddings(self, embeddings_path: str = EMBEDDINGS_PATH) -> Dict:
        """
        Index all 3-key embeddings into Elasticsearch.
        """
        # Load embeddings data
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
        
        print(f"📄 Loaded {len(embeddings_data)} documents to index")
        
        success_count = 0
        error_count = 0
        
        for idx, doc in enumerate(embeddings_data):
            try:
                self.es_client.client.index(
                    index=self.index_name,
                    id=str(idx),
                    document=doc,
                    refresh=False  # Batch refresh at end
                )
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to index doc {idx}: {e}")
                error_count += 1
        
        # Refresh index to make documents searchable
        self.es_client.client.indices.refresh(index=self.index_name)
        
        print(f"✅ Indexed {success_count} documents, {error_count} errors")
        return {"success": success_count, "errors": error_count}
    
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
            query_vector: The query embedding vector
            field: The embedding field to search (intent_embedding, description_embedding, example_embedding)
            k: Number of results to return
            num_candidates: Number of candidates to consider (higher = more accurate but slower)
        
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
            "_source": ["intent_name", "description_short", "example_utterance", "category"]
        }
        
        try:
            response = self.es_client.client.search(
                index=self.index_name,
                body=query
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'score': hit['_score'],
                    'intent_name': hit['_source'].get('intent_name', ''),
                    'description_short': hit['_source'].get('description_short', ''),
                    'example_utterance': hit['_source'].get('example_utterance', ''),
                    'category': hit['_source'].get('category', ''),
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
        strategy: str = "best_match"
    ) -> List[Dict]:
        """
        Perform KNN search across all 3 embedding keys.
        
        Args:
            query: The user's query string
            k: Number of results per key
            num_candidates: Number of candidates to consider
            strategy: 
                - "best_match": Return single best match across all keys
                - "all_keys": Return top results from each key separately
                - "combined": Combine and deduplicate results from all keys
        
        Returns:
            List of matching documents with scores
        """
        # Encode query
        query_vector = self.model.encode(query).tolist()
        
        fields = ['intent_embedding', 'description_embedding', 'example_embedding']
        all_results = {}
        
        # Search each key
        for field in fields:
            results = self.knn_search_single_key(query_vector, field, k, num_candidates)
            for r in results:
                key = r['intent_name']
                if key not in all_results or r['score'] > all_results[key]['score']:
                    all_results[key] = r
        
        if strategy == "best_match":
            # Return single best match
            if all_results:
                best = max(all_results.values(), key=lambda x: x['score'])
                return [best]
            return []
        
        elif strategy == "all_keys":
            # Return results grouped by key
            grouped = {}
            for field in fields:
                grouped[field] = self.knn_search_single_key(query_vector, field, k, num_candidates)
            return grouped
        
        elif strategy == "combined":
            # Return combined and deduplicated results, sorted by score
            combined = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
            return combined[:k]
        
        return list(all_results.values())
    
    def knn_search_multi_query(
        self,
        query: str,
        k: int = 5,
        boost_intent: float = 1.0,
        boost_description: float = 1.0,
        boost_example: float = 1.0
    ) -> List[Dict]:
        """
        Perform multi-query KNN search with boosting using Elasticsearch's bool query.
        This combines scores from all 3 keys with configurable weights.
        
        Args:
            query: The user's query string
            k: Number of results to return
            boost_intent: Weight for intent embedding matches
            boost_description: Weight for description embedding matches
            boost_example: Weight for example embedding matches
        
        Returns:
            List of matching documents with combined scores
        """
        query_vector = self.model.encode(query).tolist()
        
        # Use script_score to combine multiple KNN similarities
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
            "_source": ["intent_name", "description_short", "example_utterance", "category"]
        }
        
        try:
            response = self.es_client.client.search(
                index=self.index_name,
                body=search_query
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'score': hit['_score'],
                    'intent_name': hit['_source'].get('intent_name', ''),
                    'description_short': hit['_source'].get('description_short', ''),
                    'example_utterance': hit['_source'].get('example_utterance', ''),
                    'category': hit['_source'].get('category', ''),
                    'matched_field': 'combined_3key'
                })
            return results
        except Exception as e:
            print(f"❌ Multi-query KNN search failed: {e}")
            return []


def setup_index():
    """Setup the Elasticsearch index with 3-key embeddings."""
    searcher = ThreeKeyKNNSearch()
    
    # Test connection
    print("🔌 Testing Elasticsearch connection...")
    conn = searcher.es_client.test_connection()
    if not conn['success']:
        print("Failed to connect to Elasticsearch")
        return
    
    # Create index
    print("\n📦 Creating index with KNN mappings...")
    searcher.create_index_with_knn_mappings()
    
    # Index embeddings
    print("\n📥 Indexing 3-key embeddings...")
    searcher.index_embeddings()
    
    print("\n✅ Setup complete!")


def interactive_search():
    """Interactive search mode."""
    searcher = ThreeKeyKNNSearch()
    
    print("\n" + "="*60)
    print("🔍 3-Key KNN Search (Elasticsearch)")
    print("="*60)
    print("Strategies: best_match | combined | multi_query")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("Enter query: ").strip()
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        print("\n--- Best Match ---")
        results = searcher.knn_search_3key(query, k=1, strategy="best_match")
        for r in results:
            print(f"  Intent: {r['intent_name']}")
            print(f"  Score: {r['score']:.4f}")
            print(f"  Matched via: {r['matched_field']}")
            print(f"  Category: {r['category']}")
        
        print("\n--- Top 3 Combined ---")
        results = searcher.knn_search_3key(query, k=3, strategy="combined")
        for idx, r in enumerate(results, 1):
            print(f"  {idx}. {r['intent_name']} (Score: {r['score']:.4f}, via: {r['matched_field']})")
        
        print("\n--- Multi-Query (Weighted) ---")
        results = searcher.knn_search_multi_query(query, k=3, boost_example=1.5)
        for idx, r in enumerate(results, 1):
            print(f"  {idx}. {r['intent_name']} (Combined Score: {r['score']:.4f})")
        
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="3-Key KNN Search with Elasticsearch")
    parser.add_argument('--setup', action='store_true', help='Setup index and load embeddings')
    parser.add_argument('--search', action='store_true', help='Interactive search mode')
    parser.add_argument('--query', type=str, help='Single query to search')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_index()
    elif args.search:
        interactive_search()
    elif args.query:
        searcher = ThreeKeyKNNSearch()
        results = searcher.knn_search_3key(args.query, k=5, strategy="combined")
        print(f"\nResults for: '{args.query}'\n")
        for idx, r in enumerate(results, 1):
            print(f"{idx}. {r['intent_name']}")
            print(f"   Score: {r['score']:.4f}")
            print(f"   Matched via: {r['matched_field']}")
            print(f"   Description: {r['description_short']}")
            print()
    else:
        parser.print_help()
