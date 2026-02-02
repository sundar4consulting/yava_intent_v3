import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'esSearchintentList_47_keyword_mapping_embeddings.json')
MODEL_PATH = os.path.join(BASE_DIR, '../Transformer')

# Load embeddings
with open(EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
    embedding_data = json.load(f)

# Load model
model = SentenceTransformer(MODEL_PATH)

# Prepare matrix and metadata
embeddings = np.array([item['embedding'] for item in embedding_data])
metadata = [
    {
        'intent': item['intent'],
        'example_utterances': item['example_utterances'],
        'description': item['description']
    }
    for item in embedding_data
]

def knn_search(query, k=5):
    """
    Given a query string, return top k most similar intents using cosine similarity.
    """
    query_vec = model.encode(query)
    # Normalize vectors
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    # Cosine similarity
    scores = np.dot(emb_norm, query_norm)
    top_k_idx = np.argsort(scores)[-k:][::-1]
    results = [
        {
            'score': float(scores[i]),
            'intent': metadata[i]['intent'],
            'description': metadata[i]['description'],
            'example_utterances': metadata[i]['example_utterances']
        }
        for i in top_k_idx
    ]
    return results

if __name__ == "__main__":
    user_utterance = input("Enter member utterance: ")
    top_k = 5
    results = knn_search(user_utterance, k=top_k)
    print(f"Top {top_k} most similar intents:")
    for idx, res in enumerate(results, 1):
        print(f"{idx}. Intent: {res['intent']} (Score: {res['score']:.4f})")
        print(f"   Description: {res['description']}")
        print(f"   Example Utterances: {res['example_utterances']}")
        print()
