import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'esSearchintentList_47_3key_embeddings.json')
MODEL_PATH = os.path.join(BASE_DIR, '../Transformer')

# Load embeddings
with open(EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
    embedding_data = json.load(f)

# Load model
model = SentenceTransformer(MODEL_PATH)

# Prepare matrices and metadata
intent_embs = np.array([item['intent_embedding'] for item in embedding_data])
desc_embs = np.array([item['description_embedding'] for item in embedding_data])
ex_embs = np.array([item['example_embedding'] for item in embedding_data])
metadata = [
    {
        'intent_name': item['intent_name'],
        'description_short': item['description_short'],
        'example_utterance': item['example_utterance']
    }
    for item in embedding_data
]

def knn_search_3key(query, key='intent', k=5):
    """
    Given a query string, return top k most similar intents using cosine similarity on the selected key.
    key: 'intent', 'description', or 'example'
    """
    query_vec = model.encode(query)
    if key == 'intent':
        emb_matrix = intent_embs
    elif key == 'description':
        emb_matrix = desc_embs
    elif key == 'example':
        emb_matrix = ex_embs
    else:
        raise ValueError("key must be one of 'intent', 'description', or 'example'")
    # Normalize vectors
    emb_norm = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)
    # Cosine similarity
    scores = np.dot(emb_norm, query_norm)
    top_k_idx = np.argsort(scores)[-k:][::-1]
    results = [
        {
            'score': float(scores[i]),
            'intent_name': metadata[i]['intent_name'],
            'description_short': metadata[i]['description_short'],
            'example_utterance': metadata[i]['example_utterance']
        }
        for i in top_k_idx
    ]
    return results


if __name__ == "__main__":
    user_query = input("Enter member utterance: ")
    top_k = 5
    # Run KNN search for all three keys
    results_all = []
    for key in ["intent", "description", "example"]:
        results = knn_search_3key(user_query, key=key, k=top_k)
        for res in results:
            res_with_key = res.copy()
            res_with_key['matched_key'] = key
            results_all.append(res_with_key)
    # Find the best match overall
    best_match = max(results_all, key=lambda x: x['score'])
    print("\nBest intent match for the utterance:")
    print(f"Intent: {best_match['intent_name']} (Score: {best_match['score']:.4f}, Matched on: {best_match['matched_key']})")
    print(f"Description: {best_match['description_short']}")
    print(f"Example Utterance: {best_match['example_utterance']}")
    print()
    # Optionally, show top 3 matches overall
    print("Top 3 matches (any key):")
    for idx, res in enumerate(sorted(results_all, key=lambda x: x['score'], reverse=True)[:3], 1):
        print(f"{idx}. Intent: {res['intent_name']} (Score: {res['score']:.4f}, Matched on: {res['matched_key']})")
        print(f"   Description: {res['description_short']}")
        print(f"   Example Utterance: {res['example_utterance']}")
        print()
