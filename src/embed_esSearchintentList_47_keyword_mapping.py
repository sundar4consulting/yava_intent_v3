import json
from sentence_transformers import SentenceTransformer
import os

# Path to the JSON file
JSON_PATH = os.path.join(os.path.dirname(__file__), '../../esSearchintentList_47 _keyword_mapping.json')

# Load the JSON data
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize the sentence transformer model from local path
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../Transformer')
model = SentenceTransformer(LOCAL_MODEL_PATH)

# Prepare a list to hold embeddings and their keys
embeddings_data = []

for item in data:
    intent = item.get('intent_name', '')
    examples = item.get('training_utterances', [])
    description = item.get('description_short', '')
    # Combine all text fields for embedding
    text_to_embed = [intent, description] + examples
    # Remove empty strings
    text_to_embed = [t for t in text_to_embed if t]
    # Create a single string for embedding
    combined_text = ' '.join(text_to_embed)
    embedding = model.encode(combined_text)
    embeddings_data.append({
        'intent': intent,
        'example_utterances': examples,
        'description': description,
        'embedding': embedding.tolist()
    })

# Save the embeddings to a new file
output_path = os.path.join(os.path.dirname(__file__), 'esSearchintentList_47_keyword_mapping_embeddings.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

print(f"Embeddings saved to {output_path}")
