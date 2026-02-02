from sentence_transformers import SentenceTransformer

# Point to your folder (even if it's missing the map file)
model = SentenceTransformer('/Users/2205287/Library/CloudStorage/OneDrive-Cognizant/Aetna/Architecture/Yava/Agents/knowledge-bases/Transformer')

# If this line prints a vector, you are safe.
print(model.encode("test").shape)
