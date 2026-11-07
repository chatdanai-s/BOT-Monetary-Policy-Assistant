import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = "qwen3-embedding:0.6b"
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

texts = [
    # Monetary policy / technology
    'increase tech usage by 80%',
    'more tech usage by 80% by 2025',
    'tech usage increase 80%',
    'adopt tech 80%',

    # Animals
    'elephants are the largest land animals',
    'dogs are popular domestic pets',
    'lions hunt other prey in the wild',
    'dolphins communicate using sounds'
]

# Generate embeddings
vectors = embeddings.embed_documents(texts)

# Check embedding dimensions
print("Embedding dimensions:", len(vectors[0]))

# Calculate pairwise cosine similarities
similarity_matrix = cosine_similarity(vectors)

# Display results
print("\nCosine Similarity Matrix:")
print(f"{'':45}", end="")

for i in range(len(texts)):
    print(f"Text {i+1:>2}       ", end="")
print()

for i, text in enumerate(texts):
    print(f"Text {i+1}: {text:<40}", end="")

    for similarity in similarity_matrix[i]:
        print(f"{similarity:.4f}       ", end="")

    print()