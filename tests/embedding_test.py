from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = "nomic-embed-text" 
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

result = embeddings.embed_query("Thailand monetary policy")

print("Embedding dimensions:", len(result))