from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
EMBEDS_F = ROOT / "cache" / "movie_embeddings.npy"


class SemanticSearch():
    def __init__(self) -> None:
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text: str):
        if not text or text == '':
            raise ValueError("Text is empty")

        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: dict):
        self.documents = documents
        doc_rep = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_rep.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_rep, show_progress_bar=True)
        
        Path(ROOT / "cache").mkdir(exist_ok=True)
        np.save(EMBEDS_F, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: dict):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        
        if EMBEDS_F.is_file():
            self.embeddings = np.load(EMBEDS_F)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int) -> list:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embedding = self.generate_embedding(query)
        scores = []
        for i in range(len(self.embeddings)):
            doc_embed = self.embeddings[i]
            doc = self.documents[i]
            scores.append((cosine_similarity(embedding, doc_embed), doc))
        scores.sort(key=lambda e: e[0], reverse=True)
        
        results = []
        for score in scores:
            results.append({
                "score": score[0],
                "title": score[1]["title"],
                "description": score[1]["description"]
            })
            if len(results) == limit:
                break
        
        return results

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
    
def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    search = SemanticSearch()
    with open(ROOT / "data" / "movies.json", "r") as f:
        documents = json.load(f)["movies"]
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
