from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json
import re
from .search_utils import *


class SemanticSearch():
    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_name)
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
    with open(MOVIES_JSON_F, "r") as f:
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

def chunk_text(text: str, chunk_size: int, overlap_size: int):
    char_count = len(text)
    text_w = text.split()
            
    chunks = []
    current = []
    overlap = []
    for word in text_w:
        current.append(word)
        if len(current) == chunk_size:
            chunks.append(' '.join(current))
            if overlap_size > 0:
                overlap = current[-overlap_size:]
            else:
                overlap = []
            current = overlap[:]
            
    if current and len(current) > len(overlap):
        chunks.append(' '.join(current))
            
    print(f"Chunking {char_count} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")
        
def semantic_chunk(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    text = text.strip()
    if text == '':
        return []
    text_s = re.split(r"(?<=[.!?])\s+", text)
    if len(text_s) == 1 and not text_s[0].endswith((".", "!", "?")):
        text_s = [text]
    
    chunks = []
    current = []
    overlap = []
    for sentence in text_s:
        sentence = sentence.strip()
        if sentence == '':
            continue
        current.append(sentence)
        if len(current) == chunk_size:
            chunks.append(' '.join(current))
            if overlap_size > 0:
                overlap = current[-overlap_size:]
            else:
                overlap = []
            current = overlap[:]
            
    if current and len(current) > len(overlap):
        chunks.append(' '.join(current))
        
    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        
        chunks = []
        chunks_md = []
        for i, doc in enumerate(documents):
            if not doc["description"] or doc["description"] == "":
                continue
            
            doc_chunks = semantic_chunk(doc["description"], 4, 1)
            chunks.extend(doc_chunks)
            for j in range(len(doc_chunks)):
                chunks_md.append({
                    "movie_idx": doc["id"],
                    "chunk_idx": j,
                    "total_chunks": len(doc_chunks)
                })
        
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_md

        CACHE.mkdir(exist_ok=True)
        np.save(CHUNK_EMBEDS_F, self.chunk_embeddings)
        with open(CHUNK_MD_F, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks_md, "total_chunks": len(chunks)}, f, indent=2, ensure_ascii=False)
        
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int=10):
        query_e = self.generate_embedding(query)
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("Chunk embeddings are not loaded. Please use load_or_create_chunk_embeddings")
        
        chunk_scores = []
        for i, chunk_e in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_e, chunk_e)
            chunk_scores.append({
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score
            })
        
        movie_scores = {}
        for chunk in chunk_scores:
            if (
                chunk["movie_idx"] not in movie_scores.keys() 
                or chunk["score"] > movie_scores[chunk["movie_idx"]]
            ):
                movie_scores[chunk["movie_idx"]] = chunk["score"]
        
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for movie_idx, score in sorted_movie_scores[:limit]:
            doc = self.document_map[movie_idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": [chunk for chunk in self.chunk_metadata if chunk["movie_idx"] == movie_idx] or {}
            })
        
        return results
        
    
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if CHUNK_EMBEDS_F.is_file() and CHUNK_MD_F.is_file():
            self.chunk_embeddings = np.load(CHUNK_EMBEDS_F)
            with open(CHUNK_MD_F, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
                self.chunk_metadata = chunk_data["chunks"]
            
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)


def embed_chunks():
    with open(MOVIES_JSON_F, "r", encoding="utf-8") as f:
        documents = json.load(f)["movies"]
    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
    
def unescape_unicode(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape")

def search_chunked(query: str, limit: int):
    with open(MOVIES_JSON_F, "r", encoding="utf-8") as f:
        documents = json.load(f)["movies"]
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(documents)
    
    results = search.search_chunks(query, limit)
    for i, result in enumerate(results, 1):
        title = unescape_unicode(result["title"])
        snippet = unescape_unicode(result["document"])

        print(f"\n{i}. {title} (score: {result['score']:.4f})")
        print(f"   {snippet}...")
    