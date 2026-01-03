from .semantic_search import *
from .search_utils import *
import numpy as np


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
