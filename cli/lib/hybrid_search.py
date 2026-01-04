from pathlib import Path
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import *
import json


class HybridSearch():
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not Path.exists(INDEX_F):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        
        normalized = []
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            for _ in range(len(scores)):
                normalized.append(1.0)
        else:
            for score in scores:
                normalized.append((score - min_score) / (max_score - min_score))
        
        return normalized

    def weighted_search(self, query, alpha, limit=5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*10)

        # Convert BM25 results to dictionaries with id and score
        bm25_dicts = []
        for doc, score in bm25_results:
            bm25_dicts.append({"id": doc["id"], "score": score})

        bm25_scores = [result["score"] for result in bm25_dicts] if bm25_dicts else []
        semantic_scores = [result.get("score", 0) for result in semantic_results] if semantic_results else []
        
        normalized_bm25 = self.normalize(bm25_scores) or []
        normalized_semantic = self.normalize(semantic_scores) or []
        
        bm25_lookup = {}
        for i, result in enumerate(bm25_dicts):
            if i < len(normalized_bm25):
                bm25_lookup[result["id"]] = normalized_bm25[i]
        
        semantic_lookup = {}
        for i, result in enumerate(semantic_results):
            if i < len(normalized_semantic):
                semantic_lookup[result["id"]] = normalized_semantic[i]
        
        doc_scores = {}
        all_doc_ids = set(bm25_lookup.keys()) | set(semantic_lookup.keys())
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_lookup.get(doc_id, 0.0)
            semantic_score = semantic_lookup.get(doc_id, 0.0)
            
            doc_scores[doc_id] = {
                "id": doc_id,
                "bm25": bm25_score,
                "semantic": semantic_score,
                "hybrid": hybrid_score(bm25_score, semantic_score, alpha)
            }
        
        sorted_results = sorted(
            doc_scores.values(), 
            key=lambda s: s["hybrid"], 
            reverse=True
        )
        
        return sorted_results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    
    normalized = []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        normalized = [1.0] * len(scores)
    else:
        for score in scores:
            normalized.append((score - min_score) / (max_score - min_score))
    
    for score in normalized:
        print(f"* {score:.4f}")
    
    return normalized

def weighted_search(query, alpha, limit):
    with open(MOVIES_JSON_F, "r") as f:
        documents = json.load(f)["movies"]
    search = HybridSearch(documents)
    
    results = search.weighted_search(query, alpha, limit)
    for i, score in enumerate(results):
        # Find document by ID instead of using ID as index
        doc = next((d for d in documents if d["id"] == score["id"]), None)
        if doc:
            print(f'{i}. {doc["title"]}')
            print(f'Hybrid Score: {score["hybrid"]}')
            print(f'BM25: {score["bm25"]}, Semantic: {score["semantic"]}')
            print(f'{doc["description"][:100]}...')
        else:
            print(f'{i}. Document ID {score["id"]} not found')
        

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score
