from pathlib import Path
from google import genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import *
import json
import time
import os


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
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
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

    def rrf_search(self, query, k, limit=10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        
        doc_ranks = {}
        for i, result in enumerate(bm25_results):
            doc = result[0]
            doc_id = doc["id"]
            doc_ranks[doc_id] = {
                "doc": doc,
                "bm25_rank": i
            }
        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if doc_id not in doc_ranks.keys():
                doc_ranks[doc_id] = {
                    "doc": doc,
                    "semantic_rank": i
                }
            else:
                doc_ranks[doc_id].update(semantic_rank = i)
        
        for doc_id, score in doc_ranks.items():
            rrf = 0.0
            for key in score.keys():
                if key != "doc":
                    rrf += rrf_score(score[key], k)
            doc_ranks[doc_id].update(rrf_score = rrf)
        
        sorted_results = sorted(
            doc_ranks.values(), 
            key=lambda s: s["rrf_score"],
            reverse=True
        )
        
        return sorted_results[:limit]

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

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search(query, k, limit, enhance=None, method=None, evaluate=False, cli=True):
    # Debug: Log original query
    print(f"[DEBUG] Original query: '{query}'")
    
    documents = _load_documents()
    search = HybridSearch(documents)
    final_query = _apply_query_enhancement(query, enhance)
    
    # Debug: Log query after enhancement
    if final_query != query:
        print(f"[DEBUG] Enhanced query: '{final_query}'")
    else:
        print(f"[DEBUG] Query after enhancement: '{final_query}' (no changes)")
    
    results = _get_initial_results(search, final_query, k, limit*5, method)
    
    # Debug: Log results after RRF search
    print(f"[DEBUG] Results after RRF search: {len(results)} documents retrieved")
    if results:
        print(f"[DEBUG] Top 3 RRF results: {[r['doc']['title'] for r in results[:3]]}")
    
    if method:
        print(f"Reranking top {limit} results using {method} method...")
        results = _apply_reranking(query, results, method)
        
        # Debug: Log final results after reranking
        print(f"[DEBUG] Results after reranking: {len(results)} documents")
        if results:
            print(f"[DEBUG] Top 3 reranked results: {[r['doc']['title'] for r in results[:3]]}")
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
    
    if cli:
        _display_results(results[:limit], method)
        if evaluate:
            _llm_review(query, results[:limit])
    return results[:limit]

def _load_documents():
    """Load movie documents from JSON file"""
    with open(MOVIES_JSON_F, "r") as f:
        return json.load(f)["movies"]

def _apply_query_enhancement(query, enhance_type):
    """Apply query enhancement if requested"""
    if not enhance_type:
        return query
    
    enhanced = _get_enhanced_query(query, enhance_type)
    if enhanced != query:
        print(f"Enhanced query ({enhance_type}): '{query}' -> '{enhanced}'\n")
    return enhanced

def _get_initial_results(search, query, k, limit, method):
    """Get initial RRF search results with appropriate limit"""
    result_limit = limit * 5 if method else limit
    return search.rrf_search(query, k, result_limit)

def _apply_reranking(query, results, method):
    """Apply reranking to results based on method"""
    match method:
        case "individual":
            return _rerank_individual(query, results)
        case "batch":
            return _rerank_batch(query, results)
        case "cross_encoder":
            return _rerank_cross_encoding(query, results)
        case _:
            return results

def _rerank_individual(query, results):
    """Rerank each result individually using LLM"""
    for result in results:
        score = _get_enhanced_score(query, result["doc"], "individual")
        try:
            result["rerank_score"] = float(score.strip() if score else "0.0")
        except (ValueError, AttributeError):
            result["rerank_score"] = 0.0
        time.sleep(3)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)

def _rerank_batch(query, results):
    """Rerank all results in a single batch using LLM"""
    batch_rankings = _get_enhanced_score(query, results, "batch")
    
    try:
        if not batch_rankings:
            raise ValueError("Received empty batch rankings")
        
        ranked_ids = json.loads(batch_rankings)
        rerank_map = {doc_id: len(ranked_ids) - i for i, doc_id in enumerate(ranked_ids)}
        
        for result in results:
            doc_id = result["doc"]["id"]
            result["rerank_score"] = rerank_map.get(doc_id, 0.0)
        
        return sorted(results, key=lambda r: r["rerank_score"], reverse=True)
    
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Error parsing batch rerank results: {e}. Returning original order.")
        return results

def _rerank_cross_encoding(query, results):
    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    for i, result in enumerate(results):
        result["cross_encoder_score"] = scores[i]
    
    return sorted(results, key=lambda r: r["cross_encoder_score"], reverse=True)

def _get_gemini_client():
    """Initialize and return Gemini API client"""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables")
        return None
    
    print(f"Using key {api_key[:6]}...")
    return genai.Client(api_key=api_key)

def _get_enhanced_query(query, enhance_type):
    """Enhance search query using LLM"""
    client = _get_gemini_client()
    if not client:
        return query
    
    prompts = {
        "spell": f"""Fix any spelling errors in this movie search query.

                        Only correct obvious typos. Don't change correctly spelled words.

                        Query: "{query}"

                        If no errors, return the original query.
                        Corrected:""",
        
        "rewrite": f"""Rewrite this movie search query to be more specific and searchable.

                        Original: "{query}"

                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep it concise (under 10 words)
                        - It should be a google style search query that's very specific
                        - Don't use boolean logic

                        Examples:

                        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                        Rewritten query:""",
        
        "expand": f"""Expand this movie search query with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        This will be appended to the original query.

                        Examples:

                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        Query: "{query}"
                        """
    }
    
    if enhance_type in prompts:
        response = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompts[enhance_type])
        return response.text if response else query
    
    return query

def _get_enhanced_score(query, doc, method_type):
    """Get enhanced relevance score using LLM"""
    client = _get_gemini_client()
    if not client:
        return "0.0"
    
    prompt = _build_rerank_prompt(query, doc, method_type)
    if not prompt:
        return "0.0"
    
    response = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
    return response.text if response else "0.0"

def _build_rerank_prompt(query, doc, method_type):
    """Build prompt for reranking based on method type"""
    match method_type:
        case "individual":
            return f"""Rate how well this movie matches the search query.

                        Query: "{query}"
                        Movie: {doc.get("title", "")} - {doc.get("description", "")}

                        Consider:
                        - Direct relevance to query
                        - User intent (what they're looking for)
                        - Content appropriateness

                        Rate 0-10 (10 = perfect match).
                        Give me ONLY the number in your response, no other text or explanation.

                        Score:"""
        case "batch":
            return f"""Rank these movies by relevance to the search query from most relevant to least relevant.

                        Query: "{query}"

                        Movies:
                        {_format_movies_for_batch(doc)}

                        Consider:
                        - Direct relevance to the search query
                        - User intent and what they're looking for
                        - Content appropriateness

                        Return ONLY the movie IDs in order of relevance (best match first) as a JSON array. 
                        For example: [75, 12, 34, 2, 1]

                        Return only the JSON array, no other text:"""
        case _:
            return None

def _format_movies_for_batch(results):
    """Format movies for batch reranking prompt"""
    return "\n".join(
        f"ID: {result['doc']['id']}, Title: {result['doc']['title']}, Description: {result['doc']['description']}"
        for result in results
    )

def _display_results(results, method=None):
    for i, result in enumerate(results, start=1):
        doc = result["doc"]
        if doc:
            print(f'{i}. {doc["title"]}')
            if method:
                if method == "cross_encoder":
                    cross_encoder_score = result.get("cross_encoder_score", 0.0)
                    print(f'   Cross Encoder Score: {cross_encoder_score:.3f}')
                else:
                    rerank_score = result.get("rerank_score", 0.0)
                    print(f'   Rerank Score: {rerank_score:.3f}/10')
            print(f'   RRF Score: {result["rrf_score"]:.3f}')
            bm25_rank = result.get("bm25_rank", "N/A")
            semantic_rank = result.get("semantic_rank", "N/A")
            print(f'   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}')
            print(f'   {doc["description"][:100]}...')
            print()
        else:
            print(f'{i}. Document not found')
            print()

def _llm_review(query, results):
    """Use LLM to evaluate search result relevance"""
    client = _get_gemini_client()
    if not client:
        print("Error: Failed to load Gemini client")
        return
    
    # Format results for the prompt
    formatted_results = []
    for i, result in enumerate(results, start=1):
        doc = result["doc"]
        formatted_results.append(f"{i}. {doc['title']} - {doc['description'][:200]}...")
    
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    
    response = None
    try:
        response = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
        if not response or not response.text:
            print("Error: No response from LLM")
            return
        
        # Parse JSON response
        scores = json.loads(response.text.strip())
        
        if not isinstance(scores, list) or len(scores) != len(results):
            print(f"Error: Expected {len(results)} scores, got {len(scores) if isinstance(scores, list) else 'invalid format'}")
            return
        
        # Print evaluation report
        print("\nEvaluation Report:")
        for i, (result, score) in enumerate(zip(results, scores), start=1):
            doc = result["doc"]
            print(f"{i}. {doc['title']}: {score}/3")
    
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {response.text if response else 'None'}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
            
