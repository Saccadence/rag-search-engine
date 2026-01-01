from .text_processing import text_process
import json


def load_movies(search_query, docs) -> list:
    matches = set()
    query_tokens = text_process(search_query)
    for token in query_tokens:
        if token != '' and token in docs.index.keys():
            for doc_id in docs.index[token]:
                matches.add(doc_id)
                if len(matches) == 5:
                    break
    
    return sorted(matches)

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False