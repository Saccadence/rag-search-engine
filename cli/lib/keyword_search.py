from pathlib import Path
from .text_processing import text_process
import json

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def load_movies(search_query) -> list:
    file_path = ROOT / "data" / "movies.json"
    with file_path.open("r") as f:
        movies = json.load(f)["movies"]
    
    matches = []
    query_tokens = text_process(search_query)
    for movie in movies:
        title_tokens = text_process(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            matches.append(movie)
            
    matches.sort(key=lambda m: m["id"])
    return matches[:5]

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False