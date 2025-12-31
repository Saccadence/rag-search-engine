from pathlib import Path
import json

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

def load_movies(search_query):
    file_path = ROOT / "data" / "movies.json"
    with file_path.open("r") as f:
        movies = json.load(f)["movies"]
    
    matches = []
    for movie in movies:
        if search_query in movie["title"]:
            matches.append(movie)
    matches.sort(key=lambda m: m["id"])
    return matches[:5]