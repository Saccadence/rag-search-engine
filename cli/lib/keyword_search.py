from pathlib import Path
import json

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

def load_movies(search_query):
    file_path = ROOT / "data" / "movies.json"
    with file_path.open("r") as f:
        movies = json.load(f)
    
    results = []
    for movie in movies.keys():
        if search_query in movie["title"]:
            results.append(movie)
    return results