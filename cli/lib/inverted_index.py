from pathlib import Path
from .text_processing import text_process
import os
import json
import pickle

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

class InvertedIndex():
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
    
    def __add_document(self, doc_id, text) -> None:
        tokens = text_process(text)
        for token in tokens:
            if token != '':
                self.index.setdefault(token, set()).add(doc_id)
    
    def get_documents(self, term) -> list:
        if term not in self.index.keys():
            return []
        return sorted(self.index[term])
    
    def build(self) -> None:
        file_path = ROOT / "data" / "movies.json"
        with file_path.open("r") as f:
            movies = json.load(f)["movies"]
        
        for movie in movies:
            text = f'{movie["title"]} {movie["description"]}'
            self.__add_document(movie["id"], text)
            self.docmap[movie["id"]] = movie
    
    def save(self) -> None:
        Path(ROOT / "cache").mkdir(exist_ok=True)
        index_f = ROOT / "cache" / "index.pkl"
        docmap_f = ROOT / "cache" / "docmap.pkl"
        with open(index_f, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_f, "wb") as f:
            pickle.dump(self.docmap, f)
    
    def load(self) -> None:
        index_f = ROOT / "cache" / "index.pkl"
        docmap_f = ROOT / "cache" / "docmap.pkl"
        
        if not index_f.is_file() or not docmap_f.is_file():
            raise FileExistsError("Cache files do not exist")
        
        with open(ROOT / "cache" / "index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open(ROOT / "cache" / "docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
