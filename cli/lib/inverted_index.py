from pathlib import Path
from .text_processing import text_process
import json
import pickle
import collections

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
INDEX_F = ROOT / "cache" / "index.pkl"
DOCMAP_F = ROOT / "cache" / "docmap.pkl"
TERM_FREQ_F = ROOT / "cache" / "term_freq.pkl"

class InvertedIndex():
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_freq = {}
    
    def __add_document(self, doc_id, text) -> None:
        tokens = text_process(text)
        for token in tokens:
            if token != '':
                self.index.setdefault(token, set()).add(doc_id)
        
        if not doc_id in self.term_freq.keys():
            self.term_freq[doc_id] = collections.Counter()
        self.term_freq[doc_id].update(tokens)
    
    def get_documents(self, term) -> list:
        if term not in self.index.keys():
            return []
        return sorted(self.index[term])

    def get_tf(self, doc_id, term) -> int:
        token = text_process(term)
        if len(token) > 1:
            raise ValueError("Term does not convert to a single token")
        
        return self.term_freq.get(doc_id, 0)[token[0]]
    
    
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
        with open(INDEX_F, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_F, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQ_F, "wb") as f:
            pickle.dump(self.term_freq, f)
        
    
    def load(self) -> None:
        if not INDEX_F.is_file() or not DOCMAP_F.is_file():
            raise FileExistsError("Cache files do not exist")
        
        with open(INDEX_F, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_F, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQ_F, "rb") as f:
            self.term_freq = pickle.load(f)
        
