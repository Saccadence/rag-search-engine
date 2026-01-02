from pathlib import Path
from .text_processing import text_process
from .config import BM25_K1
import json
import math
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
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = text_process(text)
        for token in tokens:
            if token != '':
                self.index.setdefault(token, set()).add(doc_id)
        
        if not doc_id in self.term_freq.keys():
            self.term_freq[doc_id] = collections.Counter()
        self.term_freq[doc_id].update(tokens)
    
    def get_documents(self, term: str|list) -> list:
        if term not in self.index.keys():
            return []
        return sorted(self.index[term])

    def get_tf(self, doc_id: int, term: str|list) -> int:
        tokens = text_process(term)
        if len(tokens) != 1:
            raise ValueError("Term does not convert to a single token")
        token = tokens[0]

        doc_tf = self.term_freq.get(doc_id)
        if doc_tf is None:
            return 0
        return doc_tf.get(token, 0)
    
    def get_idf(self, term: str|list) -> float:
        tokens = text_process(term)
        if len(tokens) != 1:
            raise ValueError("Term does not convert to a single token")
        token = tokens[0]

        total_docs = len(self.docmap)
        docs_wt = sum(
            1
            for doc_id in self.docmap.keys()
            if self.get_tf(doc_id, token) != 0
        )
        return math.log((total_docs + 1) / (docs_wt + 1))
    
    def get_tf_idf(self, doc_id: int, query: str|list) -> float:
        tokens = text_process(query)
        score = 0.0
        for token in tokens:
            if token != '':
                score += self.get_tf(doc_id, token) * self.get_idf(token)
        return score
    
    def get_bm25_idf(self, term: str|list) -> float:
        token = text_process(term)
        if len(token) != 1:
            raise ValueError("Term does not convert to a single token")
        token = token[0]
        
        total_docs = len(self.docmap)
        docs_wt = sum(
            1
            for doc_id in self.docmap.keys()
            if self.get_tf(doc_id, token) != 0
        )
        return math.log((total_docs - docs_wt + 0.5) / (docs_wt + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str|list, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)
    
    
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


def tf_command(doc_id: int, term: str|list) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, term)

def idf_command(term: str|list) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(term)

def tf_idf_command(doc_id: int, term: str|list) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str|list) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str|list, k1=BM25_K1) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1)
