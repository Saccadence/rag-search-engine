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
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(doc_chunks)
                })
        
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_md

        np.save(CHUNK_EMBEDS_F, self.chunk_embeddings)
        with open(CHUNK_MD_F, "w") as f:
            json.dump({"chunks": chunks_md, "total_chunks": len(chunks)}, f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if CHUNK_EMBEDS_F.is_file() and CHUNK_MD_F.is_file():
            self.chunk_embeddings = np.load(CHUNK_EMBEDS_F)
            with open(CHUNK_MD_F, "r") as f:
                self.chunk_metadata = json.load(f)
            
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)


def embed_chunks():
    with open(MOVIES_JSON_F, "r") as f:
        documents = json.load(f)["movies"]
    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")