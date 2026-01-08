import os
import json
from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import MOVIES_JSON_F


class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32", docs=[]):
        self.model = SentenceTransformer(model_name, model_kwargs={"use_fast": True})
        self.docs = docs
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        
    def embed_image(self, image_path: str):
        abs_path = os.path.abspath(image_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image file not found: {abs_path}")
        
        img = Image.open(abs_path)
        return self.model.encode(img)  # type: ignore[arg-type]

    def search_with_image(self, image_path):
        img_embed = self.embed_image(image_path)
        results = []
        for i, text_embed in enumerate(self.text_embeddings):
            doc = self.docs[i]
            similarity = cosine_similarity(img_embed, text_embed)
            results.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "similarity": similarity,
                "description": doc["description"]
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path: str):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    with open(MOVIES_JSON_F, "r") as f:
        documents = json.load(f)["movies"]
    search = MultimodalSearch(docs=documents)
    return search.search_with_image(image_path)
