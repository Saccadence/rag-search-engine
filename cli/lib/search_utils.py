from pathlib import Path

# File Locations
HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CACHE = ROOT / "cache"

# Cache files
CHUNK_EMBEDS_F = ROOT / "cache" / "chunk_embeddings.npy"
CHUNK_MD_F = ROOT / "cache" / "chunk_metadata.json"
EMBEDS_F = ROOT / "cache" / "movie_embeddings.npy"
INDEX_F = ROOT / "cache" / "index.pkl"
DOCMAP_F = ROOT / "cache" / "docmap.pkl"
TERM_FREQ_F = ROOT / "cache" / "term_freq.pkl"
DOC_LEN_F = ROOT / "cache" / "doc_length.pkl"

# Data files
MOVIES_JSON_F = ROOT / "data" / "movies.json"
STOPWORDS_F = ROOT / "data" / "stopwords.txt"
GOLD_DATASET = ROOT / "data" / "golden_dataset.json"

# BM25 Constants
BM25_K1 = 1.5
BM25_B = 0.75

# Score vars
SCORE_PRECISION = 2