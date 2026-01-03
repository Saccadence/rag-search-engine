import string
from nltk.stem import PorterStemmer
from .search_utils import ROOT, STOPWORDS_F


def text_process(text) -> list:
    clean = text
    # Clean up string
    clean = clean.lower()
    clean = remove_punctuation(clean)
    
    #Tokenize
    clean = clean.split()
    
    # Remove stop words and stems
    clean = remove_stop_words(clean)
    clean = remove_stems(clean)
    
    return clean


def remove_punctuation(text) -> str:
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_stop_words(tokens) -> list:
    stopwords_path = STOPWORDS_F
    with open(stopwords_path, "r") as f:
        stopwords = {line.strip() for line in f}
    return [token for token in tokens if token not in stopwords]
                

def remove_stems(tokens) -> list:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
