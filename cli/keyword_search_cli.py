#!/usr/bin/env python3

import argparse
from lib.text_processing import text_process
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build dataset to file for faster searches")
    
    tf_parser = subparsers.add_parser("tf", help="Return the number of times a term appears in a doc")
    tf_parser.add_argument("doc_id", type=int, help="Doc to check against")
    tf_parser.add_argument("term", type=str, help="Term to return count of")
    
    idf_parser = subparsers.add_parser("idf", help="See how common a term is in the dataset")
    idf_parser.add_argument("term", help="Term to check against set")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="See how related a term is to a doc")
    tfidf_parser.add_argument("doc_id", type=int, help="Doc to check against")
    tfidf_parser.add_argument("term", type=str, help="Term to check with")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    
    match args.command:
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        
        case "tf":
            index = InvertedIndex()
            index.load()
            term_count = index.get_tf(args.doc_id, args.term)
            print(f"Term Frequency of '{args.term}' in doc '{args.doc_id}': {term_count}")
        
        case "idf":
            index = InvertedIndex()
            index.load()
            print(f"Inverse Document Frequency of '{args.term}': {index.get_idf(args.term):.2f}")
            
        case "tfidf":
            index = InvertedIndex()
            index.load()
            tf_idf = index.get_tf_idf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            
        case "search":
            print(f"Searching for: {args.query}")
            
            docs = InvertedIndex()
            try:
                docs.load()
            except Exception as e:
                print(e)
                return
            
            matches = set()
            for token in text_process(args.query):
                for match in docs.get_documents(token):
                    if len(matches) == 5:
                        break
                    matches.add(match)
            matches = sorted(matches)
            
            if not matches:
                print("No results found!")
            else:
                for i, match in enumerate(matches, start=1):
                    movie = docs.docmap[match]
                    print(f'{i}. {movie["title"]}')
        case _:
            parser.print_help()

    return

if __name__ == "__main__":
    main()