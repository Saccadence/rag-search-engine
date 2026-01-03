#!/usr/bin/env python3

import argparse
from lib.text_processing import text_process
from lib.keyword_search import *


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
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Custom limit for number of results")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    
    match args.command:
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        
        case "tf":
            print(f"Term Frequency of '{args.term}' in doc '{args.doc_id}': {tf_command(args.doc_id, args.term)}")
        
        case "idf":
            print(f"Inverse Document Frequency of '{args.term}': {idf_command(args.term):.2f}")
            
        case "tfidf":
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf_command(args.doc_id, args.term):.2f}")
        
        case "bm25idf":
            print(f"BM25 IDF score of '{args.term}': {bm25_idf_command(args.term):.2f}")
            
        case "bm25tf":
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf_command(args.doc_id, args.term, args.k1):.2f}")
        
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
                    
        case "bm25search":
            print(f"Searching for: {args.query}")
            
            docs = InvertedIndex()
            try:
                docs.load()
            except Exception as e:
                print(e)
                return
            
            matches = docs.bm25_search(args.query, args.limit)
            if not matches:
                print("No results found!")
            else:
                for i, match in enumerate(matches, start=1):
                    movie = match[0]
                    print(f'{i}. ({movie["id"]}) {movie["title"]} - Score: {match[1]:.2f}')
            
        case _:
            parser.print_help()

    return

if __name__ == "__main__":
    main()