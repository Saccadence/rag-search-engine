#!/usr/bin/env python3

import argparse
from lib.text_processing import text_process
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build dataset to file for faster searches")
    
    tf_parser = subparsers.add_parser("tf", help="Return the number of times a term appears in a doc")
    tf_parser.add_argument("doc_id", type=int, help="Doc id")
    tf_parser.add_argument("term", type=str, help="Term to return count of")

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
            print(f'In doc [{args.doc_id}], [{args.term}] appears [{term_count}] times.')
            
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