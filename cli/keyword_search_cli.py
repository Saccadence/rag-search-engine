#!/usr/bin/env python3

import argparse
from lib.keyword_search import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            matches = load_movies(args.query)
            
            if not matches:
                print("No results found!")
            else:
                for i, match in enumerate(matches, start=1):
                    print(f'{i}. {match["title"]}')
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()