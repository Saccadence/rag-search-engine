#!/usr/bin/env python3

from lib.semantic_search import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_model_parser = subparsers.add_parser("verify_model", help="Verify model information")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings for .json file")
    
    embed_parser = subparsers.add_parser("embed_text", help="Embed text for model to use")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query for model to use")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")
    
    search_parser = subparsers.add_parser("search", help="Search documents with a query and return formatted results")
    search_parser.add_argument("query", type=str, help="Query to search against documents")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit for the number of results (Default = 5)")
    
    args = parser.parse_args()

    match args.command:
        case "verify_model":
            verify_model()
            
        case "verify_embeddings":
            verify_embeddings()
            
        case "embed_text":
            embed_text(args.text)
        
        case "embedquery":
            embed_query_text(args.query)
            
        case "search":
            search = SemanticSearch()
            with open(ROOT / "data" / "movies.json", "r") as f:
                documents = json.load(f)["movies"]
            search.load_or_create_embeddings(documents)
            
            results = search.search(args.query, args.limit)
            for i in range(len(results)):
                result = results[i]
                print(f'{i}. {result["title"]} ({result["score"]})')
                print(f'   {result["description"]}')
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()