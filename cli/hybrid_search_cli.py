import argparse
from lib.hybrid_search import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalizes a set of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Set of scores to normalize")
    
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform a weighted search on a set")
    weighted_search_parser.add_argument("query", type=str, help="Query to perform the search with")
    weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Configure how to weight the search (0.0 to 1.0)")
    weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Limit how many results to show (max)")
    
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform a Reciprocal Ranked Fusion search")
    rrf_search_parser.add_argument("query", type=str, help="Query to check against documents")
    rrf_search_parser.add_argument("-k", type=int, nargs='?', default=60, help="Adjusts how to weight scores based on rank (Default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit number of results")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores)
            
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
            
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()