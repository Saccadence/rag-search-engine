#!/usr/bin/env python3

from lib.semantic_search import *
from lib.chunked_semantic_search import *
from lib.search_utils import ROOT, MOVIES_JSON_F
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
    
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed all semantic chunks in a document")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk large text into a more effective size")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Size of chunks in words")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="How much to overlap chunks (in words)")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text based on sentences")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Size of chunks in sentences (max)")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="How much to overlap chunks (in sentences)")
    
    search_parser = subparsers.add_parser("search", help="Search documents with a query and return formatted results")
    search_parser.add_argument("query", type=str, help="Query to search against documents")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit for the number of results (Default = 5)")
    
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search chunked documents against a query")
    search_chunked_parser.add_argument("query", type=str, help="Query to search against documents")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to display (max)")
    
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
        
        case "embed_chunks":
            embed_chunks()
            
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
            
        case "search":
            search = SemanticSearch()
            with open(MOVIES_JSON_F, "r") as f:
                documents = json.load(f)["movies"]
            search.load_or_create_embeddings(documents)
            
            results = search.search(args.query, args.limit)
            for i in range(len(results)):
                result = results[i]
                print(f'{i}. {result["title"]} ({result["score"]})')
                print(f'   {result["description"]}')
            
        case "search_chunked":
            search_chunked(args.query, args.limit)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()