import argparse
from lib.search_utils import MOVIES_JSON_F
from lib.hybrid_search import _get_gemini_client, rrf_search
import json


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser("summarize", help="Summarize the results with an LLM")
    summarize_parser.add_argument("query", type=str, help="Search query for summary")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit how many results to include")

    citations_parser = subparsers.add_parser("citations", help="Generate an LLM summary with citations")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit how many results to include")

    question_parser = subparsers.add_parser("question", help="Ask a question about something in the dataset")
    question_parser.add_argument("question", type=str, help="Question you wish to ask")
    question_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Limit the number of results")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag = _get_rag_results(args.command, args.query)
            results = rag[0]
            response = rag[1]

            print("Search Results")
            for result in results:
                print(f"  - {result['doc']['title']}")
            print("\nRAG Response:")
            print(f"{response.text if response is not None else 'No prompt provided!'}")
        
        case "summarize":
            rag = _get_rag_results(args.command, args.query, args.limit)
            results = rag[0]
            response = rag[1]

            print("Search Results")
            for result in results:
                print(f"  - {result['doc']['title']}")
            print("\nLLM Response:")
            print(f"{response.text if response is not None else 'No prompt provided!'}")
            
        case "citations":
            rag = _get_rag_results(args.command, args.query, args.limit)
            results = rag[0]
            response = rag[1]

            print("Search Results")
            for result in results:
                print(f"  - {result['doc']['title']}")
            print("\nLLM Answer:")
            print(f"{response.text if response is not None else 'No prompt provided!'}")
        
        case "question":
            rag = _get_rag_results(args.command, args.question, args.limit)
            results = rag[0]
            response = rag[1]

            print("Search Results")
            for result in results:
                print(f"  - {result['doc']['title']}")
            print("\nAnswer:")
            print(f"{response.text if response is not None else 'No prompt provided!'}")
            
        case _:
            parser.print_help()


def _format_documents_for_citations(results):
    """Format search results with numbered citations for LLM"""
    formatted = []
    for i, result in enumerate(results, start=1):
        doc = result['doc']
        formatted.append(f"[{i}] {doc['title']}\n{doc['description']}")
    return "\n\n".join(formatted)

def _get_rag_results(command, query, limit=5) -> tuple:
    results = rrf_search(query, limit=limit, cli=False)
    
    prompt = None
    match command:
        case "rag":
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                        Query: {query}

                        Documents:
                        {results}

                        Provide a comprehensive answer that addresses the query:"""
        case "summarize":
            prompt = f"""
                        Provide information useful to this query by synthesizing information from multiple search results in detail.
                        The goal is to provide comprehensive information so that users know what their options are.
                        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                        Query: {query}
                        Search Results:
                        {results}
                        Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
                        """
        case "citations":
            formatted_docs = _format_documents_for_citations(results)
            prompt = f"""Answer the question or provide information based on the provided documents.

                        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                        Query: {query}

                        Documents:
                        {formatted_docs}

                        Instructions:
                        - Provide a comprehensive answer that addresses the query
                        - Cite sources using [1], [2], etc. format when referencing information from specific documents
                        - If sources disagree, mention the different viewpoints
                        - If the answer isn't in the documents, say "I don't have enough information"
                        - Be direct and informative

                        Answer:"""
        case "question":
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

                            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                            Question: {query}

                            Documents:
                            {results}

                            Instructions:
                            - Answer questions directly and concisely
                            - Be casual and conversational
                            - Don't be cringe or hype-y
                            - Talk like a normal person would in a chat conversation

                            Answer:"""
        case _:
            pass
    
    if prompt is not None:
        client = _get_gemini_client()
        if client:
            response = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
        else:
            raise ValueError("Failed to load Gemini client")
    else:
        return (results, None)
    return (results, response)

if __name__ == "__main__":
    main()