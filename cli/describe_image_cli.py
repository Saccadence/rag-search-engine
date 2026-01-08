import argparse
import os
import mimetypes
from lib.hybrid_search import _get_gemini_client
from google.genai import types


def main():
    parser = argparse.ArgumentParser(description="Multimodal Evaluator")
    parser.add_argument("--image", type=str, nargs='?', required=True, help="File path from Current Working Directory to image")
    parser.add_argument("--query", type=str, nargs='?', required=True, help="Query to ask LLM")
    
    args = parser.parse_args()
    file_path = os.path.join(os.getcwd(), args.image)
    
    if file_path:
        mime, _= mimetypes.guess_type(file_path)
        mime = mime or "image/jpeg"
        with open(file_path, "rb") as f:
            img = f.read()
    else:
        raise ValueError("File not found!")
        
    client = _get_gemini_client()
    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                    - Synthesize visual and textual information
                    - Focus on movie-specific details (actors, scenes, style, etc.)
                    - Return only the rewritten query, without any additional commentary"""
    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]
    response = None
    if client:
        response = client.models.generate_content(model="gemini-2.0-flash-001", contents=parts)
    else:
        raise ValueError("Gemini Client not properly loaded!")
    
    if response.text is not None:
        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    else:
        raise ValueError("No response generated!")


if __name__ == "__main__":
    main()