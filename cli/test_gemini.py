import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    print(f"Using key {api_key[:6]}...")
else:
    print("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(model="gemini-2.0-flash-001", contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")
if response.usage_metadata:
    print(f'{response.text}')
    print(f'Prompt Tokens: {response.usage_metadata.prompt_token_count}')
    print(f'Response Tokens: {response.usage_metadata.candidates_token_count}')
else:
    print(f'Response generation failed!')
