import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def expand_query(query):
    headers = {
        'Content-Type': 'Application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers, 
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{
                "role": "system",
                "content": """Your task is to generate five different versions of the given user query
                    to retrieve relevant documents from a vector database. Provide these alternative joined with comma to the original query.
                    Queries will be related to education and dont add extra, unnecessary info.
                    """
            },
            {
                "role": "user",
                "content": query
            }],
            "max_tokens": 300
        }
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return query
    
def get_answer(user_query, top_5_chunks):
    headers = {
        'Content-Type': 'Application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }

    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers=headers, 
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{
                "role": "system",
                "content": f"""You are an AI language model. Based on these chunks, you need to generate a short answer like a rag llm to the user query.
                    The answer should beconcise and clear, and should be based on the information in the document. 
                    Strictly Don't answer if the query is not related to chunks, just say 'i dont have information about it' The chunks are:
                    {top_5_chunks}
                    """
            },
            {
                "role": "user",
                "content": user_query
            }],
            "max_tokens": 300
        }
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return response.text