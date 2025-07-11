import os
from dotenv import load_dotenv

from openai import OpenAI
from google import genai
import chromadb

import chunk

load_dotenv()
client = OpenAI()

chromadb_client = chromadb.PersistentClient("./chroma_db")
chromadb_collection = chromadb_client.get_or_create_collection("linghuchong")

# https://platform.openai.com/docs/api-reference/embeddings/create
def embed(text: str) -> list[str]:
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

def create_db() -> None:
    for idx, c in enumerate(chunk.get_chunks()):
        print(f"Process: {c}")
        embedding = embed(c)
        chromadb_collection.upsert(
            ids=str(idx),
            documents=c,
            embeddings=embedding
        )

def query_db(question):
    question_embedding = embed(question)
    result = chromadb_collection.query(
        query_embeddings=question_embedding,
        n_results=5
    )
    return result['documents'][0]

if __name__ == "__main__":
    # create_db()
    question = '令狐冲领悟了什么魔法？'
    chunks = query_db(question)
    # for c in chunks:
    #     print(c)
    #     print("----------")
    prompt = "Please answer user's question according to context\n"
    prompt += f"Question: {question}\n"
    prompt += "Context:\n"
    for c in chunks:
        prompt += f"{c}\n"
        prompt += "-------------\n"
    
    # completion = client.chat.completions.create(
    #     model=os.getenv("LLM_MODEL"),
    #     messages=[
    #             {"role": "developer", "content": f"{prompt}"},
    #             {"role": "user", "content": "Hello!"}
    #         ]
    #     )
    # print(completion.choices[0].message)
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    print(response.text)