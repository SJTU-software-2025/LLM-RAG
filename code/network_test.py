# from openai import OpenAI

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="sk-2U5HLSi3ad7VMGyMjEpIHvRkQvdNBKwVgZKsFyafxu7ig9lk",
#     base_url="https://api.chatanywhere.tech/v1"
#     # base_url="https://api.chatanywhere.org/v1"
# )

# import gradio as gr
# import random

# def greet(name, age):
#     return f"Hello {name} {age}!"

# demo = gr.Interface(
#     fn=greet,
#     inputs="text",
#     outputs="textbox",
#     title="This is a tile",
#     css="footer {visibility: hidden}"
# )

# demo.launch(share=True)

# import os

# os.environ["GOOGLE_CSE_ID"] = "6519ceb96db3c44c9"
# os.environ["GOOGLE_API_KEY"] = "AIzaSyB9oSyCqijbY3hhkLIK33pS2-nsftIv-IU"

# from langchain_core.tools import Tool
# from langchain_google_community import GoogleSearchAPIWrapper

# search = GoogleSearchAPIWrapper()

# tool = Tool(
#     name="google_search",
#     description="Search Google for recent results.",
#     func=search.run,
# )

# tool.run("Obama's first name?")

import requests
import json

url = "https://api.bochaai.com/v1/web-search"

payload = json.dumps({
  "query": "What is iGEM?",
  "summary": True,
  "count": 2
})

headers = {
  'Authorization': 'Bearer sk-1c3ed84a53684a7bafb076d00d75d0c0',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.json())