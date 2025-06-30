from dotenv import load_dotenv
from contextlib import contextmanager
import os

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper

load_dotenv()

def get_chat_model():
    return ChatOpenAI(model_name=os.getenv('LLM_MODEL'),
                    temperature=float(os.getenv('TEMPERATURE')),
                    max_tokens = int(os.getenv('MAX_TOKENS')))

# from sentence_transformers import SentenceTransformer
# def get_embedding_model():
#     return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# --- 导入正确的 LangChain 包装器 ---
# from sentence_transformers import SentenceTransformer # 不再需要直接导入这个

def get_embedding_model():
    """
    获取一个符合 LangChain 接口标准的 Embedding 模型实例。
    """
    # 指定模型的名称，这个名称来自于 Hugging Face Hub
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # 定义模型加载时的一些参数，比如指定在 CPU 上运行
    model_kwargs = {'device': 'cpu'}
    
    # 定义编码时的一些参数，比如是否要归一化向量
    # 对于 all-MiniLM-L6-v2，归一化是推荐的
    encode_kwargs = {'normalize_embeddings': True}
    
    # 实例化 LangChain 的包装器类
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embedding_model

def get_vision_model():
    return OpenAI()

@contextmanager
def temporary_proxy(proxy_url: str):
    """一个上下文管理器，用于在 with 块内临时设置代理。"""
    original_http = os.environ.get("HTTP_PROXY")
    original_https = os.environ.get("HTTPS_PROXY")
    
    print(f"临时设置代理: {proxy_url}")
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    
    try:
        yield  # with 块内部的代码将在这里执行
    finally:
        print("操作完成，取消临时代理。")
        # 恢复 HTTP_PROXY
        if original_http:
            os.environ["HTTP_PROXY"] = original_http
        elif "HTTP_PROXY" in os.environ:
            del os.environ["HTTP_PROXY"]
            
        # 恢复 HTTPS_PROXY
        if original_https:
            os.environ["HTTPS_PROXY"] = original_https
        elif "HTTPS_PROXY" in os.environ:
            del os.environ["HTTPS_PROXY"]

def proxied_google_search(query: str) -> str:
    """
    一个包装函数，在调用 Google 搜索前后临时设置和取消代理。
    （使用上下文管理器实现）
    """
    search = GoogleSearchAPIWrapper()
    proxy_url = "http://127.0.0.1:10808"

    with temporary_proxy(proxy_url):
        # 在这个 with 块内，代理是设置好的
        result = search.run(query)
    
    # 离开 with 块后，代理会自动恢复
    return result

if __name__ == "__main__":
    print(type(os.getenv("TEMPERATURE"))) # <class 'str'>

