from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
import os
def get_chat_model():
    return ChatOpenAI(model_name=os.getenv('LLM_MODEL'),
                    temperature=float(os.getenv('TEMPERATURE')),
                    max_tokens = int(os.getenv('MAX_TOKENS')))

# from sentence_transformers import SentenceTransformer
# def get_embedding_model():
#     return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# --- 导入正确的 LangChain 包装器 ---
# from sentence_transformers import SentenceTransformer # 不再需要直接导入这个
from langchain_huggingface import HuggingFaceEmbeddings

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


from openai import OpenAI
def get_vision_model():
    return OpenAI()

if __name__ == "__main__":
    print(type(os.getenv("TEMPERATURE"))) # <class 'str'>

