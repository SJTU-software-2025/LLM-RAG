from dotenv import load_dotenv
load_dotenv()

# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

# from google import genai
# client = genai.Client()
# result = client.models.embed_content(
#         model="gemini-embedding-exp-03-07",
#         contents="How does alphafold work?",
# )
# print(result.embeddings)

# from langchain.schema import Document
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma

# # 文档示例
# documents = [
#     "AlphaFold is an AI system developed by DeepMind that predicts a protein's 3D structure from its amino acid sequence.",
#     "The model uses a novel neural network architecture to interpret biological and genetic data.",
#     "It achieved revolutionary accuracy in the CASP14 protein structure prediction challenge."
# ]
# documents = [Document(page_content=d) for d in documents]

# # --- 关键步骤 ---
# # 1. 创建一个 GoogleGenerativeAIEmbeddings "模型对象"
# # 这就是你需要的那个可以被复用的嵌入器！
# # 我们在这里指定为文档生成向量，所以使用 'RETRIEVAL_DOCUMENT'
# google_embeddings = GoogleGenerativeAIEmbeddings(
#     # model = "gemini-embedding-exp-03-07",
#     model="models/embedding-001", # 这是推荐的稳定版模型
#     task_type="RETRIEVAL_DOCUMENT",
#     google_api_key=os.getenv("GEMINI_API_KEY")
#     # 如果需要，还可以添加 title 参数
#     # title="About AlphaFold Technology" 
# )

# # 2. 将这个嵌入器对象和你的文档一起传给 Chroma
# # Chroma 会在内部调用 google_embeddings.embed_documents()
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=google_embeddings  # <-- 看，这里用法和 HuggingFaceEmbeddings 完全一样！
# )

# # --- 后续使用 ---
# # 当你需要检索时，创建一个用于查询的嵌入器
# query_embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     task_type="RETRIEVAL_QUERY",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

# # LangChain 的 Retriever 会自动使用正确的嵌入器
# retriever = vectorstore.as_retriever()
# # 如果你想手动测试，需要这样
# query_vector = query_embeddings.embed_query("How does alphafold work?")
# # 然后用这个 vector 去数据库里搜索