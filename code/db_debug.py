# 文件名: debug_db.py
from pathlib import Path
import sys
import json
from langchain_chroma import Chroma
from util import get_embedding_model 

# --- 设置路径 ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
input_dir = project_root / 'db' 

# --- 加载数据库 ---
print("正在加载向量数据库...")
vdb = Chroma(
    persist_directory=str(input_dir), 
    embedding_function=get_embedding_model() # 必须使用相同的 embedding 函数
)
print("数据库加载完毕！")

# --- 从数据库中取出所有数据 ---
# .get() 方法可以获取数据库中的内容
# 它会返回一个包含 'ids', 'embeddings', 'metadatas', 'documents' 的字典
all_data = vdb.get()

# --- 分析数据 ---
documents = all_data['documents']
metadatas = all_data['metadatas']

total_count = len(documents)
print(f"\n数据库中总共包含 {total_count} 个文档块。")

# 检查内容是否重复
# 使用 set 来统计有多少个独立不重复的文档块
unique_documents = set(documents)
unique_count = len(unique_documents)
print(f"其中，不重复的文档块有 {unique_count} 个。")

if unique_count < total_count:
    print("\n警告：数据库中存在大量重复内容！")
else:
    print("\n数据库内容看起来是唯一的，没有重复。")

# --- 将内容写入 JSON 文件以便详细查看 ---
output_data = []
for i in range(total_count):
    output_data.append({
        'document_content': documents[i],
        'metadata': metadatas[i]
    })

debug_json_path = 'project_root' / 'db' / 'db_content_debug.json'
with open(debug_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n数据库的全部内容已写入调试文件: {debug_json_path}")