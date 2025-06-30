from pathlib import Path
import sys

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
input_dir = project_root / 'mineru_markdown' / 'revised_output'
markdown_files = list(input_dir.glob('*.md'))

# loader = UnstructuredMarkdownLoader(str(markdown_files[1]))
loader = TextLoader(str(markdown_files[1]), encoding='utf-8')
docs_raw = loader.load()
print(docs_raw)

# # md = '# What is flux balance analysis?\n\nFlux balance analysis is a mathematical approach for analyzing the flow of metabolites through a metabolic network. This primer covers the theoretical basis of the approach, several practical examples and a software toolbox for performing the calculations.\n\nWith metabolic models for 35 organisms already available (http://systemsbiology.ucsd.edu/ In_Silico_Organisms/Other_Organisms) and high-throughput technologies enabling the construction of many more each year5-7, FBA is an important tool for harnessing the knowledge encoded in these models.\n\n# Flux balance analysis is based on constraints\n\nThe first step in FBA is to mathematically represent metabolic reactions (Box 1 and Fig. 1).'
# md = "# Foo\n\n  Hey we go!  ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3")
# ]
# from langchain.text_splitter import MarkdownHeaderTextSplitter
# markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# md_header_splits = markdown_splitter.split_text(md)
# for split in md_header_splits:
#     print(split)
markdown_text = docs_raw[0].page_content

# --- 阶段一：结构化分割 ---
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=True # 推荐设置为 True，让内容更干净
)
md_header_splits = markdown_splitter.split_text(markdown_text)
print(f"阶段一：结构化分割后，得到 {len(md_header_splits)} 个语义块。")
print(md_header_splits)

import os
# --- 阶段二：长度分割 ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  
    chunk_overlap=120
)
all_splits_for_file = text_splitter.split_documents(md_header_splits)
print(f"阶段二：长度分割后，得到 {len(all_splits_for_file)} 个最终片段。")