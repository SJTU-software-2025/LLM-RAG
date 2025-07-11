from pathlib import Path
import sys
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

from util import get_embedding_model, get_chat_model
from prompt import RETRIEVAL_PROMPT_TPL

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
input_dir = project_root / 'db' 

vdb = Chroma(persist_directory=str(input_dir), embedding_function=get_embedding_model())
prompt = PromptTemplate.from_template(RETRIEVAL_PROMPT_TPL)
retrieval_chain = LLMChain(
    llm = get_chat_model(),
    prompt = prompt,
    verbose = bool(os.getenv('VERBOSE'))
)

def retrieval(query):
    documents = vdb.similarity_search_with_score(query, k=3)
    query_result = [doc[0].page_content for doc in documents]
    inputs = {
        'query': query,
        'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
    }
    return retrieval_chain.invoke(inputs)['text']

if __name__ == "__main__":
    # query = 'What is the first step in FBA?'
    # query = 'What is GEM?'
    # query = "In FSEOF, the targets were selected by identifying fluxes that increased upon the application of the enforced objective flux without changing the reaction's direction. How is this mathematically formulated?"
    # query = "What is ecGEM?"
    query = "What is etcGEM?" # 不知道
    print(retrieval(query))


# 检索结果：upon the application of the enforced objective flux without changing the reaction's direction. This is mathematically formulated as follows.
# Fig. S1 in the supplemental material for more information on the flux variability analysis). Eleven fluxes were found to have an increasing pattern without variability (Fig. 2B), the same as FSEOF predicted. Ten fluxes were also found to have an increasing pattern but within a narrow increasing range in which the minimum of the upper line was lower than the maximum of the bottom line (Fig. 2C). This suggests that these 10 fluxes should increase with the enforced objective flux, which is consistent with the results of FSEOF. Thus, these 21 fluxes (Fig. 2B and C) showing increasing trends are consistent with the results from FSEOF simulation. Of the remaining 14 fluxes, 5 showed an increasing pattern within a broad increasing range (Fig. 2D), 4 showed almost no change and/or a
# of $\begin{array} { r c l } { | \nu _ { j } | ^ { \mathrm { m a x } } } & { > } & { | \nu _ { j } ^ { \mathrm { i n i t i a l } } | } \end{array}$ and $\nu _ { j } ^ { \mathrm { m a x } } ~ \times ~ \nu _ { j } ^ { \mathrm { m i n } } ~ \geq ~ 0$ were implemented during FSEOF to select fluxes showing at least one period of the increasing pattern without changing the reaction's direction. The newly predicted flux values using FSEOF should be higher than the initial flux values, $\nu _ { j } .$ For predicted gene targets where the flux direction changed, they were not considered overexpression targets. This resulted in 35 out of 983 reactions being identified as initial gene amplification targets for increasing lycopene production (Fig. 2A and Table 3; also see Table S4A in the supplemental

# 在FSEOF（Flux-Strain Engineering Optimization Framework）中，选择目标是通过识别在施加客观通量（enforced objective flux）时增加且不改变反应方向的 通量来实现的。这种选择过程的数学公式如下：
# 1. 选择至少显示一个增加模式周期的通量，而不改变反应方向。这通过以下条件实现：
#    - $|\nu_j^{\text{max}}| > |\nu_j^{\text{initial}}|$
#    - $\nu_j^{\text{max}} \times \nu_j^{\text{min}} \geq 0$
# 其中，$\nu_j^{\text{max}}$ 表示通量 $j$ 的最大值，$\nu_j^{\text{initial}}$ 表示通量 $j$ 的初始值，$\nu_j^{\text{min}}$ 表示通量 $j$ 的最小值。   
# 2. 使用FSEOF预测的新通量值应该高于初始通量值，即：
#    - $\nu_j > \nu_j^{\text{initial}}$
# 3. 对于预测的基因靶点，如果通量方向发生了变化，则不考虑作为过表达靶点。
# 通过这些数学条件，FSEOF能够从983个反应中识别出35个初始基因扩增靶点，以增加番茄红素的产量。