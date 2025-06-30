# 首先从最简单的开始，用大模型自身能力来回答日常交际的问题
# 这里有一个注意事项，就是利用大模型自身能力回答问题时，要防止模型自由发挥的太过了，所以提示词要相对严格一些。

import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub

from translate import translate_query_to_english
from util import get_chat_model, get_embedding_model, proxied_google_search
from prompt import *


class Agent():
    def __init__(self):
        print("正在初始化Agent...")
        # print(os.path.join(os.path.dirname(__file__), '../db/'))
        self.vdb = Chroma(persist_directory=os.path.join(os.path.dirname(__file__), '../db/'), embedding_function=get_embedding_model())
        # --- 创建工具列表和Agent Executor ---
        self.tools = [
            Tool(
                name = 'generic_tool',
                func = self.generic_tool,
                description = '可以解答通用领域的问题，例如打招呼，问你是谁等问题'
            ),
            Tool(
                name = 'retrival_tool',
                func = lambda x: self.retrival_tool(self.query),
                description = '用于回答细胞设计领域相关问题，知识库是多篇文献，涵盖FBA、Cobrapy、FSEOF等内容',
            ),
            Tool(
                name = 'search_tool',
                func = self.search_tool,
                description = '其他工具都没有正确答案时，通过谷歌搜索引擎，回答通用类问题',
            ),
        ]
        agent_prompt = hub.pull("hwchase17/openai-tools-agent")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent = create_openai_tools_agent(
            llm=get_chat_model(),
            tools=self.tools,
            prompt=agent_prompt
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=bool(os.getenv('VERBOSE')),
            memory=memory
        )

    def generic_tool(self, query):
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_chat_model(),
            prompt = prompt,
            verbose = bool(os.getenv('VERBOSE'))
        )
        return llm_chain.invoke({'query': query})['text']

    def retrival_tool(self, query):
        documents = self.vdb.similarity_search_with_score(query, k=3)
        query_result = [doc[0].page_content for doc in documents]
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        retrival_chain = LLMChain(
            llm = get_chat_model(),
            prompt = prompt,
            verbose = bool(os.getenv('VERBOSE'))
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return retrival_chain.invoke(inputs)['text']
        
    def search_tool(self, query):
        result = proxied_google_search(query)
        prompt = PromptTemplate(
            template=SEARCH_PROMPT_TPL,
            partial_variables={'query_result': result},
            input_variables=['query']
        )
        search_chain = LLMChain(
            llm = get_chat_model(),
            prompt = prompt,
            verbose = bool(os.getenv('VERBOSE'))
        )
        return search_chain.invoke({'query': query})['text']

    def process(self, query):
        """此方法现在只调用已创建的agent_executor。"""
        self.query = translate_query_to_english(query)
        return self.agent_executor.invoke({'input': self.query})['output']

# 调用测试
if __name__ == '__main__':
    agent = Agent()
    # print(agent.generic_tool('你叫什么名字？'))
    # print(agent.retrival_tool("In FSEOF, the targets were selected by identifying fluxes that increased upon the application of the enforced objective flux without changing the reaction's direction. How is this mathematically formulated?"))
    # print(agent.retrival_tool("介绍一下COBRApy")
    # print(agent.search_tool('什么是iGEM？SJTU-Software历年表现如何？'))

    while True:
        human_input = input('问题：')
        result = agent.process(human_input)
        print('答案：', result, '\n')


