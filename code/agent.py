import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory 
from langchain import hub

from translate import translate_query_to_english
from bochaAI import bocha_websearch_tool
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
                name = 'conversational_tool',  # 名称可以更具体
                func = self.generic_tool,
                description = "A tool for simple, conversational interactions like greetings, self-introductions, or identifying its creator. It should be used for casual chat, not for answering knowledge-based questions. For example: 'Hello', 'Who are you?', 'Who made you?'."
            ),
            Tool(
                name = 'cell_design_expert_tool', # 名称更具专家色彩,
                func = lambda x: self.retrieval_tool(self.query),
                description = "The primary tool for answering expert-level questions about a specialized knowledge base on cell design and metabolic engineering. Use this for any query related to specific scientific concepts, methods, or software such as GEM (genome-scale metabolic model), ecGEM (enzyme constrained genome-scale metabolic model), FBA (Flux Balance Analysis), dFBA (Dynamic Flux Balance Analysis), COBRApy, OptKnock, FSEOF, metabolic pathways, genetic circuits, and synthetic biology. This is the go-to tool for deep academic and technical inquiries.",
            ),
            Tool(
                name = 'web_search_tool', # 名称更清晰
                func = self.search_tool,
                description = "A fallback tool that uses a web search engine to answer general knowledge questions that are outside the scope of cell design literature. Use this for topics like recent events, public figures, definitions of new or broad terms (e.g., 'What is iGEM?'), or questions about competition results (e.g., 'Who won the iGEM gold medal in a specific year?'). Use only when other tools are not suitable.",
            ),
        ]
        agent_prompt = hub.pull("hwchase17/openai-tools-agent")
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4, return_messages=True)

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

    def retrieval_tool(self, query):
        documents = self.vdb.similarity_search_with_score(query, k=3)
        query_result = [doc[0].page_content for doc in documents]
        prompt = PromptTemplate.from_template(RETRIEVAL_PROMPT_TPL)
        retrieval_chain = LLMChain(
            llm = get_chat_model(),
            prompt = prompt,
            verbose = bool(os.getenv('VERBOSE'))
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return retrieval_chain.invoke(inputs)['text']
        
    def search_tool(self, query):
        # result = proxied_google_search(query)
        result = bocha_websearch_tool(query)
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
        # 打印对话历史
        print("当前对话历史:", self.agent_executor.memory.load_memory_variables({}))
        return self.agent_executor.invoke({'input': self.query})['output']

# 调用测试
if __name__ == '__main__':
    agent = Agent()
    # print(agent.generic_tool('你叫什么名字？'))
    # print(agent.retrieval_tool("In FSEOF, the targets were selected by identifying fluxes that increased upon the application of the enforced objective flux without changing the reaction's direction. How is this mathematically formulated?"))
    # print(agent.retrieval_tool("介绍一下COBRApy")
    print(agent.search_tool('什么是iGEM？SJTU-Software历年表现如何？'))

    while True:
        human_input = input('问题：')
        result = agent.process(human_input)
        print('答案：', result, '\n')


