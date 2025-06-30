from langchain import hub

agent_prompt = hub.pull("hwchase17/openai-tools-agent")
print(agent_prompt.messages)
print(len(agent_prompt.messages))