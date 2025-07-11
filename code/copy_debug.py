import copy
from agent import Agent # 确保你的 Agent 类能被导入

print("正在创建一个 Agent 实例...")
try:
    # 尝试创建 Agent 实例
    agent_instance = Agent()
    print("Agent 实例创建成功。")

    print("\n正在尝试深拷贝 Agent 实例...")
    # 尝试深拷贝它
    copied_agent = copy.deepcopy(agent_instance)

    print("\n成功！你的 Agent 类可以被深拷贝。")

except Exception as e:
    # 如果失败，打印错误信息
    print("\n失败！你的 Agent 类不可被深拷贝。")
    print("================ 错误信息 ================")
    print(e)
    print("==========================================")