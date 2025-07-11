import os
import time

import gradio as gr

# Gradio 的 gr.Chatbot 组件在后台存储其聊天记录的数据格式是一个列表 (List)，其中每个元素都是一个包含两条字符串的元组 (Tuple)。
# 它的结构是：List[Tuple[str, str]]
# 最外层是列表 []：代表整个聊天历史。列表中的每个元素是元组 ()：代表一轮完整的对话（一次提问和一次回答）。元组中的第一个字符串：是用户的输入。元组中的第二个字符串：是机器人的回复。
# 一个具体的例子：如果进行了两轮对话，chatbot 组件内部存储的数据看起来会是这样：
# [
#     ("你好", "hi 你好"),
#     ("你叫什么名字？", "hi 你叫什么名字？")
# ]
# 这个数据结构在前端和后端之间流动：
# 从前端到后端 (作为 inputs): 当你把 chatbot 放入事件监听器的 inputs 列表时（如 inputs=[textbox, chatbot]），Gradio 会将上述的整个列表作为参数，传递给你在 fn 中指定的后端函数。
# 从后端到前端 (作为 outputs): 你的后端函数（比如 respond）在处理完逻辑后，必须返回一个符合同样格式的、更新后的完整列表。Gradio 接收到这个新列表后，会用它来完全覆盖并重新渲染前端的聊天机器人界面。

# 定义一个函数，用于同时处理响应和清空输入框，简化代码
def stream_respond(message, chat_history):
    """
    一个函数完成所有工作：
    1. 接收消息和历史记录。
    2. 更新历史记录。
    3. 以正确的格式 (history, "") yield 中间结果，供 Gradio 更新 UI。
    """
    # 在函数开头打印接收到的 chat_history
    print("----------- 函数开始 -----------")
    print(f"收到的消息 (message): {message}")
    print(f"收到的历史 (chat_history): {chat_history}")
    print(f"历史的数据类型: {type(chat_history)}")

    # --- 第一步：立即上屏用户的消息，并清空输入框 ---
    # 将用户的消息和空的机器人占位符追加到历史中
    chat_history.append((message, ""))
    # 立即 yield 一次，Gradio 会用这个元组更新 UI
    # chat_history 更新 chatbot, "" 更新 textbox
    yield chat_history, "正在生成答案..."

    # 构造机器人的回复
    bot_message = "hi " + message
    
    # # 将用户的消息和机器人的回复作为一个元组添加到聊天历史中
    # chat_history.append((message, bot_message))
    
    # # 打印即将返回的 chat_history
    # print(f"即将返回的新历史: {chat_history}")

    # --- 第二步：逐字流式生成机器人的回复 ---
    for char in bot_message:
        # 修改历史记录中的最后一个元素（也就是机器人的回复部分）
        chat_history[-1] = (chat_history[-1][0], chat_history[-1][1] + char)
        time.sleep(0.05)
        # 每新增一个字，就 yield 一次，Gradio 会平滑更新 UI
        # ""再次传递，确保输入框保持清空状态
        yield chat_history, ""
    print("----------- 函数结束 -----------")

    # 返回更新后的聊天记录
    return chat_history

# 辅助函数：用于清空输入框和聊天记录
def clear_all():
    return None, ""

# --- Gradio 界面代码 ---

# 预先定义好头像文件的路径
# 这样即使文件不存在，程序也不会在启动时报错，只是不显示头像
user_avatar_path = './images/user_avatar.png' if os.path.exists('./images/user_avatar.png') else None
bot_avatar_path = './images/bot_avatar.png' if os.path.exists('./images/bot_avatar.png') else None

# .tabs 是 Gradio 中代表每个标签页内容区域的 CSS 类
# 我们让它变成一个垂直的 flex 容器，并让它占据所有剩余空间
custom_css = """
.gradio-container { max-width: 95% !important; margin: 0 auto !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), fill_height=True, css=custom_css) as demo:
    with gr.Tab(label="Cell Design AI Assistant", scale=1):
        # with gr.Column(scale=1): # 使用 gr.Column 和 scale 来控制垂直布局
        #     with gr.Column(scale=8):
                gr.Markdown(
                    """
                    # 🧬 细胞设计智能助手
                    欢迎使用本助手。我集成了本地文献知识库、谷歌搜索和通用对话能力。请在下方输入您的问题。
                    **声明:** 本助手回答仅供研究参考，不构成任何形式的专业建议。
                    """
                )
                
                # 聊天机器人主体界面
                chatbot = gr.Chatbot(
                    label="Chat",
                    # bubble_full_width=None,
                    height=500,
                    render_markdown=True,
                    avatar_images=(user_avatar_path, bot_avatar_path)
                )
            
            # with gr.Column(scale=2):
                # 输入框和控制按钮
                with gr.Row(equal_height=True):
                    textbox = gr.Textbox(lines=2, max_lines=5, placeholder="请在此输入您的问题...", container=False, scale=7)
                                        # label info
                    submit_btn = gr.Button("🚀 发送", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ 清空记录", variant="stop", scale=1)

                # 示例问题
                gr.Examples(
                    examples=['你好，你叫什么名字？', '介绍一下COBRApy', '什么是FSEOF？', 'What is iGEM?'],
                    inputs=textbox,
                    label="示例问题 (点击即可填充)"
                )
        
    with gr.Tab(label="Build GEM"):
        gr.Markdown("## 这里是用于构建基因组尺度模型的功能区")
        gr.Textbox(label="输入物种名称")

    # 4. 事件监听器
    # 将 Textbox 的提交事件和 Button 的点击事件绑定到同一个响应函数

    submit_action = textbox.submit(
        fn=stream_respond, 
        # 修改 inputs，需要将 chatbot（历史记录）也传给函数
        inputs=[textbox, chatbot], 
        # 修改 outputs，函数会同时更新 chatbot 和 textbox
        outputs=[chatbot, textbox],
        queue=True # 使用队列以处理并发请求
    )

    submit_btn.click(
        fn=stream_respond, 
        inputs=[textbox, chatbot], 
        outputs=[chatbot, textbox],
        queue=True
    )

    # 清空按钮的逻辑
    clear_btn.click(
        fn=lambda: ([], ""), 
        inputs=None, 
        outputs=[chatbot, textbox], 
        queue=False
    )

# 5. 启动UI
if __name__ == '__main__':
    demo.launch()
