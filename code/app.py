import os
import time

import gradio as gr

from agent import Agent

def stream_respond(message, chat_history):
    print("----------- 函数开始 -----------")
    print(f"收到的消息 (message): {message}")
    print(f"收到的历史 (chat_history): {chat_history}")
    print(f"历史的数据类型: {type(chat_history)}")

    chat_history.append((message, "Generating answer..."))
    yield chat_history, ""

    # chat_history[-1][1] = "" TypeError: 'tuple' object does not support item assignment
    chat_history[-1] = (message, "")
    bot_message = agent.process(message)
    print(f"bot_message: {bot_message}")
    
    for char in bot_message:
        chat_history[-1] = (chat_history[-1][0], chat_history[-1][1] + char)
        time.sleep(0.02)
        yield chat_history, ""
    print("----------- 函数结束 -----------")

    return chat_history

agent = Agent()

user_avatar_path = './images/user_avatar.png' if os.path.exists('./images/user_avatar.png') else None
bot_avatar_path = './images/bot_avatar.png' if os.path.exists('./images/bot_avatar.png') else None

custom_css = """
.gradio-container { max-width: 95% !important; margin: 0 auto !important; } footer { visibility: hidden; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), fill_height=True, css=custom_css) as demo:
    with gr.Tab(label="Cell Design AI Assistant", scale=1):
            gr.Markdown(
                """
                # 🧬 Cell Design AI Assistant
                Welcome to the assistant. I have integrated a local literature knowledge base, Google search, and general conversational abilities. Please enter your question below.
                **Disclaimer:** The answers provided by this assistant are for research reference only and do not constitute professional advice of any kind.
                """
            )
            
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                render_markdown=True,
                avatar_images=(user_avatar_path, bot_avatar_path)
            )

            with gr.Row(equal_height=True):
                textbox = gr.Textbox(lines=2, max_lines=5, placeholder="Please enter your question here... (Shift+Enter or click the Send button to submit)", container=False, scale=7)
                                    # label info
                submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ Clear Histroy", variant="stop", scale=1)

            gr.Examples(
                examples=[
                    'Who are you?', 
                    'What is GEM?', 
                    'Tell me about COBRApy', 
                    "In FSEOF, the targets were selected by identifying fluxes that increased upon the application of the enforced objective flux without changing the reaction's direction. How is this mathematically formulated?", 
                    'What is iGEM? How has SJTU-Software performed over the years?' 
                    ],
                inputs=textbox,
                label="Example Questions (Click to fill)",
                # examples_per_page=3
            )
        
    with gr.Tab(label="Build GEM"):
        gr.Markdown("## This is the functional area for building Genome-Scale Models")
        gr.Textbox(label="Enter Species Name")

    submit_action = textbox.submit(
        fn=stream_respond, 
        inputs=[textbox, chatbot], 
        outputs=[chatbot, textbox],
        queue=True 
    )

    submit_btn.click(
        fn=stream_respond, 
        inputs=[textbox, chatbot], 
        outputs=[chatbot, textbox],
        queue=True
    )

    clear_btn.click(
        fn=lambda: ([], ""), 
        inputs=None, 
        outputs=[chatbot, textbox], 
        queue=False
    )

if __name__ == '__main__':
    demo.launch(share=True)
