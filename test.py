from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os

OPENAI_API_KEY = "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" 

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=1.0, model='gpt-3.5-turbo')

tools = load_tools(["llm-math"],llm=llm)

agent = initialize_agent(
        tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
)

def response(message, history, additional_input_info):
        history_langchain_format = []
        # additional_input_info로 받은 시스템 프롬프트를 랭체인에게 전달할 메시지에 포함시킨다.
        history_langchain_format.append(SystemMessage(content= additional_input_info))
        print(history)
        for human, ai in history:
                history_langchain_format.append(HumanMessage(content=human))
                history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = llm(history_langchain_format)
        print(gpt_response)
        return gpt_response.content

gr.ChatInterface(
        fn=response,
        textbox=gr.Textbox(placeholder="말걸어주세요..", container=False, scale=7),
        # 채팅창의 크기를 조절한다.
        chatbot=gr.Chatbot(height=1000),
        title="어떤 챗봇을 원하심미까?",
        description="물어보면 답하는 챗봇임미다.",
        theme="soft",
        examples=[["안뇽"], ["요즘 덥다 ㅠㅠ"], ["점심메뉴 추천바람, 짜장 짬뽕 택 1"]],
        retry_btn="다시보내기 ↩",
        undo_btn="이전챗 삭제 ❌",
        clear_btn="전챗 삭제 💫",
        additional_inputs=[
            gr.Textbox("", label="System Prompt를 입력해주세요", placeholder="I'm lovely chatbot.")
        ]
).launch()