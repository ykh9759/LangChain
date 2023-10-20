import gradio as gr

from langchain.schema import AIMessage, HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from llm.tools import Tools
from llm.llm import Llm

llm = Llm().get_openai()

system_template="""
        You are a chatbot that answers questions in Korean.

        Refer to search contents when answering

        {search}
"""

def response(message, history):

        history_langchain_format = []
        history_langchain_format.append(SystemMessagePromptTemplate.from_template(system_template))
        for human, ai in history:
                history_langchain_format.append(HumanMessage(content=human))
                history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessagePromptTemplate.from_template("{input}"))

        prompt = ChatPromptTemplate.from_messages(history_langchain_format)

        llm_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=True
        )

        search = Tools(llm).get_search(message)

        answer = llm_chain.run(input=message, search=search)
        return answer

gr.ChatInterface(
        fn=response,
        textbox=gr.Textbox(placeholder="질문하기...", container=True, scale=7),
        # 채팅창의 크기를 조절한다.
        chatbot=gr.Chatbot(height=1000),
        title="챗봇 테스트",
        description="물어보면 답하는 챗봇입니다.",
        theme="soft",
        examples=[["안녕"], ["오늘의 날씨"], ["점심메뉴 추천"]],
        retry_btn="다시보내기 ↩",
        undo_btn="이전 삭제 ❌",
        clear_btn="전체 삭제 💫",
).launch()