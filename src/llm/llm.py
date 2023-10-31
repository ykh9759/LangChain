"""
LLM 관련 파일
작성: 염경훈
날짜: 2023-09-21
"""
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.config.settings import Settings

class Llm:

    settings = Settings()
    
    # ChatOpenAI 모델을 생성하는 함수
    def get_openai(self) -> ChatOpenAI:

        model = ChatOpenAI(
            model = "gpt-3.5-turbo",                    #모델
            openai_api_key = self.settings.OPENAI_API_KEY,   #API_KEY
            temperature=0.7 ,                           #답변에 대한 랜덤성
            streaming=True,                             #전체 응답을 기다리지 않고 응답이 가능한 것을 바로 처리함
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        return model
    
    def get_openai_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(openai_api_key=self.settings.OPENAI_API_KEY)

