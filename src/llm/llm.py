"""
LLM 관련 파일
작성: 염경훈
날짜: 2023-09-21
"""
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.language_model import BaseLanguageModel
from src.config.settings import Settings

class Llm:

    settings = Settings()
    
    # ChatOpenAI 모델을 생성하는 함수
    def get_llm(
        self,
        company: str,
        model_name: str
    ) -> BaseLanguageModel:
        """
        input: 
            company: str AI회사명
            model_name: str 모델 회사명
            
        return: 
            model: BaseLanguageModel
        """
        
        model: BaseLanguageModel

        if company == "openai":
            model = ChatOpenAI(
                model = model_name,                    #모델
                openai_api_key = self.settings.OPENAI_API_KEY,   #API_KEY
                temperature=0.8 ,                           #답변에 대한 랜덤성
                streaming=True,                             #전체 응답을 기다리지 않고 응답이 가능한 것을 바로 처리함
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            
        return model

    def get_embeddings(
        self,
        company: str
    ) -> Embeddings:
        
        """
        input: 
            company: str AI회사명
            
        return: 
            embeddings: Embeddings
        """
        embeddings: Embeddings
        
        if company == "openai":
            embeddings = OpenAIEmbeddings(openai_api_key=self.settings.OPENAI_API_KEY)
            
        return embeddings

