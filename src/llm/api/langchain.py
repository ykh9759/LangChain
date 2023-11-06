"""
랭체인API 라우터 페이지
작성: 염경훈
날짜: 2023-09-20
"""
import time
from typing import Iterable
import tiktoken
import re

from fastapi import Depends, APIRouter, status
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain, RetrievalQA, ChatVectorDBChain, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from src.llm.tools import Tools
from src.llm.llm import Llm
from src.llm.response import chatModelsResponse
from googletrans import Translator
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.storage import (
    InMemoryStore,
    LocalFileStore,
    RedisStore,
    UpstashRedisStore,
)
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings

router = APIRouter(
    prefix="/api/langchain2",
    tags=["langchain"]
)

trans = Translator()  #구글번역

#공통 파라미터
class CommonQueryParams:
    def __init__(self, 
        model:str,                 #모델명
        q: str                    #질문
    ):   
        self.model = model 
        self.q = q                                  

@router.get(
    "/chat-models",                       #라우터경로
    status_code=status.HTTP_200_OK,       #HTTP status
    response_model=chatModelsResponse     #응답모델 지정
) 
async def chatModels2(commons: CommonQueryParams = Depends()):

    llm = getattr(Llm(), f"get_{commons.model}")()          # Llm클래스에서 model명에 맞는 함수 호출                          
    # question = trans.translate(commons.q, dest="en").text   
    question = commons.q                              

    # 랭체인
    ############################################################

    system_template="You are a chatbot that speaks {language}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template), 
        AIMessagePromptTemplate.from_template("{search}"), 
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        verbose=True
    )

    # 응답 데이터 세팅
    ############################################################

    response = {}

    try:

        search = Tools(llm).get_search(question)
        print(search)
        
        chat = chain.run(language="korean", search=search, input=question)

        response['status'] = "success"
        response['data'] = {
            "question": commons.q, 
            "answer": trans.translate(chat, dest="ko").text
        }

    except ValueError as e:
        response['status'] = "fail"

        chat = str(e)
        if chat.startswith("Could not parse LLM output: `"):
            response['message'] = "LLM파싱 실패"
        else:
            response['message'] = "LLM조회 실패"


    ############################################################

    return response

    
@router.get(
    "/web-rag",                       #라우터경로
    status_code=status.HTTP_200_OK      #HTTP status
) 
async def webRag(commons: CommonQueryParams = Depends()):

    llm = getattr(Llm(), f"get_{commons.model}")()
    embeddings = getattr(Llm(), f"get_{commons.model}_embeddings")()
    question = trans.translate(commons.q, dest="en").text
    # question = commons.q    
    search = Tools(llm).get_google_search()

    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm, 
        search=search, 
    )

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)
    
    result = qa_chain({"question": question})

    return result


@router.get(
    "/pdf-rag",                       #라우터경로
    status_code=status.HTTP_200_OK      #HTTP status
) 
async def pdfRag(commons: CommonQueryParams = Depends()):

    start_time = time.time()

    llm = getattr(Llm(), f"get_{commons.model}")()
    embeddings: OpenAIEmbeddings = getattr(Llm(), f"get_{commons.model}_embeddings")()
    question = commons.q    

    # loader = PyPDFLoader("file/test.pdf")
    # pages = loader.load()
    # cnt = 0
    # texts = ""
    # for i in pages:
    #     cnt += num_tokens_from_string(i.page_content, "cl100k_base")
    #     pattern = r'\d+\x00/\x00\d+'
    #     texts += re.sub(pattern, "" , i.page_content)

    file2 = open("file/test.txt","r", encoding="utf-8")
    documents = file2.read()
    file2.close()

    # print(f'예상 토큰 수 : {cnt}')
    section_splitter = RecursiveCharacterTextSplitter(
        separators=[".*제 [1-9] 절.*"],
        chunk_size = 1000,
        chunk_overlap=0,
        is_separator_regex=True
    )

    sub_section_splitter = RecursiveCharacterTextSplitter(
        separators=[".*제 [1-9] 관.*"],
        chunk_size = 10,
        chunk_overlap=0,
        is_separator_regex=True
    )

    article_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\s*\n\s*제\s?\d+-?\d*\s?조.*"],
        chunk_size = 50,
        chunk_overlap=0,
        is_separator_regex=True
    )

    # db = Chroma(embedding_function=embeddings)

    file = open("file/test.log","w", encoding="utf-8")

    c=0
    sections = section_splitter.split_text(documents)

    list_text = ""
    for section in sections:

        section_match = re.search(".*제 [1-9] 절.*", section)
        if section_match:
            list_text += section_match.group()+"\n"
        else:
            list_text += "\n"

        sub_sections = sub_section_splitter.split_text(section)
        for sub_section in sub_sections:

            sub_section_match = re.search(".*제 [1-9] 관.*", sub_section)
            if sub_section_match:
                list_text += "ㄴ"+sub_section_match.group()+"\n"
            else: 
                list_text += "ㄴ\n"
                
            articles = article_splitter.split_text(sub_section)
            for article in articles:

                article_match = re.search("\n\s*\n\s*제\s?\d+-?\d*\s?조.*", article)
                if article_match:
                    list_text += " ㄴ"+article_match.group()+"\n"
                else:
                    list_text += " ㄴ\n"

                text = f"{section_match.group()}\n\n{sub_section_match.group()}\n\n{article}"
                file.write(f"{str(c)} {len(text)} \n {text} \n")
                file.write("=====================================================\n\n")

    file_list = open("file/list.log","w", encoding="utf-8")
    file_list.write(list_text)
    file_list.close()
    file.close()

    # retriever = db.as_retriever(search_kwargs={"k": 3})

    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     verbose=True,
    #     retriever=retriever,
    #     return_source_documents=True
    # )
    
    # chat = chain({"question":question, "chat_history": []})

    # end_time = time.time()

    # execution_time = end_time - start_time

    # print(f"프로그램 실행 시간: {execution_time} 초")
    chat="테스트"

    return chat

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

