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

    llm = getattr(Llm(), f"get_{commons.model}")()
    embeddings: OpenAIEmbeddings = getattr(Llm(), f"get_{commons.model}_embeddings")()
    question = commons.q    

    loader = PyPDFLoader("file/test.pdf")
    pages = loader.load()
    cnt = 0
    texts = ""
    for i in pages:
        cnt += num_tokens_from_string(i.page_content, "cl100k_base")
        pattern = r'\d+\x00/\x00\d+'
        texts += re.sub(pattern, "" , i.page_content)
        # texts += i.page_content

    print(f'예상 토큰 수 : {cnt}')

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["제.*조","\n \n"],
        chunk_size = 1000,
        chunk_overlap  = 200
    )

    documents = text_splitter.split_text(texts)
    print(documents)

    # db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    file = open("file/test.log","w", encoding="utf-8")

    c=0
    for document in documents:
        c += 1
        print(str(c) + document)
        print("=====================================================")
        file.write(f"{str(c)} : {document} \n")
        file.write("=====================================================\n\n")
        
        
        # if db is None:
        #     db = Chroma.from_texts([document], cached_embedder, persist_directory="./chroma_db")
        # else:
        # db.add_texts([document])
            # time.sleep(0.1)  # wait for 60 seconds before processing the next document


    # db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    

    # retriever = db.as_retriever(search_kwargs={"k": 3})

    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     verbose=True,
    #     retriever=retriever,
    #     return_source_documents=True
    # )
    
    # chat = chain({"question":question, "chat_history": []})
    file.close()
    chat = "테스트"

    return chat

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

