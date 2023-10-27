"""
랭체인API 라우터 페이지
작성: 염경훈
날짜: 2023-09-20
"""
from fastapi import Depends, APIRouter, status
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
import tiktoken
from llm.tools import Tools
from llm.llm import Llm
from llm.response import chatModelsResponse
from googletrans import Translator
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter

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
    embeddings = getattr(Llm(), f"{commons.model}_embeddings")()
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
    embeddings: OpenAIEmbeddings = getattr(Llm(), f"{commons.model}_embeddings")()
    # question = trans.translate(commons.q, dest="en").text
    question = commons.q    
    # search = Tools(llm).get_google_search()

    loader = PyPDFLoader("file/test.pdf")
    pages = loader.load()
    print(pages)
    cnt = 0
    for i in pages:
        cnt += num_tokens_from_string(i.page_content, "cl100k_base")
    print(f'예상 토큰 수 : {cnt}')

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="/제( [1-9](-[(1-9]|) )조/", is_separator_regex=True)
    documents = text_splitter.split_documents(pages)

    db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    # db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    searchs = db.similarity_search(question)
    print(searchs)

    search = []
    for s in searchs:
        search.append(s.page_content)


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
    
    chat = chain.run(language="korean", search=search, input=question)

    return chat

@router.get(
    "/json-rag",                       #라우터경로
    status_code=status.HTTP_200_OK      #HTTP status
) 
async def jsonRag(commons: CommonQueryParams = Depends()):

    llm = getattr(Llm(), f"get_{commons.model}")()
    embeddings: OpenAIEmbeddings = getattr(Llm(), f"{commons.model}_embeddings")()
    question = trans.translate(commons.q, dest="en").text
    # question = commons.q    
    search = Tools(llm).get_google_search()

    loader = JSONLoader(
        file_path="file/fp_role_plyaing.json",
        jq_schema="."
    )
    json = loader.load()
    # cnt = 0
    # for i in pages:
    #     cnt += num_tokens_from_string(i.page_content, "cl100k_base")
    # print(f'예상 토큰 수 : {cnt}')

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="제")
    documents = text_splitter.split_documents(json)

    db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db_json")
    # db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    searh = db.similarity_search(question)
    print(searh)
    
    return searh

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

