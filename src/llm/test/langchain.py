"""
랭체인API 라우터 페이지
작성: 염경훈
날짜: 2023-09-20
"""
import time
import tiktoken
import re

from fastapi import Depends, APIRouter, status
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain, ConversationalRetrievalChain, AnalyzeDocumentChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from src.common.log import Log
from src.llm.tools import Tools
from src.llm.llm import Llm
from googletrans import Translator
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain

router = APIRouter(
    prefix="/api",
    tags=["langchain"]
)

trans = Translator()  #구글번역
log = Log()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#공통 파라미터
class CommonQueryParams:
    def __init__(self, 
        model:str,                 #모델명
        q: str                    #질문
    ):   
        self.model = model.strip() 
        self.q = q                                  

@router.get(
    "/chat-models",                       #라우터경로
    status_code=status.HTTP_200_OK       #HTTP status
) 
async def chatModels(commons: CommonQueryParams = Depends()):

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
    db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    logger = log.get_logger(name="PDF", path="log/test.log", mode="w")

    # loader = PyPDFLoader("file/test.pdf")
    # pages = loader.load()
    # cnt = 0
    # texts = ""
    # for i in pages:
    #     cnt += num_tokens_from_string(i.page_content, "cl100k_base")
    #     pattern = r'\d+\x00/\x00\d+'
    #     texts += re.sub(pattern, "" , i.page_content)
    # print(f'예상 토큰 수 : {cnt}')

    #약관로 자르는 spliter
    terms_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\se.*약관","\s특별약관.*\n.*\n.*\n무배당"],
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True,
    )

    #절로 자르는 spliter
    section_splitter = CharacterTextSplitter(
        separator="제 [1-9] 절.*",
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True,
        keep_separator=True
    )

    #관으로 자르는 spliter
    sub_section_splitter = CharacterTextSplitter(
        separator="제 [1-9] 관.*",
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True,
        keep_separator=True
    )

    #조로 자느는 spliter
    article_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\s*\n\s*제\s?\d+-?\d*\s?조.*","\n\s*【 별표"],
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True
    )

    #특별약관 splliter
    special_splitter = RecursiveCharacterTextSplitter(
        separators=["\n.* 약관 \n","\n장해분류표","\n 보험용어  해설","【부록】약관.*"],
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True
    )

    #특별약관 splliter
    splitter1 = RecursiveCharacterTextSplitter(
        separators=["1  총칙 ","2  장해분류별  판정기준"],
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True
    )

    #장해분류표 spliter
    classification_splitter = RecursiveCharacterTextSplitter(
        separators=["\d\. .* \n가\.","< 붙임 >"],
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True,
        keep_separator=True
    )

    #법률 spliter
    law_splitter = CharacterTextSplitter(
        separator="\n\s\n\s?○ .*",
        chunk_size=10,
        chunk_overlap=0,
        is_separator_regex=True,
        keep_separator=True
    )
    
    file2 = open("file/test.txt","r", encoding="utf-8")
    documents = file2.read()
    file2.close()

    # array_text = []
    # terms = terms_splitter.split_text(documents)
    # for term in terms:

    #     term_match1 = re.search("e.*약관 \n", term)
    #     term_match2 = re.search("\s특별약관.*\n.*\n.*\n무배당", term)

    #     if term_match2:
    #         specials = special_splitter.split_text(term)
    #         for special in specials:

    #             special_match1 = re.search("\n.* 약관 \n", special)
    #             special_match2 = re.search("\n장해분류표", special)
    #             special_match3 = re.search("\n 보험용어  해설", special)
    #             special_match4 = re.search("【부록】약관.*", special)

    #             if special_match1:

    #                 articles2 = article_splitter.split_text(special)
    #                 for article in articles2:
    #                     text = ""
    #                     text += special_match1.group()
    #                     text += article
    #                     array_text.append(text)
    #                     logger.info(f"{len(text)} \n {text} \n")
    #                     logger.info("=====================================================\n\n")
    #             elif special_match2:

    #                 classifications = splitter1.split_text(special)

    #                 for classification in classifications:

    #                     classification_match = re.search("1  총칙 ", classification)

    #                     if classification_match:
    #                         text = ""
    #                         text += f"{special_match2.group()}\n"
    #                         text += classification
    #                         array_text.append(text)
    #                         logger.info(f"{len(text)} \n {text} \n")
    #                         logger.info("=====================================================\n\n")
    #                     else:
    #                         nos = classification_splitter.split_text(classification)

    #                         for no in nos:
    #                             text = ""
    #                             text += f"{special_match2.group()}\n"
    #                             text += "2 장해분류별 판정기준\n\n"
    #                             text += no
    #                             array_text.append(text)
    #                             logger.info(f"{len(text)} \n {text} \n")
    #                             logger.info("=====================================================\n\n")
    #             elif special_match3:
    #                 text = ""
    #                 text += special
    #                 array_text.append(text)
    #                 logger.info(f"{len(text)} \n {text} \n")
    #                 logger.info("=====================================================\n\n")
    #             elif special_match4:
    #                 laws = law_splitter.split_text(special)
    #                 for law in laws:

    #                     law_match = re.search("○ .*", law)

    #                     article3 = article_splitter.split_text(law)
    #                     for article in article3:

    #                         article_match = re.search("\n\s*\n\s*제\s?\d+-?\d*\s?조.*", article)

    #                         if article_match:
    #                             text = ""
    #                             text += f"{special_match4.group()}\n"
    #                             text += law_match.group() if law_match else ""
    #                             text += article
    #                             array_text.append(text)
    #                             logger.info(f"{len(text)} \n {text} \n")
    #                             logger.info("=====================================================\n\n")
    #     else:
        
    #         #텍스트 안에서 절로 자른다
    #         sections = section_splitter.split_text(term)
    #         for section in sections:  

    #             #절로 잘린 텍스트 안에서 관으로 다시 자른다
    #             sub_sections = sub_section_splitter.split_text(section)
    #             for sub_section in sub_sections:

    #                 #관로 잘린 텍스트 안에서 조으로 다시 자른다
    #                 articles = article_splitter.split_text(sub_section)
    #                 for article in articles:

    #                     term_match = re.search("e.*약관 \n", term)
    #                     section_match = re.search(".*제 [1-9] 절.*", section)               #절로 자른 첫번째로 매치된 절을 가져온다.
    #                     sub_section_match = re.search(".*제 [1-9] 관.*", sub_section)       #관으로 자른 첫번째로 매치된 관을 가져온다
    #                     article_match = re.search("\n\s*\n\s*제\s?\d+-?\d*\s?조.*", article)
    #                     asterisk_match = re.search("\n\s*【 별표", article)

    #                     text = ""

    #                     if article_match:
    #                         text += term_match.group() if term_match else ""
    #                         text += f"{section_match.group()}\n" if section_match else ""
    #                         text += sub_section_match.group() if sub_section_match else ""
    #                         text += article
    #                     elif asterisk_match:
    #                         text += term_match.group() if term_match else ""
    #                         text += article

    #                     if text:
    #                         logger.info(f"{len(text)} \n {text} \n")
    #                         logger.info("=====================================================\n\n")
    #                         array_text.append(text)

    # while len(array_text) > 0:
    #     db.add_texts(array_text[0:200])
    #     del array_text[0:200] 

    retriever = db.as_retriever(search_kwargs={"k": 2})

    template = (
        "You are an AI that searches for insurance terms and conditions"
        "you answer in korean"
        "Please search for the most appropriate clause for your question and provide detailed information"
        "question: {question}"
    )
    prompt = PromptTemplate.from_template(template)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        verbose=True,
        retriever=retriever,
        condense_question_prompt=prompt,
        condense_question_llm=llm,
        return_source_documents=True
    )
    
    chat = chain({"question":question, "chat_history": []})

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"프로그램 실행 시간: {execution_time} 초")
    # chat="테스트"

    return chat

@router.get(
    "/summery",                       #라우터경로
    status_code=status.HTTP_200_OK      #HTTP status
) 
async def summary(commons: CommonQueryParams = Depends()):
    llm = getattr(Llm(), f"get_{commons.model}")()
    question = commons.q

    prompt_template = """Please answer in Korean

    You are an expert with excellent summarizing skills.
    You receive input from the conversation between the customer and the counselor and summarize the important content of the conversation.

    The summary is short and concise in one sentence.
    
    Write a concise summary of the following:
    {text}
    요약:"""
    prompt = PromptTemplate.from_template(prompt_template)

    chain = load_summarize_chain(llm, verbose=True, prompt=prompt)

    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=chain, verbose=True)
    summary = summarize_document_chain.run(question)
    # chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    # summary = chain.run(text=question)

    return summary