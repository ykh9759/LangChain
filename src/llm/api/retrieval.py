
import json

from fastapi import Depends, APIRouter, status
from langchain.embeddings import OpenAIEmbeddings
from src.common.log import Log
from src.llm.llm import Llm
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings

router = APIRouter(
    prefix="/api",
    tags=["retrieval"]
)

llms = Llm()
log = Log()

#공통 파라미터
class CommonQueryParams:
    def __init__(self, 
        model:str,                 #모델명
        q: str                    #질문
    ):   
        self.model = model
        self.q = q                                  
        self.llm = llms.get_llm(model.strip())

@router.get(
    "/retrival",                       #라우터경로
    status_code=status.HTTP_200_OK      #HTTP status
) 
async def retrival(commons: CommonQueryParams = Depends()):
    embeddings: OpenAIEmbeddings = llms.get_embeddings(commons.model)
    question = commons.q    
    db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.35})
    
    documents = retriever.get_relevant_documents(question)
    
    texts = [v.page_content for v in documents]
    
    print(json.dumps(texts, ensure_ascii=False))
    
    result = {}
    result["data"] = json.dumps(texts, ensure_ascii=False)
    
    return result