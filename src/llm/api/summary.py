from fastapi import Depends, APIRouter, status
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate
from src.common.log import Log
from src.llm.llm import Llm
from langchain.chains.summarize import load_summarize_chain

router = APIRouter(
    prefix="/api",
    tags=["summary"]
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
    "/summary",                       #라우터경로
    status_code=status.HTTP_200_OK      #HTTP status
) 
async def summary(commons: CommonQueryParams = Depends()):
    llm = commons.llm
    question = commons.q

    prompt_template = """Please answer in Korean

    You are an expert with excellent summarizing skills.
    You receive input from the conversation between the customer and the counselor and summarize the important content of the conversation.

    The summary must be short and concise, with no more than 100 characters.
    
    Write a concise summary of the following:
    {text}
    요약:"""
    prompt = PromptTemplate.from_template(prompt_template)

    chain = load_summarize_chain(llm, verbose=True, prompt=prompt)
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=chain, verbose=True)
    
    i = 0
    while True:
    
        summary = summarize_document_chain.run(question)
        
        if len(summary) < 100 or i > 5:
            break
        
        question = summary
        i += 1


    result = {}
    result["data"] = summary
    
    return result