

from fastapi import APIRouter, Depends
from src.llm.searchApi import SearchApi


router = APIRouter(
    prefix="/test/api",
    tags=["test"]
)

api = SearchApi()

#공통 파라미터
class CommonQueryParams:
    def __init__(self, 
        q: str                    #질문
    ):   
        self.q = q         
        
@router.get("/google")
async def googleApi(commons: CommonQueryParams = Depends()):
    
    result = api.google_api(commons.q)
    print(result)
    
    return result