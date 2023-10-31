"""
API 메인 페이지
작성: 염경훈
날짜: 2023-09-20
"""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import PlainTextResponse
from src.llm.api import langchain, agent, socket

app = FastAPI()                 #FastAPI객체 생성

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

app.include_router(agent.router)     #라우터 추가
app.include_router(langchain.router)     #라우터 추가
app.include_router(socket.router)     #라우터 추가

@app.get("/") 
def index():
    return {"result" : "Success!"} 