"""
API 메인 페이지
작성: 염경훈
날짜: 2023-09-20
"""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import PlainTextResponse
from src.llm.api import chat, classification, retrieval, summary
from src.llm.test import langchain, agent, socket, api

app = FastAPI()                 #FastAPI객체 생성

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

#라우터추가
app.include_router(chat.router)
# app.include_router(classification.router)
app.include_router(retrieval.router)
# app.include_router(summary.router)

#테스트 라우터 추가
app.include_router(agent.router)
app.include_router(langchain.router)
app.include_router(socket.router)
app.include_router(api.router)

@app.get("/") 
def index():
    return {"result" : "Success!"} 