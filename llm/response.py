"""
데이터 유효성 검증 클래스 모음
작성: 염경훈
날짜: 2023-10-04
"""
from pydantic import BaseModel

class chatModelsResponse(BaseModel):
    status: str
    message: str | None = None
    data: dict | None = None