"""
공통변수 모음 파일
작성: 염경훈
날짜: 2023-09-21
"""

from pydantic_settings import BaseSettings

#설정 관리를 위해 사용되는 클래스
class Settings(BaseSettings):
    OPENAI_API_KEY: str = "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"
    SERPAPI_API_KEY: str = "2697b83421fccc4d4e045ad12cda54b978a54e5b4ef40a3e663bc4b9d384b0a0"
    WOLFRAM_ALPHA_APPID: str = "5Y4GV4-969AJ8AA6X"