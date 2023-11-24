"""
공통변수 모음 파일
작성: 염경훈
날짜: 2023-09-21
"""

from pydantic_settings import BaseSettings
from starlette.config import Config

config = Config(".env")

#설정 관리를 위해 사용되는 클래스
class Settings(BaseSettings):
    OPENAI_API_KEY: str = config("OPENAI_API_KEY")
    SERPAPI_API_KEY: str = config("SERPAPI_API_KEY")
    WOLFRAM_ALPHA_APPID: str = config("WOLFRAM_ALPHA_APPID")
    GOOGLE_API_KEY: str = config("GOOGLE_API_KEY")
    GOOGLE_CSE_ID: str = config("GOOGLE_CSE_ID")
    NAVER_CLIENT_ID: str = config("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET: str = config("NAVER_CLIENT_SECRET")