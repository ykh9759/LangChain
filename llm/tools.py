"""
랭체인 Agent tools 클래스 파일
작성: 염경훈
날짜: 2023-10-10
"""

from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper, WolframAlphaAPIWrapper, GoogleSearchAPIWrapper
from langchain.chains import LLMMathChain
from langchain.agents import Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from PyNaver import Naver
import os

from common.settings import Settings

class Tools:

    settings = Settings()

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm_math_chain = LLMMathChain.from_llm(llm=llm)
        self.search = SerpAPIWrapper(serpapi_api_key=self.settings.SERPAPI_API_KEY)
        self.wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=self.settings.WOLFRAM_ALPHA_APPID)
        self.google_search = GoogleSearchAPIWrapper(google_api_key=self.settings.GOOGLE_API_KEY, google_cse_id=self.settings.GOOGLE_CES_ID)

    def get_tools(self, list) -> None:
        tools = []

        for tool in list:
            if tool == "serpapi":
                tools.append(
                    Tool(
                        name="SerpApi",
                        func=self.search.run,
                        description="useful for when you need to answer questions about current events"
                    )
                )
            elif tool == "llm-math":
                tools.append(
                    Tool(
                        name="Calculator",
                        func=self.llm_math_chain.run,
                        description="only used for math calculations"
                    )
            )
            elif tool == "wikipedia":
                tools.append(
                    Tool(
                        name="Wikipedia",
                        func=self.wikipedia.run,
                        description="used last"
                    )
            )
            elif tool == "wolfram-alpha":
                tools.append(
                    Tool(
                        name="Wolfram Alpha",
                        func=self.wolfram.run,
                        description="Use when you need to answer questions about math, science, technology, culture, society, and everyday life and use it first."
                    )
            )
            elif tool == "google-search":
                tools.append(
                    Tool(
                        name="Google Search",
                        func=self.google_search.run,
                        description="useful for when you need to answer questions about current events"
                    )
                )
                
        return tools