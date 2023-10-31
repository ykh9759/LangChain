"""
랭체인 Agent tools 클래스 파일
작성: 염경훈
날짜: 2023-10-10
"""

from googletrans import Translator
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper, WolframAlphaAPIWrapper, GoogleSearchAPIWrapper
from langchain.chains import LLMMathChain, AnalyzeDocumentChain
from langchain.agents import Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain.chains.summarize import load_summarize_chain
from PyNaver import Naver
from src.config.settings import Settings

class Tools:

    trans = Translator() 
    settings = Settings()

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm_math_chain = LLMMathChain.from_llm(llm=llm)
        self.search = SerpAPIWrapper(serpapi_api_key=self.settings.SERPAPI_API_KEY)
        self.wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=self.settings.WOLFRAM_ALPHA_APPID)
        self.google_search = GoogleSearchAPIWrapper(google_api_key=self.settings.GOOGLE_API_KEY, google_cse_id=self.settings.GOOGLE_CES_ID)
        self.naver = Naver(client_id="rnniAuJmGFFu7FZDG4JG", client_secret="D0iBI0Vs6G")

        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        self.summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)

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
    

    def get_search(self, query) -> None:

        en_query = self.trans.translate(query, dest="en").text

        search_text = ""
        # search_text += self.search.run(query) + "\r\n"
        search_text += self.wolfram.run(en_query) + "\r\n"
        # search_text += self.naver.search_news(query) + "\r\n"
        # news = self.naver.search_news(query)
        # wiki = self.wikipedia.run(query)
        # print(wiki)
        # search_text += self.google_search.run(query)

        # search = self.summarize_document_chain.run(search_text)

        return search_text
    
    def get_wolfram(self):
        return self.wolfram
    
    def get_google_search(self):
        return self.google_search