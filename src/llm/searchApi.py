
import urllib
import json
import re
import pandas as pd
import requests
from wikipediaapi import Wikipedia
from googleapiclient.discovery import build
from PyNaver import Naver
from src.config.settings import Settings

class SearchApi:
    
    settings = Settings()
    Trash_Link = ["tistory", "kin", "youtube", "blog", "book", "news", "dcinside", "fmkorea", "ruliweb", "theqoo", "clien", "mlbpark", "instiz", "todayhumor"] 
    
    def google_api(
        self,
        query: str,
        wanted_row: int = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        input : 
            query : str  검색하고 싶은 검색어 
            wanted_row : str 검색 결과를 몇 행 저장할 것인지 
        output : 
            df_google : dataframe / column = title, link,description  
            사용자로 부터 입력받은 쿼리문을 통해 나온 검색 결과를 wanted_row만큼 (100행을 입력받았으면) 100행이 저장된 데이터 프레임을 return합니다.
        """
        query= query.replace("|","OR") #쿼리에서 입력받은 | 기호를 OR 로 바꿉니다 
        query += "-filetype:pdf" # 검색식을 사용하여 file type이 pdf가 아닌 것을 제외시켰습니다 

        df_google= pd.DataFrame(columns=['Title','Link','Description']) # df_Google이라는 데이터 프레임에 컬럼명은 Title, Link, Description으로 설정했습니다.

        row_count =0 # dataframe에 정보가 입력되는 것을 카운트 하기 위해 만든 변수입니다. 

        service = build("customsearch", "v1", developerKey=self.settings.GOOGLE_API_KEY)

        for start_page in range(1,wanted_row+1000,10):
            
            print(start_page)
    
            # 1페이지, 11페이지,21페이지 마다,             
            res = service.cse().list(q=query, cx=self.settings.GOOGLE_CSE_ID, start=start_page, **kwargs).execute()
            search_items = res["items"]
            # search_items엔 검색결과 [1~ 10]개의 아이템들이 담겨있다.  start_page = 11 ~ [11~20] 
            
            try:
            #try 구문을 하는 이유: 검색 결과가 null인 경우 link를 가져올 수가 없어서 없으면 없는대로 예외처리
                for search_item in search_items:
                    # link 가져오기 
                    link = search_item["link"]
                    # link url에 출처가 신뢰도가 낮은 사이트의 정보라면 데이터프레임에 저장하지 않고 넘어갑니다. 
                    if not any(trash in link for trash in self.Trash_Link):
                        # 제목저장
                        title = search_item["title"]
                        
                        if title == "Untitled":
                            continue
                        
                        # 설명 저장 
                        descripiton = search_item["snippet"]
                        # df_google에 한줄한줄 append 
                        df_google.loc[row_count] = [title,link,descripiton] 
                        # 저장하면 행 갯수 카운트 
                        row_count+=1
                        if (row_count >= wanted_row) or (row_count == 300) :
                        #원하는 갯수만큼 저장끝나면
                            return df_google
            except:
            # 더 이상 검색결과가 없으면 df_google 리턴 후 종료 
                return df_google

        
        return df_google
    
    
    def naver_api(
        self, 
        query: str,
        wanted_row: int = 10
    ) -> pd.DataFrame:
        
        """
        input : 
            query : str  검색하고 싶은 검색어 
            wanted_row : str 검색 결과를 몇 행 저장할 것인지 
        output : 
            df : dataframe / column = title, link,description  
            사용자로 부터 입력받은 쿼리문을 통해 나온 검색 결과를 wanted_row만큼 (100행을 입력받았으면) 100행이 저장된 데이터 프레임을 return합니다.
        """

        display=100 
        #네이버 검색 API는 한 페이지를 요청했을 때 몇개의 건수를 보여줄 것인지 인자로 표시할 수 있습니다. 
        start=1
        # start page를 1로 설정합니다.
        end=wanted_row+10000
        # 끝내는 페이지를 원하는 행의 갯수보다 더 많이 설정했는데 이유는 , trashlink 보다 많은 데이터를 저장합니다.  
        sort='sim'
        # 네이버 API 검색 결과는 검색결과 데이터를 정렬하는 순서의 기준을 정합니다. 

        df= pd.DataFrame(columns=['Title','Link','Description'])
        # 마찬가지로 title,link,description의 컬럼을 가진 데이터프레임을 생성합니다. 
        row_count= 0 
        # dataframe에 정보가 입력되는 것을 카운트 하기 위해 만든 변수입니다. 
        
        naver = Naver(client_id=self.settings.NAVER_CLIENT_ID, client_secret=self.settings.NAVER_CLIENT_SECRET)
        
        for start_index in range(start,end,display):
            
            try:
                items = naver.search_webkr(query=query, display=display, start=start_index, sort=sort)
                # 전체 response를 json화 한 뒤 key값이 items로 되어있는 값에 저장을 합니다. 
                remove_tag = re.compile('<.*?>')
                # html문법의 태그들을 제거하는 컴파일러를 정규식을 패키지를 통해 생성합니다.
                for item_index in range(0,len(items)):
                    # 아이템에 링크에 접근합니다
                    link = urllib.parse.unquote(items.loc[item_index]['link'])
                    # link url에 출처가 신뢰도가 낮은 사이트의 정보라면 데이터프레임에 저장하지 않고 넘어갑니다. 
                    if not any(trash in link for trash in self.Trash_Link):
                        title = re.sub(remove_tag, '', items.loc[item_index]['title'])
                        description = re.sub(remove_tag, '', items.loc[item_index]['description'])
                        # html 태그를 제거한 후, 제목 설명,링크 저장 
                        df.loc[row_count] = [title,link,description]
                        row_count+=1
                        if (row_count >= wanted_row) or (row_count == 300):
                            return df
                            
            except:
                return df
    
    def wiki_api(
        self,
        query: str
    ) -> pd.DataFrame:
        
        """
        input : 
            query : str  검색하고 싶은 검색어 
        output : 
            df : dataframe / column = title, link,description  
        """
        
        df= pd.DataFrame(columns=['Title','Link','Description'])
        # 마찬가지로 title,link,description의 컬럼을 가진 데이터프레임을 생성합니다. 
        
        headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
        
        wiki_ko = Wikipedia(user_agent='SCT', language='ko', headers=headers)
        page_py_ko = wiki_ko.page(query)
        
        page_missing = page_py_ko.exists()
        if page_missing:
               
            title = page_py_ko.title
            link = page_py_ko.fullurl
            description = page_py_ko.summary
        
            df.loc[0] = [title, link, description]
        
        return df
    