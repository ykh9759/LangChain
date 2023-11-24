

import json
import re
import pandas as pd
import requests
import urllib3
from src.config.settings import Settings

class SearchApi:
    
    settings = Settings()
    Trash_Link = ["tistory", "kin", "youtube", "blog", "book", "news", "dcinside", "fmkorea", "ruliweb", "theqoo", "clien", "mlbpark", "instiz", "todayhumor"] 
    
    def google_api(self, query, wanted_row: int = 10):
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
        start_pages=[] # start_pages 라는 리스트를 생성합니다. 

        df_google= pd.DataFrame(columns=['Title','Link','Description']) # df_Google이라는 데이터 프레임에 컬럼명은 Title, Link, Description으로 설정했습니다.

        row_count =0 # dataframe에 정보가 입력되는 것을 카운트 하기 위해 만든 변수입니다. 


        for i in range(1,wanted_row+1000,10):
            start_pages.append(i) #구글 api는 1페이지당 10개의 결과물을 보여줘서 1,11,21순으로 로드한 페이지를 리스트에 담았습니다. 

        for start_page in start_pages:
        # 1페이지, 11페이지,21페이지 마다, 
            url = f"https://www.googleapis.com/customsearch/v1?key={self.settings.GOOGLE_API_KEY}&cx={self.settings.GOOGLE_CSE_ID}&q={query}&start={start_page}"
            print(url)
            # 요청할 URL에 사용자 정보인 API key, CSE ID를 저장합니다. 
            data = requests.get(url).json()
            # request를 requests 라이브러리를 통해서 요청하고, 결과를 json을 호출하여 데이터에 담습니다.
            search_items = data.get("items")
            # data의 하위에 items키로 저장돼있는 값을 불러옵니다. 
            # search_items엔 검색결과 [1~ 10]개의 아이템들이 담겨있다.  start_page = 11 ~ [11~20] 
            
            try:
            #try 구문을 하는 이유: 검색 결과가 null인 경우 link를 가져올 수가 없어서 없으면 없는대로 예외처리
                for i, search_item in enumerate(search_items, start=1):
                # link 가져오기 
                    link = search_item.get("link")
                    if any(trash in link for trash in self.Trash_Link):
                    # 링크에 dcinside, News 등을 포함하고 있으면 데이터를 데이터프레임에 담지 않고, 다음 검색결과로 
                        pass
                    else: 
                        # 제목저장
                        title = search_item.get("title")
                        # 설명 저장 
                        descripiton = search_item.get("snippet")
                        # df_google에 한줄한줄 append 
                        df_google.loc[start_page + i] = [title,link,descripiton] 
                        # 저장하면 행 갯수 카운트 
                        row_count+=1
                        if (row_count >= wanted_row) or (row_count == 300) :
                        #원하는 갯수만큼 저장끝나면
                            return df_google
            except:
            # 더 이상 검색결과가 없으면 df_google 리턴 후 종료 
                return df_google

        
        return df_google
    
    
    def naver_api(self, query, wanted_row: int = 10):
        query = urllib3.parse.quote(query)

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
        
        for start_index in range(start,end,display):
            url = "https://openapi.naver.com/v1/search/webkr?query="+ query +\
                "&display=" + str(display)+ \
                "&start=" + str(start_index) + \
                "&sort=" + sort

            #url에 요청할 정보에 대한 내용을 담아 변수 선언하고, 
            request = urllib.request.Request(url)
            #urllib.request 모듈을 통해 요청을만들고,  
            request.add_header("X-Naver-Client-Id",self.settings.NAVER_CLIENT_ID)
            request.add_header("X-Naver-Client-Secret",self.settings.NAVER_CLIENT_SECRET)
            # 그 요청에 헤더를 만들어서 클라이언트 아이디와, 비밀번호를 헤더에 입력합니다. 
            try:
                response = urllib.request.urlopen(request)
                # 요청하여 받은 내용을 response로 저장합니다.
                rescode = response.getcode()
                # response 객체에 담긴 응답 코드를 받아옵니다
                if(rescode==200):
                    response_body = response.read()
                    # response 내용을 읽어들여 response_body에 저장합니다. 
                    items= json.loads(response_body.decode('utf-8'))['items']
                    # 전체 response를 json화 한 뒤 key값이 items로 되어있는 값에 저장을 합니다. 
                    remove_tag = re.compile('<.*?>')
                    # html문법의 태그들을 제거하는 컴파일러를 정규식을 패키지를 통해 생성합니다.
                    for item_index in range(0,len(items)):
                        link = items[item_index]['link']
                        # 아이템에 링크에 접근합니다
                        if any(trash in link for trash in self.Trash_Link):
                        # link url에 출처가 신뢰도가 낮은 사이트의 정보라면 데이터프레임에 저장하지 않고 넘어갑니다. 
                            pass
                        else:
                            title = re.sub(remove_tag, '', items[item_index]['title'])
                            description = re.sub(remove_tag, '', items[item_index]['description'])
                            # html 태그를 제거한 후, 제목 설명,링크 저장 
                            df.loc[row_count] =[title,link,description]
                            row_count+=1
                            if (row_count >= wanted_row) or (row_count == 300):
                                return df
                            
            except:
                return df