
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate, 
    ChatPromptTemplate,
    PromptTemplate, 
    MessagesPlaceholder
)
from src.llm.tools import Tools
from src.llm.llm import Llm


router = APIRouter(
    prefix="/socket",
    tags=["socket"]
)

# html파일을 서비스 
templates = Jinja2Templates(directory="template")

#에러원인 llm변수가 아래 LLMChain의 llm파라미터에 사용되는데 llm객체를 전달하면 오류가 발생한다.
llm = Llm().get_openai()
tools = Tools(llm)

# 웹소켓 연결을 테스트 할 수 있는 웹페이지
@router.get("/client")
async def client(request: Request):

    return templates.TemplateResponse("client.html", {"request":request})

# 웹소켓 설정
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print(f"연결 완료 : {websocket.client}")
    await websocket.accept() # client의 websocket접속 허용
    await websocket.send_text(f"안녕하세요 : {websocket.client}")

    #대화내역저장
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

    try:
        while True:
            message = await websocket.receive_text()  # client 메시지 수신대기
            print(f"message received : {message} from : {websocket.client}")

            search = tools.get_search(message)

            system_template = """
                You are a chatbot that answers questions in Korean.

                Refer to search contents when answering

                search: {search}    
            """

            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template).format(search=search),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])

            llm_chain = LLMChain(
                    llm=llm,
                    prompt=prompt,
                    verbose=True,
                    memory=memory
            )

            answer = llm_chain.run(input=message)
            print(llm_chain.memory)
            print(answer)
            
            await websocket.send_text(answer) # client에 메시지 전달
    except WebSocketDisconnect:
        print(f"WebSocket 연결이 끊어짐")


# 웹소켓 연결 추적을 위한 딕셔너리
websocket_connections = {}

@router.websocket("/ws/{client_id}")
async def websocket_endpoint2(websocket: WebSocket, client_id: int):
    await websocket.accept()
    # 클라이언트 아이디를 사용하여 연결을 딕셔너리에 추가
    websocket_connections[client_id] = websocket

    try:
        while True:
            data = await websocket.receive_text()
            # 수신한 메시지를 다른 모든 클라이언트에게 브로드캐스트
            for connection_id, connection in websocket_connections.items():
                if connection_id != client_id:
                    await connection.send_text(f"{client_id} : {data}")
    except WebSocketDisconnect:
        print(f"WebSocket 연결이 끊어짐: {client_id}")
    finally:
        # 클라이언트 연결이 끊기면 딕셔너리에서 제거
        del websocket_connections[client_id]

