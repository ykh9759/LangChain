
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate, 
    ChatPromptTemplate,
    MessagesPlaceholder
)
from src.llm.tools import Tools
from src.llm.llm import Llm


router = APIRouter(
    prefix="/socket",
    tags=["chat"]
)

#에러원인 llm변수가 아래 LLMChain의 llm파라미터에 사용되는데 llm객체를 전달하면 오류가 발생한다.
llm = Llm().get_llm("openai", "gpt-3.5-turbo")
tools = Tools(llm)

# 웹소켓 연결 추적을 위한 딕셔너리
websocket_connections = {}

#고객과 상담사 채팅
@router.websocket("/ws/{client_id}")
async def websocket_endpoint2(websocket: WebSocket, client_id:str):
    await websocket.accept()
    
    print(f"연결완료: {client_id}")
    # 클라이언트 아이디를 사용하여 연결을 딕셔너리에 추가
    websocket_connections[client_id] = websocket

    try:
        while True:
            data = await websocket.receive_json()
            
            print(f"받음: {data}")
            # 수신한 메시지를 다른 모든 클라이언트에게 브로드캐스트
            for connection_id, connection in websocket_connections.items():
                if connection_id != client_id:
                    
                    result = {}
                    result["message"] = f"{client_id} : {data['message']}"
                    await connection.send_json(result)
    except WebSocketDisconnect:
        print(f"WebSocket 연결이 끊어짐: {client_id}")
    finally:
        # 클라이언트 연결이 끊기면 딕셔너리에서 제거
        del websocket_connections[client_id]

# llm과채팅
@router.websocket("/ws-llm")
async def websocket_endpoint(websocket: WebSocket):
    
    print(f"연결 완료 : {websocket.client}")
    await websocket.accept() # client의 websocket접속 허용

    #대화내역저장
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

    try:
        while True:
            data = await websocket.receive_json()  # client 메시지 수신대기
            print(f"message received : {data} from : {websocket.client}")

            search = tools.get_search(data["message"])

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

            answer = llm_chain.run(input=data["message"])
            print(llm_chain.memory)
            print(answer)
            
            result = {}
            result["message"] = answer
            
            await websocket.send_json(result) # client에 메시지 전달
    except WebSocketDisconnect:
        print(f"WebSocket 연결이 끊어짐")
    


