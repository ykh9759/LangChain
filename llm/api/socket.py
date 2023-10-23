
from fastapi import APIRouter, Request, WebSocket
from fastapi.templating import Jinja2Templates
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate,PromptTemplate, MessagesPlaceholder
from llm.tools import Tools
from llm.llm import Llm


router = APIRouter(
    prefix="/socket",
    tags=["socket"]
)

# html파일을 서비스 
templates = Jinja2Templates(directory="template")

llm = Llm().get_openai()
tools = Tools(llm)

#대화내역저장
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)


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

    while True:
        message = await websocket.receive_text()  # client 메시지 수신대기
        print(f"message received : {message} from : {websocket.client}")

        search = tools.get_search(message)

        system_template = """
            You are a chatbot that answers questions in Korean.

            Refer to search: (text) contents when answering

            search: ({search})    
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

