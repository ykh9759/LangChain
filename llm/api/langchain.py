"""
랭체인API 라우터 페이지
작성: 염경훈
날짜: 2023-09-20
"""
from fastapi import Depends, APIRouter, status
from langchain.agents import AgentExecutor, LLMSingleActionAgent, initialize_agent, AgentType
from langchain.chains import LLMChain
from llm.custom import CustomOutputParser, CustomPromptTemplate
from llm.tools import Tools
from llm.llm import Llm
from llm.response import chatModelsResponse
from googletrans import Translator

router = APIRouter(
    prefix="/api/langchain",
    tags=["langchain"]
)

#공통 파라미터
class CommonQueryParams:
    def __init__(
        self, 
        model:str,                 #모델명
        q: str,                    #질문
        llm: Llm = Depends()       #llm모델
    ):    
        self.llm = getattr(llm, f"get_{model}")() # Llm클래스에서 model명에 맞는 함수 호출
        self.q = q

@router.get(
    "/chat-models",                       #라우터경로
    status_code=status.HTTP_200_OK,       #HTTP status
    response_model=chatModelsResponse     #응답모델 지정
) 
async def chatModels(commons: CommonQueryParams = Depends()):

    trans = Translator()

    question = trans.translate(commons.q, dest="en").text                         #질문
    llm  = commons.llm                                #모델 생성

    # 랭체인
    ############################################################

    # Set up the base template
    template = """
    Based on Korean standards

    Complete the objective as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    These were previous tasks you completed:

    Begin!

    Question: {input}
    {agent_scratchpad}"""

    tools = Tools(llm).get_tools(["wolfram-alpha","wikipedia","llm-math"])

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input=question,
        input_variables=["input", "intermediate_steps"]
    )

    #체인생성
    # llm_chain = LLMChain(
    #     llm=llm,
    #     prompt=prompt
    # )

    agent = initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
        # max_execution_time=10,
        # early_stopping_method="generate"
    )


    # agent = LLMSingleActionAgent(
    #     llm_chain=llm_chain, 
    #     output_parser=CustomOutputParser(),
    #     stop=["\nObservation:"], 
    #     allowed_tools=[tool.name for tool in tools]
    # )

    # agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # 응답 데이터 세팅
    ############################################################

    response = {}

    try:
        chat = agent.run(prompt)
        # chat = agent_executor.run(question)

        #타임아웃응답
        if chat.startswith("Agent stopped"):
            response['status'] = "fail"
            response['message'] = "타임아웃"
        else:
            response['status'] = "success"
            response['data'] = {
                "question": question, 
                "answer": trans.translate(chat, dest="ko").text
            }

    except ValueError as e:
        response['status'] = "fail"

        chat = str(e)
        if chat.startswith("Could not parse LLM output: `"):
            response['message'] = "LLM파싱 실패"
        else:
            response['message'] = "LLM조회 실패"


    ############################################################

    return response

    