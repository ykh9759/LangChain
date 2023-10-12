ASGI( Asynchronous Server Gateway Interface )는 WSGI의 정신적 계승자로, 비동기 가능 Python 웹 서버, 프레임워크 및 애플리케이션 간의 표준 인터페이스를 제공하기 위한 것입니다.

WSGI가 동기 Python 앱에 대한 표준을 제공했다면 ASGI는 WSGI 이전 버전과의 호환성 구현과 여러 서버 및 애플리케이션 프레임워크를 통해 비동기 및 동기 앱 모두에 대한 표준을 제공합니다.

서버 실행 명령어
python -m uvicorn main:app --reload