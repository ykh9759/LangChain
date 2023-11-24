"""
로깅 클래스 
작성: 염경훈
날짜: 2023-11-20
"""

import logging

class Log:

    @staticmethod
    def get_logger(path, name: str = None, mode: str = "a"):
        
        
        # 로거 생성
        logger = logging.getLogger(name)

        # 파일 핸들러 생성
        file_handler = logging.FileHandler(filename=path, mode=mode, encoding="UTF-8")

        # 콘솔 핸들러 생성
        console_handler = logging.StreamHandler()

        # 로그 레벨 설정 (예: DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.INFO)      # 파일 출력 레벨을 설정
        console_handler.setLevel(logging.DEBUG)  # 콘솔 출력 레벨을 설정

        # 파일 핸들러에 포매터 추가
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 로거에 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        
        return logger
    