"""
로깅 클래스 
작성: 염경훈
날짜: 2023-11-24
"""

import tiktoken


class Func:
    
    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens