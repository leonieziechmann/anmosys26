from typing import Any, Optional, Dict, List

class GenerateContentConfig:
    def __init__(self, 
                 response_mime_type: Optional[str] = None, 
                 response_schema: Optional[Any] = None, 
                 system_instruction: Optional[str] = None,
                 temperature: Optional[float] = None,
                 tools: Optional[List[Any]] = None,
                 **kwargs):
        self.response_mime_type = response_mime_type
        self.response_schema = response_schema
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.tools = tools

class MockFunctionResponse:
    def __init__(self, name: str, response: Dict[str, Any]):
        self.name = name
        self.response = response

class Part:
    def __init__(self, text: Optional[str] = None, function_call: Optional[Any] = None, function_response: Optional[Any] = None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response

    @classmethod
    def from_text(cls, text: str):
        return cls(text=text)

    @classmethod
    def from_function_response(cls, name: str, response: Dict[str, Any]):
        return cls(function_response=MockFunctionResponse(name, response))

class Content:
    def __init__(self, role: str, parts: List[Part]):
        self.role = role
        self.parts = parts
