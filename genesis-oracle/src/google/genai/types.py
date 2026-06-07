from typing import Any, Optional, Dict

class GenerateContentConfig:
    def __init__(self, 
                 response_mime_type: Optional[str] = None, 
                 response_schema: Optional[Any] = None, 
                 system_instruction: Optional[str] = None,
                 temperature: Optional[float] = None,
                 **kwargs):
        self.response_mime_type = response_mime_type
        self.response_schema = response_schema
        self.system_instruction = system_instruction
        self.temperature = temperature
