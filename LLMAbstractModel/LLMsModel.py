# from https://github.com/qinhy/singleton-key-value-storage.git
import base64
from datetime import datetime
import fnmatch
import io
import json
import os
import re
import unittest
import uuid
from PIL import Image
from typing import Any, Dict, List, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo
from pydantic import validator, ConfigDict, Field
import requests
from typing import Dict, Any

from .BasicModel import Controller4Basic,Model4Basic

def now_utc():
    return datetime.now().replace(tzinfo=ZoneInfo("UTC"))

class Controller4LLMs:
    pass
        
class Model4LLMs:
    
    class AbstractVendor(Model4Basic.AbstractObj):
        vendor_name: str  # e.g., 'OpenAI'
        api_url: str  # e.g., 'https://api.openai.com/v1/'
        api_key: str = None  # API key for authentication, if required
        timeout: int = 30  # Default timeout for API requests in seconds
        
        def send_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            """
            Send a request to the vendor's API.
            :param endpoint: API endpoint to be called.
            :param payload: Request payload to be sent.
            :return: Response from the API as a dictionary.
            """
            pass

        def get_available_models(self) -> Dict[str, Any]:
            """
            Get a list of available models from the vendor.
            :return: Dictionary of available models and their details.
            """
            pass

        def _get_headers(self) -> Dict[str, str]:
            """
            Get headers for API requests, including authorization if API key is present.
            :return: Dictionary of headers.
            """
            headers = {
                'Content-Type': 'application/json'
            }
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            return headers

        def _build_url(self, endpoint: str) -> str:
            """
            Construct the full API URL for a given endpoint.
            :param endpoint: API endpoint to be called.
            :return: Full API URL.
            """
            return f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"

    class OpenAIVendor(AbstractVendor):
        vendor_name:str = 'OpenAI'
        api_url:str = 'https://api.openai.com/v1/chat/completions'

        def send_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            """
            Send a request to the OpenAI API.
            :param endpoint: API endpoint to be called.
            :param payload: Request payload to be sent.
            :return: Response from the API as a dictionary.
            """
            url = self._build_url(endpoint)
            headers = self._get_headers()
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()  # Raise an exception for HTTP errors
                return response.json()
            except requests.RequestException as e:
                raise Exception(f"API request failed: {e}")

        def get_available_models(self) -> Dict[str, Any]:
            """
            Get a list of available models from OpenAI.
            :return: Dictionary of available models and their details.
            """
            endpoint = 'models'
            response = self.send_request(endpoint, {})
            models = {model['id']: model for model in response.get('data', [])}
            return models

    class AbstractLLM(Model4Basic.AbstractObj):
        vendor_id:str
        model_name:str

        context_window_tokens:int
        # Context Window Size: The context window size dictates the total number of tokens the model can handle at once. For example, if a model has a context window of 4096 tokens, it can process up to 4096 tokens of combined input and output.
        max_output_tokens:int
        
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 1.0
        frequency_penalty: Optional[float] = 0.0
        presence_penalty: Optional[float] = 0.0
        system_prompt: Optional[str] = None

        def get_usage_limits(self) -> Dict[str, Any]:
            """
            Abstract method to get usage limits, such as rate limits, token limits, etc.
            """
            pass

        def validate_input(self, prompt: str) -> bool:
            """
            Validate the input prompt based on max input tokens.
            """
            if len(prompt) > self.context_window_tokens:
                raise ValueError(f"Input exceeds the maximum token limit of {self.context_window_tokens}.")
            return True

        def calculate_cost(self, tokens_used: int) -> float:
            """
            Optional method to calculate the cost based on tokens used.
            Can be overridden by subclasses to implement vendor-specific cost calculation.
            """
            return 0.0

        def get_token_count(self, text: str) -> int:
            """
            Dummy implementation for token counting. Replace with vendor-specific logic.
            """
            return len(text.split())
        
        def build_system(self, purpose='...'):
            """
            Dummy implementation for building system prompt
            """

        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4Basic.AbstractObjController = None
        def get_controller(self)->Controller4Basic.AbstractObjController: return self._controller
        def init_controller(self,store):self._controller = Controller4Basic.AbstractObjController(store,self)

    class ChatGPT4o(AbstractLLM):
        model_name:str = 'gpt-4o'

        context_window_tokens:int = 128000
        max_output_tokens:int = 4096
        
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 1.0
        frequency_penalty: Optional[float] = 0.0
        presence_penalty: Optional[float] = 0.0
        system_prompt: Optional[str] = None

        # New attributes for advanced configurations
        stop_sequences: Optional[List[str]] = Field(default_factory=list)
        n: Optional[int] = 1  # Number of completions to generate for each input prompt
        
        # Validator to ensure correct range of temperature
        @validator('temperature')
        def validate_temperature(cls, value):
            if not 0 <= value <= 1:
                raise ValueError("Temperature must be between 0 and 1.")
            return value

        # Validator to ensure correct range of top_p
        @validator('top_p')
        def validate_top_p(cls, value):
            if not 0 <= value <= 1:
                raise ValueError("top_p must be between 0 and 1.")
            return value
        
        # Method to construct the payload for API request        
        # "messages": [
        #     {"role": "system", "content": system_prompt or self.system_prompt},
        #     {"role": "user", "content": prompt}
        # ],
        def construct_payload(self, messages: list[dict]) -> Dict[str, Any]:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stop": self.stop_sequences,
                "n": self.n,
            }
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            return payload
        
        def gen(self, messages=[])->str:
            pass
        
    class ChatGPT4oMini(ChatGPT4o):
        model_name:str = 'gpt-4o-mini'


def test(materials=''):
    # ideas : 1.purpose 2.reference materials 3.previous outputs ==> current outputs 
    reference_materials = materials.split()
    previous_outputs = []

    llm = Model4LLMs.ChatGPT4oMini()
    # llm.build_system(purpose='...')
    def system_prompt(limit_words,code_path,code_lang,code,pre_explanation):
        return f'''You are an expert in code explanation, familiar with Electron, and Python.
I have an app built with Electron and Python.
I will provide pieces of the project code along with prior explanations.
Your task is to read each new code snippet and add new explanations accordingly.  
You should reply in Japanese with explanations only, without any additional information.

## Your Reply Format Example (should not over {limit_words} words)
```explanation
- This code shows ...
```
                           
## Code Snippet
code path : {code_path}
```{code_lang}
{code}
```

## Previous Explanations
```explanation
{pre_explanation}
```'''
    
    def messages(limit_words,code_path,code_lang,code,pre_explanation):
        return [
            {"role": "system", "content": system_prompt(limit_words,code_path,code_lang,code,pre_explanation)},
        ]

    def outputFormatter(output=''):
        def extract_explanation_block(text):
            matches = re.findall(r"```explanation\s*(.*)\s*```", text, re.DOTALL)
            return matches if matches else []
        return extract_explanation_block(output)[0]

    for m in reference_materials:
        preout = previous_outputs[-1] if len(previous_outputs)>0 else None
        msgs = messages(code=m,pre_explanation=preout,limit_words=100)
        tokens = llm.get_token_count(msgs)
        output = llm.gen(msgs)
        previous_outputs.append(outputFormatter(output))


