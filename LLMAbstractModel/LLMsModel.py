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
from typing import Any, Dict, List, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo
from pydantic import field_validator, ConfigDict, Field
import requests
from typing import Dict, Any

from BasicModel import Controller4Basic, Model4Basic, BasicStore


class Controller4LLMs:
    class AbstractVendorController(Controller4Basic.AbstractObjController):
        pass
    class OpenAIVendorController(Controller4Basic.AbstractObjController):
        pass
    class AbstractLLMController(Controller4Basic.AbstractObjController):
        pass
    class ChatGPT4oController(Controller4Basic.AbstractObjController):
        pass
    class ChatGPT4oMiniController(Controller4Basic.AbstractObjController):
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

        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.AbstractVendorController = None
        def get_controller(self)->Controller4LLMs.AbstractVendorController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractVendorController(store,self)

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

        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.OpenAIVendorController = None
        def get_controller(self)->Controller4LLMs.OpenAIVendorController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.OpenAIVendorController(store,self)

    class AbstractLLM(Model4Basic.AbstractObj):
        vendor_id:str
        llm_model_name:str

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
        _controller: Controller4LLMs.AbstractLLMController = None
        def get_controller(self)->Controller4LLMs.AbstractLLMController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractLLMController(store,self)

    class OpenAIChatGPT(AbstractLLM):
        llm_model_name:str

        context_window_tokens:int
        max_output_tokens:int
        
        limit_output_tokens: Optional[int]
        temperature: Optional[float]
        top_p: Optional[float]
        frequency_penalty: Optional[float]
        presence_penalty: Optional[float]
        system_prompt: Optional[str]

        # New attributes for advanced configurations
        stop_sequences: Optional[List[str]]
        n: Optional[int]
        
        # field_validator to ensure correct range of temperature
        @field_validator('temperature')
        def validate_temperature(cls, value):
            if not 0 <= value <= 1:
                raise ValueError("Temperature must be between 0 and 1.")
            return value

        # field_validator to ensure correct range of top_p
        @field_validator('top_p')
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
                "model": self.llm_model_name,
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

    class ChatGPT4o(OpenAIChatGPT):
        llm_model_name:str = 'gpt-4o'

        context_window_tokens:int = 128000
        max_output_tokens:int = 4096
        
        limit_output_tokens: Optional[int] = 1024
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 1.0
        frequency_penalty: Optional[float] = 0.0
        presence_penalty: Optional[float] = 0.0
        system_prompt: Optional[str] = None

        # New attributes for advanced configurations
        stop_sequences: Optional[List[str]] = Field(default_factory=list)
        n: Optional[int] = 1  # Number of completions to generate for each input prompt
        
        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.ChatGPT4oController = None
        def get_controller(self)->Controller4LLMs.ChatGPT4oController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.ChatGPT4oController(store,self)

    class ChatGPT4oMini(ChatGPT4o):
        llm_model_name:str = 'gpt-4o-mini'

        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.ChatGPT4oMiniController = None
        def get_controller(self)->Controller4LLMs.ChatGPT4oMiniController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.ChatGPT4oMiniController(store,self)

class LLMsStore(BasicStore):

    def _get_class(self, id: str, modelclass=Model4LLMs):
        return super()._get_class(id, modelclass)

    def add_new_openai_vendor(self,api_key: str,
                              api_url: str='https://api.openai.com/v1/chat/completions',
                              timeout: int=30) -> Model4LLMs.OpenAIVendor:
        return self.add_new_obj(Model4LLMs.OpenAIVendor(api_url=api_url,api_key=api_key,timeout=timeout))
    

    def add_new_chatgpt4o(self,vendor_id:str,
                                limit_output_tokens:int = 1024,
                                temperature:float = 0.7,
                                top_p:float = 1.0,
                                frequency_penalty:float = 0.0,
                                presence_penalty:float = 0.0,
                                system_prompt:str = None ) -> Model4LLMs.ChatGPT4o:
        
        return self.add_new_obj(Model4LLMs.ChatGPT4o(vendor_id=vendor_id,
                                limit_output_tokens=limit_output_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                system_prompt=system_prompt,))
    
    def add_new_chatgpt4omini(self,vendor_id:str,
                                limit_output_tokens:int = 1024,
                                temperature:float = 0.7,
                                top_p:float = 1.0,
                                frequency_penalty:float = 0.0,
                                presence_penalty:float = 0.0,
                                system_prompt:str = None ) -> Model4LLMs.ChatGPT4oMini:
        return self.add_new_obj(Model4LLMs.ChatGPT4oMini(vendor_id=vendor_id,
                                limit_output_tokens=limit_output_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                system_prompt=system_prompt,))
    
    def find_all_vendors(self)->list[Model4LLMs.AbstractVendor]:
        return self.find_all('*Vendor:*')
