import json
import os
import unittest
from typing import Any, Dict, List, Optional
from pydantic import field_validator, ConfigDict, Field
import requests
from typing import Dict, Any

from .BasicModel import Controller4Basic, Model4Basic, BasicStore

class Controller4LLMs:
    class AbstractVendorController(Controller4Basic.AbstractObjController):
        def __init__(self, store, model):
            super().__init__(store, model)
            self.model:Model4LLMs.AbstractVendor = model
            self._store:LLMsStore = store
    class OpenAIVendorController(AbstractVendorController):
        pass
    class AbstractLLMController(Controller4Basic.AbstractObjController):
        def __init__(self, store, model):
            super().__init__(store, model)
            self.model:Model4LLMs.AbstractLLM = model
            self._store:LLMsStore = store
        
        def get_vendor(self):
            vendor:Model4LLMs.AbstractVendor = self._store.find(self.model.vendor_id)
            return vendor
    class OpenAIChatGPTController(AbstractLLMController):
        pass
    class ChatGPT4oController(OpenAIChatGPTController):
        pass
    class ChatGPT4oMiniController(ChatGPT4oController):
        pass
        
class Model4LLMs:
    
    class AbstractVendor(Model4Basic.AbstractObj):
        vendor_name: str  # e.g., 'OpenAI'
        api_url: str  # e.g., 'https://api.openai.com/v1/'
        api_key: str = None  # API key for authentication, if required
        timeout: int = 30  # Default timeout for API requests in seconds
        
        def get_available_models(self) -> Dict[str, Any]:
            pass

        def _build_headers(self) -> Dict[str, str]:
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
        api_url:str = 'https://api.openai.com'

        def get_available_models(self) -> Dict[str, Any]:
            response = requests.get( self._build_url('/v1/models'),
                                    headers=self._build_headers(),
                                    timeout=self.timeout)
            models = {model['id']: model for model in response.json().get('data', [])}
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

        def get_vendor(self):
            return self.get_controller().get_vendor()

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
            return """
            Dummy implementation for building system prompt
            """

        def gen(self, messages=Optional[list|str])->str:
            pass

        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.AbstractLLMController = None
        def get_controller(self)->Controller4LLMs.AbstractLLMController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractLLMController(store,self)

    class OpenAIChatGPT(AbstractLLM):
        llm_model_name:str
        endpoint:str='/v1/chat/completions'

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
        
        def chat_completions(self,messages=[]):
            vendor = self.get_vendor()
            url=vendor._build_url(self.endpoint)
            headers=vendor._build_headers()
            msgs = []
            if self.system_prompt:
                msgs.append({"role":"system","content":self.system_prompt})
            data = self.construct_payload(msgs+messages)
            response = requests.post(url=url, data=json.dumps(data,ensure_ascii=False), headers=headers)
            try:
                response = json.loads(response.text,strict=False)
            except Exception as e:
                if type(response) is not dict:
                    return {'error':f'{e}'}
                else:
                    return {'error':f'{response}({e})'}
            return response
        
        def gen(self, messages=[])->str:
            if type(messages) is str:
                messages = [{"role":"user","content":messages}]
            res = self.chat_completions(messages)
            return str(res['error']) if 'error' in res else res['choices'][0]['message']['content']
            
        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.OpenAIChatGPTController = None
        def get_controller(self)->Controller4LLMs.OpenAIChatGPTController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.OpenAIChatGPTController(store,self)

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
                              api_url: str='https://api.openai.com',
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

class Tests(unittest.TestCase):
    def __init__(self,*args,**kwargs)->None:
        super().__init__(*args,**kwargs)
        self.store = LLMsStore()

    def test_all(self,num=1):
        for i in range(num):
            self.test_1()
            self.test_2()
            self.test_3()
    
    def test_1(self):
        v = self.store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
        print(v.get_available_models())

    def test_2(self):
        v = self.store.find_all('OpenAIVendor:*')[0]
        c = self.store.add_new_chatgpt4omini(vendor_id=v.get_id())
        print(c)
    
    
    def test_3(self):
        c:Model4LLMs.ChatGPT4oMini = self.store.find_all('ChatGPT4oMini:*')[0]
        print(c.gen('What is your name?'))

    