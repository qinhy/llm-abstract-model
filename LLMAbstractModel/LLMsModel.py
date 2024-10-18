from graphlib import TopologicalSorter
import inspect
import json
import os
import time
import unittest
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, field_validator, ConfigDict, Field
import requests
from typing import Dict, Any

from .BasicModel import Controller4Basic, Model4Basic, BasicStore

class Controller4LLMs:
    class AbstractObjController(Controller4Basic.AbstractObjController):
        pass
    class AbstractVendorController(AbstractObjController):
        def __init__(self, store, model):
            super().__init__(store, model)
            self.model:Model4LLMs.AbstractVendor = model
            self._store:LLMsStore = store
    class AbstractLLMController(AbstractObjController):
        def __init__(self, store, model):
            super().__init__(store, model)
            self.model:Model4LLMs.AbstractLLM = model
            self._store:LLMsStore = store
        
        def get_vendor(self,auto=False):
            if not auto:
                vendor:Model4LLMs.AbstractVendor = self._store.find(self.model.vendor_id)
                if vendor is None:
                    raise ValueError(f'vendor of {self.model.vendor_id} is not exists! Please change_vendor(...)')
                return vendor
            else:
                if type(self.model) in [Model4LLMs.ChatGPT4o,Model4LLMs.ChatGPT4oMini]:
                    # try openai vendor
                    vs = self._store.find_all('OpenAIVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of OpenAIVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]
                    # try other vendors
                    pass
                elif type(self.model) in [Model4LLMs.Gemma2, Model4LLMs.Phi3, Model4LLMs.Llama ]:
                    # try ollama vendor
                    vs = self._store.find_all('OllamaVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of OllamaVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]
                    # try other vendors
                    pass
        
        def change_vendor(self,vendor_id:str):
            vendor:Model4LLMs.AbstractVendor = self._store.find(vendor_id)
            if vendor is None:
                raise ValueError(f'vendor of {vendor_id} is not exists! Please do add new vendor')
            self.update(vendor_id=vendor.get_id())
            return vendor
    
    class WorkFlowController(AbstractObjController):
        def __init__(self, store, model):
            super().__init__(store, model)
            self.model:Model4LLMs.WorkFlow = model
            self._store:LLMsStore = store
        
        def _read_task(self,id:str):
            """Builds the dependency graph for TopologicalSorter."""
            s = self.storage()
            return s.find(id)

        def run(self,input=None):
            """Executes tasks in the correct order, handling dependencies and results."""
            tasks = self.model.tasks
            sorter = TopologicalSorter(tasks)
            # Execute tasks based on topological order
            res = None
            for i,task_id in enumerate(sorter.static_order()):
                if i==0 and input is not None:
                    if type(input) is not list:
                        dependency_results = [input]
                    else:
                        dependency_results = input
                    self.model.results['input'] = input
                else:
                    # Gather results from dependencies to pass as arguments
                    dependency_results = [self.model.results[dep] for dep in tasks[task_id]]
                # Execute the task and store its result
                res = self.model.results[task_id] = self._read_task(task_id)(*dependency_results)
            self.model.results['final'] = res
            self.update(results=self.model.results)
            return res

class KeyOrEnv(BaseModel):
    key:str

    def get(self):
        return os.getenv(self.key,self.key)
    
class Model4LLMs:
    class AbstractObj(Model4Basic.AbstractObj):
        pass
    
    class AbstractVendor(AbstractObj):
        vendor_name: str  # e.g., 'OpenAI'
        api_url: str  # e.g., 'https://api.openai.com/v1/'
        api_key: KeyOrEnv = None # API key for authentication, if required
        timeout: int = 30  # Default timeout for API requests in seconds
        
        chat_endpoint:str = None # e.g., '/v1/chat/completions'
        models_endpoint:str = None # e.g., '/v1/models'
        
        def format_llm_model_name(self,llm_model_name:str)->str:
            return llm_model_name       
        
        def get_api_key(self)->str:
            return self.api_key.get()
            # return os.getenv(self.api_key,self.api_key)

        def get_available_models(self) -> Dict[str, Any]:
            return requests.get(self._build_url(self.models_endpoint),
                                    headers=self._build_headers(),
                                    timeout=self.timeout)

        def _build_headers(self) -> Dict[str, str]:
            headers = {
                'Content-Type': 'application/json'
            }
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.get_api_key()}'
            return headers

        def _build_url(self, endpoint: str) -> str:
            """
            Construct the full API URL for a given endpoint.
            :param endpoint: API endpoint to be called.
            :return: Full API URL.
            """
            return f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"

        def chat_request(self,payload={}):
            url=self._build_url(self.chat_endpoint)
            headers=self._build_headers()
            response = requests.post(url=url, data=json.dumps(payload,ensure_ascii=True),
                                     headers=headers,timeout=self.timeout)
            try:
                response = json.loads(response.text,strict=False)
            except Exception as e:
                if type(response) is not dict:
                    return {'error':f'{e}'}
                else:
                    return {'error':f'{response}({e})'}
            return response
        
        def chat_result(self,response)->str:
            return response
        
        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.AbstractVendorController = None
        def get_controller(self)->Controller4LLMs.AbstractVendorController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractVendorController(store,self)

    class OpenAIVendor(AbstractVendor):
        vendor_name:str = 'OpenAI'
        api_url:str = 'https://api.openai.com'
        chat_endpoint:str = '/v1/chat/completions'
        models_endpoint:str = '/v1/models'

        def get_available_models(self) -> Dict[str, Any]:
            response = super().get_available_models()
            return {model['id']: model for model in response.json().get('data', [])}
        
        def chat_result(self,response)->str:
            if not self._try_binary_error(lambda:response['choices'][0]['message']['content']):
                return self._log_error(ValueError(f'cannot get result from {response}'))
            return response['choices'][0]['message']['content']

    class AbstractLLM(AbstractObj):
        vendor_id:str='auto'
        llm_model_name:str
        context_window_tokens:int
        # Context Window Size: The context window size dictates the total number of tokens the model can handle at once. For example, if a model has a context window of 4096 tokens, it can process up to 4096 tokens of combined input and output.
        max_output_tokens:int
        stream:bool = False
        
        limit_output_tokens: Optional[int] = None
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 1.0
        frequency_penalty: Optional[float] = 0.0
        presence_penalty: Optional[float] = 0.0
        system_prompt: Optional[str] = None
        
        # # field_validator to ensure correct range of temperature
        # @field_validator('temperature')
        # def validate_temperature(cls, value):
        #     if not 0 <= value <= 1:
        #         raise ValueError("Temperature must be between 0 and 1.")
        #     return value

        # # field_validator to ensure correct range of top_p
        # @field_validator('top_p')
        # def validate_top_p(cls, value):
        #     if not 0 <= value <= 1:
        #         raise ValueError("top_p must be between 0 and 1.")
        #     return value
        
        def get_vendor(self):
            return self.get_controller().get_vendor(auto=(self.vendor_id=='auto'))

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

        def construct_payload(self, messages: List[Dict]) -> Dict[str, Any]:            
            payload = {
                "model": self.get_vendor().format_llm_model_name(self.llm_model_name),
                "stream": self.stream,
                "messages": messages,
                "max_tokens": self.limit_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }
            return {k: v for k, v in payload.items() if v is not None}
        
        def construct_messages(self,messages:Optional[List|str])->list:
            msgs = []
            if self.system_prompt:
                msgs.append({"role":"system","content":self.system_prompt})
            if type(messages) is str:
                messages = [{"role":"user","content":messages}]
            return msgs+messages

        def __call__(self, messages:Optional[List|str])->str:
            payload = self.construct_payload(self.construct_messages(messages))
            vendor = self.get_vendor()
            return vendor.chat_result(vendor.chat_request(payload))

        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.AbstractLLMController = None
        def get_controller(self)->Controller4LLMs.AbstractLLMController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractLLMController(store,self)
    
    class OpenAIChatGPT(AbstractLLM):        
        # New attributes for advanced configurations
        stop_sequences: Optional[List[str]]
        n: Optional[int]
        def construct_payload(self, messages: List[Dict]) -> Dict[str, Any]:
            payload = super().construct_payload(messages)            
            payload.update({"stop": self.stop_sequences, "n": self.n,})
            return {k: v for k, v in payload.items() if v is not None}
                    
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
        
    class ChatGPT4oMini(ChatGPT4o):
        llm_model_name:str = 'gpt-4o-mini'

    class OllamaVendor(AbstractVendor):
        vendor_name:str = 'Ollama'
        api_url:str = 'http://localhost:11434'
        chat_endpoint:str = '/api/chat'
        models_endpoint:str = '/api/tags'

        def get_available_models(self) -> Dict[str, Any]:
            response = self._try_obj_error(super().get_available_models, None)
            if response is None:
                return self._log_error(ValueError(f'cannot get available models'))
            return {model['name']: model for model in response.json().get('models', [])}

        def format_llm_model_name(self,llm_model_name:str) -> str:            
            llm_model_name = llm_model_name.lower().replace('meta-','')
            if '-' in llm_model_name:
                llm_model_name = llm_model_name.split('-')
                if not self._try_binary_error(lambda:llm_model_name[2]):
                    return self._log_error(ValueError(f'cannot parse name of {llm_model_name}'))
                llm_model_name = llm_model_name[0] + llm_model_name[1] + ':' + llm_model_name[2]
            return llm_model_name
        
        def chat_result(self,response)->str:
            if not self._try_binary_error(lambda:response['message']['content']):
                return self._log_error(ValueError(f'cannot get result from {response}'))
            return response['message']['content']
        
    class Gemma2(AbstractLLM):
        llm_model_name:str = 'gemma-2-2b'

        context_window_tokens:int = -1
        max_output_tokens:int = -1
        
        limit_output_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        frequency_penalty: Optional[float] = None
        presence_penalty: Optional[float] = None
        system_prompt: Optional[str] = None

    class Phi3(AbstractLLM):
        llm_model_name:str = 'phi-3-3.8b'

        context_window_tokens:int = -1
        max_output_tokens:int = -1
        
        limit_output_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        frequency_penalty: Optional[float] = None
        presence_penalty: Optional[float] = None
        system_prompt: Optional[str] = None

    class Llama(AbstractLLM):
        llm_model_name:str = 'llama-3.2-3b'

        context_window_tokens:int = -1
        max_output_tokens:int = -1
        
        limit_output_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        frequency_penalty: Optional[float] = None
        presence_penalty: Optional[float] = None
        system_prompt: Optional[str] = None

    ##################### utils model #########
    
    class Function(AbstractObj):

        def param_descriptions(description,**descriptions):
            def decorator(func):
                func:Model4LLMs.Function = func
                func._parameters_description = descriptions
                func._description = description
                return func
            return decorator

        class Parameter(BaseModel):
            type: str
            description: str            

        name: str = 'null'
        description: str = 'null'
        _description: str = 'null'
        # arguments: Dict[str, Any] = None
        _properties: Dict[str, Parameter] = {}
        parameters: Dict[str, Any] = {"type": "object",'properties':_properties}
        required: list[str] = []        
        _parameters_description: Dict[str, str] = {}
        _string_arguments: str='\{\}'

        def __init__(self, *args, **kwargs):
            # super(self.__class__, self).__init__(*args, **kwargs)
            super().__init__(*args, **kwargs)
            self._extract_signature()

        def _extract_signature(self):
            self.name=self.__class__.__name__
            sig = inspect.signature(self.__call__)
            # try:
            #     self.__call__()
            # except Exception as e:
            #     pass
            # Map Python types to more generic strings
            type_map = {
                int: "integer",float: "number",
                str: "string",bool: "boolean",
                list: "array",dict: "object"
                # ... add more mappings if needed
            }
            self.required = []
            for name, param in sig.parameters.items():
                param_type = type_map.get(param.annotation, "object")
                self._properties[name] = Model4LLMs.Function.Parameter(
                    type=param_type, description=self._parameters_description.get(name,''))
                if param.default is inspect._empty:
                    self.required.append(name)
            self.parameters['properties']=self._properties
            self.description = self._description

        def __call__(self):
            print('this is root class , not implement')
        
        def get_description(self):
            return self.model_dump()#exclude=['arguments'])

        _controller: Controller4LLMs.AbstractObjController = None
        def get_controller(self)->Controller4LLMs.AbstractObjController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractObjController(store,self)

    class WorkFlow(AbstractObj):
        tasks: Dict[str, list[str]] # using uuids, task and dependencies
        # tasks = {
        #     "task1_uuid": ["task2_uuid", "task3_uuid"],  # task1 depends on task2 and task3
        #     "task2_uuid": ["task4_uuid"],           # task2 depends on task4
        #     "task3_uuid": [],                  # task3 has no dependencies
        #     "task4_uuid": []                   # task4 has no dependencies
        # }
        results: Dict[str, Any] = {}

        def __call__(self, *args, **kwargs):
            self.get_controller().run(*args, **kwargs)

        def get_result(self, task_uuid: str) -> Any:
            """Returns the result of a specified task."""
            return self.results.get(task_uuid, None)
        
        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.WorkFlowController = None
        def get_controller(self)->Controller4LLMs.WorkFlowController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.WorkFlowController(store,self)

    @Function.param_descriptions('Makes an HTTP request using the configured method, url, and headers, and the provided params, data, or json.',
                                params='query parameters',
                                data='form data',
                                json='JSON payload')
    class RequestsFunction(Function):
        method: str = 'GET'
        url: str
        headers: Dict[str, str] = {}

        def __call__(self,params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     json: Optional[Dict[str, Any]] = None ) -> Dict[str, Any]:
            try:
                response = requests.request(
                    method=self.method,url=self.url,
                    headers=self.headers,params=params,
                    data=data,json=json
                )
                response.raise_for_status()
                try:
                    return response.json()
                except Exception as e:
                    return {'text':response.text}
            except requests.exceptions.RequestException as e:
                return {"error": str(e), "status": getattr(e.response, "status_code", None)}

                # # Example usage:
                # request_function = RequestsFunction(
                #     method="POST",
                #     url="https://api.example.com/data",
                #     headers={"Authorization": "Bearer YOUR_TOKEN"}
                # )

                # result = request_function(json={"key": "value"})

                # if "error" in result:
                #     print(f"Error: {result['error']}")
                # else:
                #     print(f"Success: {result['data']}")
    @Function.param_descriptions('Makes an HTTP request to async Celery REST api.',
                                params='query parameters',
                                data='form data',
                                json='JSON payload')
    class AsyncCeleryWebApiFunction(Function):
        method: str = 'GET'
        url: str
        headers: Dict[str, str] = {}
        task_status_url: str = 'http://127.0.0.1:8000/tasks/status/{task_id}'

        def __call__(self,params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     json: Optional[Dict[str, Any]] = None ) -> Dict[str, Any]:
            try:
                response = requests.request(
                    method=self.method,url=self.url,
                    headers=self.headers,params=params,
                    data=data,json=json
                )
                response.raise_for_status()
                # get task id
                for k,v in response.json().items():
                    if 'id' in k:
                        task_id = v
                
                while True:
                    response = requests.request(
                        method='GET',
                        url= self.task_status_url.format(task_id=task_id),
                        headers=self.headers,params=params,
                        data=data,json=json
                    )
                    response.raise_for_status()
                    if response.json()['status'] in ['SUCCESS','FAILURE','REVOKED']:
                        break
                    time.sleep(1)
                try:
                    return response.json()
                except Exception as e:
                    return {'text':response.text}
            except requests.exceptions.RequestException as e:
                return {"error": str(e), "status": getattr(e.response, "status_code", None)}
            
class LLMsStore(BasicStore):
    MODEL_CLASS_GROUP = Model4LLMs   

    def _get_class(self, id: str, modelclass=MODEL_CLASS_GROUP):
        class_type = id.split(':')[0]
        res = {c.__name__:c for c in [i for k,i in modelclass.__dict__.items() if '_' not in k]}
        res = res.get(class_type, None)
        if res is None: raise ValueError(f'No such class of {class_type}')
        return res
    
    def add_new_openai_vendor(self,api_key: str,
                              api_url: str='https://api.openai.com',
                              timeout: int=30) -> MODEL_CLASS_GROUP.OpenAIVendor:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.OpenAIVendor(api_url=api_url,api_key=KeyOrEnv(key=api_key),timeout=timeout))
    
    def add_new_ollama_vendor(self,api_url: str='http://localhost:11434',
                              timeout: int=30) -> MODEL_CLASS_GROUP.OllamaVendor:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.OllamaVendor(api_url=api_url,api_key=KeyOrEnv(key=''),timeout=timeout))
    
    def add_new_chatgpt4o(self,vendor_id:str,
                                limit_output_tokens:int = 1024,
                                temperature:float = 0.7,
                                top_p:float = 1.0,
                                frequency_penalty:float = 0.0,
                                presence_penalty:float = 0.0,
                                system_prompt:str = None ) -> MODEL_CLASS_GROUP.ChatGPT4o:
        
        return self.add_new_obj(self.MODEL_CLASS_GROUP.ChatGPT4o(vendor_id=vendor_id,
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
                                system_prompt:str = None ) -> MODEL_CLASS_GROUP.ChatGPT4oMini:
        
        return self.add_new_obj(self.MODEL_CLASS_GROUP.ChatGPT4oMini(vendor_id=vendor_id,
                                limit_output_tokens=limit_output_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                system_prompt=system_prompt,))
    
    def add_new_gemma2(self,vendor_id:str,system_prompt:str = None) -> MODEL_CLASS_GROUP.Gemma2:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.Gemma2(vendor_id=vendor_id,system_prompt=system_prompt))
    
    def add_new_phi3(self,vendor_id:str,system_prompt:str = None) -> MODEL_CLASS_GROUP.Phi3:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.Phi3(vendor_id=vendor_id,system_prompt=system_prompt))
    
    def add_new_llama(self,vendor_id:str,system_prompt:str = None) -> MODEL_CLASS_GROUP.Llama:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.Llama(vendor_id=vendor_id,system_prompt=system_prompt))
    
    def add_new_function(self, function_obj:MODEL_CLASS_GROUP.Function)->MODEL_CLASS_GROUP.Function:
        return self.add_new_obj(function_obj)
    
    def add_new_request(self, url:str, method='GET', headers={})->MODEL_CLASS_GROUP.RequestsFunction:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.RequestsFunction(method=method,url=url,headers=headers))
    
    def add_new_celery_request(self, url:str, method='GET', headers={},
                               task_status_url: str = 'http://127.0.0.1:8000/tasks/status/{task_id}'
                               )->MODEL_CLASS_GROUP.AsyncCeleryWebApiFunction:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.AsyncCeleryWebApiFunction(method=method,url=url,
                                                        headers=headers,task_status_url=task_status_url))
    
    def add_new_workflow(self, tasks:Optional[Dict[str,list[str]]|list[str]], metadata={})->MODEL_CLASS_GROUP.WorkFlow:
        if type(tasks) is list:
            tasks = tasks[::-1]
            ds    = [[t] for t in tasks[1:]] + [[]]
            tasks = {t:d for t,d in zip(tasks,ds)}
        return self.add_new_obj(Model4LLMs.WorkFlow(tasks=tasks,metadata=metadata))
    
    def find_function(self,function_id:str) -> MODEL_CLASS_GROUP.Function:
        return self.find(function_id)
    
    def find_all_vendors(self)->list[MODEL_CLASS_GROUP.AbstractVendor]:
        return self.find_all('*Vendor:*')

    @staticmethod    
    def chain_dumps(cl:list[Model4Basic.AbstractObj]):
        res = {}
        for i in list(map(lambda x:{x.get_id():json.loads(x.model_dump_json())},cl)):
            res.update(i)
        return json.dumps(res)

    def chain_loads(self,cl_json):
        data:dict = json.loads(cl_json)
        ks = data.keys()
        tmp_store = LLMsStore()
        tmp_store.loads(cl_json)
        vs = [tmp_store.find(k) for k in ks]
        for v in vs:v._id=None
        return [self.add_new_obj(v) for v in vs]

class Tests(unittest.TestCase):
    def __init__(self,*args,**kwargs)->None:
        super().__init__(*args,**kwargs)
        self.store = LLMsStore()

    def test_all(self,num=1):        
        for i in range(num):
            self.test_openai()
            self.test_ollama()
        
        print(self.store.dumps())
        
    def test_ollama(self):
        self.test_ollama_1()
        self.test_ollama_2()
        self.test_ollama_3()

    def test_openai(self):
        self.test_openai_1()
        self.test_openai_2()
        self.test_openai_3()

    def test_openai_1(self):
        v = self.store.add_new_openai_vendor(api_key=KeyOrEnv(key='OPENAI_API_KEY'))
        print(v.get_available_models())

    def test_openai_2(self):
        v = self.store.find_all('OpenAIVendor:*')[0]
        c = self.store.add_new_chatgpt4omini(vendor_id=v.get_id())
        print(c)
    
    def test_openai_3(self):
        c:Model4LLMs.ChatGPT4oMini = self.store.find_all('ChatGPT4oMini:*')[0]
        print(c('What is your name?'))
    
    def test_ollama_1(self):
        v = self.store.add_new_ollama_vendor()
        print(v.get_available_models())

    def test_ollama_2(self):
        v = self.store.find_all('OllamaVendor:*')[0]
        c = self.store.add_new_gemma2(vendor_id=v.get_id())
        print(c('What is your name?'))

    def test_ollama_3(self):
        v = self.store.find_all('OllamaVendor:*')[0]
        c = self.store.add_new_llama(vendor_id=v.get_id())
        print(c('What is your name?'))