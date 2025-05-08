from graphlib import TopologicalSorter
import inspect
import json
import math
import os
import time
import unittest
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
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
                if 'ChatGPT' in self.model.__class__.__name__:
                    # try openai vendor
                    vs = self._store.find_all('OpenAIVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of OpenAIVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]
                    
                elif 'Claude' in self.model.__class__.__name__:
                    vs = self._store.find_all('AnthropicVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of AnthropicVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]

                elif 'Grok' in self.model.__class__.__name__:
                    vs = self._store.find_all('XaiVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of XaiVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]
                    
                elif 'DeepSeek' in self.model.__class__.__name__:
                    vs = self._store.find_all('DeepSeekVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of DeepSeekVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]
                    
                elif type(self.model) in [Model4LLMs.Gemma2, Model4LLMs.Phi3, Model4LLMs.Llama ]:
                    # try ollama vendor
                    vs = self._store.find_all('OllamaVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of OllamaVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]
                    
            raise ValueError(f'not support vendor of {self.model.vendor_id}')
        
        def change_vendor(self,vendor_id:str):
            vendor:Model4LLMs.AbstractVendor = self._store.find(vendor_id)
            if vendor is None:
                raise ValueError(f'vendor of {vendor_id} is not exists! Please do add new vendor')
            self.update(vendor_id=vendor.get_id())
            return vendor
        
    class AbstractEmbeddingController(AbstractObjController):
        def __init__(self, store, model):
            super().__init__(store, model)
            self.model:Model4LLMs.AbstractEmbedding = model
            self._store:LLMsStore = store
        
        def get_vendor(self,auto=False):
            if not auto:
                vendor:Model4LLMs.AbstractVendor = self._store.find(self.model.vendor_id)
                if vendor is None:
                    raise ValueError(f'vendor of {self.model.vendor_id} is not exists! Please change_vendor(...)')
                return vendor
            else:
                if type(self.model) in [Model4LLMs.TextEmbedding3Small]:
                    # try openai vendor
                    vs:list[Model4LLMs.AbstractVendor] = self._store.find_all('OpenAIVendor:*')
                    if len(vs)==0:raise ValueError(f'auto get endor of OpenAIVendor:* is not exists! Please (add and) change_vendor(...)')
                    else: return vs[0]
                    # try other vendors
                    pass
            raise ValueError(f'not support vendor of {self.model.vendor_id}')
    class WorkFlowController(AbstractObjController):
        def __init__(self, store, model):
            super().__init__(store, model)
            self.model: Model4LLMs.WorkFlow = model
            self._store: LLMsStore = store
            
        def _read_task(self, task_id: str):
            """Fetches a task from storage."""
            return self.storage().find(task_id)

        def run(self, **kwargs: Dict[str, Any]) -> Any:
            """Executes tasks in the correct order, handling dependencies and results."""
            tasks: Dict[str, List[str]] = self.model.tasks
            sorter = TopologicalSorter(tasks)
            
            # Reset results if 'final' is in them
            if 'final' in self.model.results:
                self.model.results.clear()
                                    
            # Initialize results from kwargs
            self.model.results.update({
                task_id: result for task_id, result in kwargs.items() if result is not None})
            
            # Execute tasks in topological order
            result = None
            todo_list = list(sorter.static_order())
            for task_id in todo_list:
                if task_id in self.model.results: continue
                # Gather results from dependencies to pass as arguments
                dependency_results = [self.model.results[dep] for dep in tasks[task_id]]
                all_args, all_kwargs = self._extract_args_kwargs(dependency_results)

                # Execute the task and store its result
                try:
                    if task_id != 'final':
                        result = self._read_task(task_id)(*all_args, **all_kwargs)
                    else:
                        result = [all_args, all_kwargs]
                        
                    self.model.results[task_id] = result
                except Exception as e:
                    raise ValueError(f'[WorkFlow]: Error at {task_id}: {e}')
            
            # Set the final result
            if result is not None:
                self.model.results['final'] = result
            
            self.update(results=self.model.results)
            return result

        def _extract_args_kwargs(self, dependency_results: List[Any]):
            """Extracts positional arguments and keyword arguments from dependency results."""
            all_args = []
            all_kwargs = {}
            for args_kwargs in dependency_results:
                if isinstance(args_kwargs, list) and len(args_kwargs) == 2 and isinstance(args_kwargs[1], dict):
                    args, kwargs = args_kwargs
                    if isinstance(args, (list, tuple)):
                        all_args.extend(args)
                    else:
                        all_args.append(args)
                    all_kwargs.update(kwargs)
                else:
                    all_args.append(args_kwargs)
            return all_args, all_kwargs
        
# class KeyOrEnv(BaseModel):
#     key:str

#     def get(self):
#         return os.getenv(self.key,self.key)
    
class Model4LLMs:
    class AbstractObj(Model4Basic.AbstractObj):
        pass
    
    from .ModelInterface import AbstractVendor
    class AbstractVendor(AbstractVendor,AbstractObj):
        _controller: Controller4LLMs.AbstractVendorController = None
        def get_controller(self)->Controller4LLMs.AbstractVendorController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractVendorController(store,self)        

    from .ModelInterface import AbstractLLM
    class AbstractLLM(AbstractLLM,AbstractObj):
        _controller: Controller4LLMs.AbstractLLMController = None
        def get_controller(self)->Controller4LLMs.AbstractLLMController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractLLMController(store,self)

    from .ModelInterface import OpenAIVendor
    class OpenAIVendor(OpenAIVendor, AbstractVendor, AbstractObj):
        pass

    from.ModelInterface import OpenAIChatGPT
    class OpenAIChatGPT(OpenAIChatGPT,AbstractLLM):
        pass

    class ChatGPT4o(OpenAIChatGPT):
        llm_model_name:str = 'gpt-4o'
        context_window_tokens:int = 128000
        max_output_tokens:int = 4096
        
    class ChatGPT4oMini(ChatGPT4o):
        llm_model_name:str = 'gpt-4o-mini'

    class ChatGPT41(OpenAIChatGPT):
        llm_model_name:str = 'gpt-4.1'
        context_window_tokens:int = 1047576
        max_output_tokens:int = 32768
        
    class ChatGPT41Mini(ChatGPT41):
        llm_model_name:str = 'gpt-4.1-mini'
        
    class ChatGPT41Nano(ChatGPT41):
        llm_model_name:str = 'gpt-4.1-nano'
        
    class ChatGPTO3(OpenAIChatGPT):
        limit_output_tokens: Optional[int] = 2048
        llm_model_name: str = 'o3'
        context_window_tokens: int = 128000
        max_output_tokens: int = 32768
        temperature: Optional[float] = 1.0

        def construct_payload(self, messages: List[Dict]) -> Dict[str, Any]:            
            return self.openai_o_models_construct_payload(messages)
            
        def construct_messages(self,messages:Optional[List|str])->list:
            return self.openai_o_models_construct_messages(messages)
        
    class ChatGPTO3Mini(ChatGPTO3):
        llm_model_name: str = 'o3-mini'
        context_window_tokens: int = 1047576
        max_output_tokens: int = 32768
        
    class DeepSeekVendor(OpenAIVendor):
        vendor_name: str = "DeepSeek"
        api_url: str = "https://api.deepseek.com"
        chat_endpoint: str = "/v1/chat/completions"
        models_endpoint: str = "/v1/models"
        rate_limit: Optional[int] = None  # Example rate limit for xAI
        
    class DeepSeek(OpenAIChatGPT):
        llm_model_name:str = 'deepseek-chat'
        context_window_tokens:int = 64000
        max_output_tokens:int = 4096*2
        
    class XaiVendor(OpenAIVendor):
        vendor_name: str = "xAI"
        api_url: str = "https://api.x.ai"
        chat_endpoint: str = "/v1/chat/completions"
        models_endpoint: str = "/v1/models"
        embeddings_endpoint: str = "/v1/embeddings"
        rate_limit: Optional[int] = None  # Example rate limit for xAI
        
    class Grok(OpenAIChatGPT):
        llm_model_name:str = 'grok-beta'
        context_window_tokens:int = 128000
        max_output_tokens:int = 4096
        
    class OllamaVendor(AbstractVendor):
        vendor_name:str = 'Ollama'
        api_url:str = 'http://localhost:11434'
        chat_endpoint:str = '/api/chat'
        models_endpoint:str = '/api/tags'
        embeddings_endpoint:str = 'NULL'

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

    class Phi3(AbstractLLM):
        llm_model_name:str = 'phi-3-3.8b'

    class Llama(AbstractLLM):
        llm_model_name:str = 'llama-3.2-3b'
    
    class AnthropicVendor(AbstractVendor):
        vendor_name: str = "Anthropic"
        api_url: str = "https://api.anthropic.com"
        chat_endpoint: str = "/v1/messages"
        models_endpoint: str = "/v1/models"
        embeddings_endpoint: str = "NULL"
        default_timeout: int = 30
        rate_limit: Optional[int] = None

        def chat_result(self, response) -> str:
            if "content" in response:
                return response["content"][0]["text"]
            return self._log_error(ValueError(f'cannot get result from {response}'))

        def _build_headers(self) -> Dict[str, str]:
            headers = super()._build_headers()
            headers["anthropic-version"] = "2023-06-01"
            headers["x-api-key"] = self.get_api_key()
            headers.pop("Authorization", None)  # Remove Bearer token
            return headers
    
    class Claude(OpenAIChatGPT):
        context_window_tokens: int = 200_000
        max_output_tokens: int = 4096
        
        def construct_payload(self, messages: List[Dict]) -> Dict[str, Any]:   
            return self.claude_construct_payload(messages)
        
        def construct_messages(self, messages: Optional[List | str]) -> list:
            return self.claude_construct_messages(messages)

    class Claude35(Claude):
        llm_model_name: str = "claude-3-5-sonnet-latest"
        max_output_tokens: int = 8192

    class Claude37(Claude):
        llm_model_name: str = "claude-3-7-sonnet-latest"
        max_output_tokens: int = 64000

    ##################### embedding model #####
    class AbstractEmbedding(AbstractObj):
        vendor_id: str = 'auto'                # Vendor identifier (e.g., OpenAI, Google)
        embedding_model_name: str              # Model name (e.g., "text-embedding-3-small")
        embedding_dim: int                     # Dimensionality of the embeddings, e.g., 768 or 1024
        normalize_embeddings: bool = True      # Whether to normalize the embeddings to unit vectors
        
        max_input_length: Optional[int] = None     # Optional limit on input length (e.g., max tokens or chars)
        pooling_strategy: Optional[str] = 'mean'   # Pooling strategy if working with sentence embeddings (e.g., "mean", "max")
        distance_metric:  Optional[str] = 'cosine' # Metric for comparing embeddings ("cosine", "euclidean", etc.)
        
        cache_embeddings: bool = False         # Option to cache embeddings to improve efficiency
        cache: Optional[dict[str,List[float]]] = None
        embedding_context: Optional[str] = None # Optional context or description to customize embedding generation
        additional_features: Optional[List[str]] = None  # Additional features for embeddings, e.g., "entity", "syntax"
        
        def __call__(self, input_text: str) -> List[float]:
            return self.generate_embedding(input_text)

        def get_vendor(self):
            return self.get_controller().get_vendor(auto=(self.vendor_id=='auto'))
        
        def generate_embedding(self, input_text: str) -> List[float]:
            raise NotImplementedError("This method should be implemented by subclasses.")
        
        def similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
            raise NotImplementedError("This method should be implemented by subclasses.")

        model_config = ConfigDict(arbitrary_types_allowed=True)    
        _controller: Controller4LLMs.AbstractEmbeddingController = None
        def get_controller(self)->Controller4LLMs.AbstractEmbeddingController: return self._controller
        def init_controller(self,store):self._controller = Controller4LLMs.AbstractEmbeddingController(store,self)
    
    class TextEmbedding3Small(AbstractEmbedding):
        vendor_id: str = "auto"
        embedding_model_name: str = "text-embedding-3-small"
        embedding_dim: int = 1536  # As specified by OpenAI's "text-embedding-3-small" model
        normalize_embeddings: bool = True
        max_input_length: int = 8192  # Default max token length for text-embedding-3-small
        
        def generate_embedding(self, input_text: str) -> List[float]:
            # Check for cached result
            if self.cache_embeddings and input_text in self.cache:
                return self.cache[input_text]
            
            # Generate embedding using OpenAI API
            embedding = self.get_vendor().get_embedding(input_text, model=self.embedding_model_name)
            
            # Normalize if specified
            if self.normalize_embeddings:
                embedding = self._normalize_embedding(embedding)
            
            # Cache result if caching is enabled
            if self.cache_embeddings:
                self.cache[input_text] = embedding
            
            return embedding

        def similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
            if self.distance_metric == "cosine":
                return self._cosine_similarity(embedding1, embedding2)
            elif self.distance_metric == "euclidean":
                return self._euclidean_distance(embedding1, embedding2)
            else:
                raise ValueError("Unsupported distance metric. Choose 'cosine' or 'euclidean'.")

        def _normalize_embedding(self, embedding: List[float]) -> List[float]:
            norm = math.sqrt(sum(x * x for x in embedding))
            return [x / norm for x in embedding] if norm != 0 else embedding

        def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
            dot_product = sum(x * y for x, y in zip(embedding1, embedding2))
            norm1 = math.sqrt(sum(x * x for x in embedding1))
            norm2 = math.sqrt(sum(y * y for y in embedding2))
            return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0

        def _euclidean_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(embedding1, embedding2)))

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
            raise ValueError('this is root class , not implement')
        
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
            if '__input__' in  self.tasks:
                del  self.tasks['__input__']

            # if a sequential task input
            if len(args)!=0:
                first_task_id = self.todo_list()[0]
                if first_task_id != '__input__':
                    first_task_deps = self.tasks[first_task_id]
                    if '__input__' not in first_task_deps:
                        self.tasks[first_task_id].append('__input__')
                kwargs['__input__'] = [args,{}]
            return self.get_controller().run(**kwargs)

        
        def todo_list(self):
            return list(TopologicalSorter(self.tasks).static_order())
        
        def find_dependency_results(self,task_id):
            return [self.results[dep] for dep in self.tasks[task_id]]

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
                     json: Optional[Dict[str, Any]] = None,
                     debug=False,
                     debug_data=None) -> Dict[str, Any]:
            try:
                if debug: return debug_data
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
                     json: Optional[Dict[str, Any]] = None,
                     debug=False,
                     debug_data=None) -> Dict[str, Any]:
            try:
                if debug: return debug_data
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
                    if response is None:
                        break
                    response.raise_for_status()
                    if response.json() is None:
                        break
                    if response.json()['status'] in ['SUCCESS','FAILURE','REVOKED']:
                        if response.json()['status'] == 'FAILURE':
                            raise ValueError(f'{response.json()}')
                        break
                    time.sleep(1)

                try:
                    return response.json()
                except Exception as e:
                    raise ValueError(f'"text":{response.text}')
            except requests.exceptions.RequestException as e:
                raise ValueError(f'"error": {e} "status": {getattr(e.response, "status_code", None)}')
            
class LLMsStore(BasicStore):
    MODEL_CLASS_GROUP = Model4LLMs   

    def _get_class(self, id: str, modelclass=MODEL_CLASS_GROUP):
        class_type = id.split(':')[0]
        res = {c.__name__:c for c in [i for k,i in modelclass.__dict__.items() if '_' not in k]}
        res = res.get(class_type, None)
        if res is None: raise ValueError(f'No such class of {class_type}')
        return res
    
    def add_new_vendor(self,vendor_class_type=MODEL_CLASS_GROUP.OpenAIVendor):
        def add_vendor(api_key: str='',
                       api_url: str='https://api.openai.com',
                       timeout: int=30)->Model4LLMs.AbstractVendor:
            return self.add_new(vendor_class_type)(api_key=api_key,api_url=api_url,timeout=timeout)
        return add_vendor
    
    def add_new_llm(self,llm_class_type=MODEL_CLASS_GROUP.AbstractLLM):
        def add_llm(vendor_id:str,
                    limit_output_tokens:int = 1024,
                    temperature:float = 0.7,
                    top_p:float = 1.0,
                    frequency_penalty:float = 0.0,
                    presence_penalty:float = 0.0,
                    system_prompt:str = None,
                    id:str=None)->Model4LLMs.AbstractLLM:
            return self.add_new(llm_class_type)(vendor_id=vendor_id,
                        limit_output_tokens=limit_output_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        system_prompt=system_prompt,
                        id=id)
        return add_llm
    
    def add_new_function(self, function_obj:MODEL_CLASS_GROUP.Function, id:str=None)->MODEL_CLASS_GROUP.Function:
        return self.add_new_obj(function_obj,id=id)
    
    def add_new_request(self, url:str, method='GET', headers={}, id:str=None)->MODEL_CLASS_GROUP.RequestsFunction:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.RequestsFunction(method=method,url=url,headers=headers),id=id)
    
    def add_new_celery_request(self, url:str, method='GET', headers={},
                               task_status_url: str = 'http://127.0.0.1:8000/tasks/status/{task_id}', id:str=None
                               )->MODEL_CLASS_GROUP.AsyncCeleryWebApiFunction:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.AsyncCeleryWebApiFunction(method=method,url=url,
                                                        headers=headers,task_status_url=task_status_url),id=id)
    
    def add_new_workflow(self, tasks:Optional[Dict[str,list[str]]|list[str]], metadata={}, id:str=None)->MODEL_CLASS_GROUP.WorkFlow:
        if type(tasks) is list:
            tasks = tasks[::-1]
            ds    = [[t] for t in tasks[1:]] + [[]]
            tasks = {t:d for t,d in zip(tasks,ds)}
        return self.add_new_obj(Model4LLMs.WorkFlow(tasks=tasks,metadata=metadata),id=id)
    
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
        v = self.store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')
        print(v.get_available_models())

    def test_openai_2(self):
        v = self.store.find_all('OpenAIVendor:*')[0]
        c = self.store.add_new_llm(Model4LLMs.ChatGPT4oMini)(vendor_id='auto')
        print(c)
    
    def test_openai_3(self):
        c:Model4LLMs.ChatGPT4oMini = self.store.find_all('ChatGPT4oMini:*')[0]
        print(c('What is your name?'))
    
    def test_ollama_1(self):
        v = self.store.add_new_vendor(Model4LLMs.OllamaVendor)()
        print(v.get_available_models())

    def test_ollama_2(self):
        v = self.store.find_all('OllamaVendor:*')[0]
        c = self.store.add_new_llm(Model4LLMs.Gemma2)(vendor_id=v.get_id())
        print(c('What is your name?'))

    def test_ollama_3(self):
        v = self.store.find_all('OllamaVendor:*')[0]
        c = self.store.add_new_llm(Model4LLMs.Llama)(vendor_id=v.get_id())
        print(c('What is your name?'))