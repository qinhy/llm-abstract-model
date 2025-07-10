from graphlib import TopologicalSorter
import inspect
import json
import math
import os
import time
import unittest
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model
import requests
from typing import Dict, Any

from .BasicModel import Controller4Basic, Model4Basic, BasicStore
from .ModelInterface import AbstractVendor
from .ModelInterface import AbstractLLM
from .ModelInterface import OpenAIVendor
from .ModelInterface import AbstractGPTModel
from .ModelInterface import AbstractEmbedding    
from .MermaidWorkflowEngine import GraphNode, MermaidWorkflowFunction as MWFFunction
from .MermaidWorkflowEngine import MermaidWorkflowEngine

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
        def get_vendor(self, auto=False):
            if not auto:
                vendor = self._store.find(self.model.vendor_id)
                if vendor is None:
                    raise ValueError(f'Vendor {self.model.vendor_id} does not exist. Please use change_vendor(...)')
                return vendor
            # Mapping of model types to vendor types
            vendor_mapping = {
                'ChatGPT': 'OpenAIVendor:*',
                'Claude': 'AnthropicVendor:*', 
                'Grok': 'XaiVendor:*',
                'DeepSeek': 'DeepSeekVendor:*'
            }
            # Check model name against mapping
            for model_type, vendor_type in vendor_mapping.items():
                if model_type in self.model.__class__.__name__:
                    vendors = self._store.find_all(vendor_type)
                    if not vendors:
                        raise ValueError(f'No {vendor_type} vendor found. Please add and set vendor.')
                    return vendors[0]
            # Special case for Ollama models
            ollama_models = [Model4LLMs.Gemma2, Model4LLMs.Phi3, Model4LLMs.Llama]
            if type(self.model) in ollama_models:
                vendors = self._store.find_all('OllamaVendor:*')
                if not vendors:
                    raise ValueError('No OllamaVendor found. Please add and set vendor.')
                return vendors[0]
            raise ValueError(f'Unsupported vendor type for model {self.model.vendor_id}')
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

    class OpenAIVendorController(AbstractVendorController): pass
    class AbstractChatGPTController(AbstractLLMController): pass
    class ChatGPT4oController(AbstractLLMController): pass
    class ChatGPT4oMiniController(AbstractLLMController): pass
    class ChatGPT41Controller(AbstractLLMController): pass
    class ChatGPT41MiniController(AbstractLLMController): pass
    class ChatGPT41NanoController(AbstractLLMController): pass
    class ChatGPT45Controller(AbstractLLMController): pass
    class ChatGPTO3Controller(AbstractLLMController): pass
    class ChatGPTO3MiniController(AbstractLLMController): pass
    class DeepSeekVendorController(AbstractVendorController): pass
    class DeepSeekController(AbstractLLMController): pass
    class XaiVendorController(AbstractVendorController): pass
    class GrokController(AbstractLLMController): pass
    class OllamaVendorController(AbstractVendorController): pass
    class Gemma2Controller(AbstractLLMController): pass
    class Phi3Controller(AbstractLLMController): pass
    class LlamaController(AbstractLLMController): pass
    class AnthropicVendorController(AbstractVendorController): pass
    class AbstractClaudeController(AbstractLLMController): pass
    class Claude35Controller(AbstractLLMController): pass
    class Claude37Controller(AbstractLLMController): pass
    class TextEmbedding3SmallController(AbstractEmbeddingController): pass
    class FunctionController(AbstractObjController): pass
    class ParameterController(AbstractObjController): pass
    class RequestsFunctionController(AbstractObjController): pass
    class AsyncCeleryWebApiFunctionController(AbstractObjController): pass
    class RegxExtractorController(AbstractObjController): pass
    class StringTemplateController(AbstractObjController): pass
    class ClassificationTemplateController(AbstractObjController): pass
    class MermaidWorkflowFunctionController(AbstractObjController): pass
    class MermaidWorkflowController(AbstractObjController): pass
    
class Model4LLMs:
    class AbstractObj(Model4Basic.AbstractObj):
        
        def _get_controller_class(self,modelclass=Controller4LLMs):
            class_type = self.__class__.__name__+'Controller'
            res = {c.__name__:c for c in [i for k,i in modelclass.__dict__.items() if '_' not in k]}
            res = res.get(class_type, None)
            if res is None: 
                print(f'[warning]: No such class of {class_type}, use Controller4LLMs.AbstractObjController')
                res = Controller4LLMs.AbstractObjController
            return res
    
    class AbstractVendor(AbstractVendor,AbstractObj):
        controller: Optional[Controller4LLMs.AbstractVendorController] = None

    class AbstractLLM(AbstractGPTModel,AbstractLLM,AbstractObj):
        controller: Optional[Controller4LLMs.AbstractLLMController] = None
        def get_vendor(self)->AbstractVendor:
            return self.controller.get_vendor(auto=(self.vendor_id=='auto'))

        def set_mcp_tools(self, mcp_tools_json = '{}'):            
            super().set_mcp_tools(mcp_tools_json)
            self.controller.update(mcp_tools=self.mcp_tools)

    class OpenAIVendor(OpenAIVendor, AbstractVendor, AbstractObj):
        pass

    class AbstractChatGPT(AbstractLLM):
        def get_tools(self) -> List[Dict[str, Any]]:
            if not self.mcp_tools:return []            
            return [t.to_openai_tool() for t in self.mcp_tools]         
        def construct_payload(self, messages):
            return self.openai_construct_payload(messages)

    class ChatGPT4o(AbstractChatGPT):
        llm_model_name:str = 'gpt-4o'
        context_window_tokens:int = 128000
        max_output_tokens:int = 4096
        
    class ChatGPT4oMini(ChatGPT4o):
        llm_model_name:str = 'gpt-4o-mini'

    class ChatGPT41(AbstractChatGPT):
        llm_model_name:str = 'gpt-4.1'
        context_window_tokens:int = 1047576
        max_output_tokens:int = 32768
        
    class ChatGPT41Mini(ChatGPT41):
        llm_model_name:str = 'gpt-4.1-mini'
        
    class ChatGPT41Nano(ChatGPT41):
        llm_model_name:str = 'gpt-4.1-nano'

    class ChatGPT45(AbstractChatGPT):
        llm_model_name:str = 'gpt-4.5'
        context_window_tokens:int = 128000
        max_output_tokens:int = 128000

    class ChatGPTO3(AbstractChatGPT):
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
        
    class DeepSeek(AbstractLLM):
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
        
    class Grok(AbstractLLM):
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

        def format_llm_model_name(self,llm_model_name:str) -> str:            
            return llm_model_name.lower().replace('.','-')

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
    
    class AbstractClaude(AbstractLLM):
        context_window_tokens: int = 200_000
        max_output_tokens: int = 4096
        
        def construct_payload(self, messages: List[Dict]) -> Dict[str, Any]:   
            return self.claude_construct_payload(messages)
        
        def construct_messages(self, messages: Optional[List | str]) -> list:
            return self.claude_construct_messages(messages)

    class Claude35(AbstractClaude):
        llm_model_name: str = "claude-3.5-sonnet-latest"
        max_output_tokens: int = 8192

    class Claude37(AbstractClaude):
        llm_model_name: str = "claude-3.7-sonnet-latest"
        max_output_tokens: int = 64000

    ##################### embedding model #####
    
    class AbstractEmbedding(AbstractEmbedding, AbstractObj):
        
        def generate_embedding(self, input_text: str) -> np.ndarray:
            """Generate embedding for the given text."""
            if not input_text:
                raise ValueError("Input text cannot be empty")

            if self.max_input_length and len(input_text) > self.max_input_length:
                input_text = input_text[:self.max_input_length]
            
            vendor:AbstractVendor = self.controller.get_vendor(auto=(self.vendor_id=='auto'))

            # Generate embedding
            embedding = np.array(vendor.get_embedding(
                input_text, 
                model=self.embedding_model_name
            ))

            # Normalize if specified
            if self.normalize_embeddings:
                embedding = self._normalize_embedding(embedding)

            return embedding

        controller: Optional[Controller4LLMs.AbstractEmbeddingController] = None
    
    class TextEmbedding3Small(AbstractEmbedding, AbstractObj):
        vendor_id: str = "auto"
        embedding_model_name: str = "text-embedding-3-small"
        embedding_dim: int = 1536  # As specified by OpenAI's "text-embedding-3-small" model
        normalize_embeddings: bool = True
        max_input_length: int = 8192  # Default max token length for text-embedding-3-small
        
    ##################### utils model #########
    class MermaidWorkflowFunction(MWFFunction, AbstractObj):
        description: str = Field(..., description="description of this function.")        
        controller: Optional[Controller4LLMs.MermaidWorkflowFunctionController] = None

    class MermaidWorkflow(MermaidWorkflowEngine, AbstractObj):
        results: Dict[str, dict] = {}
        builds: str = r'{}'

        # def __call__(self, *args, **kwargs):
        #     return self.run(*args, **kwargs)['final']

        def build(self, **kwargs):
            self.controller.update(builds=json.dumps(kwargs))
            fields = kwargs.keys()
            arg_str = ', '.join(fields)
            format_args = ', '.join([f"{key}= {repr(value)}" for key, value in kwargs.items()])

            # Dynamically generate a callable method with correct argument signature
            func_code = f"""
from typing import Any, Dict
def __call__(self, {arg_str})->Dict[str, Any]:
    res = self.run({format_args})
    return res['final']
"""
            # Compile the function
            local_vars = {}
            exec(func_code, {}, local_vars)
            dynamic_call = local_vars['__call__']

            # Dynamically create a subclass and assign the new __call__ method
            dynamic_cls = create_model(f'MermaidWorkflowTemplateDynamic{id(self)}', __base__=Model4LLMs.MermaidWorkflow)
            self.__class__ = dynamic_cls
            setattr(self.__class__, '__call__', dynamic_call)
            return self

        def run(self, **initial_args) -> Dict[str, dict]:
            # import pdb; pdb.set_trace()
            def ignite_func(instance:MWFFunction, cls_data:dict, self=self):                
                # Get parameters of the __call__ method (excluding 'self')
                parameters = inspect.signature(instance.__call__).parameters
                if len(parameters) == 0 or list(parameters.keys()) == ['self']:
                    return instance()
                else:
                    return instance(**cls_data.get('args',{}))
                
            self.results = super().run(ignite_func=ignite_func,initial_args=initial_args)
            return self.results

        def parse_mermaid(self, mermaid_text: str=None) -> Dict[str, Dict[str, Any]]:
            self._graph = {}
            if mermaid_text is None:
                mermaid_text = self.mermaid_text
            mermaid_text_lines = list(map(lambda l:l.replace(':','__of__',1),mermaid_text.split('\n')))
            mermaid_text_lines = [l for l in mermaid_text_lines if len(l)>0]
            mermaid_text_lines = [l.split('-->',1) for l in mermaid_text_lines]
            mermaid_text_lines = [[l[0],l[1].replace(':','__of__',1)] if len(l)>1 else l for l in mermaid_text_lines]
            mermaid_text_lines = [l[0]+'-->'+l[1] if len(l)>1 else l[0] for l in mermaid_text_lines]
            mermaid_text = '\n'.join(mermaid_text_lines)
            res:dict[str,GraphNode] = super().parse_mermaid(mermaid_text)
            for k,v in res.items():
                res[k] = v.model_dump()
            res = json.loads(json.dumps(res).replace('__of__',':'))
            model_registry = {}
            for k,v in res.items():
                func:Model4LLMs.MermaidWorkflowFunction = self.controller.storage().find(k)       

                if not callable(func) and hasattr(func,'build') and not hasattr(func,'builds'):
                    func:MWFFunction = func.build()          

                elif hasattr(func,'builds') and func.builds and hasattr(func,'build'):
                    func:MWFFunction = func.build(**json.loads(func.builds))

                model_registry[k] = (func.__class__,func)
                res[k] = GraphNode(**v)
            self.model_register(model_registry)
            self.mermaid_text = mermaid_text
            self._graph = res
            return res
        
        controller: Optional[Controller4LLMs.MermaidWorkflowController] = None

    class RequestsFunction(MermaidWorkflowFunction):
        description:str = Field('Makes an HTTP request using the configured method, url, and headers, and the provided params, data, or json.')
        
        class Parameter(BaseModel):            
            method: str = 'GET'
            url: str
            headers: Dict[str, str] = {}

        class Arguments(BaseModel):
            params: Optional[Dict[str, Any]] = Field(...,description='query parameters'),
            data:   Optional[Dict[str, Any]] = Field(...,description='form data'),
            json_payload:   Optional[Dict[str, Any]] = Field(...,description='JSON payload'),

        class Returness(BaseModel):
            data: dict = {}

        para: Parameter
        args: Arguments
        rets: Returness = Returness()
                
        def __call__(self,
                     debug=False,
                     debug_data=None) -> Dict[str, Any]:
            try:
                if debug: self.rets.data = debug_data
                response = requests.request(
                    method=self.para.method,url=self.para.url,
                    headers=self.para.headers,params=self.args.params,
                    data=self.args.data,json=self.args.json_payload,
                )
                response.raise_for_status()
                try:
                    self.rets.data = response.json()
                except Exception as e:
                    self.rets.data = {'text':response.text}
            except requests.exceptions.RequestException as e:
                self.rets.data = {"error": str(e), "status": getattr(e.response, "status_code", None)}

            return self
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
    
    # @Function.param_descriptions('Makes an HTTP request to async Celery REST api.',
    #                             params='query parameters',
    #                             data='form data',
    #                             json='JSON payload')
    # class AsyncCeleryWebApiFunction(Function):
    #     method: str = 'GET'
    #     url: str
    #     headers: Dict[str, str] = {}
    #     task_status_url: str = 'http://127.0.0.1:8000/tasks/status/{task_id}'

    #     def __call__(self,params: Optional[Dict[str, Any]] = None,
    #                  data: Optional[Dict[str, Any]] = None,
    #                  json: Optional[Dict[str, Any]] = None,
    #                  debug=False,
    #                  debug_data=None) -> Dict[str, Any]:
    #         try:
    #             if debug: return debug_data
    #             response = requests.request(
    #                 method=self.method,url=self.url,
    #                 headers=self.headers,params=params,
    #                 data=data,json=json
    #             )
    #             response.raise_for_status()
    #             # get task id
    #             for k,v in response.json().items():
    #                 if 'id' in k:
    #                     task_id = v
                
    #             while True:
    #                 response = requests.request(
    #                     method='GET',
    #                     url= self.task_status_url.format(task_id=task_id),
    #                     headers=self.headers,params=params,
    #                     data=data,json=json
    #                 )
    #                 if response is None:
    #                     break
    #                 response.raise_for_status()
    #                 if response.json() is None:
    #                     break
    #                 if response.json()['status'] in ['SUCCESS','FAILURE','REVOKED']:
    #                     if response.json()['status'] == 'FAILURE':
    #                         raise ValueError(f'{response.json()}')
    #                     break
    #                 time.sleep(1)

    #             try:
    #                 return response.json()
    #             except Exception as e:
    #                 raise ValueError(f'"text":{response.text}')
    #         except requests.exceptions.RequestException as e:
    #             raise ValueError(f'"error": {e} "status": {getattr(e.response, "status_code", None)}')
            
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
    
    def add_new_function(self, function_obj:MODEL_CLASS_GROUP.MermaidWorkflowFunction, id:str=None)->MODEL_CLASS_GROUP.MermaidWorkflowFunction:
        return self.add_new_obj(function_obj,id=id)
    
    def add_new_request(self, url:str, method='GET', headers={}, id:str=None)->MODEL_CLASS_GROUP.RequestsFunction:
        return self.add_new_obj(self.MODEL_CLASS_GROUP.RequestsFunction(method=method,url=url,headers=headers),id=id)
    
    # def add_new_celery_request(self, url:str, method='GET', headers={},
    #                            task_status_url: str = 'http://127.0.0.1:8000/tasks/status/{task_id}', id:str=None
    #                            )->MODEL_CLASS_GROUP.AsyncCeleryWebApiFunction:
    #     return self.add_new_obj(self.MODEL_CLASS_GROUP.AsyncCeleryWebApiFunction(method=method,url=url,
    #                                                     headers=headers,task_status_url=task_status_url),id=id)
    
    # def add_new_workflow(self, tasks:Optional[Dict[str,list[str]]|list[str]], metadata={}, id:str=None)->MODEL_CLASS_GROUP.WorkFlow:
    #     if type(tasks) is list:
    #         tasks = tasks[::-1]
    #         ds    = [[t] for t in tasks[1:]] + [[]]
    #         tasks = {t:d for t,d in zip(tasks,ds)}
    #     return self.add_new_obj(Model4LLMs.WorkFlow(tasks=tasks,metadata=metadata),id=id)
    
    def find_function(self,function_id:str) -> MODEL_CLASS_GROUP.MermaidWorkflowFunction:
        return self.find(function_id)
    
    def find_all_vendors(self)->list[MODEL_CLASS_GROUP.AbstractVendor]:
        return self.find_all('*Vendor:*')
    
    def find_all_llms(self) -> list[MODEL_CLASS_GROUP.AbstractLLM]:
        """Find all concrete LLM model classes (excluding abstract/utility classes)"""
        llms = []
        all_llms_item = dict(filter(
            lambda item: all([
                '_' not in item[0],
                'Function' not in item[0], 
                'WorkFlow' not in item[0],
                'Vendor' not in item[0],
                'TextEmbedding' not in item[0],
                'Abstract' not in item[0],
                'utils' not in str(item[1])
            ]),
            self.MODEL_CLASS_GROUP.__dict__.items()
        ))
        for k,v in all_llms_item.items():
            llms += self.find_all(f'{k}:*')
        return llms

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