import inspect
import json
import os
import unittest
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, field_validator, ConfigDict, Field
import requests
from typing import Dict, Any
import mt5

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
class Model4LLMs:
    class AbstractObj(Model4Basic.AbstractObj):
        pass
    
    class AbstractVendor(AbstractObj):
        vendor_name: str  # e.g., 'OpenAI'
        api_url: str  # e.g., 'https://api.openai.com/v1/'
        api_key: str = None  # API key for authentication, if required
        timeout: int = 30  # Default timeout for API requests in seconds
        
        chat_endpoint:str = None # e.g., '/v1/chat/completions'
        models_endpoint:str = None # e.g., '/v1/models'
        
        def format_llm_model_name(self,llm_model_name:str)->str:
            return llm_model_name       
        
        def get_api_key(self)->str:
            return os.getenv(self.api_key,self.api_key)

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
    class MT5Account(AbstractObj):
        account_id: int = None
        password: str = ''
        account_server: str = ''

        def is_valid(self):
            if self.account_id is None:raise ValueError('account_id is not set')
            if self.password == '':raise ValueError('password is not set')
            if self.account_server == '':raise ValueError('account_server is not set')
            return True
    
    class Book(AbstractObj):
        class Controller(BaseModel):        
            @staticmethod
            def _try(func):
                try:
                    return func()
                except Exception as e:
                    print(e)
                    return False

            class Null(BaseModel):
                type:str = 'Null'
                def send(self,book):
                    raise ValueError(f'This is a {self.type} state')
                def close(self,book):
                    raise ValueError(f'This is a {self.type} state')
                def changeP(self,book,p):
                    raise ValueError(f'This is a {self.type} state')
                def changeTS(self,book,tp,sl):
                    raise ValueError(f'This is a {self.type} state')

            class Plan(BaseModel):
                type:str = 'Plan'
                def send(self,book):
                    book:Model4LLMs.Book = book
                    res = Model4LLMs.Book.Controller._try(lambda:book._make_order())
                    book.state = Model4LLMs.Book.Controller.Order() if res else Model4LLMs.Book.Controller.Plan()
                def close(self,book):
                    raise ValueError('This is just a Plan')
                def changeP(self,book,p):
                    book:Model4LLMs.Book = book
                    book.price_open = p
                def changeTS(self,book,tp,sl):
                    book:Model4LLMs.Book = book
                    book.tp,book.sl=tp,sl
                
            class Order(BaseModel):
                type:str = 'Order'
                def send(self,book):
                    raise ValueError('This is a exists Order')
                def close(self,book):
                    book:Model4LLMs.Book = book
                    res = Model4LLMs.Book.Controller._try(lambda:book._close_order())
                    if res : book.state = Model4LLMs.Book.Controller.Null()
                def changeP(self,book,p):
                    raise ValueError('This is a exists Order, You can close it.')
                def changeTS(self,book,tp,sl):
                    book:Model4LLMs.Book = book
                    res = Model4LLMs.Book.Controller._try(lambda:book._changeOrderTPSL(tp,sl))
                    if res : book.tp,book.sl=tp,sl

            class Position(BaseModel):
                type:str = 'Position'
                def send(self,book):
                    raise ValueError('This is a exists Position')
                def close(self,book):
                    book:Model4LLMs.Book = book
                    res = Model4LLMs.Book.Controller._try(lambda:book._close_position())
                    if res : book.state = Model4LLMs.Book.Controller.Null()
                def changeP(self,book,p):
                    raise ValueError('This is a exists Position, can not change price open')
                def changeTS(self,book,tp,sl):
                    book:Model4LLMs.Book = book
                    res = Model4LLMs.Book.Controller._try(lambda:book._changePositionTPSL(tp,sl))
                    if res : book.tp,book.sl=tp,sl

        symbol: str = ''
        sl: float = 0.0
        tp: float = 0.0
        price_open: float = 0.0
        volume: float = 0.0
        magic:int = 901000
                    
        state: Controller.Plan = Controller.Plan()
        _book: Any = None# mt5_order_position
        _is_order: bool = False
        _is_position: bool = False
        _ticket: int = 0
        _type: str = ''
        _swap: int = 0

        def as_plan(self):
            self.state = Model4LLMs.Book.Controller.Plan()
            return self
        
        def send(self):
            self.state.send(self)
            return self
        def close(self):
            self.state.close(self)
            return self
        def changeP(self,p):
            self.state.changeP(self,p)
            return self
        def changeTS(self,tp,sl):
            self.state.changeTS(self,tp,sl)
            return self

        def getBooks(self):
            return [ Model4LLMs.Book().set_mt5_book(book=op) for op in mt5.orders_get()+mt5.positions_get() ]
        
        def account_info(self):
            # Example operation: Getting account information
            account_info = mt5.account_info()
            if account_info is None:
                return "Failed to get account info"
            else:
                return account_info
            
        def set_mt5_book(self,book):
            self._book = book
            self.symbol = self._book.symbol
            self.sl = self._book.sl
            self.tp = self._book.tp
            self.price_open = self._book.price_open
            self._ticket = self._book.ticket
            self._type = self._book.type
            self._is_order=False
            self._is_position=False
            self._swap = 0
            
            if self._book.__class__.__name__ == "TradeOrder" : 
                self._is_order=True
                self.state = Model4LLMs.Book.Controller.Order()
            elif self._book.__class__.__name__ == "TradePosition": 
                self._is_position=True
                self.state = Model4LLMs.Book.Controller.Position()
            if hasattr(self._book,'volume_current'):
                self._is_order=True
                self.volume=self._book.volume_current
            elif hasattr(self._book,'volume'):
                self._is_position=True
                self.volume=self._book.volume
                self._swap = self._book.swap
            else:
                raise 'Unkown type!'
            
            self._book = ''
            return self

        def isBuy(self):
            if self._is_order: 
                    return self._type in [mt5.ORDER_TYPE_BUY,mt5.ORDER_TYPE_BUY_LIMIT ,
                                        mt5.ORDER_TYPE_BUY_STOP ,mt5.ORDER_TYPE_BUY_STOP_LIMIT]
            elif self._is_position: 
                    return self._type == mt5.POSITION_TYPE_BUY
            return True
        
        def _sendRequest(self, request):    
            result=mt5.order_send(request)    
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                print('send request failed',result)
                return False
            return True

        def _changeOrderTPSL(self, tp=0.0,sl=0.0):
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": self._ticket,
                "price": self.price_open,
                "tp": tp,
                "sl": sl
            }
            return self._sendRequest(request)

        def _changePositionTPSL(self, tp=0.0,sl=0.0):
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": self._ticket,
                "symbol": self.symbol,
                "tp": tp,
                "sl": sl
            }
            return self._sendRequest(request)

        def _changeTPSL(self, tp=0.0,sl=0.0):
            if self._is_order: 
                return self._changeOrderTPSL(tp,sl)
            elif self._is_position: 
                return self._changePositionTPSL(tp,sl)
            return False
        
        def _close_position(self):
            #https://www.mql5.com/ja/docs/constants/structures/mqltraderequest
            if self._type == mt5.ORDER_TYPE_BUY:
                price = mt5.symbol_info_tick(self.symbol).bid
                type_tmp = mt5.ORDER_TYPE_SELL
            elif self._type == mt5.ORDER_TYPE_SELL:
                price = mt5.symbol_info_tick(self.symbol).ask
                type_tmp = mt5.ORDER_TYPE_BUY
            else:
                raise ValueError('unknow position type(nor buy or sell) error.')

            deviation=20
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.volume,
                "type": type_tmp,
                "position": self._ticket,
                "price": price,
                "deviation": deviation,
                "magic": self.magic,
                "comment": "script close",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC,
            }
            return self._sendRequest(request)

        def _close_order(self):
            #https://www.mql5.com/en/forum/365968
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": self._ticket,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            return self._sendRequest(request)

        def _make_order(self, profit_risk_ratio: float=None):
            # ProfitRiskRatio = self._ProfitRiskRatio
            # Determine order type and calculate stop loss based on parameters
            going_long = self.tp > self.price_open
            current_price_info = mt5.symbol_info_tick(self.symbol)
            if current_price_info is None:
                return f"Error getting current price for {self.symbol}"

            if going_long:
                current_price = current_price_info.ask
                order_type = mt5.ORDER_TYPE_BUY_STOP if self.price_open > current_price else mt5.ORDER_TYPE_BUY_LIMIT
            else:
                current_price = current_price_info.bid
                order_type = mt5.ORDER_TYPE_SELL_STOP if self.price_open < current_price else mt5.ORDER_TYPE_SELL_LIMIT

            if profit_risk_ratio is not None:
                self.sl = self.price_open + (self.price_open - self.tp) / profit_risk_ratio

            digitsnum = mt5.symbol_info(self.symbol).digits
            self.price_open,self.sl,self.tp = list(map(lambda x:round(x*10**digitsnum)/10**digitsnum,
                                                            [self.price_open,self.sl,self.tp]))
            # Prepare trade request
            deviation=20
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": self.symbol,
                "volume": self.volume,
                "type": order_type,
                "price": self.price_open,
                "sl": self.sl,
                "tp": self.tp,
                "deviation": deviation,
                "magic": self.magic,
                "comment": "auto order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            return self._sendRequest(request)
        
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

class LLMsStore(BasicStore):

    def _get_class(self, id: str, modelclass=Model4LLMs):
        return super()._get_class(id, modelclass)
    
    def add_new_openai_vendor(self,api_key: str,
                              api_url: str='https://api.openai.com',
                              timeout: int=30) -> Model4LLMs.OpenAIVendor:
        return self.add_new_obj(Model4LLMs.OpenAIVendor(api_url=api_url,api_key=api_key,timeout=timeout))
    
    def add_new_ollama_vendor(self,api_url: str='http://localhost:11434',
                              timeout: int=30) -> Model4LLMs.OllamaVendor:
        return self.add_new_obj(Model4LLMs.OllamaVendor(api_url=api_url,api_key='',timeout=timeout))
    
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
    
    def add_new_gemma2(self,vendor_id:str,system_prompt:str = None) -> Model4LLMs.Gemma2:
        return self.add_new_obj(Model4LLMs.Gemma2(vendor_id=vendor_id,system_prompt=system_prompt))
    
    def add_new_phi3(self,vendor_id:str,system_prompt:str = None) -> Model4LLMs.Phi3:
        return self.add_new_obj(Model4LLMs.Phi3(vendor_id=vendor_id,system_prompt=system_prompt))
    
    def add_new_llama(self,vendor_id:str,system_prompt:str = None) -> Model4LLMs.Llama:
        return self.add_new_obj(Model4LLMs.Llama(vendor_id=vendor_id,system_prompt=system_prompt))

    def add_new_function(self, function_obj:Model4LLMs.Function)->Model4LLMs.Function:  
        function_name = function_obj.__class__.__name__
        setattr(Model4LLMs,function_name,function_obj.__class__)
        return self._add_new_obj(function_obj)
    
    def find_function(self,function_id:str) -> Model4LLMs.Function:
        return self.find(function_id)
    
    def find_all_vendors(self)->list[Model4LLMs.AbstractVendor]:
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
        v = self.store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
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