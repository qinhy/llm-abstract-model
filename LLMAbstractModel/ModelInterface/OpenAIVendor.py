from pydantic import BaseModel
from typing import Dict, Any, Union, Optional
from .AbstractVendor import AbstractVendor
    
class OpenAIVendor(AbstractVendor):
    vendor_name:str = 'OpenAI'
    api_url:str = 'https://api.openai.com'
    chat_endpoint:str = '/v1/chat/completions'
    models_endpoint:str = '/v1/models'
    embeddings_endpoint:str = '/v1/embeddings'
    default_timeout: int = 30               # Timeout for API requests in seconds
    rate_limit: Optional[int] = None        # Requests per minute, if applicable
    
    def get_available_models(self) -> Dict[str, Any]:
        response = super().get_available_models()
        return {model['id']: model for model in response.json().get('data', [])}


    def chat_result(self, response) -> Union[str, Dict[str, Any]]:
        # print(response)
        if not self._try_binary_error(lambda: response['choices']):
            raise ValueError(f'cannot get choices from {response}')
        choice = response['choices'][0]
        content = ''
        if self._try_binary_error(lambda: response['choices'][0]['message']['content']):
            content = response['choices'][0]['message']['content']
        # Handle function_call (legacy function call support)
        if 'function_call' in choice['message']:
            return {
                'content':content,
                'type': 'function_call',
                'name': choice['message']['function_call']['name'],
                'arguments': choice['message']['function_call'].get('arguments')
            }

        # Handle tool_calls (newer API style with multiple tool calls)
        if 'tool_calls' in choice['message']:
            return {
                'content':content,
                'type': 'tool_calls',
                'calls': choice['message']['tool_calls']
            }

        # Standard chat message
        if content:return content

        self._log_error(ValueError(f'cannot get result from {response}'))

    def get_embedding(self, text: str, model: str='text-embedding-3-small') -> Dict[str, Any]:
        payload = {"model": model,"input": text}
        return self.embedding_request(payload)
