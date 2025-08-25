from pydantic import BaseModel
from typing import Dict, Any, Union, Optional
from .AbstractVendor import AbstractVendor
    
class OpenAIVendor(AbstractVendor):
    vendor_name:str = 'OpenAI'
    api_url:str = 'https://api.openai.com'
    chat_endpoint:str = '/v1/responses'
    models_endpoint:str = '/v1/models'
    embeddings_endpoint:str = '/v1/embeddings'
    default_timeout: int = 30               # Timeout for API requests in seconds
    rate_limit: Optional[int] = None        # Requests per minute, if applicable
    
    def get_available_models(self) -> Dict[str, Any]:
        response = super().get_available_models()
        return {model['id']: model for model in response.get('data', [])}

    def chat_completions_result(self, response) -> Union[str, Dict[str, Any]]:
        # legacy support for chat completions API
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

    def chat_result(self, response) -> Union[str, Dict[str, Any]]:
        """
        Parse both OpenAI Responses API and legacy Chat Completions results.

        Returns:
        - str: assistant text (preferred)
        - dict: structured object for tool calls (so caller can execute tools)
        """

        # --- 0) Basic sanity & error passthrough --------------------------------
        if not isinstance(response, dict):
            self._log_error(ValueError(f"Unexpected response type: {type(response)}"))
            return response

        if "error" in response and response["error"]:
            # Preserve vendor's error shape
            return response

        # --- 1) New: Responses API ----------------------------------------------
        # Fast path: convenience field
        if isinstance(response.get("output_text"), str) and response["output_text"].strip():
            return response["output_text"]

        # Tool calls (Responses): appear as items in 'output' with type 'tool_call'
        output_items = response.get("output", [])
        if isinstance(output_items, list) and output_items:
            # 1a) Collect text from message/content parts
            text_chunks = []
            tool_calls = []

            for item in output_items:
                itype = item.get("type")

                # Tool call block (one or many)
                if itype == "tool_call":
                    # Typical shape: { type: "tool_call", name, arguments, call_id, … }
                    tool_calls.append(item)
                    continue

                # Message block with typed content
                if itype == "message":
                    for part in item.get("content", []):
                        # Typical text part: {type:"output_text", "text": "..."} or {type:"text","text":"..."}
                        if isinstance(part, dict):
                            if isinstance(part.get("text"), str):
                                text_chunks.append(part["text"])
                            # Some variants may place text on 'output_text' key
                            elif isinstance(part.get("output_text"), str):
                                text_chunks.append(part["output_text"])

            # If tool calls are present, return them so caller can orchestrate
            if tool_calls:
                return {
                    "type": "tool_calls",
                    "calls": tool_calls,
                    # Optional: include any assistant preface text that came alongside
                    "content": "".join(text_chunks).strip() if text_chunks else ""
                }

            # Otherwise, if we assembled text, return it
            if text_chunks:
                return "".join(text_chunks).strip()

        # --- 3) Nothing matched: surface something useful -----------------------
        self._log_error(ValueError(f"Unrecognized response shape: {response}"))
        return response
    def get_embedding(self, text: str, model: str='text-embedding-3-small') -> Dict[str, Any]:
        payload = {"model": model,"input": text}
        return self.embedding_request(payload)
