
import os
import json
import requests
from typing import Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class BasicModel(BaseModel):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('This method should be implemented by subclasses.')
    
    def _log_error(self, e):
        return f"[{self.__class__.__name__}] Error: {str(e)}"
    
    def _try_error(self, func, default_value=('NULL',None)):
        try:
            return (True,func())
        except Exception as e:
            self._log_error(e)
            return (False,default_value)
        
    def _try_binary_error(self, func):
        return self._try_error(func)[0]
    
    def _try_obj_error(self, func, default_value=('NULL',None)):
        return self._try_error(func,default_value)[1]
    
class AbstractVendor(BasicModel):
    """Base class for LLM vendor API integrations.
    
    This abstract class provides a common interface for interacting with
    various LLM vendor APIs (like OpenAI, Anthropic, etc.).
    
    Attributes:
        vendor_name: Name of the LLM vendor (e.g., 'OpenAI')
        api_url: Base URL for the vendor's API
        api_key: API key for authentication
        timeout: Default timeout for API requests in seconds
        chat_endpoint: Endpoint for chat completions
        models_endpoint: Endpoint for listing available models
        embeddings_endpoint: Endpoint for embedding requests
        rate_limit: Maximum requests per minute, if applicable
    """
    vendor_name: str
    api_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    
    chat_endpoint: Optional[str] = None
    models_endpoint: Optional[str] = None
    embeddings_endpoint: Optional[str] = None
    rate_limit: Optional[int] = None

    cache_embeddings: bool = False
    # Option to cache embeddings to improve efficiency
    embeddings_cache: Dict[str, Any] = {}

    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def format_llm_model_name(self, llm_model_name: str) -> str:
        """Format the model name according to vendor requirements.
        
        Args:
            llm_model_name: The generic model name
            
        Returns:
            The vendor-specific formatted model name
        """
        return llm_model_name
    
    def get_api_key(self) -> str:
        """Get the API key from environment variables or use the provided value.
        
        Returns:
            The API key as a string
        """
        return os.getenv(self.api_key, self.api_key)

    def get_available_models(self) -> Dict[str, Any]:
        """Retrieve the list of available models from the vendor API.
        
        Returns:
            Dictionary containing the available models information
        """
        try:
            response = requests.get(
                self._build_url(self.models_endpoint),
                headers=self._build_headers(),
                timeout=self.timeout
            )
            return self._handle_response(response)
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def _build_headers(self) -> Dict[str, str]:
        """Build the HTTP headers for API requests.
        
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            'Content-Type': 'application/json'
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.get_api_key()}'
        return headers

    def _build_url(self, endpoint: str) -> str:
        """Construct the full API URL for a given endpoint.
        
        Args:
            endpoint: API endpoint to be called
            
        Returns:
            Full API URL
        """
        return f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Process the API response and handle errors.
        
        Args:
            response: The requests.Response object
            
        Returns:
            Processed response data or error information
        """
        try:
            return response.json()
        except Exception as e:
            return {'error': f'Failed to parse response: {str(e)}'}

    def chat_request(self, payload: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Send a chat completion request to the vendor API.
        
        Args:
            payload: Request payload containing messages and parameters
            
        Returns:
            The API response as a dictionary
        """
        if not self.chat_endpoint:
            return {'error': 'Chat endpoint not defined for this vendor.'}
            
        url = self._build_url(self.chat_endpoint)
        headers = self._build_headers()
        
        try:
            response = requests.post(
                url=url, 
                data=json.dumps(payload, ensure_ascii=True),
                headers=headers, 
                timeout=self.timeout
            )
            return self._handle_response(response)
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def chat_result(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the chat completion response.
        
        Args:
            response: The raw API response
            
        Returns:
            Processed response data
        """
        return response
    
    def embedding_request(self, payload: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Send an embedding request to the vendor API.
        
        Args:
            payload: Request payload containing model and input text
            
        Returns:
            The embedding vector or error information
        """
        if not self.embeddings_endpoint:
            return {'error': 'Embeddings endpoint not defined for this vendor.'}

        url = self._build_url(self.embeddings_endpoint)
        headers = self._build_headers()
        
        try:
            response = requests.post(
                url=url, 
                data=json.dumps(payload, ensure_ascii=True),
                headers=headers, 
                timeout=self.timeout
            )
            response_data = self._handle_response(response)
            
            if 'error' in response_data:
                return response_data
                
            return response_data['data'][0]['embedding']
        except (KeyError, IndexError) as e:
            return {'error': f'Failed to extract embedding: {str(e)}'}
        except requests.RequestException as e:
            return {'error': f'Request failed: {str(e)}'}
    
    def get_embedding(self, text: str, model: str) -> Dict[str, Any]:
        """Get embedding vector for the provided text.
        
        Args:
            text: The input text to embed
            model: The embedding model to use
            
        Returns:
            The embedding vector or error information
        """
                
        # Check cache
        if self.cache_embeddings and text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        payload = {"model": model, "input": text}
        embedding = self.embedding_request(payload)
        
        # Cache result
        if self.cache_embeddings:
            self.embeddings_cache[text] = embedding
        
        return embedding

