from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union

# Import the required modules
from .AbstractLLM import AbstractLLM,MCPTool,MCPToolAnnotations

class AbstractGPTModel(AbstractLLM, BaseModel):
    """Implementation of OpenAI's ChatGPT models.
    
    This class provides the configuration and functionality needed to interact
    with OpenAI's GPT models through their chat completions API.
    """
    
    # Basic configuration parameters
    limit_output_tokens: Optional[int] = 1024

    # Advanced configuration parameters
    stop_sequences: Optional[List[str]] = Field(default_factory=list)
    n_generations: Optional[int] = 1  # Number of completions to generate
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the tools available to this model in OpenAI format.
        
        Returns:
            List[Dict[str, Any]]: List of tools in OpenAI format
        """
        raise NotImplementedError("Subclasses must implement get_tools()")

    # def construct_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """Construct the payload for the LLM API request.
        
    #     Args:
    #         messages: The messages to include in the payload
            
    #     Returns:
    #         Dict[str, Any]: The payload for the API request
    #     """
    #     payload = {
    #         "model": self.get_vendor().format_llm_model_name(self.llm_model_name),
    #         "stream": self.stream,
    #         "messages": messages,
    #         "max_tokens": self.limit_output_tokens,
    #         "temperature": self.temperature,
    #         "top_p": self.top_p,
    #         "frequency_penalty": self.frequency_penalty,
    #         "presence_penalty": self.presence_penalty,
    #     }
    #     return {k: v for k, v in payload.items() if v is not None}

    def openai_construct_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construct payload for OpenAI API with standard parameters and tools.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dict[str, Any]: Payload for OpenAI API request
        """
        payload = super().construct_payload(messages) 
        payload.update({
            # "stop": self.stop_sequences,
            "n": self.n_generations,
        })
        
        if self.mcp_tools:
            # Convert tools to MCPTool objects if they aren't already
            tools = [
                t if isinstance(t, MCPTool) else 
                MCPTool(**t) for t in self.mcp_tools
            ]
            payload.update({"tools": [t.to_openai_tool() for t in tools]})
            
        return {k: v for k, v in payload.items() if v is not None}
    
    def openai_o_models_construct_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construct payload for OpenAI 'o' models (o3, o3-mini, etc.).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dict[str, Any]: Payload for OpenAI API request
        """
        payload = super().construct_payload(messages)
        del payload["temperature"]
        return {k: v for k, v in payload.items() if v is not None}
        
    def openai_o_models_construct_messages(self, messages: Optional[Union[List, str]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI 'o' models, handling string inputs and system prompts.
        
        Args:
            messages: String message or list of message dictionaries
            
        Returns:
            List[Dict[str, Any]]: Formatted messages for API request
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
            
        # Ensure messages is not empty before trying to modify it
        if messages and self.system_prompt:
            messages[0]["content"] = f"{self.system_prompt}\n{messages[0]['content']}"
            
        return messages

    def claude_construct_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construct payload for Claude API with appropriate parameters.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dict[str, Any]: Payload for Claude API request
        """
        payload = super().construct_payload(messages)        
        
        if self.system_prompt:
            payload["system"] = self.system_prompt
            
        # Only include allowed fields for Claude API
        allowed_keys = {
            "model", "messages", "max_tokens", "temperature", 
            "top_p", "system", "stream"
        }
        return {k: v for k, v in payload.items() if k in allowed_keys and v is not None}
    
    def claude_construct_messages(self, messages: Optional[Union[List, str]]) -> List[Dict[str, Any]]:
        """Format messages for Claude API, handling system messages differently.
        
        Args:
            messages: String message or list of message dictionaries
            
        Returns:
            List[Dict[str, Any]]: Formatted messages for Claude API
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Extract and remove any system message from the list
        cleaned_messages = []
        for msg in messages or []:
            if msg.get("role") == "system":
                self.system_prompt = msg.get("content", "")
            else:
                cleaned_messages.append(msg)
                
        return cleaned_messages