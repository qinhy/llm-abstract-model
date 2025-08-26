import json
from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, ConfigDict, field_validator

from .AbstractVendor import AbstractVendor


class MCPToolAnnotations(BaseModel):
    """Annotations for MCP tools providing metadata about tool behavior."""
    
    title: Optional[str] = None
    """A human-readable title for the tool."""

    readOnlyHint: Optional[bool] = None
    """
    If true, the tool does not modify its environment.
    Default: false
    """

    destructiveHint: Optional[bool] = None
    """
    If true, the tool may perform destructive updates to its environment.
    If false, the tool performs only additive updates.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: true
    """

    idempotentHint: Optional[bool] = None
    """
    If true, calling the tool repeatedly with the same arguments 
    will have no additional effect on its environment.
    (This property is meaningful only when `readOnlyHint == false`)
    Default: false
    """

    openWorldHint: Optional[bool] = None
    """
    If true, this tool may interact with an "open world" of external
    entities. If false, the tool's domain of interaction is closed.
    For example, the world of a web search tool is open, whereas that
    of a memory tool is not.
    Default: true
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MCPTool(BaseModel):
    """Definition for a tool the client can call."""
    
    name: str
    """The name of the tool."""
    
    description: Optional[str] = None
    """A human-readable description of the tool."""
    
    inputSchema: Dict[str, Any]
    """A JSON Schema object defining the expected parameters for the tool."""
    
    annotations: Optional[MCPToolAnnotations] = None
    """Optional additional tool information."""

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert the Tool instance to an OpenAI tool format.
        
        Returns:
            Dict[str, Any]: The tool in OpenAI's function calling format
        """
        openai_tool = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.inputSchema, 
        }
        return json.loads(json.dumps(openai_tool))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AbstractLLM(BaseModel):
    """Abstract base class for Large Language Model implementations.
    
    This class provides a common interface for interacting with various LLM
    implementations, handling model parameters, message construction, and API calls.
    
    Attributes:
        vendor_id: Identifier for the LLM vendor, 'auto' for automatic detection
        llm_model_name: Name of the specific LLM model to use
        context_window_tokens: Maximum number of tokens the model can process
        max_output_tokens: Maximum number of tokens the model can generate
        stream: Whether to stream the response or return it all at once
        limit_output_tokens: Optional limit on output token count
        temperature: Controls randomness (0-1)
        top_p: Controls diversity via nucleus sampling (0-1)
        frequency_penalty: Reduces repetition of token sequences (0-2)
        presence_penalty: Reduces repetition of topics (0-2)
        system_prompt: Optional system message to control model behavior
        mcp_tools: Optional list of tools the model can use
    """

    vendor_id: str = 'auto'
    llm_model_name: str
    context_window_tokens: int
    max_output_tokens: int
    stream: bool = False
    
    limit_output_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    system_prompt: Optional[str] = None
    mcp_tools: Optional[List[MCPTool]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('temperature')
    def validate_temperature(cls, value: float) -> float:
        """Validate that temperature is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("Temperature must be between 0 and 1.")
        return value

    @field_validator('top_p')
    def validate_top_p(cls, value: float) -> float:
        """Validate that top_p is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        return value
    
    def get_vendor(self)->AbstractVendor:
        """Get the vendor for this LLM.
        
        Returns:
            AbstractVendor: The vendor instance for this LLM
        """
        raise NotImplementedError()
        return self.controller.get_vendor(auto=(self.vendor_id=='auto'))

    def get_controller(self):
        """Get the controller for this LLM.
        
        This method should be implemented by subclasses.
        
        Returns:
            AbstractLLMController: The controller for this LLM
        """
        raise NotImplementedError("Subclasses must implement controller")

    def get_usage_limits(self) -> Dict[str, Any]:
        """Get usage limits for this LLM.
        
        Returns:
            Dict[str, Any]: A dictionary of usage limits
        """
        raise NotImplementedError("Subclasses must implement get_usage_limits()")

    def validate_input(self, prompt: str) -> bool:
        """Validate the input prompt based on max input tokens.
        
        Args:
            prompt: The input prompt to validate
            
        Returns:
            bool: True if the input is valid
            
        Raises:
            ValueError: If the input exceeds the maximum token limit
        """
        token_count = self.get_token_count(prompt)
        if token_count > self.context_window_tokens:
            raise ValueError(f"Input exceeds the maximum token limit of {self.context_window_tokens}.")
        return True

    def calculate_cost(self, tokens_used: int) -> float:
        """Calculate the cost based on tokens used.
        
        Args:
            tokens_used: The number of tokens used
            
        Returns:
            float: The cost in the vendor's currency
        """
        return 0.0

    def get_token_count(self, text: str) -> int:
        """Count the number of tokens in the text.
        
        This is a dummy implementation. Subclasses should implement
        vendor-specific token counting logic.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: The number of tokens in the text
        """
        # This is a naive implementation - should be overridden by subclasses
        return len(text.split())
    
    def build_system(self, purpose: str = '...') -> str:
        """Build a system prompt for the given purpose.
        
        Args:
            purpose: The purpose of the system prompt
            
        Returns:
            str: The system prompt
        """
        raise NotImplementedError("Subclasses must implement build_system()")

    def set_mcp_tools(self, mcp_tools_json: Union[str, Dict, List] = '{}') -> None:
        """Set the MCP tools for this LLM.
        
        Args:
            mcp_tools_json: JSON string, dict, or list of tool definitions
        """
        if not isinstance(mcp_tools_json, str):
            mcp_tools_json = json.dumps(mcp_tools_json)
        self.mcp_tools = [MCPTool(**tool) for tool in json.loads(mcp_tools_json)]
        # self.controller.update(mcp_tools=self.mcp_tools)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the tools available to this LLM in the format expected by the vendor.
        
        Returns:
            List[Dict[str, Any]]: The tools in vendor-specific format
        """
        raise NotImplementedError("Subclasses must implement get_tools()")

    def construct_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construct the payload for the LLM API request.
        
        Args:
            messages: The messages to include in the payload
            
        Returns:
            Dict[str, Any]: The payload for the API request
        """
        payload = {
            "model": self.get_vendor().format_llm_model_name(self.llm_model_name),
            "stream": self.stream,
            "messages": messages,
            "max_output_tokens": self.limit_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "frequency_penalty": self.frequency_penalty,
            # "presence_penalty": self.presence_penalty,
        }
        return {k: v for k, v in payload.items() if v is not None}
    
    def construct_messages(self, messages: Optional[Union[List[Dict[str, Any]], str]]) -> List[Dict[str, Any]]:
        """Construct the messages for the LLM API request.
        
        Args:
            messages: The messages to construct, either a string or a list of message dicts
            
        Returns:
            List[Dict[str, Any]]: The constructed messages
        """
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif messages is None:
            messages = []
            
        return msgs + messages

    def __call__(self, messages: Optional[Union[List[Dict[str, Any]], str, Any]], auto_str: bool = True) -> str:
        """Call the LLM with the given messages.
        
        Args:
            messages: The messages to send to the LLM
            auto_str: Whether to automatically convert non-list, non-string messages to strings
            
        Returns:
            str: The LLM's response
        """
        if not isinstance(messages, list) and not isinstance(messages, str):
            if auto_str:
                messages = str(messages)
        messages = self.construct_messages(messages)
        payload = self.construct_payload(messages)
        vendor = self.get_vendor()
        return vendor.chat_result(vendor.chat_request(payload))