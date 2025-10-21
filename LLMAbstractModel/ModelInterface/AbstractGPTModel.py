import base64
import json
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union

import requests

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
            # "n": self.n_generations,
        })
        
        if self.mcp_tools:
            # Convert tools to MCPTool objects if they aren't already
            tools = [
                t if isinstance(t, MCPTool) else 
                MCPTool(**t) for t in self.mcp_tools
            ]
            payload.update({"tools": [t.to_openai_tool() for t in tools]})
            
        return {k: v for k, v in payload.items() if v is not None}
    
    def openai_responses_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a payload for the Responses API from Chat-Completions-style messages.
        Maps:
          - system -> instructions
          - messages -> input (typed parts)
          - token limit -> max_output_tokens
          - tools -> tools (if provided)
        """
        def _messages_to_responses_input(messages: list) -> list:
            """
            Convert Chat Completions-style messages into Responses 'input' blocks.

            - role stays ('user' / 'assistant' / 'system' allowed)
            - content becomes a list of typed parts; we default to input_text
            - if a message already has typed parts, we keep them
            """
            result = []
            for m in messages or []:
                if "role" in m:
                    role = m["role"]
                    content = m.get("content", "")
                    if isinstance(content, list) and all(isinstance(p, dict) and "type" in p for p in content):
                        parts = content
                    else:
                        parts = [{"type": "input_text" if role!='assistant' else 'output_text', "text": content}]
                    result.append({"role": role, "content": parts})
                else:
                    result.append(m)
            return result
        
        # Build the messages list with your existing helper (system_prompt handled there)
        # This ensures consistent behavior with your current pipeline.
        payload = self.openai_construct_payload(messages)

        # Split out system messages for 'instructions'
        for m in messages:
            if m.get("role") == "system":
                self.system_prompt = m.get("content", "")
                break
            
        instructions = self.system_prompt
        non_system = [m for m in messages or [] if m.get("role") != "system"]
        payload.pop("messages", None)  # Remove messages; we'll use input instead
        payload.update({
            "model": self.get_vendor().format_llm_model_name(self.llm_model_name),
            "input": _messages_to_responses_input(non_system) or [
                {"role": "user", "content": [{"type": "input_text", "text": ""}]}
            ],
            # Keep your streaming setting
            "stream": self.stream or None,
            # Temperature/top_p are supported by most models via Responses
            # "temperature": self.temperature,
            "top_p": self.top_p,
            # Prefer an explicit output limit; fall back if None
            "max_output_tokens": self.limit_output_tokens if self.limit_output_tokens is not None else self.max_output_tokens
        })

        if instructions:
            payload["instructions"] = instructions
        
        if hasattr(self,'reasoning_effort'):
            payload["reasoning"] = { "effort": self.reasoning_effort }

        # Tools wiring (prefer explicit get_tools(); otherwise use mcp_tools if present)
        tools = None
        try:
            tools = self.get_tools()
        except NotImplementedError:
            pass
        if not tools and self.mcp_tools:
            # If you rely on MCPTool objects having .to_openai_tool()
            tools = [t.to_openai_tool() if hasattr(t, "to_openai_tool") else t for t in self.mcp_tools]

        if tools:
            payload["tools"] = tools

        # Clean out Nones for a tidy payload
        return {k: v for k, v in payload.items() if v is not None}

    # # --- (optional) For o* models that ignore temperature --------------------
    def openai_o_models_responses_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = self.openai_responses_payload(messages)
        payload.pop("temperature", None)
        return payload

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
        
        payload['max_tokens'] = self.limit_output_tokens
            
        # Only include allowed fields for Claude API
        allowed_keys = {
            "model", "messages", "max_tokens", "temperature", 
            "top_p", "system", "stream"
        }
        return {k: v for k, v in payload.items() if k in allowed_keys and v is not None}
    
    def claude_construct_messages(self, messages: Optional[Union[List[Dict], str]],
                                  available_content_types: Optional[List[str]] = None
                                  ) -> List[Dict[str, Any]]:
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

            if isinstance(msg.get("content"), list):
                for i,c in enumerate(msg["content"]):
                    if isinstance(c, dict):
                        if "text" in c.get("type"):
                            if available_content_types and "text" not in available_content_types:
                                raise ValueError(f"Text content not supported by this model/vendor.")
                            msg["content"][i]["type"] = "text"

                        elif "image" in c.get("type"):
                            if available_content_types and "image" not in available_content_types:
                                raise ValueError(f"Image content not supported by this model/vendor.")                            
                            msg["content"][i]["type"] = "image"

                            if "image_url" in c:
                                msg["content"][i]["source"] = {"type": "url", "url":c.pop("image_url")}
        return cleaned_messages
        
    def gemini_construct_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # The gemini_construct_contents helper handles message formatting and
        # extracts the system prompt.
        formatted_contents = messages
        
        # Gemini nests generation parameters inside a 'generationConfig' object.
        generation_config = {
            "temperature": self.temperature,
            "topP": self.top_p,
            "maxOutputTokens": self.limit_output_tokens,
            "stopSequences": self.stop_sequences or [],
        }
        
        payload = {
            "contents": formatted_contents,
            "generationConfig": {k: v for k, v in generation_config.items() if v is not None},
        }

        # Add system prompt using the dedicated 'system_instruction' field.
        if self.system_prompt:
            payload["system_instruction"] = {"parts": [{"text": self.system_prompt}]}
            
        # Handle and format tools if they are provided.
        if self.mcp_tools:
            # Assumes your MCPTool class can produce an OpenAI-compatible format.
            tools = [
                t if isinstance(t, MCPTool) else 
                MCPTool(**t) for t in self.mcp_tools
            ]
            # We convert from OpenAI's format to Gemini's by extracting the 'function' object.
            function_declarations = [t.to_openai_tool()["function"] for t in tools]
            payload["tools"] = [{"function_declarations": function_declarations}]
        # else:
        #     payload["tool_config"] =  {
        #         "function_calling_config": {
        #         "mode": "NONE"
        #         }
        #     }

        # payload["safetySettings"] = [
        #     {
        #     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #     "threshold": "BLOCK_NONE"
        #     },
        #     {
        #     "category": "HARM_CATEGORY_HARASSMENT",
        #     "threshold": "BLOCK_NONE"
        #     }
        # ]
        
        # Return a clean payload, removing any keys with empty or None values.
        return {k: v for k, v in payload.items() if v}

    def gemini_construct_messages(self, messages: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Transforms a list of messages into the Gemini API's 'role' and 'parts' format,
        handling standard text, tool calls, tool responses, and multi-modal content.

        Args:
            messages: A list of message dictionaries. Each dictionary should have a 'role' and 'content'.
                    The 'content' can be a string or a list of multi-modal objects.

        Returns:
            A list of dictionaries formatted for the Gemini API.
        """
        contents = []
        system_prompt = None

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Extract the last system message found in the history.
        for msg in reversed(messages or []):
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break

        for msg in messages or []:
            role = msg.get("role")

            # Skip system messages; Gemini handles them implicitly, not as a dedicated role.
            if role == "system":
                continue

            # --- Map roles to Gemini's expected values.
            gemini_role = "user"
            if role == "assistant":
                gemini_role = "model"
            elif role == "tool":
                gemini_role = "function"

            parts = []

            # --- Handle standard text or multi-modal content in the 'content' field.
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                # Standard text message
                parts.append({"text": content})
            elif isinstance(content, list):
                # Multi-modal content (text + image)
                for item in content:
                    item_type = item.get("type")
                    if item_type == "input_text" and isinstance(item.get("text"), str):
                        parts.append({"text": item["text"]})
                    elif item_type == "input_image" and isinstance(item.get("image_url"), str):
                        try:
                            # Add a common User-Agent header to mimic a web browser
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                            }
                            
                            response = requests.get(item["image_url"], headers=headers, stream=True)
                            response.raise_for_status() # This will catch 4xx and 5xx errors
                            
                            image_data = response.content
                            mime_type = response.headers.get("Content-Type", "image/jpeg")
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                            
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_image
                                }
                            })
                        except requests.exceptions.RequestException as e:
                            print(f"Error fetching image from URL: {e}")
                            continue
            
            # --- Handle tool calls initiated by the model ('assistant' role).
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    function_call = tool_call.get("function", {})
                    arguments = {}
                    # Gemini expects a dictionary, not a JSON string.
                    if isinstance(function_call.get("arguments"), str):
                        try:
                            arguments = json.loads(function_call["arguments"])
                        except json.JSONDecodeError:
                            print("Warning: Failed to decode tool arguments.")
                            pass # Default to empty dict on error.
                    
                    parts.append({
                        "functionCall": {
                            "name": function_call.get("name"),
                            "args": arguments
                        }
                    })
            
            # --- Handle tool responses ('tool' role)
            if role == "tool" and msg.get("name"):
                parts.append({
                    "functionResponse": {
                        "name": msg.get("name"),
                        "response": {"content": msg.get("content")}
                    }
                })
            
            # --- Append the final message to the list of contents.
            if parts:
                contents.append({"role": gemini_role, "parts": parts})
                
        return contents