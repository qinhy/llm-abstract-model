import unittest
from unittest.mock import MagicMock, patch
import json
from typing import Dict, List, Any, Optional

# Import the classes to test
from LLMAbstractModel.ModelInterface.AbstractLLM import AbstractLLM, MCPTool, MCPToolAnnotations


class TestMCPToolAnnotations(unittest.TestCase):
    """Test cases for the MCPToolAnnotations class."""

    def test_initialization(self):
        """Test proper initialization of MCPToolAnnotations."""
        annotations = MCPToolAnnotations(
            title="Test Tool",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False
        )
        
        self.assertEqual(annotations.title, "Test Tool")
        self.assertTrue(annotations.readOnlyHint)
        self.assertFalse(annotations.destructiveHint)
        self.assertTrue(annotations.idempotentHint)
        self.assertFalse(annotations.openWorldHint)

    def test_default_values(self):
        """Test default values when initializing without parameters."""
        annotations = MCPToolAnnotations()
        
        self.assertIsNone(annotations.title)
        self.assertIsNone(annotations.readOnlyHint)
        self.assertIsNone(annotations.destructiveHint)
        self.assertIsNone(annotations.idempotentHint)
        self.assertIsNone(annotations.openWorldHint)


class TestMCPTool(unittest.TestCase):
    """Test cases for the MCPTool class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
        
        self.annotations = MCPToolAnnotations(
            title="Search Tool",
            readOnlyHint=True
        )
        
        self.tool = MCPTool(
            name="search",
            description="Search for information",
            inputSchema=self.input_schema,
            annotations=self.annotations
        )

    def test_initialization(self):
        """Test proper initialization of MCPTool."""
        self.assertEqual(self.tool.name, "search")
        self.assertEqual(self.tool.description, "Search for information")
        self.assertEqual(self.tool.inputSchema, self.input_schema)
        self.assertEqual(self.tool.annotations, self.annotations)

    def test_to_openai_tool(self):
        """Test conversion to OpenAI tool format."""
        openai_tool = self.tool.to_openai_tool()
        
        self.assertEqual(openai_tool["type"], "function")
        self.assertEqual(openai_tool["function"]["name"], "search")
        self.assertEqual(openai_tool["function"]["description"], "Search for information")
        self.assertEqual(openai_tool["function"]["parameters"], self.input_schema)

    def test_minimal_initialization(self):
        """Test initialization with only required parameters."""
        minimal_tool = MCPTool(
            name="minimal",
            inputSchema={"type": "object"}
        )
        
        self.assertEqual(minimal_tool.name, "minimal")
        self.assertIsNone(minimal_tool.description)
        self.assertEqual(minimal_tool.inputSchema, {"type": "object"})
        self.assertIsNone(minimal_tool.annotations)


if __name__ == '__main__':
    unittest.main()