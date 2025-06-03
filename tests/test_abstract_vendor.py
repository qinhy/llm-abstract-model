import unittest
from unittest.mock import patch, MagicMock
import os
import json
import requests
from requests.exceptions import RequestException, Timeout

# Import the AbstractVendor class
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LLMAbstractModel.ModelInterface.AbstractVendor import AbstractVendor


class TestAbstractVendor(unittest.TestCase):
    """Test cases for the AbstractVendor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a concrete instance of AbstractVendor for testing
        self.vendor = AbstractVendor(
            vendor_name="TestVendor",
            api_url="https://api.test-vendor.com",
            api_key="test_api_key",
            chat_endpoint="/chat",
            models_endpoint="/models",
            embeddings_endpoint="/embeddings",
            rate_limit=100
        )

    def test_initialization(self):
        """Test proper initialization of AbstractVendor instance."""
        self.assertEqual(self.vendor.vendor_name, "TestVendor")
        self.assertEqual(self.vendor.api_url, "https://api.test-vendor.com")
        self.assertEqual(self.vendor.api_key, "test_api_key")
        self.assertEqual(self.vendor.chat_endpoint, "/chat")
        self.assertEqual(self.vendor.models_endpoint, "/models")
        self.assertEqual(self.vendor.embeddings_endpoint, "/embeddings")
        self.assertEqual(self.vendor.rate_limit, 100)
        self.assertEqual(self.vendor.timeout, 30)  # Default value

    def test_format_llm_model_name(self):
        """Test the format_llm_model_name method."""
        model_name = "gpt-4"
        formatted_name = self.vendor.format_llm_model_name(model_name)
        self.assertEqual(formatted_name, model_name)  # Default implementation returns the same name

    @patch.dict(os.environ, {"TEST_API_KEY": "env_api_key"})
    def test_get_api_key_from_env(self):
        """Test retrieving API key from environment variables."""
        # Set up vendor with an environment variable name as the api_key
        vendor = AbstractVendor(
            vendor_name="TestVendor",
            api_url="https://api.test-vendor.com",
            api_key="TEST_API_KEY"
        )
        self.assertEqual(vendor.get_api_key(), "env_api_key")

    def test_get_api_key_direct(self):
        """Test retrieving API key directly when not in environment."""
        self.assertEqual(self.vendor.get_api_key(), "test_api_key")

    def test_build_headers_with_api_key(self):
        """Test building headers with API key."""
        headers = self.vendor._build_headers()
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Authorization"], "Bearer test_api_key")

    def test_build_headers_without_api_key(self):
        """Test building headers without API key."""
        vendor = AbstractVendor(
            vendor_name="TestVendor",
            api_url="https://api.test-vendor.com",
            api_key=None
        )
        headers = vendor._build_headers()
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertNotIn("Authorization", headers)

    def test_build_url(self):
        """Test URL construction with various endpoint formats."""
        # Test with leading slash in endpoint
        url = self.vendor._build_url("/test/endpoint")
        self.assertEqual(url, "https://api.test-vendor.com/test/endpoint")

        # Test without leading slash in endpoint
        url = self.vendor._build_url("test/endpoint")
        self.assertEqual(url, "https://api.test-vendor.com/test/endpoint")

        # Test with trailing slash in API URL
        vendor = AbstractVendor(
            vendor_name="TestVendor",
            api_url="https://api.test-vendor.com/",
            api_key="test_api_key"
        )
        url = vendor._build_url("/test/endpoint")
        self.assertEqual(url, "https://api.test-vendor.com/test/endpoint")

    def test_handle_response_success(self):
        """Test handling a successful API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": ["model1", "model2"]}
        
        result = self.vendor._handle_response(mock_response)
        
        self.assertEqual(result, {"status": "success", "data": ["model1", "model2"]})
        mock_response.json.assert_called_once()

    def test_handle_response_error(self):
        """Test handling an API response that can't be parsed as JSON."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        result = self.vendor._handle_response(mock_response)
        
        self.assertIn("error", result)
        self.assertIn("Failed to parse response", result["error"])
        self.assertIn("Invalid JSON", result["error"])

    @patch("requests.get")
    def test_get_available_models_success(self, mock_get):
        """Test successful retrieval of available models."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": ["model1", "model2"]}
        mock_get.return_value = mock_response
        
        result = self.vendor.get_available_models()
        
        # Verify the result
        self.assertEqual(result, {"models": ["model1", "model2"]})
        
        # Verify the request was made correctly
        mock_get.assert_called_once_with(
            "https://api.test-vendor.com/models",
            headers=self.vendor._build_headers(),
            timeout=30
        )

    @patch("requests.get")
    def test_get_available_models_error(self, mock_get):
        """Test error handling when retrieving available models."""
        # Set up the mock to raise an exception
        mock_get.side_effect = RequestException("Connection error")
        
        # Call the method which should handle the exception
        result = self.vendor.get_available_models()
        
        # Verify the error message is in the result
        self.assertIn("error", result)
        self.assertIn("Connection error", result["error"])

    @patch("requests.post")
    def test_chat_request_success(self, mock_post):
        """Test successful chat request."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_post.return_value = mock_response
        
        # Set up the request payload
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        with patch.object(self.vendor, '_handle_response', return_value=mock_response.json.return_value):
            result = self.vendor.chat_request(payload)
        
        # Verify the result
        self.assertEqual(result, {"choices": [{"message": {"content": "Hello!"}}]})
        
        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            url="https://api.test-vendor.com/chat",
            data=json.dumps(payload, ensure_ascii=True),
            headers=self.vendor._build_headers(),
            timeout=30
        )

    def test_chat_request_no_endpoint(self):
        """Test chat request when chat_endpoint is not defined."""
        # Create a vendor without a chat endpoint
        vendor = AbstractVendor(
            vendor_name="TestVendor",
            api_url="https://api.test-vendor.com",
            api_key="test_api_key",
            chat_endpoint=None
        )
        
        result = vendor.chat_request({"model": "gpt-4"})
        
        # Verify the error message
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Chat endpoint not defined for this vendor.")

    @patch("requests.post")
    def test_chat_request_network_error(self, mock_post):
        """Test chat request with network error."""
        # Set up the mock to raise an exception
        mock_post.side_effect = RequestException("Connection error")
        
        result = self.vendor.chat_request({"model": "gpt-4"})
        
        # Verify the error message
        self.assertIn("error", result)
        self.assertIn("Request failed", result["error"])
        self.assertIn("Connection error", result["error"])

    def test_chat_result(self):
        """Test the chat_result method."""
        # The default implementation just returns the response
        response = {"choices": [{"message": {"content": "Hello!"}}]}
        result = self.vendor.chat_result(response)
        self.assertEqual(result, response)

    @patch("requests.post")
    def test_embedding_request_success(self, mock_post):
        """Test successful embedding request."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{
                "embedding": [0.1, 0.2, 0.3]
            }]
        }
        mock_post.return_value = mock_response
        
        # Set up the request payload
        payload = {
            "model": "text-embedding-ada-002",
            "input": "Hello, world!"
        }
        
        with patch.object(self.vendor, '_handle_response', return_value=mock_response.json.return_value):
            result = self.vendor.embedding_request(payload)
        
        # Verify the result
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            url="https://api.test-vendor.com/embeddings",
            data=json.dumps(payload, ensure_ascii=True),
            headers=self.vendor._build_headers(),
            timeout=30
        )

    def test_embedding_request_no_endpoint(self):
        """Test embedding request when embeddings_endpoint is not defined."""
        # Create a vendor without an embeddings endpoint
        vendor = AbstractVendor(
            vendor_name="TestVendor",
            api_url="https://api.test-vendor.com",
            api_key="test_api_key",
            embeddings_endpoint=None
        )
        
        result = vendor.embedding_request({"model": "text-embedding-ada-002", "input": "Hello"})
        
        # Verify the error message
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Embeddings endpoint not defined for this vendor.")

    @patch("requests.post")
    def test_embedding_request_network_error(self, mock_post):
        """Test embedding request with network error."""
        # Set up the mock to raise an exception
        mock_post.side_effect = RequestException("Connection error")
        
        result = self.vendor.embedding_request({"model": "text-embedding-ada-002", "input": "Hello"})
        
        # Verify the error message
        self.assertIn("error", result)
        self.assertIn("Request failed", result["error"])
        self.assertIn("Connection error", result["error"])

    @patch("requests.post")
    def test_embedding_request_parsing_error(self, mock_post):
        """Test embedding request with response parsing error."""
        # Set up the mock response with invalid structure
        mock_response = MagicMock()
        mock_response.json.return_value = {"invalid": "structure"}  # Missing 'data' key
        mock_post.return_value = mock_response
        
        with patch.object(self.vendor, '_handle_response', return_value=mock_response.json.return_value):
            result = self.vendor.embedding_request({"model": "text-embedding-ada-002", "input": "Hello"})
        
        # Verify the error message
        self.assertIn("error", result)
        self.assertIn("Failed to extract embedding", result["error"])

    @patch.object(AbstractVendor, "embedding_request")
    def test_get_embedding(self, mock_embedding_request):
        """Test the get_embedding method."""
        # Set up the mock to return a sample embedding
        mock_embedding_request.return_value = [0.1, 0.2, 0.3]
        
        result = self.vendor.get_embedding("Hello, world!", "text-embedding-ada-002")
        
        # Verify the result
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # Verify embedding_request was called with the correct payload
        mock_embedding_request.assert_called_once_with({
            "model": "text-embedding-ada-002",
            "input": "Hello, world!"
        })

    @patch.object(AbstractVendor, "embedding_request")
    def test_get_embedding_error(self, mock_embedding_request):
        """Test the get_embedding method with an error."""
        # Set up the mock to return an error
        mock_embedding_request.return_value = {"error": "Something went wrong"}
        
        result = self.vendor.get_embedding("Hello, world!", "text-embedding-ada-002")
        
        # Verify the error is passed through
        self.assertEqual(result, {"error": "Something went wrong"})


if __name__ == "__main__":
    unittest.main()