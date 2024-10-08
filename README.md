# LLMAbstractModel

## Overview

`LLMAbstractModel` is a Python library designed to provide a standardized interface for integrating various large language models (LLMs) from different vendors. The library allows users to manage multiple LLMs, define interaction chains, and process text efficiently. This README provides a quick start guide on how to use the `LLMAbstractModel` library with the provided example code.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **LLMAbstractModel Library**: This library provides the `LLMsStore` and related classes.
- **OpenAI API Key**: Required if you intend to use OpenAI models.
- **Ollama (Optional)**: If you plan to use Ollama models, ensure Ollama is installed and configured on your system.

## Installation

1. **Clone the Repository**

   ```bash
    git clone https://github.com/qinhy/llm-abstract-model.git
    cd llm-abstract-model
   ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Set Up Environment Variables**

   The script requires the OpenAI API key to interact with OpenAI models. You can set it as an environment variable:

   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

   *On Windows, use:*

   ```cmd
   set OPENAI_API_KEY=your-openai-api-key
   ```

2. **(Optional) Configure Ollama**

   If you plan to use Ollama models, ensure Ollama is installed and properly configured on your system. Refer to the [Ollama documentation](https://ollama.com/docs) for setup instructions.

## Usage

The provided script initializes the `LLMsStore`, adds vendors and models, and demonstrates how to interact with the models.

```python
import os
from LLMAbstractModel import LLMsStore

# Initialize the LLMs store
store = LLMsStore()

# Add OpenAI vendor with API key from environment variable
vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY', 'null'))

# Add ChatGPT-4 Omnichannel model to the OpenAI vendor
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id())

## If you have Ollama installed, you can add Ollama models as follows:
# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id())
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id())
# llama32 = store.add_new_llama(vendor_id=vendor.get_id())

# Simple query example
print(chatgpt4omini('hi! What is your name?'))
# Expected Output:
# Hello! I’m called Assistant. How can I help you today?

# Push messages with roles example
messages = [
    {'role': 'system', 'content': 'You are a highly skilled professional English translator.'},
    {'role': 'user', 'content': '"こんにちは！"'}
]
print(chatgpt4omini(messages))
# Expected Output:
# Hello! I'm an AI language model created by OpenAI, and I don't have a personal name, but you can call me Assistant. How can I help you today?
# "Hello!"
```

### Running the Script

1. **Ensure the Environment Variable is Set**

   Make sure the `OPENAI_API_KEY` is set in your environment.

2. **Execute the Script**

   Save the script to a file, e.g., `HowToUse.py`, and run:

   ```bash
   python HowToUse.py
   ```

## Examples

### Simple Query

```python
response = chatgpt4omini('hi! What is your name?')
print(response)
```

**Output:**
```
Hello! I’m called Assistant. How can I help you today?
```

### Pushing Messages with Roles

```python
messages = [
    {'role': 'system', 'content': 'You are a highly skilled professional English translator.'},
    {'role': 'user', 'content': '"こんにちは！"'}
]
response = chatgpt4omini(messages)
print(response)
```

**Output:**
```
Hello! I'm an AI language model created by OpenAI, and I don't have a personal name, but you can call me Assistant. How can I help you today?
"Hello!"
```
## License

This project is licensed under the [MIT License](LICENSE).
