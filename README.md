# LLMAbstractModel README

## Overview

`LLMAbstractModel` is a Python library designed to provide a standardized interface for integrating various large language models (LLMs) from different vendors. The library allows users to manage multiple LLMs, define interaction chains, and process text efficiently. This README provides a quick start guide on how to use the `LLMAbstractModel` library with the provided example code.

## Installation

To install `LLMAbstractModel`, you need to clone the repository and install the required dependencies. Assuming you have Python installed, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/qinhy/llm-abstract-model.git
    cd llm-abstract-model
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

The following example demonstrates how to use `LLMAbstractModel` to set up and interact with multiple LLM vendors, create interaction templates, and process text files with summarization tasks.

### 1. Import Required Modules

```python
import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.utils import TextFile
```

### 2. Initialize the LLMsStore

Create an instance of `LLMsStore` to manage LLM vendors and models:

```python
store = LLMsStore()
```

### 3. Add LLM Vendors and Models

Add LLM vendors and their corresponding models to the store. In this example, we're using OpenAI and Ollama vendors with multiple models:

```python
# Add OpenAI vendor
vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY', 'null'))
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(), system_prompt='You are an expert in text summary.')

# Add Ollama vendor with multiple models
vendor = store.add_new_ollama_vendor()
gemma2 = store.add_new_gemma2(vendor_id=vendor.get_id(), system_prompt='You are an expert in text summary.')
phi3 = store.add_new_phi3(vendor_id=vendor.get_id(), system_prompt='You are an expert in text summary.')
llama32 = store.add_new_llama(vendor_id=vendor.get_id(), system_prompt='You are an expert in text summary.')
```

### 4. Create a Message Template and Extractor

Define a message template and a regular expression extractor to standardize input messages and extract responses:

````python
msg_template = store.add_new_str_template('''I will provide pieces of the text along with prior summarizations.
Your task is to read each new text snippet and add new summarizations accordingly.  
You should reply in Japanese with summarizations only, without any additional information.

## Your Reply Format Example (should not over {} words)
```summarization
- This text shows ...
```

## Text Snippet
```text
{}
```

## Previous Summarizations
```summarization
{}
```''')
````
````python
res_ext = store.add_new_regx_extractor(r"```summarization\s*(.*)\s*```")
````

### 5. Text Summarization Function

Create a function that processes a text file in chunks and generates summaries using a selected LLM:

```python
def test_summary(llm=llama32,
                 f='The Adventures of Sherlock Holmes.txt', 
                 limit_words=1000, chunk_lines=100, 
                 overlap_lines=30):
    
    pre_summarization = None

    text_file = TextFile(file_path=f,
                        chunk_lines=chunk_lines,
                        overlap_lines=overlap_lines)
                        
    for i, chunk in enumerate(text_file):
        msg = msg_template(limit_words, '\n'.join(chunk), pre_summarization)
        output = llm(msg)
        output = res_ext(output)
        pre_summarization = output
        yield output
```

### 6. Chain Creation and Testing

You can create custom chains of functions for processing inputs and outputs:

```python
from functools import reduce

def rcompose(*funcs):
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)

def compose(*funcs):
    return lambda x: reduce(lambda v, f: f(v), funcs, x)

chain_list = [
    msg_template,
    chatgpt4omini,
    res_ext
]

chain = compose(*chain_list)
print(chain((100, 'NULL', '')))
print(LLMsStore.chain_dumps(chain_list))
print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))
```

## License

This library is licensed under the MIT License.

## Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are welcome!
