from pydantic import Field
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions

store = LLMsStore()

system_prompt = 'You are smart assistant'

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto',system_prompt=system_prompt)

# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt=system_prompt)

## add custom function
@descriptions('generate Fibonacci sequence up to n-th number',
              n='the position in the Fibonacci sequence to compute')
class FibonacciFunction(Model4LLMs.Function):
    def __call__(self, n: int):
        def fibonacci(n):
            if n <= 1 : return n
            return fibonacci(n-1)+fibonacci(n-2)
        return fibonacci(n)

get_fibo = store.add_new_function(FibonacciFunction())
print(get_fibo.model_dump())
# -> {'rank': [0], 'create_time': datetime.datetime(2024, 10, 10, 11, 2, 18, 54621, tzinfo=TzInfo(UTC)), 'update_time': datetime.datetime(2024, 10, 10, 11, 2, 18, 54621, tzinfo=TzInfo(UTC)), 'status': '', 'metadata': {}, 'name': 'FibonacciFunction', 'description': 'generate Fibonacci sequence up to n-th number', 'parameters': {'type': 'object', 'properties': {'n': {'type': 'integer', 'description': 'the position in the Fibonacci sequence to compute'}}}, 'required': ['n']}
print(store.find_all("FibonacciFunction:*")[0](7))
# -> 13

## add custom Obj
class FibonacciObj(Model4LLMs.AbstractObj):
    n:int

fb = store.add_new_obj(FibonacciObj(n=7))
print(store.find_all("FibonacciObj:*")[0].model_dump())
# -> {'rank': [0], 'create_time': datetime.datetime(2024, 10, 10, 11, 2, 18, 56621, tzinfo=TzInfo(UTC)), 'update_time': datetime.datetime(2024, 10, 10, 11, 2, 18, 56621, tzinfo=TzInfo(UTC)), 'status': '', 'metadata': {}, 'n': 7}

print(store.dumps())
# -> {"OpenAIVendor:e4c273e4-3c3c-4b5f-bdf9-07d213776d61": {"rank": [0], "create_time": "2024-10-10T11:02:18.050786Z", "update_time": "2024-10-10T11:02:18.052575Z", "status": "", "metadata": {}, "vendor_name": "OpenAI", "api_url": "https://api.openai.com", "api_key": "OPENAI_API_KEY", "timeout": 30, "chat_endpoint": "/v1/chat/completions", "models_endpoint": "/v1/models"}, "ChatGPT4oMini:d79acb03-2f3f-4bc6-9762-67d706b3f36e": {"rank": [0], "create_time": "2024-10-10T11:02:18.052575Z", "update_time": "2024-10-10T11:02:18.052575Z", "status": "", "metadata": {}, "vendor_id": "auto", "llm_model_name": "gpt-4o-mini", "context_window_tokens": 128000, "max_output_tokens": 4096, "stream": false, "limit_output_tokens": 1024, "temperature": 0.7, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "system_prompt": "You are smart assistant", "stop_sequences": [], "n": 1}, "FibonacciFunction:6ca3e5c3-2539-4f38-a5c7-39665af8bc31": {"rank": [0], "create_time": "2024-10-10T11:02:18.054621Z", "update_time": "2024-10-10T11:02:18.054621Z", "status": "", "metadata": {}, "name": "FibonacciFunction", "description": "generate Fibonacci sequence up to n-th number", "parameters": {"type": "object", "properties": {"n": {"type": "integer", "description": "the position in the Fibonacci sequence to compute"}}}, "required": ["n"]}, "FibonacciObj:c13de394-1999-4828-8ab6-38b3840777ad": {"rank": [0], "create_time": "2024-10-10T11:02:18.056621Z", "update_time": "2024-10-10T11:02:18.056621Z", "status": "", "metadata": {}, "n": 7}}