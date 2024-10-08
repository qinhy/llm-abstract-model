from pydantic import Field
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions

store = LLMsStore()

system_prompt = 'You are smart assistant'

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto',system_prompt=system_prompt)

# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt=system_prompt)

@descriptions('generate Fibonacci sequence up to n-th number',
              n='the position in the Fibonacci sequence to compute')
class FibonacciFunction(Model4LLMs.Function):
    def __call__(self, n: int):
        def fibonacci(n):
            if n <= 1 : return n
            return fibonacci(n-1)+fibonacci(n-2)
        return fibonacci(n)

get_fibo = store.add_new_function(FibonacciFunction())
print(get_fibo)
# -> FibonacciFunction(rank=[0], create_time=datetime.datetime(2024, 10, 8, 10, 4, 56, 796501, tzinfo=TzInfo(UTC)), update_time=datetime.datetime(2024, 10, 8, 10, 4, 56, 796501, tzinfo=TzInfo(UTC)), status='', metadata={}, name='FibonacciFunction', description='generate Fibonacci sequence up to n-th number', parameters={'type': 'object', 'properties': {'n': Parameter(type='integer', description='the position in the Fibonacci sequence to compute')}}, required=['n'])
print(store.dumps())
# -> {"OpenAIVendor:1e38315a-f59d-4e1e-80bd-bf6ef9f6416a": {"rank": [0], "create_time": "2024-10-08T10:08:32.053810Z", "update_time": "2024-10-08T10:08:32.056719Z", "status": "", "metadata": {}, "vendor_name": "OpenAI", "api_url": "https://api.openai.com", "api_key": "OPENAI_API_KEY", "timeout": 30, "chat_endpoint": "/v1/chat/completions", "models_endpoint": "/v1/models"}, "ChatGPT4oMini:15e73aca-0fa3-4644-b75e-9f4973ab0805": {"rank": [0], "create_time": "2024-10-08T10:08:32.056719Z", "update_time": "2024-10-08T10:08:32.056719Z", "status": "", "metadata": {}, "vendor_id": "auto", "llm_model_name": "gpt-4o-mini", "context_window_tokens": 128000, "max_output_tokens": 4096, "stream": false, "limit_output_tokens": 1024, "temperature": 0.7, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "system_prompt": "You are smart assistant", "stop_sequences": [], "n": 1}, "FibonacciFunction:b0535cb1-f865-49a7-b466-fa064768b4e9": {"rank": [0], "create_time": "2024-10-08T10:08:32.058724Z", "update_time": "2024-10-08T10:08:32.058724Z", "status": "", "metadata": {}, "name": "FibonacciFunction", "description": "generate Fibonacci sequence up to n-th number", "parameters": {"type": "object", "properties": {"n": {"type": "integer", "description": "the position in the Fibonacci sequence to compute"}}}, "required": ["n"]}}
print(store.find_all("FibonacciFunction:*")[0](7))
# -> 13