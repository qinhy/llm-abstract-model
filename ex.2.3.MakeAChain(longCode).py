import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.utils import RegxExtractor, StringTemplate, TextFile
store = LLMsStore()

system_prompt = '''Your task is continue to complete the given "Refactored Code" by referencing the logic and details from the provided "Reference Code".


For Example:

```Reference
    for i in range(1, 11):
        result += x ** i / i
    return result

# additional logic to handle input and special cases
def adjust_value(x):
    if x < 0:
        return -x + 5
    return x
```


```Refactored
def complex_math(x):
    return sum(x ** i / i for i in range(1, 11))
```


Your reply format should be as follows:
```Continue
def adjust_value(x):
    return -x + 5 if x < 0 else x
```
'''

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY',timeout=60)
llm = store.add_new_chatgpt4o(vendor_id=vendor.get_id(),system_prompt=system_prompt,limit_output_tokens=2048)

# vendor = store.add_new_Xai_vendor(api_key=os.environ.get('XAI_API_KEY','null'),timeout=60)
# llm = grok = store.add_new_grok(vendor_id=vendor.get_id(),system_prompt=system_prompt,limit_output_tokens=2048)

# vendor = store.add_new_deepseek_vendor(api_key=os.environ.get('DEEPSEEK_API_KEY','null'),timeout=600)
# llm = deepseek = store.add_new_deepseek(vendor_id=vendor.get_id(),llm_model_name='deepseek-chat')

# Please reply refactored code in {}, and should not over {} words.
msg_template = store.add_new_function(StringTemplate(string='''
```Reference
{}
```

```Refactored
{}
```'''))

res_ext = store.add_new_function(RegxExtractor(regx=r"```Continue\s*(.*)\s*\n```"))

############# make a custom chain 
from functools import reduce
def compose(*funcs):
    def chained_function(*args, **kwargs):
        return reduce(lambda acc, f: f(*acc if isinstance(acc, tuple) else (acc,), **kwargs), funcs, args)
    return chained_function

chain_list = [msg_template, llm, res_ext]
chain = compose(*chain_list)
# print(LLMsStore.chain_dumps(chain_list))
# print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))

pre_code = '# -*- coding: utf-8 -*-'
for i,chunk_lines in enumerate(TextFile(file_path='./LLMAbstractModel/RSA.py',
                                  chunk_lines=100, overlap_lines=30)):
        print(msg_template(''.join(chunk_lines),pre_code))
        code = chain(''.join(chunk_lines),pre_code)
        with open('./tmp/RSA.py','a') as f:
             f.write('\n'+code)
        print('#########################')
        print(code)
        pre_code = code
        # break