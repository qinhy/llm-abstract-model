import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
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

vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY',timeout=60)
llm = store.add_new_llm(Model4LLMs.ChatGPT4o)(
                        vendor_id=vendor.get_id(),system_prompt=system_prompt,limit_output_tokens=2048)

# Please reply refactored code in {}, and should not over {} words.
msg_template = store.add_new_function(StringTemplate(para=dict(
     string='''
```Reference
{ref}
```

```Refactored
{rfa}
```'''
))).build()

res_ext = store.add_new_function(RegxExtractor(para=dict(regx=r"```Continue\s*(.*)\s*\n```")))

############# make a custom chain 
def compose(*funcs):
    def composed(*args, **kwargs):
        result = funcs[0](*args, **kwargs)  # start with last (innermost) function
        for f in funcs[1:]:
            result = f(result)
        return result
    return composed

chain_list = [msg_template, llm, res_ext]
chain = compose(*chain_list)
# print(LLMsStore.chain_dumps(chain_list))
# print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))

pre_code = '# -*- coding: utf-8 -*-'
for i,chunk_lines in enumerate(TextFile(file_path='./LLMAbstractModel/RSA.py',
                                  chunk_lines=100, overlap_lines=30)):
        print(msg_template(ref=''.join(chunk_lines),rfa=pre_code))
        code = chain(ref=''.join(chunk_lines),rfa=pre_code)
        with open('./tmp/RSA.py','a') as f:
             f.write('\n'+code)
        print('#########################')
        print(code)
        pre_code = code
        break