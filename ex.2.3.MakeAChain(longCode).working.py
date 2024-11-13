import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.utils import RegxExtractor, StringTemplate, TextFile
store = LLMsStore()

system_prompt = '''I will provide original Python code segments and your previously refactored versions.
Please review each new Python snippet and produce a refined, graceful refactoring.
Respond with only the refactored python code, without additional commentary or repetition.
'''

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
chatgpt4o = store.add_new_chatgpt4o(vendor_id=vendor.get_id(),system_prompt=system_prompt)

# Please reply refactored code in {}, and should not over {} words.
msg_template = store.add_new_function(StringTemplate(string='''
## Original python code Snippet
```python
{}
```
## Previous graceful refactored code
```python
{}
```'''))

res_ext = store.add_new_function(RegxExtractor(regx=r"```python\s*(.*)\s*\n```"))

def test_summary(llm = chatgpt4o,
                 f='The Adventures of Sherlock Holmes.txt',
                 limit_words=1000,chunk_lines=100, overlap_lines=30):
    
    pre_code = None
    text_file = TextFile(file_path=f, chunk_lines=chunk_lines, overlap_lines=overlap_lines)
    for i,chunk in enumerate(text_file):        
        
        msg    = msg_template('\n'.join(chunk),pre_code)        
        output = llm(msg)
        print('#########################')
        print(output)
        output = res_ext(output)

        pre_code = output
        yield output


############# make a custom chain 
from functools import reduce
def compose(*funcs):
    def chained_function(*args, **kwargs):
        return reduce(lambda acc, f: f(*acc if isinstance(acc, tuple) else (acc,), **kwargs), funcs, args)
    return chained_function

chain_list = [msg_template, chatgpt4o, res_ext]
chain = compose(*chain_list)
# print(LLMsStore.chain_dumps(chain_list))
# print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))

pre_code = ''
for i,chunk_lines in enumerate(TextFile(file_path='./LLMAbstractModel/RSA.py',
                                  chunk_lines=100, overlap_lines=30)):
        code = chain('\n'.join(chunk_lines),pre_code)
        with open('./tmp/RSA.py','a') as f:
             f.write('\n'+code)
        print('#########################')
        print(code)
        pre_code = code
        # break

# for i,p in enumerate(['file1','file2','file3','filen']):
#     ts = test_summary(p)
#     res = ''
#     for s in ts:
#         with open(p.replace('file','output'),'a') as f:
#             f.write(s)
#             res += s
#     with open(f'allinone','a') as f:
#         f.write(f'## {p}\n')
#         f.write(res)