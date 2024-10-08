import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.utils import RegxExtractor, StringTemplate, TextFile
store = LLMsStore()

system_prompt = '''I will provide pieces of the text along with prior summarizations.
Your task is to read each new text snippet and add new summarizations accordingly.
You should reply summarizations only, without any additional information.

## Your Reply Format Example
```summarization
- This text shows ...
```'''

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(),system_prompt=system_prompt)

vendor  = store.add_new_ollama_vendor()
gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt=system_prompt)
phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt=system_prompt)
llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt=system_prompt)

msg_template = store.add_new_function(StringTemplate(string='''
Please reply summarizations in {}, and should not over {} words.
## Text Snippet
```text
{}
```
## Previous Summarizations
```summarization
{}
```'''))

res_ext = store.add_new_function(RegxExtractor(regx=r"```summarization\s*(.*)\s*\n```"))

def test_summary(llm = llama32,
                 f='The Adventures of Sherlock Holmes.txt',
                 limit_words=1000,chunk_lines=100, overlap_lines=30):
    
    pre_summarization = None
    text_file = TextFile(file_path=f, chunk_lines=chunk_lines, overlap_lines=overlap_lines)
    for i,chunk in enumerate(text_file):        
        
        msg    = msg_template('Japanese', limit_words,'\n'.join(chunk),pre_summarization)        
        output = llm(msg)
        output = res_ext(output)

        pre_summarization = output
        yield output


############# make a custom chain 
from functools import reduce
def compose(*funcs):
    def chained_function(*args, **kwargs):
        return reduce(lambda acc, f: f(*acc if isinstance(acc, tuple) else (acc,), **kwargs), funcs, args)
    return chained_function

chain_list = [msg_template, chatgpt4omini, res_ext]
chain = compose(*chain_list)
# print(LLMsStore.chain_dumps(chain_list))
# print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))

pre_summarization = ''
for i,chunk_lines in enumerate(TextFile(file_path='The Adventures of Sherlock Holmes.txt',
                                  chunk_lines=100, overlap_lines=30)):
        summarization = chain('Japanese', 100,'\n'.join(chunk_lines),pre_summarization)
        pre_summarization = summarization
        print(summarization)
        break

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