import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.utils import TextFile
store = LLMsStore()

vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(),system_prompt='You are an expert in text summary.')

vendor  = store.add_new_ollama_vendor()
gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt='You are an expert in text summary.')
phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt='You are an expert in text summary.')
llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt='You are an expert in text summary.')

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

res_ext = store.add_new_regx_extractor(r"```summarization\s*(.*)\s*```")

def test_summary(llm = llama32,
                 f='The Adventures of Sherlock Holmes.txt',
                 limit_words=1000,chunk_lines=100, overlap_lines=30):
    
    pre_summarization = None
    text_file = TextFile(file_path=f, chunk_lines=chunk_lines, overlap_lines=overlap_lines)
    for i,chunk in enumerate(text_file):        
        
        msg    = msg_template(limit_words,'\n'.join(chunk),pre_summarization)        
        output = llm(msg)
        output = res_ext(output)

        pre_summarization = output
        yield output


############# make a custom chain 
from functools import reduce
def compose(*funcs):
    return lambda x: reduce(lambda v, f: f(v), funcs, x)

chain_list = [msg_template, chatgpt4omini, res_ext]
chain = compose(*chain_list)
print(chain( [100,'NULL',''] )) # limit_words, the text, pre_summarization
# print(LLMsStore.chain_dumps(chain_list))
# print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))

pre_summarization = ''
for i,chunk_lines in enumerate(TextFile(file_path='The Adventures of Sherlock Holmes.txt',
                                  chunk_lines=100, overlap_lines=30)):
        summarization = chain( [100,'\n'.join(chunk_lines),pre_summarization] )
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