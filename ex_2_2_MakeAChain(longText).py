import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
from LLMAbstractModel.utils import RegxExtractor, StringTemplate, TextFile

store = LLMsStore()

system_prompt = '''I will provide pieces of the text along with prior summarizations.
Your task is to read each new text snippet and add new summarizations accordingly.
You should reply summarizations only, without any additional information.

## Your Reply Format Example
```summarization
- This text shows ...
```'''

vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key="OPENAI_API_KEY")#auto check os.environ
llm = chatgpt = store.add_new_llm(Model4LLMs.ChatGPT41Nano)(vendor_id='auto',system_prompt=system_prompt)

msg_template = store.add_new_function(StringTemplate(para=dict(
     string='''
Please reply summarizations in {lang}, and should not over {limit_words} words.


## Text Snippet
```text
{txt}
```
## Previous Summarizations
```summarization
{summary}

```'''
))).build()

res_ext = store.add_new_function(RegxExtractor(para=dict(regx=r"```summarization\s*(.*)\s*\n```")))

def test_summary(llm = llm,
                 f='The Adventures of Sherlock Holmes.txt',
                 limit_words=1000,chunk_lines=100, overlap_lines=30):
    
    pre_summarization = None
    text_file = TextFile(file_path=f, chunk_lines=chunk_lines, overlap_lines=overlap_lines)
    for i,chunk in enumerate(text_file):        
        
        msg    = msg_template(lang='Japanese', limit_words=limit_words,txt='\n'.join(chunk),summary=pre_summarization)
        output = llm(msg)
        output = res_ext(output)

        pre_summarization = output
        yield output


############# make a custom chain 
def compose(*funcs):
    def composed(*args, **kwargs):
        result = funcs[0](*args, **kwargs)  # start with last (innermost) function
        for f in funcs[1:]:
            result = f(result)
        return result
    return composed

chain_list = [msg_template, chatgpt, res_ext]
chain = compose(*chain_list)
# print(LLMsStore.chain_dumps(chain_list))
# print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))

pre_summarization = ''
for i,chunk_lines in enumerate(TextFile(file_path='The Adventures of Sherlock Holmes.txt',
                                  chunk_lines=100, overlap_lines=30)):
        summarization = chain(lang='Japanese', limit_words=100,txt='\n'.join(chunk_lines),summary=pre_summarization)
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