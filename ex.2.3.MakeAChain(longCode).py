import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.utils import RegxExtractor, StringTemplate, TextFile
store = LLMsStore()

system_prompt = '''Your task is to refactor "Python Code Snippet" into a cleaner and more efficient version.
Append the refactored code to the "Previously Graceful Refactored Code" by considering further improvements.

## Instructions
- Focus on improving elegance, refactoring redundant sections, and utilizing Python best practices.
- Prioritize concise, readable, and maintainable code.
- Take advantage of Python built-in libraries to simplify complex logic whenever possible.
- Avoid adding comments or explanations.

## Output Format
- Provide only the newly enhanced refactored Python code without any explanations, old code, or inline comments. 
- Ensure consistency with the stylistic choices used in the "Previously Graceful Refactored Code".

## Example
### Question Example
#### Original Python Code Snippet:
```python
for i in range(1, 11):
    result += x ** i / i
return result

# additional logic to handle input and special cases
def adjust_value(x):
    if x < 0:
        return -x + 5
    return x
```
#### Previously Graceful Refactored Code:
```python
def complex_math(x):
    return sum(x ** i / i for i in range(1, 11))
```

### Reply Example:
```python
def adjust_value(x):
    return -x + 5 if x < 0 else x
```

### Target for Refinement:
```python
# additional logic to handle input and special cases
def adjust_value(x):
    if x < 0:
        return -x + 5
    return x
```
'''

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY',timeout=60)
llm = store.add_new_chatgpt4o(vendor_id=vendor.get_id(),system_prompt=system_prompt,limit_output_tokens=2048)

# vendor = store.add_new_Xai_vendor(api_key=os.environ.get('XAI_API_KEY','null'),timeout=60)
# llm = grok = store.add_new_grok(vendor_id=vendor.get_id(),system_prompt=system_prompt,limit_output_tokens=2048)

# Please reply refactored code in {}, and should not over {} words.
msg_template = store.add_new_function(StringTemplate(string='''
#### Original Python Code Snippet
```python
{}
```
##### Previously Graceful Refactored Code
```python
{}
```'''))

res_ext = store.add_new_function(RegxExtractor(regx=r"```python\s*(.*)\s*\n```"))

def test_summary(llm = llm,
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

chain_list = [msg_template, llm, res_ext]
chain = compose(*chain_list)
# print(LLMsStore.chain_dumps(chain_list))
# print(store.chain_loads(LLMsStore.chain_dumps(chain_list)))

pre_code = ''
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