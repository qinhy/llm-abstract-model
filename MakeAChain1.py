import json
import re
import sys
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
store = LLMsStore()

system_prompt='''You are an expert in English translation. I will provide you with the text. Please translate it. You should reply with translations only, without any additional information.
## Your Reply Format Example
```translation
...
```'''

vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")#auto check os.environ
llm = chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto',#vendor.get_id(),
                                                  system_prompt=system_prompt)

# vendor  = store.add_new_ollama_vendor()
# llm = llama32 = store.add_new_llama(vendor_id='auto',#vendor.get_id(),
#                                     system_prompt=system_prompt)
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt=system_prompt)

print('############# make a message template')
translate_template = store.add_new_str_template(
'''
```text
{}
```''')
# the usage of template is tmp( [args1,args2,...] ) is the same of sting.format(*[...])
print(translate_template('こんにちは！はじめてのチェーン作りです！'))
# -> ...

print(llm( translate_template('こんにちは！はじめてのチェーン作りです！')))
# -> ```translation
# -> Hello! This is my first time making a chain!
# -> ```


print('############# make a "translation" extractor, strings between " ```translation " and " ``` "')
get_result = store.add_new_regx_extractor(r"```translation\s*(.*)\s*\n```")

descriptions = Model4LLMs.Function.param_descriptions
@descriptions('Extract text by regx pattern',
              regx='regx pattern')
class RegxExtractor(Model4LLMs.Function):
    regx:str
    def __call__(self,*args,**kwargs):
        return self.extract(*args,**kwargs)
    
    def extract(self,text):
        matches = re.findall(self.regx, text, re.DOTALL)        
        if not self._try_binary_error(lambda:matches[0]):
            self._log_error(ValueError(f'cannot match {self.regx} at {text}'))
            return text
        return matches[0]

store.add_new_function(RegxExtractor(regx=r"```translation\s*(.*)\s*\n```"))
get_result2 = store.find_function('RegxExtractor')

# this just a chain like processs
print(get_result2(
        llm(
            translate_template('こんにちは！はじめてのチェーン作りです！') 
            )))
# -> Hello! This is my first time making a chain!

print(store.dumps())

sys.exit(0)

print('############# make the chain more graceful and simple')
# chain up functions
from functools import reduce
def compose(*funcs):
    def chained_function(*args, **kwargs):
        return reduce(lambda acc, f: f(*acc if isinstance(acc, tuple) else (acc,), **kwargs), funcs, args)
    return chained_function

def say_a(x):return f'{x}a'
def say_b(x):return f'{x}b'
def say_c(x):return f'{x}c'

say_abc = compose(say_a,say_b,say_c)
print('test chain up:',say_abc(''))

translator_chain = [translate_template, llm, get_result]
translator = compose(*translator_chain)

print(translator('こんにちは！はじめてのチェーン作りです！'))
# -> Hello! This is my first time making a chain!

print(translator('常識とは、18歳までに身に付けた偏見のコレクションである。'))
# -> Common sense is a collection of prejudices acquired by the age of 18.

print(translator('为政以德，譬如北辰，居其所而众星共之。'))
# -> Governing with virtue is like the North Star, which remains in its place while all the other stars revolve around it.


print('############# save/load chain json')
print(LLMsStore.chain_dumps(translator_chain))

loaded_chain = store.chain_loads(LLMsStore.chain_dumps(translator_chain))
print(loaded_chain)
translator = compose(*loaded_chain)
print(translator('こんにちは！はじめてのチェーン作りです！'))
# -> Hello! It's my first time making a chain!


print('############# additional template usage')

llm = store.add_new_chatgpt4omini(vendor_id='auto')
# llm = store.add_new_llama(vendor_id='auto')

# we use raw json to do template
translate_template = store.add_new_str_template(
'''[
    {{"role":"system","content":"You are an expert in translation text.I will you provide text. Please tranlate it.\\nYou should reply translations only, without any additional information.\\n\\n## Your Reply Format Example\\n```translation\\n...\\n```"}},
    {{"role":"user","content":"\\nPlease translate the text in{}.\\n```text\\n{}\\n```"}}
]''')
# the usage of template is tmp( [args1,args2,...] ) is the same of sting.format(*[...])
print(translate_template('to English','こんにちは！はじめてのチェーン作りです！'))
# -> [
# ->     {"role":"system","content":"You are an expert in translation text.I will you provide text. Please tranlate it.\nYou should reply translations only, without any additional information.\n\n## -> Your Reply Format Example\n```translation\n...\n```"},
# ->     {"role":"user","content":"\nPlease translate the text into English.\n```text\nこんにちは！はじめてのチェーン作りです！\n```"}
# -> ]

msg = json.loads( translate_template('to English','こんにちは！はじめてのチェーン作りです！'))
print(msg)
# -> [{'role': 'system', 'content': 'You are an expert in translation text.I will you provide text. Please tranlate it.\nYou should reply translations only, without any additional information.\n\n## Your Reply Format Example\n```translation\n...\n```'}, {'role': 'user', 'content': '\nPlease translate the text into English.\n```text\nこんにちは！はじめてのチェーン作りです！\n```'}]

print(llm(msg))
# -> ```translation
# -> Hello! This is my first time making a chain!
# -> ```

translator_chain = [translate_template, json.loads, llm, get_result]
translator = compose(*translator_chain)
print(translator('to Chinese','こんにちは！はじめてのチェーン作りです！'))
# -> 你好！这是第一次制作链条！
print(translator('to Japanese','Hello! This is my first time making a chain!'))
# -> こんにちは！これは私の初めてのチェーン作りです！