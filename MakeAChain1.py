import json
import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.utils import TextFile
store = LLMsStore()

vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(),
                                            system_prompt='You are an expert in English translation.')

# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt='You are an expert in English translation.')
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt='You are an expert in English translation.')
# llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt='You are an expert in English translation.')

##################### make a message template
translate_template = store.add_new_str_template(
'''I will provide text. Please tranlate it.
You should reply translations only, without any additional information.

## Your Reply Format Example
```translation
...
```
## The Text
```text
{}
```                                          
```''')
# the usage of template is tmp( [args1,args2,...] ) is the same of sting.format(*[...])
print(translate_template( ['こんにちは！はじめてのチェーン作りです！',] ))
# -> ...

print(chatgpt4omini( translate_template( ['こんにちは！はじめてのチェーン作りです！',] ) ))
# -> ```translation
# -> Hello! This is my first time making a chain!
# -> ```


# #################### make a "translation" extractor, strings between " ```translation " and " ``` "
get_result = store.add_new_regx_extractor(r"```translation\s*(.*)\s*```")

# this just a chain like processs
print(get_result(
        chatgpt4omini(
            translate_template( ['こんにちは！はじめてのチェーン作りです！',] ) 
            )))
# -> Hello! This is my first time making a chain!

##################### make the chain more graceful and simple
# chain up functions
from functools import reduce
def compose(*funcs): return lambda x: reduce(lambda v, f: f(v), funcs, x)

def say_a(x):return f'{x}a'
def say_b(x):return f'{x}b'
def say_c(x):return f'{x}c'

say_abc = compose(say_a,say_b,say_c)
print('test chain up:',say_abc(''))

translator_chain = [translate_template, chatgpt4omini, get_result]
translator = compose(*translator_chain)

print(translator(['こんにちは！はじめてのチェーン作りです！',]))
# -> Hello! This is my first time making a chain!

print(translator(['常識とは、18歳までに身に付けた偏見のコレクションである。',]))
# -> Common sense is a collection of prejudices acquired by the age of 18.

print(translator(['为政以德，譬如北辰，居其所而众星共之。',]))
# -> Governing with virtue is like the North Star, which remains in its place while all the other stars revolve around it.

############# save/load chain json
print(LLMsStore.chain_dumps(translator_chain))

loaded_chain = store.chain_loads(LLMsStore.chain_dumps(translator_chain))
print(loaded_chain)
translator = compose(*loaded_chain)
print(translator(['こんにちは！はじめてのチェーン作りです！',]))
# -> Hello! It's my first time making a chain!






############# additional template usage

chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id())

# we use raw json to do template
translate_template = store.add_new_str_template(
'''[
    {{"role":"system","content":"You are an expert in translation text ({})."}},
    {{"role":"user","content":"I will provide text. Please tranlate it.\\nYou should reply translations only, without any additional information.\\n\\n## Your Reply Format Example\\n```translation\\n...\\n```\\n## The Text\\n```text\\n{}\\n```\\n```"}}
]''')
# the usage of template is tmp( [args1,args2,...] ) is the same of sting.format(*[...])
print(translate_template( ['to English','こんにちは！はじめてのチェーン作りです！',] ))
# -> [
# ->     {"role":"system","content":"You are an expert in translation text (to English)."},
# ->     {"role":"user","content":"I will provide text. Please tranlate it.\nYou should reply translations only, without any additional information.\n\n## Your Reply Format Example\n```translation\n...\n```\n## The Text\n```text\nこんにちは！はじめてのチェーン作りです！\n```\n```"}
# -> ]

msg = json.loads( translate_template( ['to English','こんにちは！はじめてのチェーン作りです！',] ))
print(msg)
# -> [{'role': 'system', 'content': 'You are an expert in translation text (to English).'}, {'role': 'user', 'content': 'I will provide text. Please tranlate it.\nYou should reply translations only, without any additional information.\n\n## Your Reply Format Example\n```translation\n...\n```\n## The Text\n```text\nこんにちは！はじめてのチェーン作りです！\n```\n```'}]

print(chatgpt4omini(msg))
# -> ```translation
# -> Hello! This is my first time making a chain!
# -> ```

translator_chain = [translate_template, json.loads, chatgpt4omini, get_result]
translator = compose(*translator_chain)
print(translator( ['to Chinese','こんにちは！はじめてのチェーン作りです！',] ))
# -> 你好！这是第一次制作链条！
print(translator( ['to Japanese','Hello! This is my first time making a chain!',] ))
# -> こんにちは！これは私の初めてのチェーン作りです！