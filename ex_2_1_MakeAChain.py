
import json
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
from LLMAbstractModel.utils import StringTemplate,RegxExtractor

store = LLMsStore()

system_prompt='''You are an expert in English translation. I will provide you with the text. Please translate it. You should reply with translations only, without any additional information.
## Your Reply Format Example
```translation
...
```'''

vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key="OPENAI_API_KEY")#auto check os.environ
llm = store.add_new_llm(Model4LLMs.ChatGPT41Nano)(vendor_id='auto',#vendor.get_id(),
                                                  system_prompt=system_prompt)

print('############# make a message template')
translate_template = store.add_new_obj(
    StringTemplate(para=dict(
        string='''
```text
{text}
```'''
    ))).build()
# the usage of template is tmp( [args1,args2,...] ) is the same of sting.format(*[...])
print(translate_template(text='こんにちは！はじめてのチェーン作りです！'))
# -> ...

print(llm( translate_template(text='こんにちは！はじめてのチェーン作りです！')))
# -> ```translation
# -> Hello! This is my first time making a chain!
# -> ```

store:LLMsStore = store
print('############# make a "translation" extractor, strings between " ```translation " and " ``` "')
get_result = store.add_new_obj(RegxExtractor(para=dict(regx=r"```translation\s*(.*)\s*\n```")))

# this just a chain like processs
print(get_result(
        llm(
            translate_template('こんにちは！はじめてのチェーン作りです！') 
            )))
# -> Hello! This is my first time making a chain!

print('############# make the chain more graceful and simple')
# chain up functions
def compose(*funcs):
    def composed(*args, **kwargs):
        result = funcs[0](*args, **kwargs)  # start with last (innermost) function
        for f in funcs[1:]:
            result = f(result)
        return result
    return composed

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
loaded_chain[0] = loaded_chain[0].build()
translator = compose(*loaded_chain)
print(translator('こんにちは！はじめてのチェーン作りです！'))
# -> Hello! It's my first time making a chain!


print('############# additional template usage')

llm = store.add_new_llm(Model4LLMs.ChatGPT41Nano)(vendor_id='auto')

# we use raw json to do template
translate_template = store.add_new_obj(
    StringTemplate(para=dict(
        string='''[
    {{"role":"system",
        "content":"You are an expert in translation text.I will you provide text. Please tranlate it.\\nYou should reply translations only, without any additional information.\\n\\n## Your Reply Format Example\\n```translation\\n...\\n```"}},
    {{"role":"user","content":"\\nPlease translate the text in{lng}.\\n```text\\n{txt}\\n```"}}
]'''.replace('\n','').replace('\n','\\n')))).build()
# the usage of template is tmp( [args1,args2,...] ) is the same of sting.format(*[...])
print(translate_template(lng='to English',txt='こんにちは！はじめてのチェーン作りです！'))
# -> [
# ->     {"role":"system","content":"You are an expert in translation text.I will you provide text. Please tranlate it.\nYou should reply translations only, without any additional information.\n\n## -> Your Reply Format Example\n```translation\n...\n```"},
# ->     {"role":"user","content":"\nPlease translate the text into English.\n```text\nこんにちは！はじめてのチェーン作りです！\n```"}
# -> ]

msg = json.loads( translate_template(lng='to English',txt='こんにちは！はじめてのチェーン作りです！'))
print(msg)
# -> [{'role': 'system', 'content': 'You are an expert in translation text.I will you provide text. Please tranlate it.\nYou should reply translations only, without any additional information.\n\n## Your Reply Format Example\n```translation\n...\n```'}, {'role': 'user', 'content': '\nPlease translate the text into English.\n```text\nこんにちは！はじめてのチェーン作りです！\n```'}]

print(llm(msg))
# -> ```translation
# -> Hello! This is my first time making a chain!
# -> ```

translator_chain = [translate_template, json.loads, llm, get_result]
translator = compose(*translator_chain)
print(translator(lng='to Chinese',txt='こんにちは！はじめてのチェーン作りです！'))
# -> 你好！这是第一次制作链条！
print(translator(lng='to Japanese',txt='Hello! This is my first time making a chain!'))
# -> こんにちは！これは私の初めてのチェーン作りです！