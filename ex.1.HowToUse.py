import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs

store = LLMsStore()

vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key=os.environ.get('OPENAI_API_KEY','null'))
llm = store.add_new(Model4LLMs.ChatGPT41Nano)(vendor_id=vendor.get_id())
# llm = store.add_new_chatgpto1mini(vendor_id=vendor.get_id())
text_embedding = store.add_new(Model4LLMs.TextEmbedding3Small)(vendor_id=vendor.get_id())

# vendor = store.add_new_vendor(Model4LLMs.XaiVendor)(api_key=os.environ.get('XAI_API_KEY','null'))
# llm = grok = store.add_new_llm(Model4LLMs.Grok)(vendor_id=vendor.get_id())

## if you have ollam
# vendor  = store.add_new_vendor(Model4LLMs.OllamaVendor)()
# gemma2  = store.add_new_llm(Model4LLMs.Gemma2)(vendor_id=vendor.get_id())
# phi3    = store.add_new_llm(Model4LLMs.Phi3)(vendor_id=vendor.get_id())
# llama32 = store.add_new_llm(Model4LLMs.Llama)(vendor_id=vendor.get_id())

# vendor = store.add_new_vendor(Model4LLMs.DeepSeekVendor)(api_key=os.environ.get('DEEPSEEK_API_KEY','null'))
# llm = deepseek = store.add_new_llm(Model4LLMs.DeepSeek)(vendor_id=vendor.get_id())

# just asking
print(llm('hi! What is your name?'))
# -> Hello! I’m called Assistant. How can I help you today?

# push messages
print(llm([
    {'role':'system','content':'You are a highly skilled professional English translator.'},
    {'role':'user','content':'"こんにちは！"'}    
]))
# -> Hello! I'm an AI language model created by OpenAI, and I don't have a personal name, but you can call me Assistant. How can I help you today?
# -> "Hello!"


print(text_embedding('hi! What is your name?')[:10], '...')
# -> [0.0118862, -0.0006172658, -0.008183353, 0.02926386, -0.03759078, -0.031130238, -0.02367668 ...