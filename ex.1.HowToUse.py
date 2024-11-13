import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
store = LLMsStore()

vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id())
text_embedding = store.add_new_obj(Model4LLMs.TextEmbedding3Small())

## if you have ollam
# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id())
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id())
# llama32 = store.add_new_llama(vendor_id=vendor.get_id())

# just asking
print(chatgpt4omini('hi! What is your name?'))
# -> Hello! I’m called Assistant. How can I help you today?

# push messages
print(chatgpt4omini([
    {'role':'system','content':'You are a highly skilled professional English translator.'},
    {'role':'user','content':'"こんにちは！"'}    
]))
# -> Hello! I'm an AI language model created by OpenAI, and I don't have a personal name, but you can call me Assistant. How can I help you today?
# -> "Hello!"


print(text_embedding('hi! What is your name?')[:10], '...')
# -> [0.0118862, -0.0006172658, -0.008183353, 0.02926386, -0.03759078, -0.031130238, -0.02367668 ...