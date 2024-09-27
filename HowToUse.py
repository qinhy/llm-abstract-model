import os
from LLMAbstractModel import LLMsStore
store = LLMsStore()

vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id())

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