import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs

store = LLMsStore()

vendor = store.add_new(Model4LLMs.OpenAIVendor)(
                            api_key='OPENAI_API_KEY')
text_embedding = store.add_new(Model4LLMs.TextEmbedding3Small)(
                            vendor_id=vendor.get_id())
llm = store.add_new(Model4LLMs.ChatGPTDynamic)(
            llm_model_name='gpt-5-nano',
            vendor_id=vendor.get_id())

# vendor = store.add_new(Model4LLMs.AnthropicVendor)(api_key='ANTHROPIC_API_KEY')
# llm = store.add_new(Model4LLMs.ClaudeDynamic)(vendor_id=vendor.get_id())

# vendor = store.add_new(Model4LLMs.DeepSeekVendor)(api_key='DEEPSEEK_API_KEY')
# llm = deepseek = store.add_new(Model4LLMs.DeepSeekDynamic)(vendor_id=vendor.get_id())

# vendor = store.add_new_vendor(Model4LLMs.XaiVendor)(api_key='XAI_API_KEY')
# llm = grok = store.add_new(Model4LLMs.Grok)(vendor_id=vendor.get_id())

## if you have ollam
# vendor  = store.add_new_vendor(Model4LLMs.OllamaVendor)()
# gemma2  = store.add_new(Model4LLMs.Gemma2)(vendor_id=vendor.get_id())
# phi3    = store.add_new(Model4LLMs.Phi3)(vendor_id=vendor.get_id())
# llama32 = store.add_new(Model4LLMs.Llama)(vendor_id=vendor.get_id())

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

print(llm([
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What do you see in this image?"},
            {"type": "input_image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/9/99/Black_square.jpg"}
        ]
    }
]))
# -> A solid black square centered on a white background, with a white border around it. There are no visible details or patterns—it's basically a blank/void image.

print(text_embedding('hi! What is your model name?')[:10], '...')
# -> [0.0118862, -0.0006172658, -0.008183353, 0.02926386, -0.03759078, -0.031130238, -0.02367668 ...

# save all to json
store.dump('./tmp/ex.1.1HowToUse.json')

# # optional: save all to json with RSA encryption
# store.dump_RSA(path='path/to/your/private_data.rjson',
#                 public_key_path='path/to/your/public_key.pem',
#                 compress=True)
# # optional: load all from json with RSA decryption
# store.load_RSA(path='path/to/your/private_data.rjson',
#                 private_key_path='path/to/your/private_key.pem',
#                 compress=True)