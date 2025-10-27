import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs

store = LLMsStore()


vendor = store.add_new_obj(Model4LLMs.OpenAIVendor(
                            api_key='OPENAI_API_KEY',timeout=60))
text_embedding = store.add_new(Model4LLMs.TextEmbedding3Small)(
                            vendor_id=vendor.get_id())
llm = store.add_new_obj(Model4LLMs.ChatGPTDynamic(
            vendor_id=vendor.get_id(),
            llm_model_name='gpt-5-mini',
            reasoning_effort="medium",
            limit_output_tokens=4096))

# vendor = store.add_new(Model4LLMs.AnthropicVendor)(api_key='ANTHROPIC_API_KEY')
# llm = store.add_new(Model4LLMs.ClaudeDynamic)(vendor_id=vendor.get_id())

# vendor = store.add_new(Model4LLMs.DeepSeekVendor)(api_key='DEEPSEEK_API_KEY')
# llm = deepseek = store.add_new(Model4LLMs.DeepSeekDynamic)(vendor_id=vendor.get_id())

# vendor = store.add_new_vendor(Model4LLMs.GeminiVendor)(api_key='GEMINI_API_KEY')
# text_embedding = store.add_new(Model4LLMs.TextGeminiEmbedding001)(
#                             vendor_id=vendor.get_id())
# llm = gemini = store.add_new(Model4LLMs.Gemini25Flash)(vendor_id=vendor.get_id())

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

# print(llm([
#     {
#         "role": "user",
#         "content": [
#             {"type": "input_text", "text": "What do you see in this image?"},
#             {"type": "input_image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/9/99/Black_square.jpg"}
#         ]
#     }
# ]))
# -> A solid black square centered on a white background, with a white border around it. There are no visible details or patterns—it's basically a blank/void image.
text = store.add_new_obj(Model4LLMs.TextContent(text='hi! What is your model name?'))
emb = store.add_new_obj(Model4LLMs.EmbeddingContent(target_id=text.get_id(),vec=text_embedding(text.text)))
print(emb.get_target_data().raw,": ",emb.get_vec()[:10], '...')
# -> hi! What is your model name?: [0.0118862, -0.0006172658, -0.008183353, 0.02926386, -0.03759078, -0.031130238, -0.02367668 ...


class SimpleHistory:
    def __init__(self, store=store):
        self.store = store
        self.root = store.add_new_obj(Model4LLMs.ContentGroup(owner_id='null',depth=0,))
    def add_user(self, msg: str):
        return self.add_msg(msg, author="user")
    def add_assistant(self, msg: str):
        return self.add_msg(msg, author="assistant")
    def add_msg(self,msg,author='user'):
        msg_obj = store.add_new_obj(Model4LLMs.TextContent(text=msg,author_id=author))
        msg_group = self.store.add_new_obj(Model4LLMs.ContentGroup(owner_id=msg_obj.get_id()))
        self.root.controller.add_child(msg_group)    
        return f"{author}: {msg}"

    def to_list(self):
        ms:list[Model4LLMs.TextContent] = []
        ms = [self.store.find(g.owner_id) for g,_ in self.root.yield_children_recursive()]
        return [{"role": m.author_id, "content": m.text} for m in ms]
    
    def print(self):
        for item in self.to_list():
            print(f"{item['role'] or 'unknown'}: {item['content']}")


hist = SimpleHistory()
print(hist.add_user('hi! What is your model name?'))
print(hist.add_assistant(llm(hist.to_list()[-4:])))
print(hist.add_user('How big is your body?'))
print(hist.add_assistant(llm(hist.to_list()[-4:])))
print(hist.to_list())

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