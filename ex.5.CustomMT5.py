import os
from pydantic import Field
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, KeyOrEnv, Model4LLMs
from mt5utils import MT5Account
descriptions = Model4LLMs.Function.param_descriptions

store = LLMsStore()

system_prompt = 'You are smart assistant'

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto',system_prompt=system_prompt)

# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt=system_prompt)

acc = store.add_new_obj(
        MT5Account(account_id=int(os.environ['MT5ACC_1']),
                password=KeyOrEnv(key=f"{os.environ['MT5ACC_1']}_PASS"),
                account_server=KeyOrEnv(key=f"{os.environ['MT5ACC_1']}_SERVER"))
)

print(acc.model_dump_json_dict())