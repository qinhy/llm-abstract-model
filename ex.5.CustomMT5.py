import os
from pydantic import Field
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, KeyOrEnv, Model4LLMs
from mt5utils import MT5Account, MT5CopyLastRates, MT5Manager
descriptions = Model4LLMs.Function.param_descriptions

store = LLMsStore()

system_prompt = 'You are smart assistant'

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
llm = chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto',system_prompt=system_prompt)

# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt=system_prompt)

# env set
# import os
# os.environ["MT5ACC_1"]="your account id"
# os.environ["9102374_PASS"]="password"
# os.environ["9102374_SERVER"]="server name"

# add a account to store
acc = store.add_new_obj(
        MT5Account(account_id=int(os.environ['MT5ACC_1']),
                password=KeyOrEnv(key=f"{os.environ['MT5ACC_1']}_PASS"),
                account_server=KeyOrEnv(key=f"{os.environ['MT5ACC_1']}_SERVER"))
)
print(acc.model_dump_json_dict())

# init a mt5 terminal
manager = MT5Manager().get_singleton()
manager.add_terminal('XXXXXFX',"C:/Program Files/XXXXX FX MetaTrader 5/terminal64.exe")

# add a mt5 function to store
cpr = store.add_new_obj(MT5CopyLastRates(account=acc,symbol="USDJPY",timeframe="H4",count=30))

# do function and get result
manager.do(cpr)
print(manager.results[cpr.get_id()])

