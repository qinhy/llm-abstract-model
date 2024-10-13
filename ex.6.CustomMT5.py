import json
import os
from pydantic import Field
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, KeyOrEnv, Model4LLMs
from LLMAbstractModel.utils import RegxExtractor
from mt5utils import MT5Account, MT5ActiveBooks, MT5CopyLastRates, MT5MakeOder, MT5Manager
descriptions = Model4LLMs.Function.param_descriptions

store = LLMsStore()

def add_tag_to_obj(obj:Model4LLMs.AbstractObj,tag:str):
    tags = obj.metadata.get('tags',[]) + [tag]
    obj.get_controller().update_metadata('tags',tags)

def filter_by_tag(objs:list[Model4LLMs.AbstractObj],tag:str):
    return [o for o in objs if tag in o.metadata.get('tags',[])]

# env set
import os
os.environ["MT5ACC_1"]="9102374"#your account id
os.environ["9102374_PASS"]="password"
os.environ["9102374_SERVER"]="XXXXXFX"#server name

# # init a mt5 terminal
# manager = MT5Manager().get_singleton()
# manager.add_terminal('XXXXXFX',"C:/Program Files/XXXXX FX MetaTrader 5/terminal64.exe")

# add a account to store
acc:MT5Account = store.add_new_obj(
        MT5Account(account_id=int(os.environ['MT5ACC_1']),
                password=KeyOrEnv(key=f"{os.environ['MT5ACC_1']}_PASS"),
                account_server=KeyOrEnv(key=f"{os.environ['MT5ACC_1']}_SERVER"),
                metadata={'tags':[os.environ['MT5ACC_1']]})
)
print(acc.model_dump_json_dict())


# add a mt5 function to store
get_rates = store.add_new_obj(
    MT5CopyLastRates(account=acc,symbol="USDJPY",timeframe="H4",count=30,debug=True,
                     metadata={'tags':['USDJPY']}))


# # do function and get result
# manager.do(get_rates)
# print(manager.results[get_rates.get_id()])

# same result
print(get_rates())

# find by tag
print(filter_by_tag(store.find_all('MT5CopyLastRates:*'),'USDJPY')[0])

# get active books
get_books = store.add_new_obj(
    MT5ActiveBooks(account=acc,debug=True,
                     metadata={'tags':[str(acc.account_id)]}))
print(get_books())

##########
system_prompt = 'You are smart finacial assistant'
vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
llm = chatgpt4o = store.add_new_chatgpt4o(vendor_id='auto',system_prompt=system_prompt)

@descriptions('Return a constant string')
class MockLLM(Model4LLMs.Function):
    def __call__(self,p):
        return '```json\n{\n"Symbol": "USDJPY",\n"EntryPrice": 146.5,\n"TakeProfitPrice": 149.0,\n"ProfitRiskRatio": 2\n}\n````'

llmdebug = store.add_new_function(MockLLM())

extract_json = store.add_new_function(RegxExtractor(regx=r"```json\s*(.*)\s*\n```",is_json=True))

make_order = store.add_new_obj(
    MT5MakeOder(account=acc,debug=True,
                metadata={'tags':[str(acc.account_id)]}))

print(
    make_order(
            extract_json(
                llmdebug(
                    get_rates()
))))

workflow = store.add_new_workflow(
    tasks=[
        get_rates.get_id(),
        llmdebug.get_id(),
        extract_json.get_id(),
        make_order.get_id(),
    ],
    metadata={'tags':[str(acc.account_id)]})


# save and load workflow
data = store.dumps()
store.clean()
store.loads(data)
workflow:Model4LLMs.WorkFlow = store.find_all('WorkFlow:*')[0]

res = workflow()
print("Result:", res)
print(json.dumps(workflow.model_dump_json_dict(), indent=2))







