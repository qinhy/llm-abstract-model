import json
import os
import time
from pydantic import Field
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, KeyOrEnv, Model4LLMs
from LLMAbstractModel.utils import RegxExtractor
from mt5utils import MT5Account, MT5ActiveBooks, MT5CopyLastRates, MT5MakeOder, MockLLM
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

# Configuration settings
os.environ["MT5ACC_1"] = "9102374"  # Your account ID
os.environ["9102374_PASS"] = "password"  # Password
os.environ["9102374_SERVER"] = "XXXXXFX"  # Server name

monitor_pairs = ['USDJPY']
system_prompt = 'You are a smart financial assistant'
debug = True

# Initialize MT5 terminal manager
# manager = MT5Manager().get_singleton()
# manager.add_terminal('XXXXXFX', "C:/Program Files/XXXXXFX MetaTrader 5/terminal64.exe")

# Initialize the LLM store
store = LLMsStore()

# Helper functions
def add_tag_to_obj(obj: Model4LLMs.AbstractObj, tag: str):
    tags = obj.metadata.get('tags', []) + [tag]
    obj.get_controller().update_metadata('tags', tags)

def filter_by_tag(objs: list[Model4LLMs.AbstractObj], tag: str):
    return [o for o in objs if tag in o.metadata.get('tags', [])]

## try celery web requests, need a valid url(here is local one)
store.clean()
req = store.add_new_celery_request(url='http://localhost:8000/terminals/add',method='POST')
myprint('req.model_dump()')
# ->  {'rank': [0], 'create_time': datetime.datetime(2024, 10, 18, 9, 44, 4, 502305, tzinfo=TzInfo(UTC)), 'update_time': datetime.datetime(2024, 10, 18, 9, 44, 4, 502305, tzinfo=TzInfo(UTC)), 'status': '', 'metadata': {}, 'name': 'RequestsFunction', 'description': 'Makes an HTTP request using the configured method, url, and headers, and the provided params, data, or json.', 'parameters': {'type': 'object', 'properties': {'params': {'type': 'object', 'description': 'query parameters'}, 'data': {'type': 'object', 'description': 'form data'}, 'json': {'type': 'object', 'description': 'JSON payload'}}}, 'required': [], 'method': 'GET', 'url': 'http://localhost:8000/tasks/status/some-task-id', 'headers': {}} 
myprint('req(params={"broker": "brokerX", "path": "/path/to/termial"})')
# ->  {'task_id': 'some-task-id', 'status': 'STARTED', 'result': {'message': 'Task is started'}} 

# # Add an MT5 account to the store
# account_id = int(os.environ["MT5ACC_1"])
# acc = store.add_new_obj(
#     MT5Account(
#         account_id=account_id,
#         password=KeyOrEnv(key=f"{account_id}_PASS"),
#         account_server=KeyOrEnv(key=f"{account_id}_SERVER"),
#         metadata={'tags': [str(account_id)]}
#     )
# )
# print(acc.model_dump_json_dict())

# # Add MT5 functions to the store
# get_rates = store.add_new_obj(
#     MT5CopyLastRates(
#         account=acc, 
#         symbol="USDJPY", 
#         timeframe="H4", 
#         count=30, 
#         debug=debug,
#         metadata={'tags': ['USDJPY']}
#     )
# )
# print(get_rates())

# print(filter_by_tag(store.find_all('MT5CopyLastRates:*'), 'USDJPY')[0])

# get_books = store.add_new_obj(
#     MT5ActiveBooks(
#         account=acc, 
#         debug=debug,
#         metadata={'tags': [str(account_id)]}
#     )
# )
# print(get_books())

# # Initialize LLM vendor and add to the store
# vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
# llm = store.add_new_chatgpt4o(vendor_id='auto', system_prompt=system_prompt)

# llm = store.add_new_function(MockLLM()) if debug else llm

# # Add functions to the store
# extract_json = store.add_new_function(RegxExtractor(regx=r"```json\s*(.*)\s*\n```", is_json=True))

# make_order = store.add_new_obj(
#     MT5MakeOder(
#         account=acc, 
#         debug=debug,
#         metadata={'tags': [str(account_id)]}
#     )
# )

# # Execute workflow
# print(
#     make_order(
#         extract_json(
#             llm(
#                 get_rates()
#             )
#         )
#     )
# )

# # Define and save workflow
# workflow = store.add_new_workflow(
#     tasks=[
#         get_rates.get_id(),
#         llm.get_id(),
#         extract_json.get_id(),
#         make_order.get_id(),
#     ],
#     metadata={'tags': [str(account_id)]}
# )

# # Save and reload workflow data
# data = store.dumps()
# store.clean()
# store.loads(data)
# workflow:Model4LLMs.WorkFlow = store.find_all('WorkFlow:*')[0]
# workflow()
# print(json.dumps(workflow.model_dump_json_dict(), indent=2))

# store.dump('ex.6.CustomMT5.workflow.txt')

# # Monitoring loop
# while True:
#     try:
#         for currency in set(monitor_pairs) - {book.symbol for book in get_books()}:
#             get_rates.get_controller().update(symbol=currency)
#             workflow_result = workflow()
#             print(
#                 "Result:", 
#                 workflow.results[extract_json.get_id()], 
#                 workflow.results[make_order.get_id()]
#             )
#             with open('../llm-abstract-model-logs.txt', 'a') as log_file:
#                 log_file.write(workflow.results[llm.get_id()] + '\n')
#         time.sleep(10)
#     except Exception as e:
#         print("Error:", e)