import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, KeyOrEnv, Model4LLMs
from LLMAbstractModel.utils import RegxExtractor
from mt5utils import MT5Account, Book, MT5MakeOder, MockLLM, RatesReturn
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

# Configuration settings
os.environ["MT5ACCs"] = "[9102374,]"  # Your account ID
os.environ["9102374_PASS"] = "password"  # Password
os.environ["9102374_SERVER"] = "XXXXXFX"  # Server name

monitor_pairs = ['USDJPY']
system_prompt = 'You are a smart financial assistant'

debug = True

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
myprint('req(params={"broker": "brokerX", "path": "/path/to/termial"},debug=debug,debug_data=dict())')
# ->  {'task_id': 'some-task-id', 'status': 'STARTED', 'result': {'message': 'Task is started'}} 

req = store.add_new_celery_request(url='http://localhost:8000/terminals/',method='GET')
myprint('req(debug=debug,debug_data=dict())')

# Add an MT5 accounts to the store
accs:list[MT5Account] = [ store.add_new_obj(
            MT5Account(
                account_id=i,
                password=KeyOrEnv(key=f"{i}_PASS"),
                account_server=KeyOrEnv(key=f"{i}_SERVER"),
                metadata={'tags': [str(i)]}
            )
        ) for i in eval(os.environ["MT5ACCs"])]
myprint('accs[0].model_dump_json_dict()')

acc=dict(account_id=accs[0].account_id,
        password=accs[0].password.get(),
        account_server=accs[0].account_server.get())
book=dict(symbol='USDJPY',price_open = 100.0,volume= 0.01)

# get account info
account_info = store.add_new_celery_request(url='http://localhost:8000/accounts/info/',method='GET')
myprint('''account_info(json=acc,debug=debug,debug_data=dict())''')

# get books info
books_info = store.add_new_celery_request(url='http://localhost:8000/books/',method='GET')
myprint('''books_info(json=acc,debug=debug,debug_data=dict())''')

# get rates
get_rates = store.add_new_celery_request(url='http://localhost:8000/rates/',method='GET')
myprint('''get_rates(json=acc,params=dict(symbol='USDJPY',timeframe='H4',count=30),debug=debug,debug_data=dict())''')

decode_rates = store.add_new_function(RatesReturn())
myprint('''decode_rates(
            get_rates(json=acc,params=dict(symbol='USDJPY',timeframe='H4',count=30),debug=debug,debug_data=dict()))''')

# Initialize LLM vendor and add to the store
vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
llm = store.add_new_chatgpt4o(vendor_id='auto', system_prompt=system_prompt)

llm = store.add_new_function(MockLLM()) if debug else llm

# Add functions to the store
extract_json = store.add_new_function(RegxExtractor(regx=r"```json\s*(.*)\s*\n```", is_json=True))
to_book_plan = store.add_new_function(MT5MakeOder())
books_send = store.add_new_celery_request(url='http://localhost:8000/books/send',method='GET')
myprint('''books_send(json=dict(acc=acc,book=book),debug=debug,debug_data=dict())''')

# manual workflow

to_book_plan(
                extract_json(
                    llm(
                        decode_rates(
                            get_rates(json=acc,
                                        params=dict(symbol='USDJPY',timeframe='H4',count=30))
                        )
                    )
                )
            )

# res = books_send(
#         json=dict(
#             acc=acc,
#             book=
#             )
# )
# print(res)

# Define and save workflow
# workflow = store.add_new_workflow(
#     tasks={
#         get_rates.get_id():['acc','rate_param'],
#         llm.get_id():['init_deps'],
#         get_rates.get_id():['init_deps','rates_param'],
#     },
#     metadata={'tags': [str(accs[0].account_id)]}
# )
# workflow = store.add_new_workflow(
#     tasks=[
#         account_info.get_id(),
#         # books_info.get_id(),
#         # get_rates.get_id(),

#         # get_rates.get_id(),
#         # llm.get_id(),
#         # extract_json.get_id(),
#         # make_order.get_id(),
#     ],
#     metadata={'tags': [str(accs[0].account_id)]}
# )

# Save and reload workflow data
# data = store.dumps()
# store.clean()
# store.loads(data)
# workflow:Model4LLMs.WorkFlow = store.find_all('WorkFlow:*')[0]
# workflowf = lambda:workflow(
#     init_deps=[(),
#                 dict(
#                     json=dict(
#                         account_id=accs[0].account_id,
#                         password=accs[0].password.get(),
#                         account_server=accs[0].account_server.get(),
#                     )
#                 )],
#     rates_param=[(),
#                 dict(
#                     params=dict(symbol='USDJPY',timeframe='H4',count=30)
#                 )]
# )
# myprint('json.dumps(workflow.model_dump_json_dict(), indent=2)')

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