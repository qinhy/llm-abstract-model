import json
import os
import time
from pydantic import Field
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, KeyOrEnv, Model4LLMs
from LLMAbstractModel.utils import RegxExtractor
from mt5utils import MT5Account, MT5MakeOder, MockLLM, RatesReturn
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

def init_store():
    # Initialize the LLM store
    store = LLMsStore()
    store.set('monitor_pairs',{'monitor_pairs':monitor_pairs})

    # try celery web requests, need a valid url(here is local one)
    store.clean()
    req = store.add_new_celery_request(url='http://localhost:8000/terminals/add',method='POST')
    # myprint('req.model_dump()')
    # ->  {'rank': [0], 'create_time': datetime.datetime(2024, 10, 18, 9, 44, 4, 502305, tzinfo=TzInfo(UTC)), 'update_time': datetime.datetime(2024, 10, 18, 9, 44, 4, 502305, tzinfo=TzInfo(UTC)), 'status': '', 'metadata': {}, 'name': 'RequestsFunction', 'description': 'Makes an HTTP request using the configured method, url, and headers, and the provided params, data, or json.', 'parameters': {'type': 'object', 'properties': {'params': {'type': 'object', 'description': 'query parameters'}, 'data': {'type': 'object', 'description': 'form data'}, 'json': {'type': 'object', 'description': 'JSON payload'}}}, 'required': [], 'method': 'GET', 'url': 'http://localhost:8000/tasks/status/some-task-id', 'headers': {}} 
    # myprint('req(params={"broker": "TitanFX", "path": "C:/Program Files/Titan FX MetaTrader 5/terminal64.exe"})')
    # ->  {'task_id': 'some-task-id', 'status': 'STARTED', 'result': {'message': 'Task is started'}} 

    req = store.add_new_celery_request(url='http://localhost:8000/terminals/',method='GET')
    # myprint('req()')

    # Add an MT5 accounts to the store
    accs:list[MT5Account] = [ store.add_new_obj(
                MT5Account(
                    account_id=i,
                    password=KeyOrEnv(key=os.environ[f"{i}_PASS"]),
                    account_server=KeyOrEnv(key=os.environ[f"{i}_SERVER"]),
                    metadata={'tags': [str(i)]}
                )
            ) for i in eval(os.environ["MT5ACCs"])]
    # myprint('accs[0].model_dump_json_dict()')

    acc=dict(account_id=accs[0].account_id,
            password=accs[0].password.get(),
            account_server=accs[0].account_server.get())
    book=dict(symbol='USDJPY',price_open = 100.0,volume= 0.01)

    # get account info
    account_info = store.add_new_celery_request(url='http://localhost:8000/accounts/info/',
                                                method='GET',id='AsyncCeleryWebApiFunction:books_info')
    # myprint('''account_info(json=acc)''')

    # get books info
    books_info = store.add_new_celery_request(url='http://localhost:8000/books/',method='GET')
    # myprint('''books_info(json=acc)''')

    # get rates
    get_rates = store.add_new_celery_request(url='http://localhost:8000/rates/',method='GET')
    # myprint('''get_rates(json=acc,params=dict(symbol='USDJPY',timeframe='H4',count=30))''')

    decode_rates = store.add_new_function(RatesReturn())
    # myprint('''decode_rates(get_rates(json=acc,params=dict(symbol='USDJPY',timeframe='H4',count=30)))''')

    # Initialize LLM vendor and add to the store
    vendor = store.add_new_openai_vendor(api_key=os.environ['OPENAI_API_KEY'])
    llm = store.add_new_chatgpt4o(vendor_id='auto', system_prompt=system_prompt)
    # llm = store.add_new_function(MockLLM()) if debug else llm

    # Add functions to the store
    extract_json = store.add_new_function(RegxExtractor(regx=r"```json\s*(.*)\s*\n```", is_json=True))
    to_book_plan = store.add_new_function(MT5MakeOder())
    books_send = store.add_new_celery_request(url='http://localhost:8000/books/send',
                                            method='POST',id='AsyncCeleryWebApiFunction:books_send')
    # myprint('''books_send(json=dict(acc=acc,book=book))''')

    # manual workflow
    # res = books_send(
    #         json=dict(
    #             acc=acc,
    #             book=to_book_plan(
    #                       extract_json(
    #                         llm(
    #                             decode_rates(
    #                                 get_rates(json=acc,
    #                                             params=dict(symbol='USDJPY',timeframe='H4',count=30))
    #                             )
    #                         )
    #                     )
    #                 )
    #             )
    # )

    # Define and save workflow
    # workflow = store.add_new_workflow(
    #     tasks=[
    #         account_info.get_id(),
    #         # get_rates.get_id(),
    #         # llm.get_id(),
    #         # extract_json.get_id(),
    #         # make_order.get_id(),
    #     ],
    #     metadata={'tags': [str(accs[0].account_id)]}
    # )
    workflow = store.add_new_workflow(
        tasks={
            get_rates.get_id():['init_deps','rates_param'],
            decode_rates.get_id():[get_rates.get_id()],
            llm.get_id():[decode_rates.get_id()],
            extract_json.get_id():[llm.get_id()],
            to_book_plan.get_id():[extract_json.get_id()],
        },
        metadata={'tags': [str(accs[0].account_id)]}
    )
    workflowf = lambda acc,symbol:workflow(
        init_deps=[(),
                    dict(json=acc)],
        rates_param=[(),
                    dict(
                        params=dict(symbol=symbol,timeframe='H4',count=30)
                    )]
    )
    return store

# Functions for secure data storage and retrieval using RSA key pair
def save_secure(store: LLMsStore):
    store.dump_RSA('./tmp/ex.20.store.rjson', './tmp/public_key.pem')

def load_secure():
    # Load stored RSA-encrypted data and initialize the agent
    store = LLMsStore()
    store.load_RSA('./tmp/ex.20.store.rjson', './tmp/private_key.pem')
    return store

store = init_store()
# save_secure(store)
# store.clean()
# store = load_secure()

workflow:Model4LLMs.WorkFlow = store.find_all('WorkFlow:*')[0]
myprint('json.dumps(workflow.model_dump_json_dict(), indent=2)')
    
# Monitoring loop
def start_monitoring():
    llm = store.find_all('ChatGPT4o:*')[0]
    monitor_pairs = store.get('monitor_pairs')['monitor_pairs']
    workflow = store.find_all('WorkFlow:*')[0]
    accs = store.find_all('MT5Account:*')
    acc = accs[0]
    books_send = store.find_all('AsyncCeleryWebApiFunction:books_send')
    books_info = store.find_all('AsyncCeleryWebApiFunction:books_info')

    workflowf = lambda acc,symbol:workflow(
        init_deps=[(),dict(json=acc)],
        rates_param=[(),dict(params=dict(symbol=symbol,timeframe='H4',count=30))]
    )
    
    class SyncAccounts:
        def __init__(self,accs:list[MT5Account]):
            self.accs = [
                dict(account_id=acc.account_id,
                        password=acc.password.get(),
                        account_server=acc.account_server.get())
                        for acc in accs]

        def books_send(self,plan):
            for acc in self.accs:
                books_send(json=dict(acc=acc,book=plan))

        # def book_close(self,order):
        #     for acc in self.accs:
        #         books_close(json=dict(acc=acc,book=order))

    sa = SyncAccounts(accs)
    acc = sa.accs[0]
    try:
        alls = set(monitor_pairs) - {book['symbol'] for book in json.loads(books_info(json=acc)['result']).values()}
        plans = []
        for currency in list(alls):
            workflow_result = workflowf(acc,currency)
            print("Result:", workflow_result)
            with open('../llm-abstract-model-logs.txt', 'a') as log_file:
                log_file.write(workflow.results[llm.get_id()] + '\n')
            plans.append(workflow_result)

        for p in plans:sa.books_send(p)
        
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
        
        
        
from datetime import datetime, timedelta
import pytz
import time

# Define market open and close times in JST timezone
MARKET_OPEN = {"day": 0, "hour": 5}  # Monday 5 AM JST
MARKET_CLOSE = {"day": 5, "hour": 5}  # Saturday 5 AM JST

# Timezone for FX market hours (Japan Time)
JST = pytz.timezone("Asia/Tokyo")

def is_market_open() -> bool:
    """
    Checks if the FX market is currently open.
    Returns True if open, False if closed.
    """
    now = datetime.now(JST)
    weekday = now.weekday()
    hour = now.hour

    # Market is closed if it's Saturday after 5 AM or Sunday
    if weekday > MARKET_CLOSE["day"] or (weekday == MARKET_CLOSE["day"] and hour >= MARKET_CLOSE["hour"]):
        return False
    # Market is closed if it's before Monday 5 AM
    if weekday < MARKET_OPEN["day"] or (weekday == MARKET_OPEN["day"] and hour < MARKET_OPEN["hour"]):
        return False
    return True

def monitor_fx_market():
    """
    Function to monitor the FX market. This function will only run if the market is open.
    """
    while True:
        if is_market_open():
            # Place your monitoring logic here
            start_monitoring()
            time.sleep(10)
        else:
            print("Market is closed. Monitoring function will not run.")
            time.sleep(600)
        
# Start monitoring
monitor_fx_market()