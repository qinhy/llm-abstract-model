import json
import os
import time

from datetime import datetime
import pytz
import time

from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs
from LLMAbstractModel.utils import RegxExtractor
from mt5utils import MT5Account, MT5MakeOder, RatesReturn

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

    # Add an MT5 accounts to the store
    accs:list[MT5Account] = [ store.add_new_obj(
                MT5Account(
                    account_id      = i,
                    password        = os.environ[f"{i}_PASS"],
                    account_server  = os.environ[f"{i}_SERVER"],
                    metadata        = {'tags': [str(i)]})
            ) for i in eval(os.environ["MT5ACCs"])]
    # myprint('accs[0].model_dump_json_dict()')

    acc=dict(account_id=accs[0].account_id,
            password=accs[0].password,
            account_server=accs[0].account_server)
    book=dict(symbol='USDJPY',price_open = 100.0,volume= 0.01)

    # get account info
    account_info = store.add_new_celery_request(url='http://localhost:8000/mt5accountinfo/',
                                            method='POST',id='AsyncCeleryWebApiFunction:account-info')
    
    account_info(json_data={
        "param": acc,
    }).ret['info']['books']

    # get books info
    books_info = account_info

    books_info(json_data={
        "param": acc,
    }).ret['info']['books']

    # get rates
    get_rates = store.add_new_celery_request(url='http://localhost:8000/mt5copylastratesservice/',
                                             method='POST',id='AsyncCeleryWebApiFunction:get-rates')
    get_rates(json_data={
        "param": acc,
        "args": dict(symbol='USDJPY',timeframe='H4',count=30)
    }).ret['rates']
    # myprint('''get_rates(json_data=acc,params=dict(symbol='USDJPY',timeframe='H4',count=30))''')

    decode_rates = store.add_new_function(RatesReturn())
    # myprint('''decode_rates(get_rates(json_data=acc,params=dict(symbol='USDJPY',timeframe='H4',count=30)))''')

    # Initialize LLM vendor and add to the store    
    vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key=os.environ['OPENAI_API_KEY'],timeout=60)
    llm = store.add_new(Model4LLMs.ChatGPTDynamic)(
            llm_model_name='o3',system_prompt=system_prompt,limit_output_tokens=4096,
            vendor_id=vendor.get_id())
    # llm = store.add_new_function(MockLLM()) if debug else llm

    # Add functions to the store
    extract_json = store.add_new_function(RegxExtractor(
                        para=dict(regx=r"```json\s*(.*)\s*\n```",is_json=True)))
    to_book_plan = store.add_new_function(MT5MakeOder())
    books_send   = store.add_new_celery_request(url='http://localhost:8000/booksendservice/',
                                            method='POST',id='AsyncCeleryWebApiFunction:books-send')
    
    # myprint('''books_send(json_data=dict(acc=acc,book=book))''')

    # manual workflow
    # order = to_book_plan(
    #                 extract_json(
    #                 llm(
    #                     decode_rates(
    #                             get_rates(json_data={
    #                                 "param": acc,
    #                                 "args": dict(symbol='USDJPY',timeframe='H4',count=30)
    #                             }).ret
    #                     )
    #                 )
    #             )
    #         )
    # res = books_send(order)
    djson={
        "param": acc,
        "args": dict(symbol='USDJPY',timeframe='H4',count=30)
    }
    workflow:Model4LLMs.MermaidWorkflow = store.add_new_obj(
        Model4LLMs.MermaidWorkflow(
            mermaid_text=f'''
    graph TD
        {get_rates.get_id()}["{{ 'args': {{'json_data':{djson} }} }}"]
        {get_rates.get_id()} -- "{{'ret':'data'}}" --> {decode_rates.get_id()}
        {decode_rates.get_id()} -- "{{'prompt':'messages'}}" --> {llm.get_id()}
        {llm.get_id()} -- "{{'data':'text'}}" --> {extract_json.get_id()}
        {extract_json.get_id()} -- "{{'data':'data'}}" --> {to_book_plan.get_id()}
    '''))
    workflow.parse_mermaid()
    workflow.run()
    books_send(json_data={
        "param": acc,
        "args": workflow.results['final']
    })
    # parse_mermaid the workflow
    # print(workflow.parse_mermaid())
    # print(res)
    # myprint('workflow.parse_mermaid()')
    ## -> 13
    # Run the workflow
    # myprint('workflow.run()')
    ## -> 13
    return store

# Functions for secure data storage and retrieval using RSA key pair
def save_secure(store: LLMsStore):
    store.dump_RSA('./tmp/ex.20.store.rjson', './tmp/public_key.pem', True)

def load_secure():
    # Load stored RSA-encrypted data and initialize the agent
    store = LLMsStore()
    store.load_RSA('./tmp/ex.20.store.rjson', './tmp/private_key.pem')
    return store

# tmp_store = init_store()
# save_secure(store)
# store.clean()
# store = load_secure()

# workflow:Model4LLMs.WorkFlow = store.find_all('WorkFlow:*')[0]
# myprint('json.dumps(workflow.model_dump_json_dict(), indent=2)')
    
# Monitoring loop
def start_monitoring(store):
    llm = store.find_all('ChatGPT*:*')[0]
    monitor_pairs = store.get('monitor_pairs')['monitor_pairs']
    workflow:Model4LLMs.MermaidWorkflow = store.find_all('MermaidWorkflow:*')[0]
    accs = store.find_all('MT5Account:*')
    acc = accs[0]
    books_send = store.find('AsyncCeleryWebApiFunction:books-send')
    books_info = store.find('AsyncCeleryWebApiFunction:account-info')
    get_rates = store.find('AsyncCeleryWebApiFunction:get-rates')
    decode_rates = store.find_all('RatesReturn:*')[0]
    extract_json = store.find_all('RegxExtractor:*')[0]
    to_book_plan = store.find_all('MT5MakeOder:*')[0]
    
    class SyncAccounts:
        def __init__(self,accs:list[MT5Account]):
            self.accs = [
                dict(account_id=acc.account_id,
                        password=acc.password,
                        account_server=acc.account_server)
                        for acc in accs]

        def books_send(self,plan):
            for acc in self.accs:                
                books_send(json_data={
                    "param": acc,
                    "args": plan
                })

        # def book_close(self,order):
        #     for acc in self.accs:
        #         books_close(json_data=dict(acc=acc,book=order))

    sa = SyncAccounts(accs)
    acc = sa.accs[0]
    try:
        bs = [json.loads(b) for b in books_info(json_data={
            "param": acc,
        }).ret['info']['books']]
        alls = set(monitor_pairs) - {book['symbol'] for book in bs if book['volume']==0.01}
        plans = []
        for currency in list(alls):
            # workflow_result = workflowf(acc,currency)
            res = get_rates(json_data={"param": acc, "args":dict(symbol=currency,timeframe='H4',count=30)}).ret
            res = decode_rates(data=res)
            llmr = res = llm(str(res))
            with open('../llm-abstract-model-logs.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(llmr + '\n')
            res = extract_json(res).data
            p = to_book_plan(data=res)
            print("Result:", p)
            plans.append(p.model_dump())

        for p in plans:sa.books_send(p)
        
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
        
def is_market_open() -> bool:
    # Define market open and close times in JST timezone
    MARKET_OPEN = {"day": 0, "hour": 5}  # Monday 5 AM JST
    MARKET_CLOSE = {"day": 5, "hour": 5}  # Saturday 5 AM JST

    # Timezone for FX market hours (Japan Time)
    JST = pytz.timezone("Asia/Tokyo")
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
    store = load_secure()
    while True:
        if is_market_open():
            # Place your monitoring logic here
            start_monitoring(store)
            time.sleep(10)
        else:
            print("Market is closed. Monitoring function will not run.")
            time.sleep(600)
        
# Start monitoring
# monitor_fx_market()