from multiprocessing import Lock
import random
import time
import uuid
from pydantic import BaseModel
import json
from typing import Any, Dict, List
from LLMAbstractModel.LLMsModel import KeyOrEnv, LLMsStore
try:
    import MetaTrader5 as mt5
except Exception as e:
    print(e)
from LLMAbstractModel import BasicModel, Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions

class MT5Account(Model4LLMs.AbstractObj):
    account_id: int = None
    password: KeyOrEnv = None
    account_server: KeyOrEnv = None

    def is_valid(self):
        if self.account_id is None:raise ValueError('account_id is not set')
        # if self.password == '':raise ValueError('password is not set')
        # if self.account_server == '':raise ValueError('account_server is not set')
        return True

class Book(BaseModel):
    class Controller(BaseModel):

        class Null(BaseModel):
            type:str = 'Null'

        class Plan(BaseModel):
            type:str = 'Plan'
            
        class Order(BaseModel):
            type:str = 'Order'

        class Position(BaseModel):
            type:str = 'Position'

    symbol: str = ''
    sl: float = 0.0
    tp: float = 0.0
    price_open: float = 0.0
    volume: float = 0.0
    magic:int = 901000
                
    state: Controller.Plan = Controller.Plan()
    _book: Any = None# mt5_order_position
    _is_order: bool = False
    _is_position: bool = False
    _ticket: int = 0
    _type: str = ''
    _swap: int = 0

    def as_plan(self):
        self.state = Book.Controller.Plan()
        return self
    
    def getBooks(self,book_list:list[dict]=[]):
        return [ Book(**op) for op in book_list]
     

@descriptions('Create an MT5 order based on symbol, entry price, exit price.',
        Symbol='The financial instrument for the order (e.g., USDJPY).',
        EntryPrice='Price at which to enter the trade.',
        TakeProfitPrice='Price at which to take profit in the trade.',
        ProfitRiskRatio='The ratio of profit to risk.')
class MT5MakeOder(Model4LLMs.Function):    
    def __call__(Symbol, EntryPrice, TakeProfitPrice, ProfitRiskRatio):
        going_long = TakeProfitPrice > EntryPrice
        if going_long:
            stop_loss = EntryPrice - (TakeProfitPrice - EntryPrice) / ProfitRiskRatio
        else:
            stop_loss = EntryPrice + (EntryPrice - TakeProfitPrice) / ProfitRiskRatio
        digitsnum = {'AUDJPY':3,'CADJPY':3,'CHFJPY':3,'CNHJPY':3,'EURJPY':3,
                        'GBPJPY':3,'USDJPY':3,'NZDJPY':3,'XAUJPY':0,'JPN225':1,'US500':1}[Symbol]
        EntryPrice,stop_loss,TakeProfitPrice = list(
                        map(lambda x:round(x*10**digitsnum)/10**digitsnum,
                                                        [EntryPrice,stop_loss,TakeProfitPrice]))
        # Prepare trade request
        return dict(symbol=Symbol,volume= 0.01,price_open=EntryPrice,sl=stop_loss,tp=TakeProfitPrice)


@descriptions('Return rates as string')
class RatesReturn(Model4LLMs.Function):
    class Rates(BaseModel):
        symbol: str = "USDJPY"
        timeframe: str = "H1"
        count: int = 10
        rates: list = None
        digitsnum: int = 0
        error: tuple = None
        header: str='```{symbol} {count} Open, High, Low, Close (OHLC) data points for the {timeframe} timeframe\n{join_formatted_rates}\n```'

        def __str__(self):
            if self.rates is None:
                return f"Error: {self.error}"

            if self.digitsnum > 0:
                n = self.digitsnum
                formatted_rates = [
                    f'{r[1]:.{n}f}\n{r[2]:.{n}f}\n{r[3]:.{n}f}\n{r[4]:.{n}f}\n'
                    for r in self.rates
                ]
            else:
                formatted_rates = [
                    f'{int(r[1])}\n{int(r[2])}\n{int(r[3])}\n{int(r[4])}\n'
                    for r in self.rates
                ]

            # Join the formatted rates into a single string
            join_formatted_rates = '\n'.join(formatted_rates)

            # Use the customizable header format to return the final output
            return self.header.format(
                symbol=self.symbol,
                count=self.count,
                timeframe=self.timeframe,
                join_formatted_rates=join_formatted_rates
            )
        
        def __call__(self, rates):
            return str(RatesReturn.Rates(**rates))

    
# Define and add a mock LLM function if in debug mode
@descriptions('Return a constant string')
class MockLLM(Model4LLMs.Function):
    def __call__(self, msg):
        return (
            '```json\n{\n'
            '"Symbol": "USDJPY",\n'
            '"EntryPrice": 146.5,\n'
            '"TakeProfitPrice": 149.0,\n'
            '"ProfitRiskRatio": 2\n'
            '}\n```'
        )

@descriptions('Return a Dict')
class MakeDict(Model4LLMs.Function):
    def __call__(self, **kwargs):
        return dict(kwargs)

store = LLMsStore()
acc = store.add_new_obj(
    MT5Account(
        account_id=123,
        password=KeyOrEnv(key=f"123_PASS"),
        account_server=KeyOrEnv(key=f"123_SERVER")
    )
)

store.add_new_function(
    MockLLM()
).get_controller().delete()
store.add_new_function(
    MT5MakeOder()
).get_controller().delete()
store.add_new_function(
    RatesReturn()
).get_controller().delete()
store.add_new_function(
    MakeDict()
).get_controller().delete()

acc.get_controller().delete()

