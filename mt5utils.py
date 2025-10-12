from pydantic import BaseModel, Field
import json
from typing import Any, Optional
from LLMAbstractModel.LLMsModel import LLMsStore
from LLMAbstractModel import Model4LLMs

class MT5Account(Model4LLMs.AbstractObj):
    account_id: int = None
    password: str = None
    account_server: str = None

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
     

class MT5MakeOder(Model4LLMs.MermaidWorkflowFunction):    
        description:str = Field('Create an MT5 order based on symbol, entry price, exit price.')
        
        class Parameter(BaseModel):
            volume:float = 0.01
            digits:dict[str,int] = {'AUDJPY':3,'CADJPY':3,'CHFJPY':3,'CNHJPY':3,'EURJPY':3,
                            'GBPJPY':3,'USDJPY':3,'NZDJPY':3,'XAUJPY':0,'JPN225':1,'US500':1}

        class Arguments(BaseModel):
            Symbol:str=Field(...,description='The financial instrument for the order (e.g., USDJPY).')
            EntryPrice:float=Field(...,description='Price at which to enter the trade.')
            TakeProfitPrice:float=Field(...,description='Price at which to take profit in the trade.')
            ProfitRiskRatio:float=Field(...,description='The ratio of profit to risk.')

        class Returness(BaseModel):
            symbol: str = Field(..., description='The financial instrument for the order (e.g., USDJPY).')
            volume: float = Field(..., description='Volume of the order.')
            price_open: float = Field(..., description='Price at which to enter the trade.')
            sl: float = Field(..., description='Stop loss price for the order.')
            tp: float = Field(..., description='Take profit price for the order.')

        para: Parameter = Parameter()
        args: Optional[Arguments] = None
        rets: Optional[Returness] = None

        def __call__(self,args,volume=0.01):
            self.args = MT5MakeOder.Arguments(**args)
            EntryPrice = self.args.EntryPrice
            TakeProfitPrice = self.args.TakeProfitPrice
            ProfitRiskRatio = self.args.ProfitRiskRatio

            going_long = TakeProfitPrice > EntryPrice
            if going_long:
                stop_loss = EntryPrice - (TakeProfitPrice - EntryPrice) / ProfitRiskRatio
            else:
                stop_loss = EntryPrice + (EntryPrice - TakeProfitPrice) / ProfitRiskRatio
            digitsnum = self.para.digits[self.args.Symbol]
            EntryPrice,stop_loss,TakeProfitPrice = list(
                            map(lambda x:round(x*10**digitsnum)/10**digitsnum,
                            [EntryPrice,stop_loss,TakeProfitPrice]))
            self.rets = self.Returness(symbol=self.args.Symbol,volume=volume,
                                       price_open=EntryPrice,sl=stop_loss,tp=TakeProfitPrice)
            return  self.rets


class RatesReturn(Model4LLMs.MermaidWorkflowFunction):
    description:str = Field('Return rates as string')

    class Parameter(BaseModel):
        pass
    class Arguments(BaseModel):
        rates: list = []
    class Returness(BaseModel):
        symbol: str = "USDJPY"
        timeframe: str = "H1"
        count: int = 10
        rates: list = []
        digitsnum: int = 0
        error: Optional[str] = ''
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
   
    para: Parameter = Parameter()
    args: Arguments = Arguments()
    rets: Returness = Returness()
    def __call__(self, rates: list = []):
        self.rets = self.Returness(rates=rates)
        return self.rets

    
# Define and add a mock LLM function if in debug mode
class MockLLM(Model4LLMs.MermaidWorkflowFunction):
    description:str = Field('Define and add a mock LLM function if in debug mode')

    class Parameter(BaseModel):
        pass
    class Arguments(BaseModel):
        pass
    class Returness(BaseModel):
        res:str = 'null'
        
    para: Parameter = Parameter()
    args: Arguments = Arguments()
    rets: Returness = Returness()

    def __call__(self, msg):
        self.rets = self.Returness(res=(
            '```json\n{\n'
            '"Symbol": "USDJPY",\n'
            '"EntryPrice": 146.5,\n'
            '"TakeProfitPrice": 149.0,\n'
            '"ProfitRiskRatio": 2\n'
            '}\n```'
        ))
        return self.rets

class MakeDict(Model4LLMs.MermaidWorkflowFunction):
    description:str = Field('class MakeDict(Model4LLMs.MermaidWorkflowFunction)')
    
    class Parameter(BaseModel):
        pass
    class Arguments(BaseModel):
        pass
    class Returness(BaseModel):
        res:dict = {}
        
    para: Parameter = Parameter()
    args: Arguments = Arguments()
    rets: Returness = Returness()   
        
    def __call__(self, **kwargs):
        self.rets = self.Returness(res=dict(kwargs))
        return self.rets

store = LLMsStore()
acc = store.add_new_obj(
    MT5Account(
        account_id=123,
        password=f"123_PASS",
        account_server=f"123_SERVER"
    )
)

store.add_new_function(
    MockLLM()
).controller.delete()
store.add_new_function(
    MT5MakeOder()
).controller.delete()
store.add_new_function(
    RatesReturn()
).controller.delete()
store.add_new_function(
    MakeDict()
).controller.delete()

acc.controller.delete()
