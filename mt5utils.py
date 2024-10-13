from multiprocessing import Lock
import random
import time
import uuid
from pydantic import BaseModel
import json
from typing import Any, Dict, List
from LLMAbstractModel.LLMsModel import KeyOrEnv
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

class MT5Manager:
    # statics for singleton
    _uuid = uuid.uuid4()
    _results:Dict[str,List[Any]] = {}
    _terminals:Dict[str,List] = {}
    _is_singleton = True
    _meta = {}

    class TerminalLock:
        def __init__(self,exe_path="path/to/your/terminal64.exe"):
            self.exe_path=exe_path
            self._lock = Lock()
        def acquire(self):
            # print("acquired", self)
            self._lock.acquire()
        def release(self):
            # print("released", self)
            self._lock.release()
        def __enter__(self):
            self.acquire()
        def __exit__(self, type, value, traceback):
            self.release()

        
    def __init__(self,id=None,results=None,terminals=None,is_singleton=None):
        self.uuid = uuid.uuid4() if id is None else id
        self.results:Dict[str,List[Any]] = None if results is None else results
        self.terminals:Dict[str,List[MT5Manager.TerminalLock]] = None if terminals is None else terminals
        self.is_singleton:bool = False if is_singleton is None else is_singleton
    # {
    #     'TitanFX':[
    #         MT5Manager.TerminalLock(exe_path="path/to/your/terminal64.exe")
    #     ],
    #     'XMTrading':[ MT5Manager.TerminalLock(exe_path="path/to/your/terminal64.exe") ],
    # }
    
    def get_singleton(self):
        return self.__class__(self._uuid,self._results,self._terminals,self._is_singleton)
    
    def add_terminal(self, account_server='XMTrading', exe_path="path/to/your/terminal64.exe"):
        if account_server not in self.terminals:self.terminals[account_server]=[]
        self.terminals.get(account_server,[]).append(
            MT5Manager.TerminalLock(exe_path=exe_path))
    
    def _get_terminal_lock(self, account_server='XMTrading'):
        broker = account_server.split('-')[0]
        t_locks = self.terminals.get(broker,[])
        if len(t_locks)==0:raise ValueError('the broker is not support!')
        return random.choice(t_locks)

    def do(self, action:Model4LLMs.Function):
        account:MT5Account = action.account
        # m = Manager()
        # m.addExe

        # m.do(
        #     new Action class do some thing
        #     timeout 
        #     retry_times on timeout
        #     retry_times on error
        # )
        
        # get lock
        l = self._get_terminal_lock(account.account_server.get())
        try:
            l.acquire()
            if not mt5.initialize(path=l.exe_path):
                raise ValueError(f"Failed to initialize MT5 for executable path: {l.exe_path}")
            
            if account is None:raise ValueError('account is not set')
            account.is_valid()

            if not mt5.login(account.account_id,
                             password=account.password.get(),
                             server=account.account_server.get()):                
                raise ValueError(f"Failed to log in with account ID: {account.account_id}")
            
            if action.get_id() not in self.results:self.results[action.get_id()]=[]
            
            retry_times_on_error = action.retry_times_on_error
            res = None
            while retry_times_on_error>0:
                try:
                    res = action.run()
                    break
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    retry_times_on_error -= 1
            self.results[action.get_id()].append(res)            
        finally:
            mt5.shutdown()  # Ensure shutdown is called even if an error occurs
            l.release()
        # release lock

class Book(BaseModel):
    class Controller(BaseModel):        
        @staticmethod
        def _try(func):
            try:
                return func()
            except Exception as e:
                print(e)
                return False

        class Null(BaseModel):
            type:str = 'Null'
            def send(self,book):
                raise ValueError(f'This is a {self.type} state')
            def close(self,book):
                raise ValueError(f'This is a {self.type} state')
            def changeP(self,book,p):
                raise ValueError(f'This is a {self.type} state')
            def changeTS(self,book,tp,sl):
                raise ValueError(f'This is a {self.type} state')

        class Plan(BaseModel):
            type:str = 'Plan'
            def send(self,book):
                book:Book = book
                res = Book.Controller._try(lambda:book._make_order())
                book.state = Book.Controller.Order() if res else Book.Controller.Plan()
            def close(self,book):
                raise ValueError('This is just a Plan')
            def changeP(self,book,p):
                book:Book = book
                book.price_open = p
            def changeTS(self,book,tp,sl):
                book:Book = book
                book.tp,book.sl=tp,sl
            
        class Order(BaseModel):
            type:str = 'Order'
            def send(self,book):
                raise ValueError('This is a exists Order')
            def close(self,book):
                book:Book = book
                res = Book.Controller._try(lambda:book._close_order())
                if res : book.state = Book.Controller.Null()
            def changeP(self,book,p):
                raise ValueError('This is a exists Order, You can close it.')
            def changeTS(self,book,tp,sl):
                book:Book = book
                res = Book.Controller._try(lambda:book._changeOrderTPSL(tp,sl))
                if res : book.tp,book.sl=tp,sl

        class Position(BaseModel):
            type:str = 'Position'
            def send(self,book):
                raise ValueError('This is a exists Position')
            def close(self,book):
                book:Book = book
                res = Book.Controller._try(lambda:book._close_position())
                if res : book.state = Book.Controller.Null()
            def changeP(self,book,p):
                raise ValueError('This is a exists Position, can not change price open')
            def changeTS(self,book,tp,sl):
                book:Book = book
                res = Book.Controller._try(lambda:book._changePositionTPSL(tp,sl))
                if res : book.tp,book.sl=tp,sl

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
    
    def send(self):
        self.state.send(self)
        return self
    def close(self):
        self.state.close(self)
        return self
    def changeP(self,p):
        self.state.changeP(self,p)
        return self
    def changeTS(self,tp,sl):
        self.state.changeTS(self,tp,sl)
        return self

    def getBooks(self):
        return [ Book().set_mt5_book(book=op) for op in mt5.orders_get()+mt5.positions_get() ]
    
    def account_info(self):
        # Example operation: Getting account information
        account_info = mt5.account_info()
        if account_info is None:
            return "Failed to get account info"
        else:
            return account_info
        
    def set_mt5_book(self,book):
        self._book = book
        self.symbol = self._book.symbol
        self.sl = self._book.sl
        self.tp = self._book.tp
        self.price_open = self._book.price_open
        self._ticket = self._book.ticket
        self._type = self._book.type
        self._is_order=False
        self._is_position=False
        self._swap = 0
        
        if self._book.__class__.__name__ == "TradeOrder" : 
            self._is_order=True
            self.state = Book.Controller.Order()
        elif self._book.__class__.__name__ == "TradePosition": 
            self._is_position=True
            self.state = Book.Controller.Position()
        if hasattr(self._book,'volume_current'):
            self._is_order=True
            self.volume=self._book.volume_current
        elif hasattr(self._book,'volume'):
            self._is_position=True
            self.volume=self._book.volume
            self._swap = self._book.swap
        else:
            raise 'Unkown type!'
        
        self._book = ''
        return self

    def isBuy(self):
        if self._is_order: 
                return self._type in [mt5.ORDER_TYPE_BUY,mt5.ORDER_TYPE_BUY_LIMIT ,
                                    mt5.ORDER_TYPE_BUY_STOP ,mt5.ORDER_TYPE_BUY_STOP_LIMIT]
        elif self._is_position: 
                return self._type == mt5.POSITION_TYPE_BUY
        return True
    
    def _sendRequest(self, request):    
        result=mt5.order_send(request)    
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print('send request failed',result)
            return False
        return True

    def _changeOrderTPSL(self, tp=0.0,sl=0.0):
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": self._ticket,
            "price": self.price_open,
            "tp": tp,
            "sl": sl
        }
        return self._sendRequest(request)

    def _changePositionTPSL(self, tp=0.0,sl=0.0):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": self._ticket,
            "symbol": self.symbol,
            "tp": tp,
            "sl": sl
        }
        return self._sendRequest(request)

    def _changeTPSL(self, tp=0.0,sl=0.0):
        if self._is_order: 
            return self._changeOrderTPSL(tp,sl)
        elif self._is_position: 
            return self._changePositionTPSL(tp,sl)
        return False
    
    def _close_position(self):
        #https://www.mql5.com/ja/docs/constants/structures/mqltraderequest
        if self._type == mt5.ORDER_TYPE_BUY:
            price = mt5.symbol_info_tick(self.symbol).bid
            type_tmp = mt5.ORDER_TYPE_SELL
        elif self._type == mt5.ORDER_TYPE_SELL:
            price = mt5.symbol_info_tick(self.symbol).ask
            type_tmp = mt5.ORDER_TYPE_BUY
        else:
            raise ValueError('unknow position type(nor buy or sell) error.')

        deviation=20
        request={
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": type_tmp,
            "position": self._ticket,
            "price": price,
            "deviation": deviation,
            "magic": self.magic,
            "comment": "script close",
            "type_time": mt5.ORDER_TIME_GTC,
            #"type_filling": mt5.ORDER_FILLING_IOC,
        }
        return self._sendRequest(request)

    def _close_order(self):
        #https://www.mql5.com/en/forum/365968
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": self._ticket,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return self._sendRequest(request)

    def _make_order(self, profit_risk_ratio: float=None):
        # ProfitRiskRatio = self._ProfitRiskRatio
        # Determine order type and calculate stop loss based on parameters
        going_long = self.tp > self.price_open
        current_price_info = mt5.symbol_info_tick(self.symbol)
        if current_price_info is None:
            return f"Error getting current price for {self.symbol}"

        if going_long:
            current_price = current_price_info.ask
            order_type = mt5.ORDER_TYPE_BUY_STOP if self.price_open > current_price else mt5.ORDER_TYPE_BUY_LIMIT
        else:
            current_price = current_price_info.bid
            order_type = mt5.ORDER_TYPE_SELL_STOP if self.price_open < current_price else mt5.ORDER_TYPE_SELL_LIMIT

        if profit_risk_ratio is not None:
            self.sl = self.price_open + (self.price_open - self.tp) / profit_risk_ratio

        digitsnum = mt5.symbol_info(self.symbol).digits
        self.price_open,self.sl,self.tp = list(map(lambda x:round(x*10**digitsnum)/10**digitsnum,
                                                        [self.price_open,self.sl,self.tp]))
        # Prepare trade request
        deviation=20
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": order_type,
            "price": self.price_open,
            "sl": self.sl,
            "tp": self.tp,
            "deviation": deviation,
            "magic": self.magic,
            "comment": "auto order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        return self._sendRequest(request)        

@descriptions('Retrieve MT5 last N bars data in MetaTrader 5 terminal.',
            # account='MT5Account object for login.',
            # symbol='Financial instrument name (e.g., EURUSD).',
            # timeframe='Timeframe from which the bars are requested. {M1, H1, ...}',
            # # start_pos='Index of the first bar to retrieve.',
            # count='Number of bars to retrieve.'
            )

class MT5CopyLastRates(Model4LLMs.Function):
    account:MT5Account
    symbol:str
    timeframe:str
    count:int
    retry_times_on_error:int=3
    debug:bool=False

    _start_pos=0
    _digitsnum = {'AUDJPY':3,'CADJPY':3,'CHFJPY':3,'CNHJPY':3,'EURJPY':3,
                    'GBPJPY':3,'USDJPY':3,'NZDJPY':3,'XAUJPY':0,'JPN225':1,'US500':1}

    def __call__(self,*args):
        if self.debug:
            return '```USDJPY H4 OHLC\n\n142.520\n143.087\n142.382\n142.511\n\n142.509\n142.606\n142.068\n142.266\n\n142.173\n142.954\n142.128\n142.688\n\n142.687\n142.846\n142.080\n142.127\n\n142.127\n142.579\n141.643\n142.534\n\n142.537\n143.004\n142.406\n142.945\n\n142.949\n143.370\n142.746\n143.112\n\n143.112\n143.914\n142.940\n143.624\n\n143.624\n144.125\n143.369\n143.966\n\n143.966\n144.397\n143.661\n144.279\n\n144.277\n144.528\n143.699\n143.807\n\n143.808\n144.069\n143.561\n144.041\n\n144.039\n144.072\n142.972\n143.635\n\n143.634\n143.922\n143.326\n143.553\n\n143.547\n143.881\n143.423\n143.818\n\n143.817\n144.190\n143.561\n143.735\n\n143.733\n144.329\n143.532\n144.328\n\n144.327\n145.446\n144.076\n145.370\n\n145.370\n146.261\n145.298\n146.029\n\n146.030\n146.514\n145.967\n146.454\n\n146.454\n147.054\n146.258\n146.992\n\n146.993\n147.240\n146.676\n146.724\n\n146.723\n146.863\n146.301\n146.749\n\n146.749\n146.993\n146.517\n146.772\n\n146.778\n147.179\n146.470\n146.716\n\n146.716\n146.964\n146.578\n146.922\n\n146.922\n146.932\n146.617\n146.646\n\n146.645\n146.681\n146.152\n146.230\n\n146.230\n146.411\n145.917\n146.341\n\n146.342\n148.061\n146.340\n147.975\n```'
        manager = MT5Manager().get_singleton()
        manager.do(self)
        return manager.results.get(self.get_id(),[None])[-1]

    def run(self):
        symbol=self.symbol
        timeframe=self.timeframe
        count=self.count
        digitsnum = mt5.symbol_info(symbol).digits
        tf = {   'M1':mt5.TIMEFRAME_M1,
                        'M2':mt5.TIMEFRAME_M2,
                        'M3':mt5.TIMEFRAME_M3,
                        'M4':mt5.TIMEFRAME_M4,
                        'M5':mt5.TIMEFRAME_M5,
                        'M6':mt5.TIMEFRAME_M6,
                        'M10':mt5.TIMEFRAME_M10,
                        'M12':mt5.TIMEFRAME_M12,
                        'M12':mt5.TIMEFRAME_M12,
                        'M20':mt5.TIMEFRAME_M20,
                        'M30':mt5.TIMEFRAME_M30,
                        'H1':mt5.TIMEFRAME_H1,
                        'H2':mt5.TIMEFRAME_H2,
                        'H3':mt5.TIMEFRAME_H3,
                        'H4':mt5.TIMEFRAME_H4,
                        'H6':mt5.TIMEFRAME_H6,
                        'H8':mt5.TIMEFRAME_H8,
                        'H12':mt5.TIMEFRAME_H12,
                        'D1':mt5.TIMEFRAME_D1,
                        'W1':mt5.TIMEFRAME_W1,
                        'MN1':mt5.TIMEFRAME_MN1,
                    }.get(timeframe,mt5.TIMEFRAME_H1)
        # Retrieve the bar data from MetaTrader 5
        rates = mt5.copy_rates_from_pos(symbol, tf, self._start_pos, count)
        if rates is None:
            return None, mt5.last_error()  # Return error details if retrieval fails
        if digitsnum>0:
            return '\n'.join([f'```{symbol} {count} Open, High, Low, Close (OHLC) data points for the {timeframe} timeframe\n']+[f'{r[1]:.{digitsnum}f}\n{r[2]:.{digitsnum}f}\n{r[3]:.{digitsnum}f}\n{r[4]:.{digitsnum}f}\n' for r in rates]+['```'])
        else:
            return '\n'.join([f'```{symbol} {count} Open, High, Low, Close (OHLC) data points for the {timeframe} timeframe\n']+[f'{int(r[1])}\n{int(r[2])}\n{int(r[3])}\n{int(r[4])}\n' for r in rates]+['```'])


@descriptions('Create an MT5 order based on symbol, entry price, exit price.',
        Symbol='The financial instrument for the order (e.g., USDJPY).',
        EntryPrice='Price at which to enter the trade.',
        TakeProfitPrice='Price at which to take profit in the trade.',
        ProfitRiskRatio='The ratio of profit to risk.')
class MT5MakeOder(Model4LLMs.Function):
    account:MT5Account
    
    retry_times_on_error:int=3
    debug:bool=False
    
    _Symbol:str = ''
    _EntryPrice:float = None
    _TakeProfitPrice:float = None
    _ProfitRiskRatio: float = 2
    _volume: float = 0.01

    def __call__(self, Symbol: str, EntryPrice: float=None, TakeProfitPrice: float=None, ProfitRiskRatio: float=2.0):
        if EntryPrice is None and TakeProfitPrice is None:
            args = Symbol
            Symbol, EntryPrice, TakeProfitPrice, ProfitRiskRatio = [args[i] for i in ['Symbol', 'EntryPrice', 'TakeProfitPrice', 'ProfitRiskRatio']]
            
        if self.debug:
            return ['SUCESS!',Symbol, EntryPrice, TakeProfitPrice, ProfitRiskRatio]
        
        self._Symbol,self._EntryPrice,self._TakeProfitPrice,self._ProfitRiskRatio = Symbol, EntryPrice, TakeProfitPrice, ProfitRiskRatio
        manager = MT5Manager().get_singleton()
        manager.do(self)
        return manager.results.get(self.get_id(),[None])[-1]
        
    def run(self):
        # make book
        # book = Book()
        # MT5Manager().run(self.get_action(book))

        # ProfitRiskRatio = self._ProfitRiskRatio
        # Determine order type and calculate stop loss based on parameters
        Symbol, EntryPrice, TakeProfitPrice, ProfitRiskRatio = self._Symbol,self._EntryPrice,self._TakeProfitPrice,self._ProfitRiskRatio

        going_long = TakeProfitPrice > EntryPrice
        current_price_info = mt5.symbol_info_tick(Symbol)
        if current_price_info is None:
            return f"Error getting current price for {Symbol}"

        if going_long:
            current_price = current_price_info.ask
            order_type = mt5.ORDER_TYPE_BUY_STOP if EntryPrice > current_price else mt5.ORDER_TYPE_BUY_LIMIT
            stop_loss = EntryPrice - (TakeProfitPrice - EntryPrice) / ProfitRiskRatio
        else:
            current_price = current_price_info.bid
            order_type = mt5.ORDER_TYPE_SELL_STOP if EntryPrice < current_price else mt5.ORDER_TYPE_SELL_LIMIT
            stop_loss = EntryPrice + (EntryPrice - TakeProfitPrice) / ProfitRiskRatio

        digitsnum = mt5.symbol_info(Symbol).digits
        EntryPrice,stop_loss,TakeProfitPrice = list(map(lambda x:round(x*10**digitsnum)/10**digitsnum,
                                                        [EntryPrice,stop_loss,TakeProfitPrice]))
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": Symbol,
            "volume": self._volume,
            "type": order_type,
            "price": EntryPrice,
            "sl": stop_loss,
            "tp": TakeProfitPrice,
            "deviation": 20,
            "magic": 234000,
            "comment": "auto order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            res = f"Trade failure {result.retcode}"
            # return f"order_send failed, retcode={result.retcode}"
        else:
            res = "Trade successful"
            # return "Trade successful"
        return res

@descriptions('Retrieve MT5 active orders and positions.')
class MT5ActiveBooks(Model4LLMs.Function):
    account:MT5Account
    retry_times_on_error:int=3
    debug:bool=False
    def __call__(self):
        if self.debug:
            return []
        manager = MT5Manager().get_singleton()
        manager.do(self)
        return manager.results.get(self.get_id(),[None])[-1]
        
    def run(self):
        return Book().getBooks()
    # class OrdersGet(Function):
    #     description: str = 'Retrieve active MT5 orders with optional filters for symbol, group, or ticket.'
    #     _parameters_description = dict(
    #         symbol='Symbol name for specific orders.',
    #         group='Filter for grouping symbols.',
    #         ticket='Specific order ticket.'
    #     )

    #     def __call__(self, symbol: str = None, group: str = None, ticket: int = None):
    #         # Determine which mode of operation based on the parameters provided
    #         if symbol:
    #             orders = mt5.orders_get(symbol=symbol)
    #         elif group:
    #             orders = mt5.orders_get(group=group)
    #         elif ticket:
    #             orders = mt5.orders_get(ticket=ticket)
    #         else:
    #             orders = mt5.orders_get()

    #         if orders is None:
    #             return None, mt5.last_error()  # Return error details if orders retrieval fails

    #         return orders, None  # Return orders and None for error

    #     def __init__(self, *args, **kwargs):
    #         super(self.__class__, self).__init__(*args, **kwargs)
    #         self._extract_signature()
            
    # class PositionsGet(Function):
    #     description: str = 'Retrieve trading MT5 positions with optional filters for symbol, group, or ticket.'
    #     _parameters_description = dict(
    #         symbol='Symbol name for specific positions.',
    #         group='Filter for grouping symbols.',
    #         ticket='Specific position ticket.'
    #     )

    #     def __call__(self, symbol: str = None, group: str = None, ticket: int = None):
    #         # Determine which mode of operation based on the parameters provided
    #         if symbol:
    #             positions = mt5.positions_get(symbol=symbol)
    #         elif group:
    #             positions = mt5.positions_get(group=group)
    #         elif ticket:
    #             positions = mt5.positions_get(ticket=ticket)
    #         else:
    #             positions = mt5.positions_get()

    #         if positions is None:
    #             return None, mt5.last_error()  # Return error details if positions retrieval fails

    #         return positions, None  # Return positions and None for error

    #     def __init__(self, *args, **kwargs):
    #         super(self.__class__, self).__init__(*args, **kwargs)
    #         self._extract_signature()

    # class HistoryDealsGet(Function):
    #     description: str = 'Retrieve trade MT5 history within a specified time interval, with optional filters for symbol group, ticket, or position.'
    #     _parameters_description = dict(
    #         date_from='The starting date for requested trades."YYYY-MM-DD"',
    #         date_to='The ending date for requested trades."YYYY-MM-DD"',
    #         group='Filter for selecting trades by currency pair symbol group.',
    #         ticket='Ticket for specific order trades.',
    #         position='Ticket for specific position trades.'
    #     )

    #     def __call__(self, date_from: str, date_to: str, group: str = None, ticket: int = None, position: int = None):
    #         # Convert string dates to datetime objects
    #         hours = 3
    #         if isinstance(date_from, str):
    #             date_from = datetime.strptime(date_from, "%Y-%m-%d") + relativedelta.relativedelta(hours=hours)
    #         if isinstance(date_to, str):
    #             date_to = datetime.strptime(date_to, "%Y-%m-%d") + relativedelta.relativedelta(hours=hours)
    #         # Check for parameter type and existence
    #         if not all([isinstance(date_from, (datetime, int)), isinstance(date_to, (datetime, int))]):
    #             return None, "date_from and date_to must be datetime objects or integers."

    #         if group:
    #             deals = mt5.history_deals_get(date_from, date_to, group=group)
    #         elif ticket:
    #             deals = mt5.history_deals_get(ticket=ticket)
    #         elif position:
    #             deals = mt5.history_deals_get(position=position)
    #         else:
    #             deals = mt5.history_deals_get(date_from, date_to)

    #         if deals is None:
    #             return None, mt5.last_error()  # Return error details if deals retrieval fails

    #         return deals, None  # Return deals and None for error

    #     def __init__(self, *args, **kwargs):
    #         super(self.__class__, self).__init__(*args, **kwargs)
    #         self._extract_signature()

